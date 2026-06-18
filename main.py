"""
AcrylicAuto - main.py
写真 → AI背景除去 → 白版 → カットラインSVG → PDF → ダウンロード
Railway対応版
"""
from __future__ import annotations

import gc
import io
import logging
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
import svgwrite
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse
)
from PIL import Image

# ── ログ設定 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 定数 ─────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DPI               = 300
ALPHA_THRESH      = 10
CUTLINE_OFFSET_MM = 2.0
ALLOWED_EXT       = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
MAX_MB            = 50
MAX_INPUT_PX      = 1500  # メモリ節約のため入力画像をリサイズ

# ── rembg 遅延ロード ──────────────────────────────────────────
_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        logger.info("rembg モデルロード中 (isnet-general-use)…")
        from rembg import new_session
        _rembg_session = new_session("isnet-general-use")
        logger.info("rembg ロード完了")
    return _rembg_session

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="AcrylicAuto", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# 画像処理コア
# ══════════════════════════════════════════════════════════════

def load_and_resize(path: Path) -> np.ndarray:
    """画像読み込み＆メモリ節約リサイズ"""
    ext = path.suffix.lower()
    if ext == ".heic":
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ValueError("HEICにはpillow-heifが必要です")
        pil = Image.open(path).convert("RGBA")
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            pil = Image.open(path).convert("RGBA")
        else:
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGBA)
            elif raw.shape[2] == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
            elif raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            pil = Image.fromarray(raw.astype(np.uint8), "RGBA")

    # メモリ節約リサイズ
    pil.thumbnail((MAX_INPUT_PX, MAX_INPUT_PX), Image.LANCZOS)
    return np.array(pil)


def remove_bg(image: np.ndarray) -> np.ndarray:
    """AI背景除去"""
    from rembg import remove
    pil_in  = Image.fromarray(image.astype(np.uint8), "RGBA")
    pil_out = remove(pil_in, session=get_rembg_session())
    result  = np.array(pil_out.convert("RGBA"))
    del pil_in, pil_out
    gc.collect()

    if not np.any(result[:, :, 3] > ALPHA_THRESH):
        raise ValueError("被写体を検出できませんでした。別の写真をお試しください。")
    return result


def clean_alpha(alpha: np.ndarray) -> np.ndarray:
    """アルファマスクをクリーンアップ（ギザギザ除去・穴埋め）"""
    # 穴埋め（内側の隙間を塞ぐ）
    kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel15)

    # 細かい毛・フリンジを除去
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.erode(alpha, kernel5, iterations=2)

    # エッジなめらか化
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    _, alpha = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # 孤立ピクセル除去
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        alpha, connectivity=8
    )
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha = np.where(labels == largest, alpha, 0).astype(np.uint8)

    gc.collect()
    return alpha


def make_white(transparent: np.ndarray) -> np.ndarray:
    """白版生成"""
    mask  = transparent[:, :, 3] > ALPHA_THRESH
    white = np.zeros_like(transparent)
    white[mask] = [255, 255, 255, 255]
    return white


def make_cutline(transparent: np.ndarray, out_path: Path) -> Path:
    """カットラインSVG生成（外側2mm膨張）"""
    alpha    = transparent[:, :, 3].copy()
    h_px, w_px = alpha.shape

    _, binary = cv2.threshold(alpha, ALPHA_THRESH, 255, cv2.THRESH_BINARY)
    offset_px = round(CUTLINE_OFFSET_MM * DPI / 25.4)
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (offset_px * 2 + 1, offset_px * 2 + 1)
    )
    dilated = cv2.dilate(binary, kernel)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("輪郭を検出できませんでした。")

    contour = max(contours, key=cv2.contourArea)
    approx  = cv2.approxPolyDP(contour, 2.0, closed=True)

    px_per_mm = DPI / 25.4
    w_mm = w_px / px_per_mm
    h_mm = h_px / px_per_mm

    dwg = svgwrite.Drawing(
        str(out_path),
        size=(f"{w_mm:.3f}mm", f"{h_mm:.3f}mm"),
        viewBox=f"0 0 {w_mm:.3f} {h_mm:.3f}",
    )
    points = [
        (float(p[0][0]) / px_per_mm, float(p[0][1]) / px_per_mm)
        for p in approx
    ]
    dwg.add(dwg.polygon(
        points=points, stroke="red", stroke_width="0.1", fill="none"
    ))
    dwg.save()
    return out_path


def resize_to_mm(
    image: np.ndarray, width_mm: float, height_mm: float
) -> np.ndarray:
    """指定mmサイズにリサイズ"""
    w_px = round(width_mm * DPI / 25.4)
    h_px = round(height_mm * DPI / 25.4)
    return cv2.resize(image, (w_px, h_px), interpolation=cv2.INTER_LANCZOS4)


def make_preview_pdf(
    color_path: Path, white_path: Path, out_path: Path
) -> Path:
    """プレビューPDF生成"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas

        c = rl_canvas.Canvas(str(out_path), pagesize=A4)
        pw, ph = A4

        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, ph - 20 * mm, "AcrylicAuto - Preview")
        c.setFont("Helvetica", 9)
        c.drawString(20 * mm, ph - 30 * mm, f"DPI: {DPI}")

        # 元画像
        if color_path.exists():
            img = Image.open(str(color_path))
            iw, ih = img.size
            max_w, max_h = (pw - 40 * mm) / 2, (ph - 80 * mm) / 2
            ratio  = min(max_w / iw, max_h / ih)
            dw, dh = iw * ratio, ih * ratio
            c.drawString(20 * mm, ph - 45 * mm, "Color (BG Removed)")
            c.drawImage(
                str(color_path), 20 * mm, ph - 50 * mm - dh,
                width=dw, height=dh,
                preserveAspectRatio=True, mask="auto"
            )

        # 白版
        if white_path.exists():
            img = Image.open(str(white_path))
            iw, ih = img.size
            max_w, max_h = (pw - 40 * mm) / 2, (ph - 80 * mm) / 2
            ratio  = min(max_w / iw, max_h / ih)
            dw2, dh2 = iw * ratio, ih * ratio
            c.drawString(pw / 2 + 5 * mm, ph - 45 * mm, "White Layer")
            c.drawImage(
                str(white_path), pw / 2 + 5 * mm, ph - 50 * mm - dh2,
                width=dw2, height=dh2,
                preserveAspectRatio=True, mask="auto"
            )

        c.save()
    except Exception as e:
        logger.warning("PDF生成エラー: %s", e)
        out_path.write_bytes(b"%PDF-1.4")
    return out_path


def process(
    src_path: Path,
    job_id: str,
    width_mm: float = 55.0,
    height_mm: float = 55.0,
) -> dict[str, Path]:
    """メイン処理パイプライン"""
    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] ① 画像ロード", job_id)
    image = load_and_resize(src_path)

    logger.info("[%s] ② AI背景除去", job_id)
    transparent = remove_bg(image)
    del image
    gc.collect()

    logger.info("[%s] ③ エッジクリーンアップ", job_id)
    alpha_clean = clean_alpha(transparent[:, :, 3].copy())
    transparent[:, :, 3] = alpha_clean

    logger.info("[%s] ④ サイズ変換 (%.0f×%.0fmm)", job_id, width_mm, height_mm)
    transparent = resize_to_mm(transparent, width_mm, height_mm)

    color_path = out_dir / "color.png"
    Image.fromarray(transparent.astype(np.uint8), "RGBA").save(
        str(color_path), dpi=(DPI, DPI)
    )

    logger.info("[%s] ⑤ 白版生成", job_id)
    white = make_white(transparent)
    white_path = out_dir / "white.png"
    Image.fromarray(white.astype(np.uint8), "RGBA").save(
        str(white_path), dpi=(DPI, DPI)
    )
    del white
    gc.collect()

    logger.info("[%s] ⑥ カットライン生成", job_id)
    cutline_path = out_dir / "cutline.svg"
    make_cutline(transparent, cutline_path)
    del transparent
    gc.collect()

    logger.info("[%s] ⑦ プレビューPDF生成", job_id)
    pdf_path = out_dir / "preview.pdf"
    make_preview_pdf(color_path, white_path, pdf_path)

    logger.info("[%s] 完了", job_id)
    return {
        "color":   color_path,
        "white":   white_path,
        "cutline": cutline_path,
        "pdf":     pdf_path,
    }


# ══════════════════════════════════════════════════════════════
# エンドポイント
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path("templates/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>AcrylicAuto is running. POST /process to start.</h2>")


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    width_mm: float = 55.0,
    height_mm: float = 55.0,
):
    """画像処理エンドポイント"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, detail=f"未対応の形式: {ext}")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"{MAX_MB}MB以下にしてください")

    job_id   = uuid.uuid4().hex[:8].upper()
    src_path = UPLOAD_DIR / f"{job_id}{ext}"
    src_path.write_bytes(content)
    logger.info("受付: %s (%s, %dKB)", job_id, file.filename, len(content) // 1024)

    try:
        files = process(src_path, job_id, width_mm, height_mm)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("処理エラー [%s]", job_id)
        raise HTTPException(500, detail=f"処理エラー: {e}")

    return JSONResponse({
        "job_id":   job_id,
        "status":   "done",
        "width_mm":  width_mm,
        "height_mm": height_mm,
    })


@app.get("/preview/{job_id}/color")
def get_color(job_id: str):
    path = OUTPUT_DIR / job_id / "color.png"
    if not path.exists():
        raise HTTPException(404, detail="ファイルが見つかりません")
    return FileResponse(str(path), media_type="image/png")


@app.get("/preview/{job_id}/white")
def get_white(job_id: str):
    path = OUTPUT_DIR / job_id / "white.png"
    if not path.exists():
        raise HTTPException(404, detail="ファイルが見つかりません")
    return FileResponse(str(path), media_type="image/png")


@app.get("/preview/{job_id}/cutline")
def get_cutline(job_id: str):
    path = OUTPUT_DIR / job_id / "cutline.svg"
    if not path.exists():
        raise HTTPException(404, detail="ファイルが見つかりません")
    return FileResponse(str(path), media_type="image/svg+xml")


@app.get("/preview/{job_id}/pdf")
def get_pdf(job_id: str):
    path = OUTPUT_DIR / job_id / "preview.pdf"
    if not path.exists():
        raise HTTPException(404, detail="ファイルが見つかりません")
    return FileResponse(
        str(path), media_type="application/pdf",
        filename=f"acrylic_{job_id}_preview.pdf"
    )


@app.get("/preview/{job_id}/download")
def download_zip(job_id: str):
    """全ファイルをZIPでダウンロード"""
    out_dir = OUTPUT_DIR / job_id
    if not out_dir.exists():
        raise HTTPException(404, detail="ジョブが見つかりません")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["color.png", "white.png", "cutline.svg", "preview.pdf"]:
            fpath = out_dir / fname
            if fpath.exists():
                zf.write(fpath, fname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=acrylic_{job_id}.zip"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
