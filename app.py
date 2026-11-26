from io import BytesIO
from pathlib import Path
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pymupdf
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Paths & Defaults
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
INPUT_DIR = APP_DIR / "input"
OUTPUT_DIR = APP_DIR / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PREVIEW_HEIGHT = 320
DEFAULT_GRID_COLS = 3
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

FORMAT_SPECS = {
    "JPG": ("JPEG", ".jpg", "image/jpeg"),
    "PNG": ("PNG", ".png", "image/png"),
    "WEBP": ("WEBP", ".webp", "image/webp"),
}


def sanitize_stem(raw_name: str, fallback: str) -> str:
    """Clean user-entered filename stem so saving and downloading never fail."""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", raw_name.strip())
    cleaned = Path(cleaned).stem.strip(" .")
    return cleaned if cleaned else fallback


def pdf_bytes_to_pages(pdf_bytes: bytes, source_label: str, dpi: int = 250) -> list[tuple[str, Image.Image]]:
    """Render each page of a PDF into an RGB PIL image."""
    pages: list[tuple[str, Image.Image]] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:
                rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                rgb = img_np.copy()
            page_label = f"{source_label} [p.{page_idx + 1}]"
            pages.append((page_label, Image.fromarray(rgb)))
    finally:
        doc.close()
    return pages


def orient_image(pil_img: Image.Image, rot_deg: int, flip_h: bool, flip_v: bool) -> Image.Image:
    """Apply clockwise rotation (0/90/180/270) and optional horizontal/vertical flip."""
    out = pil_img.copy()
    deg = rot_deg % 360
    if deg == 90:
        out = out.rotate(-90, expand=True)
    elif deg == 180:
        out = out.rotate(180, expand=True)
    elif deg == 270:
        out = out.rotate(90, expand=True)
    if flip_h:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return out


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Extracting PDF pages...")
def load_pdf_pages_cached(pdf_bytes: bytes, source_label: str, dpi: int) -> list[tuple[str, Image.Image]]:
    """Cache rendered PDF pages so changing range or clicking buttons does not re-render the PDF."""
    return pdf_bytes_to_pages(pdf_bytes, source_label, dpi=dpi)


@st.cache_data(show_spinner="Loading input folder items...")
def load_local_input_cached(input_dir_str: str, pdf_dpi: int) -> list[tuple[str, list[tuple[str, bytes]]]]:
    """Collect image paths and PDF raw bytes from input/ without re-decoding on every rerun."""
    root = Path(input_dir_str)
    collected: list[tuple[str, list[tuple[str, bytes]]]] = []
    if not root.exists():
        return collected
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext_lower = p.suffix.lower()
        rel_label = str(p.relative_to(root))
        if ext_lower in SUPPORTED_IMAGE_EXTS:
            collected.append(("img", [(rel_label, p.read_bytes())]))
        elif ext_lower == ".pdf":
            collected.append(("pdf", [(rel_label, p.read_bytes())]))
    return collected


# 2. Backend Dynamic Analysis
# ---------------------------------------------------------------------------
def analyze_image(pil_img: Image.Image) -> dict[str, float | bool]:
    """Inspect tone, saturation, shadows, and detail to recommend balanced settings."""
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean_sat = float(np.mean(hsv[:, :, 1]))
    high_sat_ratio = float(np.mean(hsv[:, :, 1] > 65.0))
    mean_val = float(np.mean(hsv[:, :, 2]))
    dark_ratio = float(np.mean(gray < 55.0))
    contrast_std = float(np.std(gray))
    edge_sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    texture_var = float(np.var(gray))

    is_sketch = mean_sat < 28.0 and high_sat_ratio < 0.08

    if is_sketch:
        rec_sat = 1.02
        rec_vibrance = 1.03
        rec_contrast = 1.14 if contrast_std < 55.0 else 1.08
        rec_sharpness = 1.20 if edge_sharpness < 450.0 else 1.10
        rec_gamma = 0.98 if mean_val < 165.0 else 1.00
        rec_brightness = 1.08 if mean_val < 165.0 else 1.03
        rec_warmth = 0.04
        rec_shadow_lift = 0.16 if dark_ratio > 0.10 else 0.08
        rec_clarity = 1.15
        rec_bg_pop = 1.05
        rec_grain = False
    else:
        if mean_sat < 65.0:
            rec_sat = 1.20
            rec_vibrance = 1.22
        elif mean_sat < 105.0:
            rec_sat = 1.12
            rec_vibrance = 1.16
        else:
            rec_sat = 1.05
            rec_vibrance = 1.10

        rec_contrast = 1.12 if contrast_std < 50.0 else (1.05 if contrast_std > 75.0 else 1.09)
        rec_sharpness = 1.15 if edge_sharpness < 350.0 else 1.08
        rec_gamma = 0.94 if mean_val < 150.0 else 0.97
        rec_brightness = 1.08 if mean_val < 140.0 else 1.04
        rec_warmth = 0.05
        rec_shadow_lift = 0.20 if dark_ratio > 0.08 else 0.10
        rec_clarity = 1.12
        rec_bg_pop = 1.18
        rec_grain = texture_var < 1100.0

    return {
        "sat": round(rec_sat, 2),
        "vibrance": round(rec_vibrance, 2),
        "contrast": round(rec_contrast, 2),
        "sharpness": round(rec_sharpness, 2),
        "gamma": round(rec_gamma, 2),
        "brightness": round(rec_brightness, 2),
        "warmth": round(rec_warmth, 2),
        "shadow_lift": round(rec_shadow_lift, 2),
        "clarity": round(rec_clarity, 2),
        "bg_pop": round(rec_bg_pop, 2),
        "grain": rec_grain,
    }


# ---------------------------------------------------------------------------
import json as _json

PRESETS_FILE = APP_DIR / "custom_presets.json"


def load_custom_presets() -> dict[str, dict[str, float | str]]:
    """Load user-defined presets from custom_presets.json."""
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = _json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_custom_presets(presets: dict[str, dict[str, float | str]]) -> None:
    """Persist user-defined presets to custom_presets.json."""
    PRESETS_FILE.write_text(_json.dumps(presets, indent=2), encoding="utf-8")


MEDIUM_PROFILES = [
    "Auto-Detect",
    "Pastel",
    "Pencil",
    "Digital (Krita)",
    "Sketch Pen / Marker",
]


def resolve_medium_and_base(pil_img: Image.Image, chosen_medium: str) -> tuple[str, dict[str, float | bool]]:
    """Return effective medium name and tailored baseline knobs without altering Pastel defaults."""
    base = analyze_image(pil_img)
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean_sat = float(np.mean(hsv[:, :, 1]))
    high_sat_ratio = float(np.mean(hsv[:, :, 1] > 65.0))
    mean_val = float(np.mean(hsv[:, :, 2]))
    contrast_std = float(np.std(gray))
    edge_sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    if chosen_medium == "Auto-Detect":
        effective_medium = "Pencil" if (mean_sat < 28.0 and high_sat_ratio < 0.08) else "Pastel"
    else:
        effective_medium = chosen_medium

    if effective_medium == "Pastel":
        return effective_medium, enhance_pastel_backend(pil_img, base)

    if effective_medium == "Pencil":
        rec = dict(base)
        rec["sat"] = 1.00
        rec["vibrance"] = 1.00
        rec["bg_pop"] = 1.00
        rec["warmth"] = 0.02 if mean_val > 195.0 else 0.0
        rec["contrast"] = round(1.26 if contrast_std < 55.0 else 1.18, 2)
        rec["sharpness"] = round(1.32 if edge_sharpness < 500.0 else 1.20, 2)
        rec["clarity"] = round(1.24 if edge_sharpness < 500.0 else 1.14, 2)
        rec["gamma"] = round(0.95 if mean_val < 165.0 else 0.98, 2)
        rec["brightness"] = round(1.10 if mean_val < 170.0 else 1.04, 2)
        rec["shadow_lift"] = round(0.22 if contrast_std < 50.0 else 0.14, 2)
        rec["grain"] = False
        return effective_medium, rec

    if effective_medium == "Digital (Krita)":
        rec = dict(base)
        rec["sat"] = round(min(1.08, float(base["sat"])), 2)
        rec["vibrance"] = round(min(1.08, float(base["vibrance"])), 2)
        rec["contrast"] = round(1.05 if contrast_std < 55.0 else 1.02, 2)
        rec["sharpness"] = round(1.06 if edge_sharpness < 400.0 else 1.02, 2)
        rec["clarity"] = 1.05
        rec["gamma"] = 1.00
        rec["brightness"] = 1.01
        rec["shadow_lift"] = 0.04
        rec["bg_pop"] = 1.05
        return effective_medium, rec

    # Sketch Pen / Marker
    rec = dict(base)
    rec["sat"] = round(1.14 if mean_sat < 85.0 else 1.06, 2)
    rec["vibrance"] = round(1.14 if mean_sat < 85.0 else 1.06, 2)
    rec["contrast"] = round(1.22 if contrast_std < 60.0 else 1.14, 2)
    rec["sharpness"] = round(1.28 if edge_sharpness < 550.0 else 1.16, 2)
    rec["clarity"] = 1.20
    rec["gamma"] = 0.96
    rec["brightness"] = round(1.07 if mean_val < 165.0 else 1.02, 2)
    rec["shadow_lift"] = 0.12
    rec["bg_pop"] = 1.10
    rec["grain"] = False
    return effective_medium, rec


import hashlib as _hashlib

CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_item_cache_token(raw_img: Image.Image, rot_val: int, fh_val: bool, fv_val: bool, sig: tuple) -> str:
    """Create a deterministic hash key for disk & session caching."""
    buf = BytesIO()
    raw_img.save(buf, format="PNG")
    h = _hashlib.sha256()
    h.update(buf.getvalue())
    h.update(f"|rot={rot_val}|fh={fh_val}|fv={fv_val}|sig={repr(sig)}".encode("utf-8"))
    return h.hexdigest()


def load_disk_pair(token: str) -> tuple[Image.Image, Image.Image] | None:
    """Load cached (oriented_orig, edited_img) pair from disk if present."""
    orig_p = CACHE_DIR / f"{token}_orig.png"
    edit_p = CACHE_DIR / f"{token}_edit.png"
    if orig_p.exists() and edit_p.exists():
        try:
            with Image.open(orig_p) as o_im, Image.open(edit_p) as e_im:
                return o_im.convert("RGB"), e_im.convert("RGB")
        except Exception:
            return None
    return None


def save_disk_pair(token: str, orig_im: Image.Image, edit_im: Image.Image) -> None:
    """Persist (oriented_orig, edited_img) pair to disk cache."""
    try:
        orig_im.save(CACHE_DIR / f"{token}_orig.png", format="PNG", optimize=False)
        edit_im.save(CACHE_DIR / f"{token}_edit.png", format="PNG", optimize=False)
    except Exception:
        pass


def auto_trim_border(pil_img: Image.Image, pad_px: int = 8) -> Image.Image:
    """Trim uniform bright/scan margins around artwork while keeping a small margin."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Detect non-near-white pixels (threshold < 242)
    mask = gray < 242
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return pil_img
    h, w = gray.shape
    x0 = max(0, int(xs.min()) - pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    x1 = min(w, int(xs.max()) + pad_px + 1)
    y1 = min(h, int(ys.max()) + pad_px + 1)
    if (x1 - x0) < int(w * 0.25) or (y1 - y0) < int(h * 0.25):
        return pil_img
    return pil_img.crop((x0, y0, x1, y1))


def apply_vignette(pil_img: Image.Image, strength: float) -> Image.Image:
    """Apply a soft radial vignette (strength in [0.0, 1.0])."""
    if strength <= 0.001:
        return pil_img
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2.0) / max(1.0, w / 2.0)
    ny = (yy - h / 2.0) / max(1.0, h / 2.0)
    dist = np.sqrt(nx * nx + ny * ny)
    vig = 1.0 - strength * np.clip((dist - 0.35) / 0.95, 0.0, 1.0) ** 1.7
    arr *= vig[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def enhance_pastel_backend(pil_img: Image.Image, base_rec: dict[str, float | bool]) -> dict[str, float | bool]:
    """Refine dynamic backend recommendations specifically for pastel artwork."""
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))
    highlight_ratio = float(np.mean(gray > 225.0))
    contrast_std = float(np.std(gray))

    rec = dict(base_rec)
    # Lift soft pastel colors cleanly while keeping high-saturation areas natural
    if mean_sat < 70.0:
        rec["sat"] = round(min(1.32, float(rec["sat"]) + 0.06), 2)
        rec["vibrance"] = round(min(1.34, float(rec["vibrance"]) + 0.06), 2)
    else:
        rec["sat"] = round(min(1.22, float(rec["sat"]) + 0.03), 2)
        rec["vibrance"] = round(min(1.26, float(rec["vibrance"]) + 0.04), 2)

    # Gentle midtone contrast and clarity for pastel tooth/texture
    rec["contrast"] = round(1.14 if contrast_std < 55.0 else 1.09, 2)
    rec["clarity"] = round(1.16 if contrast_std < 65.0 else 1.10, 2)
    rec["bg_pop"] = round(min(1.26, float(rec["bg_pop"]) + 0.04), 2)

    # Protect blown highlights on bright pastel scans
    if highlight_ratio > 0.12:
        rec["brightness"] = round(max(0.98, float(rec["brightness"]) - 0.03), 2)
        rec["gamma"] = round(min(1.03, float(rec["gamma"]) + 0.02), 2)
    elif mean_val < 155.0:
        rec["brightness"] = round(min(1.14, float(rec["brightness"]) + 0.03), 2)

    return rec


# 3. Backend Editing Pipeline
# ---------------------------------------------------------------------------
def apply_art_edit(
    pil_img: Image.Image,
    sat_boost_max: float,
    contrast_factor: float,
    sharpness_factor: float,
    gamma: float,
    brightness_factor: float,
    warmth_shift: float,
    vibrance_factor: float,
    shadow_lift: float,
    clarity_factor: float,
    bg_pop_factor: float,
    add_subtle_grain: bool,
) -> Image.Image:
    """Apply natural brightness, skin-friendly warm tones, background pop, and detail."""
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

    if shadow_lift > 0.001:
        shadow_mask = np.power(np.clip(1.0 - rgb, 0.0, 1.0), 2.0)
        rgb = np.clip(rgb + shadow_lift * 0.28 * shadow_mask, 0.0, 1.0)

    if abs(warmth_shift) > 0.001:
        shift = warmth_shift * 0.05
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + shift, 0.0, 1.0)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] - shift, 0.0, 1.0)

    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    if abs(gamma - 1.0) > 0.001:
        inv_gamma = 1.0 / max(0.3, gamma)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        rgb_u8 = cv2.LUT(rgb_u8, lut)

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    mean_sat = float(np.mean(s))

    skin_like = (h >= 2.0) & (h <= 24.0) & (s >= 18.0) & (s <= 155.0) & (v >= 95.0)
    bg_mask = ~skin_like

    if vibrance_factor > 1.0:
        sat_norm = np.clip(s / 255.0, 0.0, 1.0)
        vib_gain = (1.0 - sat_norm * 0.75) * (vibrance_factor - 1.0)
        s = np.clip(s * (1.0 + vib_gain), 0.0, 255.0)

    if mean_sat < 115.0 and sat_boost_max > 1.0:
        deficit = max(0.0, (115.0 - mean_sat) / 115.0)
        scale = 1.0 + deficit * (sat_boost_max - 1.0)
        s[skin_like] = np.clip(s[skin_like] * (1.0 + (scale - 1.0) * 0.55), 0.0, 255.0)
        s[bg_mask] = np.clip(s[bg_mask] * scale, 0.0, 255.0)

    if abs(bg_pop_factor - 1.0) > 0.001:
        s[bg_mask] = np.clip(s[bg_mask] * bg_pop_factor, 0.0, 255.0)
        v[bg_mask] = np.clip(v[bg_mask] * (1.0 + (bg_pop_factor - 1.0) * 0.35), 0.0, 255.0)

    rgb_f = cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if abs(clarity_factor - 1.0) > 0.001:
        blurred = cv2.GaussianBlur(np.clip(rgb_f, 0, 255).astype(np.uint8), (0, 0), sigmaX=5.0).astype(np.float32)
        rgb_f = np.clip(rgb_f + (rgb_f - blurred) * ((clarity_factor - 1.0) * 0.65), 0.0, 255.0)

    out_pil = Image.fromarray(np.clip(rgb_f, 0, 255).astype(np.uint8))
    out_pil = ImageEnhance.Brightness(out_pil).enhance(brightness_factor)
    out_pil = ImageEnhance.Contrast(out_pil).enhance(contrast_factor)
    out_pil = ImageEnhance.Sharpness(out_pil).enhance(sharpness_factor)

    if add_subtle_grain:
        arr = np.array(out_pil, dtype=np.float32)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if float(np.var(gray)) < 1100.0:
            noise = np.random.normal(0, 1.5, arr.shape).astype(np.float32)
            out_pil = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    return out_pil


def proportional_preview(img: Image.Image, max_height: int) -> Image.Image:
    """Scale proportionally to max_height without stretching or distorting aspect ratio."""
    h_limit = max(140, int(max_height))
    if img.height <= h_limit or img.height <= 0:
        return img.copy()
    scale = h_limit / float(img.height)
    new_w = max(40, int(round(img.width * scale)))
    return img.resize((new_w, h_limit), Image.Resampling.LANCZOS)


def encode_output_image(img: Image.Image, fmt_label: str, quality: int) -> tuple[bytes, str, str]:
    """Encode PIL image into JPG, PNG, or WEBP bytes and return (bytes, ext, mime)."""
    pil_fmt, ext, mime = FORMAT_SPECS.get(fmt_label, FORMAT_SPECS["JPG"])
    buf = BytesIO()
    rgb_img = img.convert("RGB")
    if pil_fmt == "PNG":
        rgb_img.save(buf, format="PNG", optimize=True)
    elif pil_fmt == "WEBP":
        rgb_img.save(buf, format="WEBP", quality=quality, method=4)
    else:
        rgb_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), ext, mime


def build_default_stem(label: str, index_1based: int, rename_mode: str, prefix: str, start_num: int) -> str:
    """Compute default output filename stem from the chosen batch naming rule."""
    num_str = str(start_num + index_1based - 1)
    clean_prefix = sanitize_stem(prefix, "")
    if rename_mode == "Numbers Only (1, 2, 3...)":
        return num_str
    if rename_mode == "Prefix + Number":
        return f"{clean_prefix}_{num_str}" if clean_prefix else num_str
    base = label.replace(" [p.", "_p").replace("]", "")
    return sanitize_stem(base, num_str)


# ---------------------------------------------------------------------------
# 4. Streamlit Web UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Art Studio", layout="wide")
st.title("Art Studio")
st.caption("Edit images or multi-page PDFs, fix page orientation, select a range for large batches, and track live progress.")

with st.sidebar:
    st.header("1. Source")
    source_mode = st.radio(
        "Input source",
        ["Local Folder (`input/`)", "Upload Files (Browser / Cloud)"],
    )

    st.header("2. Medium & Main Knobs")
    custom_presets_store = load_custom_presets()
    custom_preset_names = [f"Custom: {k}" for k in sorted(custom_presets_store.keys())]
    all_preset_options = MEDIUM_PROFILES + custom_preset_names
    default_preset_idx = all_preset_options.index("Pastel") if "Pastel" in all_preset_options else 0
    medium_preset = st.selectbox("Medium / Custom preset", all_preset_options, index=default_preset_idx)
    knob_mode = st.radio(
        "Slider behavior",
        ["Fine-Tune Per-Image Auto", "Direct Values for All"],
    )
    is_fine_tune = knob_mode.startswith("Fine-Tune")

    if is_fine_tune:
        st.caption("1.00 = keep each image's automatic backend recommendation.")
        sat_slider = st.slider("Saturation Boost", 0.70, 1.35, 1.00, 0.02)
        contrast_slider = st.slider("Contrast", 0.75, 1.35, 1.00, 0.02)
        sharpness_slider = st.slider("Sharpness", 0.75, 1.35, 1.00, 0.02)
        gamma_slider = st.slider("Gamma", 0.75, 1.25, 1.00, 0.02)
    else:
        st.caption("Applies these exact values to every image.")
        sat_slider = st.slider("Saturation Boost", 1.00, 1.60, 1.12, 0.02)
        contrast_slider = st.slider("Contrast", 0.80, 1.50, 1.09, 0.02)
        sharpness_slider = st.slider("Sharpness", 0.75, 1.75, 1.15, 0.02)
        gamma_slider = st.slider("Gamma", 0.75, 1.30, 0.96, 0.02)

    st.header("3. Layout")
    preview_height = st.slider("Preview Height (px)", 160, 720, DEFAULT_PREVIEW_HEIGHT, 20)
    grid_cols = st.slider("Artworks per Row", 1, 4, DEFAULT_GRID_COLS, 1)
    fast_preview = st.checkbox("Fast Preview (faster screen rendering)", value=True)

tab_editor, tab_scheduler = st.tabs(["1. Batch Edit & Enhance", "2. Feed Scheduler & Mixer"])

with tab_editor:
    top_c1, top_c2, top_c3, top_c4 = st.columns([1.2, 1.6, 1.2, 1.6])
    with top_c1:
        output_format = st.selectbox("Output Format", ["JPG", "PNG", "WEBP"], index=0)
    with top_c2:
        if output_format in {"JPG", "WEBP"}:
            quality_val = st.slider(f"{output_format} Quality", 75, 100, 95, 1)
        else:
            quality_val = 95
            st.caption("PNG saves losslessly without a quality slider.")
    with top_c3:
        pdf_dpi = st.selectbox("PDF Scan DPI", [150, 250, 300], index=1)
    with top_c4:
        if source_mode.startswith("Local Folder"):
            save_to_output = st.checkbox(f"Save edited files to output/ ({output_format})", value=False)
        else:
            save_to_output = False

    st.markdown("#### Batch Rename All Files")
    ren_c1, ren_c2, ren_c3, ren_c4 = st.columns([1.8, 1.5, 1.0, 1.4])
    with ren_c1:
        bulk_rename_mode = st.selectbox(
            "Bulk naming rule",
            ["Keep File / Page Name", "Numbers Only (1, 2, 3...)", "Prefix + Number"],
            index=2,
        )
    with ren_c2:
        bulk_prefix = st.text_input(
            "Prefix (for Prefix + Number)",
            value="art",
            disabled=(bulk_rename_mode != "Prefix + Number"),
        )
    with ren_c3:
        bulk_start_num = st.number_input(
            "Start #",
            min_value=1,
            max_value=9999,
            value=1,
            step=1,
            disabled=(bulk_rename_mode == "Keep File / Page Name"),
        )
    with ren_c4:
        allow_individual_override = st.checkbox("Allow per-card name box override", value=False)

    loaded_items: list[tuple[str, Image.Image]] = []

    if source_mode.startswith("Local Folder"):
        raw_entries = load_local_input_cached(str(INPUT_DIR), int(pdf_dpi))
        for kind, payload in raw_entries:
            label, data_bytes = payload[0]
            if kind == "img":
                loaded_items.append((label, Image.open(BytesIO(data_bytes)).convert("RGB")))
            elif kind == "pdf":
                loaded_items.extend(load_pdf_pages_cached(data_bytes, label, int(pdf_dpi)))
    else:
        uploads = st.file_uploader(
            "Upload artwork images or multi-page PDF scans",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            for uf in uploads:
                ext_lower = Path(uf.name).suffix.lower()
                raw_bytes = uf.getvalue()
                if ext_lower == ".pdf":
                    loaded_items.extend(load_pdf_pages_cached(raw_bytes, uf.name, int(pdf_dpi)))
                else:
                    loaded_items.append((uf.name, Image.open(BytesIO(raw_bytes)).convert("RGB")))

    total_found = len(loaded_items)
    if total_found == 0:
        st.info("Place images or PDFs in `input/` (including any subfolders), or switch to **Upload Files**.")
        st.stop()

    st.info(f"**Total items detected:** {total_found} image(s)/page(s).")
    rng_c1, rng_c2, rng_c3 = st.columns([1.2, 1.2, 2.6])
    with rng_c1:
        start_idx = int(st.number_input("Start from #", min_value=1, max_value=total_found, value=1, step=1))
    with rng_c2:
        default_end = min(total_found, 5)
        end_idx = int(st.number_input("End at #", min_value=start_idx, max_value=total_found, value=default_end, step=1))
    with rng_c3:
        st.write("")
        run_clicked = st.button("Process Selected Range", type="primary")

    if "active_range" not in st.session_state:
        st.session_state["active_range"] = None

    if run_clicked:
        st.session_state["active_range"] = (start_idx, end_idx)

    active_range = st.session_state["active_range"]
    if active_range is None:
        st.caption(f"Set **Start from #** and **End at #** (1 to {total_found}), then click **Process Selected Range**.")
        st.stop()

    run_start, run_end = active_range
    run_start = max(1, min(run_start, total_found))
    run_end = max(run_start, min(run_end, total_found))
    selected_items = loaded_items[run_start - 1 : run_end]

    with st.expander("More Settings, Extra Sliders & Dynamic Backend Table", expanded=False):
        st.markdown("#### Extra Optional Sliders")
        ex_c1, ex_c2, ex_c3 = st.columns(3)
        with ex_c1:
            brightness_extra = st.slider("Brightness Lift", 0.80, 1.30, 1.00, 0.02)
            warmth_extra = st.slider("Warmth Shift", -0.50, 0.50, 0.00, 0.02)
        with ex_c2:
            vibrance_extra = st.slider("Vibrance Multiplier", 0.75, 1.35, 1.00, 0.02)
            clarity_extra = st.slider("Clarity Multiplier", 0.75, 1.35, 1.00, 0.02)
        with ex_c3:
            shadow_extra = st.slider("Shadow Lift Multiplier", 0.50, 1.60, 1.00, 0.05)
            bg_pop_extra = st.slider("Background Pop Multiplier", 0.80, 1.40, 1.00, 0.02)

        tint_extra = st.slider("Color Tint Shift (- cool/green, + warm/magenta)", -0.50, 0.50, 0.00, 0.02)

        grain_choice = st.selectbox(
            "Subtle Texture (smooth scans)",
            ["Per-Image Auto", "Always On", "Always Off"],
            index=0, key="grain_choice_select")

        f_c1, f_c2 = st.columns(2)
        with f_c1:
            trim_borders = st.checkbox("Auto-Trim Scan Borders", value=False, key="trim_borders_chk")
        with f_c2:
            vignette_strength = st.slider("Soft Vignette Strength", 0.00, 0.60, 0.00, 0.02, key="vignette_strength_slider")

        if "edit_cache" in st.session_state and st.session_state["edit_cache"]:
            st.markdown("#### Live Preview While Adjusting Extra Sliders")
            lp_show = st.checkbox("Show live preview right here while tweaking extra sliders", value=True, key="live_slider_preview_chk")
            if lp_show:
                st.caption("Updates as you move the extra sliders; click **Process Selected Range** if you change range or orientation.")
                lp_cols = st.columns(min(3, max(1, grid_cols)), gap="medium")
                preview_pairs: list[tuple[str, Image.Image, Image.Image]] = []
                for p_off, (p_lbl, p_raw) in enumerate(selected_items[: len(lp_cols)]):
                    p_num = run_start + p_off
                    p_rot = int(st.session_state.get(f"rot_{p_num}", 0))
                    p_fh = bool(st.session_state.get(f"fh_{p_num}", False))
                    p_fv = bool(st.session_state.get(f"fv_{p_num}", False))
                    p_ori = orient_image(p_raw, rot_deg=p_rot, flip_h=p_fh, flip_v=p_fv)
                    if medium_preset.startswith("Custom: "):
                        c_name = medium_preset.removeprefix("Custom: ")
                        c_cfg = custom_presets_store.get(c_name, {})
                        base_med = str(c_cfg.get("base_medium", "Pastel"))
                        _, p_rec = resolve_medium_and_base(p_ori, base_med)
                        p_mode = str(c_cfg.get("apply_mode", "Override Backend"))
                        for k_m, c_k in [
                            ("sat", "sat_boost"),
                            ("contrast", "contrast"),
                            ("sharpness", "sharpness"),
                            ("gamma", "gamma"),
                            ("brightness", "brightness_lift"),
                            ("warmth", "warmth_shift"),
                            ("vibrance", "vibrance_mult"),
                            ("clarity", "clarity_mult"),
                            ("shadow_lift", "shadow_lift_mult"),
                            ("bg_pop", "bg_pop_mult"),
                        ]:
                            if c_k in c_cfg:
                                v_c = float(c_cfg[c_k])
                                p_rec[k_m] = round(float(p_rec[k_m]) * v_c, 3) if p_mode == "Scale Over Backend" else v_c
                    else:
                        _, p_rec = resolve_medium_and_base(p_ori, medium_preset)

                    e_sat = (float(p_rec["sat"]) * sat_slider) if is_fine_tune else sat_slider
                    e_con = (float(p_rec["contrast"]) * contrast_slider) if is_fine_tune else contrast_slider
                    e_shp = (float(p_rec["sharpness"]) * sharpness_slider) if is_fine_tune else sharpness_slider
                    e_gam = (float(p_rec["gamma"]) * gamma_slider) if is_fine_tune else gamma_slider
                    e_brt = float(p_rec["brightness"]) * brightness_extra
                    e_wrm = float(p_rec["warmth"]) + warmth_extra
                    e_vib = float(p_rec["vibrance"]) * vibrance_extra
                    e_clr = float(p_rec["clarity"]) * clarity_extra
                    e_shd = float(p_rec["shadow_lift"]) * shadow_extra
                    e_bgp = float(p_rec["bg_pop"]) * bg_pop_extra
                    e_grn = True if grain_choice == "Always On" else (False if grain_choice == "Always Off" else bool(p_rec["grain"]))

                    p_ed = apply_art_edit(
                        p_ori,
                        sat_boost_max=e_sat,
                        contrast_factor=e_con,
                        sharpness_factor=e_shp,
                        gamma=e_gam,
                        brightness_factor=e_brt,
                        warmth_shift=e_wrm,
                        vibrance_factor=e_vib,
                        shadow_lift=e_shd,
                        clarity_factor=e_clr,
                        bg_pop_factor=e_bgp,
                        add_subtle_grain=e_grn,
                    )
                    if abs(tint_extra) > 0.001:
                        t_a = np.array(p_ed.convert("RGB"), dtype=np.float32)
                        t_d = tint_extra * 14.0
                        t_a[:, :, 0] = np.clip(t_a[:, :, 0] + t_d, 0, 255)
                        t_a[:, :, 1] = np.clip(t_a[:, :, 1] - t_d * 0.8, 0, 255)
                        t_a[:, :, 2] = np.clip(t_a[:, :, 2] + t_d * 0.6, 0, 255)
                        p_ed = Image.fromarray(t_a.astype(np.uint8))
                    if trim_borders:
                        p_ori = auto_trim_border(p_ori)
                        p_ed = auto_trim_border(p_ed)
                    if vignette_strength > 0.001:
                        p_ed = apply_vignette(p_ed, vignette_strength)
                    preview_pairs.append((f"#{p_num} {p_lbl}", p_ori, p_ed))

                for idx_p, (p_title, p_o, p_e) in enumerate(preview_pairs):
                    with lp_cols[idx_p]:
                        st.caption(p_title)
                        sub_l, sub_r = st.columns(2)
                        with sub_l:
                            st.image(proportional_preview(p_o, 180), caption="Orig")
                        with sub_r:
                            st.image(proportional_preview(p_e, 180), caption="Live Edit")




        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("Clear Disk & Session Cache", key="clear_all_cache_btn"):
                st.session_state["edit_cache"] = {}
                for cf in CACHE_DIR.glob("*.png"):
                    cf.unlink(missing_ok=True)
                st.success("Cleared disk and session cache.")
                st.rerun()
        with c_act2:
            st.caption(f"Cached PNG pairs on disk: {len(list(CACHE_DIR.glob('*_edit.png')))}")

        st.markdown("#### Per-Image Dynamic Backend Settings")
        backend_rows = []
        for offset, (item_label, raw_item_img) in enumerate(selected_items):
            p_num = run_start + offset
            p_rot = int(st.session_state.get(f"rot_{p_num}", 0))
            p_fh = bool(st.session_state.get(f"fh_{p_num}", False))
            p_fv = bool(st.session_state.get(f"fv_{p_num}", False))
            p_oriented = orient_image(raw_item_img, rot_deg=p_rot, flip_h=p_fh, flip_v=p_fv)
            p_rec = analyze_image(p_oriented)
            backend_rows.append(
                {
                    "#": p_num,
                    "Item": item_label,
                    "Sat": p_rec["sat"],
                    "Vib": p_rec["vibrance"],
                    "Contrast": p_rec["contrast"],
                    "Sharp": p_rec["sharpness"],
                    "Gamma": p_rec["gamma"],
                    "Bright": p_rec["brightness"],
                    "Warmth": p_rec["warmth"],
                    "Shadow": p_rec["shadow_lift"],
                    "Clarity": p_rec["clarity"],
                    "BG Pop": p_rec["bg_pop"],
                    "Grain": "On" if p_rec["grain"] else "Off",
                }
            )
        if backend_rows:
            st.dataframe(backend_rows, use_container_width=True, hide_index=True)

        st.markdown("#### Save / Manage Custom Preset (`custom_presets.json`)")
        st.caption("Saves your current slider values as a reusable preset in `custom_presets.json`.")
        pm_c1, pm_c2, pm_c3, pm_c4 = st.columns([1.8, 1.3, 1.4, 1.3])
        with pm_c1:
            preset_name_input = st.text_input("Preset name", value="My Pastel Pop", key="custom_preset_name_input")
        with pm_c2:
            preset_base_choice = st.selectbox(
                "Base fallback",
                ["Pastel", "Pencil", "Digital (Krita)", "Sketch Pen / Marker"],
                index=0,
                key="custom_preset_base_choice",
            )
        with pm_c3:
            preset_apply_mode = st.selectbox(
                "Preset mode",
                ["Override Backend", "Scale Over Backend"],
                index=0,
                key="custom_preset_apply_mode",
            )
        with pm_c4:
            st.write("")
            save_preset_btn = st.button("Save / Update Preset", key="save_custom_preset_btn")

        if save_preset_btn:
            clean_p_name = sanitize_stem(preset_name_input, "")
            if clean_p_name:
                updated = load_custom_presets()
                updated[clean_p_name] = {
                    "base_medium": preset_base_choice,
                    "apply_mode": preset_apply_mode,
                    "sat_boost": float(sat_slider),
                    "contrast": float(contrast_slider),
                    "sharpness": float(sharpness_slider),
                    "gamma": float(gamma_slider),
                    "brightness_lift": float(brightness_extra),
                    "warmth_shift": float(warmth_extra),
                    "vibrance_mult": float(vibrance_extra),
                    "clarity_mult": float(clarity_extra),
                    "shadow_lift_mult": float(shadow_extra),
                    "bg_pop_mult": float(bg_pop_extra),
                    "tint_shift": float(tint_extra),
                }
                save_custom_presets(updated)
                st.success(f"Saved preset **{clean_p_name}** to `custom_presets.json`.")
                st.rerun()
            else:
                st.warning("Enter a valid preset name before saving.")

        existing_custom = load_custom_presets()
        if existing_custom:
            del_c1, del_c2 = st.columns([2.5, 1.5])
            with del_c1:
                preset_to_delete = st.selectbox(
                    "Saved custom presets",
                    sorted(existing_custom.keys()),
                    key="delete_custom_preset_select",
                )
            with del_c2:
                st.write("")
                if st.button("Delete Selected Preset", key="delete_custom_preset_btn"):
                    existing_custom.pop(preset_to_delete, None)
                    save_custom_presets(existing_custom)
                    st.success(f"Deleted preset **{preset_to_delete}**.")
                    st.rerun()

    if "edit_cache" not in st.session_state:
        st.session_state["edit_cache"] = {}
    cache: dict[tuple, tuple[Image.Image, Image.Image]] = st.session_state["edit_cache"]

    settings_sig = (
        is_fine_tune,
        sat_slider,
        contrast_slider,
        sharpness_slider,
        gamma_slider,
        medium_preset,
        brightness_extra,
        warmth_extra,
        vibrance_extra,
        clarity_extra,
        shadow_extra,
        bg_pop_extra,
        tint_extra,
        trim_borders,
        vignette_strength,
        grain_choice,
    )

    missing_offsets: list[int] = []
    for offset, (label, _) in enumerate(selected_items):
        item_num = run_start + offset
        rot_val = int(st.session_state.get(f"rot_{item_num}", 0))
        fh_val = bool(st.session_state.get(f"fh_{item_num}", False))
        fv_val = bool(st.session_state.get(f"fv_{item_num}", False))
        cache_key = (item_num, label, rot_val, fh_val, fv_val, settings_sig)
        if cache_key not in cache:
            missing_offsets.append(offset)

    prepared_records: list[dict[str, object]] = []
    total_missing = len(missing_offsets)

    if total_missing > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
        t_start = time.perf_counter()
        done_count = 0

        for offset, (label, raw_orig) in enumerate(selected_items):
            item_num = run_start + offset
            step_idx = offset + 1
            rot_key = f"rot_{item_num}"
            fh_key = f"fh_{item_num}"
            fv_key = f"fv_{item_num}"
            rot_val = int(st.session_state.get(rot_key, 0))
            fh_val = bool(st.session_state.get(fh_key, False))
            fv_val = bool(st.session_state.get(fv_key, False))
            cache_key = (item_num, label, rot_val, fh_val, fv_val, settings_sig)

            if cache_key in cache:
                oriented_orig, edited_img = cache[cache_key]
            else:
                done_count += 1
                elapsed = time.perf_counter() - t_start
                status_text.write(
                    f"Computing **{done_count}/{total_missing}** item(s) "
                    f"(#{item_num}: *{label}*) — Elapsed: `{elapsed:.1f}s`"
                )
                progress_bar.progress(done_count / total_missing)

                oriented_orig = orient_image(raw_orig, rot_deg=rot_val, flip_h=fh_val, flip_v=fv_val)
                if medium_preset.startswith("Custom: "):
                    c_name = medium_preset.removeprefix("Custom: ")
                    c_cfg = custom_presets_store.get(c_name, {})
                    base_medium = str(c_cfg.get("base_medium", "Pastel"))
                    _, rec = resolve_medium_and_base(oriented_orig, base_medium)
                    apply_mode = str(c_cfg.get("apply_mode", "Override Backend"))
                    for k_map, cfg_key in [
                        ("sat", "sat_boost"),
                        ("contrast", "contrast"),
                        ("sharpness", "sharpness"),
                        ("gamma", "gamma"),
                        ("brightness", "brightness_lift"),
                        ("warmth", "warmth_shift"),
                        ("vibrance", "vibrance_mult"),
                        ("clarity", "clarity_mult"),
                        ("shadow_lift", "shadow_lift_mult"),
                        ("bg_pop", "bg_pop_mult"),
                    ]:
                        if cfg_key in c_cfg:
                            val = float(c_cfg[cfg_key])
                            if apply_mode == "Scale Over Backend":
                                rec[k_map] = round(float(rec[k_map]) * val, 3)
                            else:
                                rec[k_map] = val
                else:
                    _, rec = resolve_medium_and_base(oriented_orig, medium_preset)

                if is_fine_tune:
                    eff_sat = float(rec["sat"]) * sat_slider
                    eff_con = float(rec["contrast"]) * contrast_slider
                    eff_sharp = float(rec["sharpness"]) * sharpness_slider
                    eff_gamma = float(rec["gamma"]) * gamma_slider
                else:
                    eff_sat = sat_slider
                    eff_con = contrast_slider
                    eff_sharp = sharpness_slider
                    eff_gamma = gamma_slider

                eff_bright = float(rec["brightness"]) * brightness_extra
                eff_warmth = float(rec["warmth"]) + warmth_extra
                eff_vib = float(rec["vibrance"]) * vibrance_extra
                eff_clar = float(rec["clarity"]) * clarity_extra
                eff_shadow = float(rec["shadow_lift"]) * shadow_extra
                eff_bg_pop = float(rec["bg_pop"]) * bg_pop_extra

                if grain_choice == "Always On":
                    eff_grain = True
                elif grain_choice == "Always Off":
                    eff_grain = False
                else:
                    eff_grain = bool(rec["grain"])

                edited_img = apply_art_edit(
                    oriented_orig,
                    sat_boost_max=eff_sat,
                    contrast_factor=eff_con,
                    sharpness_factor=eff_sharp,
                    gamma=eff_gamma,
                    brightness_factor=eff_bright,
                    warmth_shift=eff_warmth,
                    vibrance_factor=eff_vib,
                    shadow_lift=eff_shadow,
                    clarity_factor=eff_clar,
                    bg_pop_factor=eff_bg_pop,
                    add_subtle_grain=eff_grain,
                )
                if abs(tint_extra) > 0.001:
                    t_arr = np.array(edited_img.convert("RGB"), dtype=np.float32)
                    t_delta = tint_extra * 14.0
                    t_arr[:, :, 0] = np.clip(t_arr[:, :, 0] + t_delta, 0, 255)
                    t_arr[:, :, 1] = np.clip(t_arr[:, :, 1] - t_delta * 0.8, 0, 255)
                    t_arr[:, :, 2] = np.clip(t_arr[:, :, 2] + t_delta * 0.6, 0, 255)
                    edited_img = Image.fromarray(t_arr.astype(np.uint8))
                cache[cache_key] = (oriented_orig, edited_img)

            prepared_records.append(
                {
                    "item_num": item_num,
                    "step_idx": step_idx,
                    "label": label,
                    "orig": oriented_orig,
                    "edited": edited_img,
                    "rot_key": rot_key,
                    "fh_key": fh_key,
                    "fv_key": fv_key,
                }
            )

        total_elapsed = time.perf_counter() - t_start
        progress_bar.progress(1.0)
        status_text.success(f"Computed {total_missing} item(s) in `{total_elapsed:.1f}s`.")
    else:
        for offset, (label, _) in enumerate(selected_items):
            item_num = run_start + offset
            step_idx = offset + 1
            rot_key = f"rot_{item_num}"
            fh_key = f"fh_{item_num}"
            fv_key = f"fv_{item_num}"
            rot_val = int(st.session_state.get(rot_key, 0))
            fh_val = bool(st.session_state.get(fh_key, False))
            fv_val = bool(st.session_state.get(fv_key, False))
            cache_key = (item_num, label, rot_val, fh_val, fv_val, settings_sig)
            oriented_orig, edited_img = cache[cache_key]
            prepared_records.append(
                {
                    "item_num": item_num,
                    "step_idx": step_idx,
                    "label": label,
                    "orig": oriented_orig,
                    "edited": edited_img,
                    "rot_key": rot_key,
                    "fh_key": fh_key,
                    "fv_key": fv_key,
                }
            )

    zip_entries: list[tuple[str, bytes]] = []
    used_names: dict[str, int] = {}

    for row_start in range(0, len(prepared_records), grid_cols):
        row_items = prepared_records[row_start : row_start + grid_cols]
        cols = st.columns(grid_cols, gap="large")

        for col_idx, rec_obj in enumerate(row_items):
            item_num = int(rec_obj["item_num"])
            step_idx = int(rec_obj["step_idx"])
            label = str(rec_obj["label"])
            orig_img = rec_obj["orig"]
            edited_img = rec_obj["edited"]
            rot_key = str(rec_obj["rot_key"])
            fh_key = str(rec_obj["fh_key"])
            fv_key = str(rec_obj["fv_key"])

            batch_default_stem = build_default_stem(
                label=label,
                index_1based=step_idx,
                rename_mode=bulk_rename_mode,
                prefix=bulk_prefix,
                start_num=int(bulk_start_num),
            )

            with cols[col_idx]:
                st.markdown(f"**#{item_num} — {label}**")

                # Orientation fix buttons (rotate 90° left/right, flip horizontal/vertical)
                o1, o2, o3, o4 = st.columns(4)
                with o1:
                    if st.button("↺ 90°", key=f"rotl_{item_num}", help="Rotate 90° counter-clockwise"):
                        st.session_state[rot_key] = (int(st.session_state.get(rot_key, 0)) - 90) % 360
                        st.rerun()
                with o2:
                    if st.button("↻ 90°", key=f"rotr_{item_num}", help="Rotate 90° clockwise"):
                        st.session_state[rot_key] = (int(st.session_state.get(rot_key, 0)) + 90) % 360
                        st.rerun()
                with o3:
                    if st.button("↔ Flip", key=f"fliph_{item_num}", help="Flip horizontally"):
                        st.session_state[fh_key] = not bool(st.session_state.get(fh_key, False))
                        st.rerun()
                with o4:
                    if st.button("↕ Flip", key=f"flipv_{item_num}", help="Flip vertically"):
                        st.session_state[fv_key] = not bool(st.session_state.get(fv_key, False))
                        st.rerun()
                if rot_val != 0 or fh_val or fv_val:
                    if st.button("Reset Orientation", key=f"reset_ori_{item_num}"):
                        st.session_state[rot_key] = 0
                        st.session_state[fh_key] = False
                        st.session_state[fv_key] = False
                        st.rerun()

                if allow_individual_override:
                    custom_raw = st.text_input(
                        "Output file name",
                        value=batch_default_stem,
                        key=f"rename_{item_num}_{bulk_rename_mode}_{bulk_prefix}_{bulk_start_num}",
                        label_visibility="collapsed",
                    )
                    chosen_stem = sanitize_stem(custom_raw, batch_default_stem)
                else:
                    chosen_stem = batch_default_stem

                out_bytes, ext, mime = encode_output_image(edited_img, output_format, quality_val)

                candidate_name = f"{chosen_stem}{ext}"
                count = used_names.get(candidate_name, 0) + 1
                used_names[candidate_name] = count
                out_filename = candidate_name if count == 1 else f"{chosen_stem}_{count}{ext}"

                zip_entries.append((out_filename, out_bytes))

                if save_to_output:
                    (OUTPUT_DIR / out_filename).write_bytes(out_bytes)

                pair_l, pair_r = st.columns(2, gap="small")
                with pair_l:
                    st.image(proportional_preview(orig_img, min(preview_height, 260) if fast_preview else preview_height), caption="Original")
                with pair_r:
                    st.image(proportional_preview(edited_img, min(preview_height, 260) if fast_preview else preview_height), caption="Edited")

                st.download_button(
                    label=f"Download {out_filename}",
                    data=out_bytes,
                    file_name=out_filename,
                    mime=mime,
                    key=f"dl_{item_num}_{out_filename}",
                )

        st.divider()

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in zip_entries:
            zf.writestr(fname, data)

    st.download_button(
        label=f"Download Selected {len(zip_entries)} Files (.zip)",
        data=zip_buf.getvalue(),
        file_name=f"edited_art_{start_idx}_to_{end_idx}_{output_format.lower()}.zip",
        mime="application/zip",
    )


# ---------------------------------------------------------------------------
# 5. Tab 2: Instagram Feed Scheduler & Batch Interleaver
# ---------------------------------------------------------------------------
def interleave_groups(
    group_a: list[tuple[str, Image.Image]],
    group_b: list[tuple[str, Image.Image]],
    count_a: int,
    count_b: int,
) -> list[tuple[str, str, Image.Image]]:
    """Interleave group_a and group_b in cycles of (count_a, count_b) until both are exhausted."""
    result: list[tuple[str, str, Image.Image]] = []
    idx_a, idx_b = 0, 0
    len_a, len_b = len(group_a), len(group_b)
    while idx_a < len_a or idx_b < len_b:
        for _ in range(max(0, count_a)):
            if idx_a < len_a:
                lbl, im = group_a[idx_a]
                result.append(("Group A", lbl, im))
                idx_a += 1
        for _ in range(max(0, count_b)):
            if idx_b < len_b:
                lbl, im = group_b[idx_b]
                result.append(("Group B", lbl, im))
                idx_b += 1
    return result


def is_low_sat_sketch(pil_img: Image.Image) -> bool:
    rgb = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 1])) < 28.0 and float(np.mean(hsv[:, :, 1] > 65)) < 0.08


with tab_scheduler:
    st.subheader("Feed Scheduler & Batch Interleaver")
    st.caption(
        "Mix Pastel and Pencil artworks in a repeatable ratio (e.g., 2 Pastel : 2 Pencil or 2 Pencil : 1 Pastel), "
        "preview the 3-column grid order, and export numbered files (`01_...`, `02_...`) for one-shot uploading."
    )

    sched_source = st.radio(
        "Where should the scheduler load images from?",
        ["Upload Two Groups (Group A & Group B)", "Auto-Split Local `output/` Folder (Color vs. Pencil/Sketch)"],
        key="sched_source_radio",
    )

    group_a_items: list[tuple[str, Image.Image]] = []
    group_b_items: list[tuple[str, Image.Image]] = []

    if sched_source.startswith("Upload Two Groups"):
        up_c1, up_c2 = st.columns(2, gap="large")
        with up_c1:
            files_a = st.file_uploader(
                "Group A (e.g., Pastel / Color artworks)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sched_up_a",
            )
            if files_a:
                for f in sorted(files_a, key=lambda x: x.name):
                    group_a_items.append((f.name, Image.open(BytesIO(f.getvalue())).convert("RGB")))
        with up_c2:
            files_b = st.file_uploader(
                "Group B (e.g., Pencil / Sketch artworks)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sched_up_b",
            )
            if files_b:
                for f in sorted(files_b, key=lambda x: x.name):
                    group_b_items.append((f.name, Image.open(BytesIO(f.getvalue())).convert("RGB")))
    else:
        out_files = sorted(
            [p for p in OUTPUT_DIR.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )
        for p in out_files:
            im = Image.open(BytesIO(p.read_bytes())).convert("RGB")
            if is_low_sat_sketch(im):
                group_b_items.append((p.name, im))
            else:
                group_a_items.append((p.name, im))
        st.info(
            f"Detected in `output/`: **{len(group_a_items)}** Color/Pastel (Group A) and "
            f"**{len(group_b_items)}** Pencil/Sketch (Group B)."
        )

    pat_c1, pat_c2, pat_c3, pat_c4 = st.columns([1.4, 1.4, 1.4, 1.6])
    with pat_c1:
        pat_preset = st.selectbox(
            "Quick pattern",
            ["2 Group A : 2 Group B", "2 Group B : 1 Group A", "1 Group A : 1 Group B", "Custom Counts"],
            index=0,
            key="sched_pat_preset",
        )
    if pat_preset == "2 Group A : 2 Group B":
        def_a, def_b = 2, 2
    elif pat_preset == "2 Group B : 1 Group A":
        def_a, def_b = 1, 2
    elif pat_preset == "1 Group A : 1 Group B":
        def_a, def_b = 1, 1
    else:
        def_a, def_b = 2, 1

    with pat_c2:
        take_a = int(
            st.number_input(
                "Group A per cycle",
                min_value=0,
                max_value=20,
                value=def_a,
                step=1,
                disabled=(pat_preset != "Custom Counts"),
                key="sched_take_a",
            )
        )
    with pat_c3:
        take_b = int(
            st.number_input(
                "Group B per cycle",
                min_value=0,
                max_value=20,
                value=def_b,
                step=1,
                disabled=(pat_preset != "Custom Counts"),
                key="sched_take_b",
            )
        )
    with pat_c4:
        sched_prefix = st.text_input("Output filename prefix", value="post", key="sched_prefix_in")

    if not group_a_items and not group_b_items:
        st.caption("Add images to Group A and/or Group B above to preview and export your scheduled sequence.")
    elif take_a == 0 and take_b == 0:
        st.warning("Set at least one group count per cycle above 0.")
    else:
        mixed = interleave_groups(group_a_items, group_b_items, take_a, take_b)
        st.success(f"Scheduled **{len(mixed)}** artworks ({len(group_a_items)} Group A + {len(group_b_items)} Group B).")

        sched_zip_buf = BytesIO()
        pad_width = max(2, len(str(len(mixed))))
        clean_sprefix = sanitize_stem(sched_prefix, "post")

        with zipfile.ZipFile(sched_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as szf:
            for idx_1, (grp_tag, orig_lbl, im) in enumerate(mixed, start=1):
                seq_str = str(idx_1).zfill(pad_width)
                orig_stem = sanitize_stem(orig_lbl, f"item_{seq_str}")
                out_name = f"{seq_str}_{clean_sprefix}_{orig_stem}.jpg"
                enc_bytes, _, _ = encode_output_image(im, "JPG", 95)
                szf.writestr(out_name, enc_bytes)

        dl_c1, dl_c2 = st.columns([2.0, 2.0])
        with dl_c1:
            st.download_button(
                label=f"Download Scheduled Sequence ({len(mixed)} files .zip)",
                data=sched_zip_buf.getvalue(),
                file_name=f"scheduled_{clean_sprefix}_{len(mixed)}_items.zip",
                mime="application/zip",
                type="primary",
                key="sched_download_zip_btn",
            )
        with dl_c2:
            if st.button("Also Save Numbered Sequence to `output/scheduled/`", key="sched_save_disk_btn"):
                sched_dir = OUTPUT_DIR / "scheduled"
                sched_dir.mkdir(parents=True, exist_ok=True)
                for idx_1, (_, orig_lbl, im) in enumerate(mixed, start=1):
                    seq_str = str(idx_1).zfill(pad_width)
                    orig_stem = sanitize_stem(orig_lbl, f"item_{seq_str}")
                    out_name = f"{seq_str}_{clean_sprefix}_{orig_stem}.jpg"
                    enc_bytes, _, _ = encode_output_image(im, "JPG", 95)
                    (sched_dir / out_name).write_bytes(enc_bytes)
                st.success(f"Saved {len(mixed)} numbered files to `{sched_dir}`.")

        st.markdown("#### 3-Column Feed Order Preview (Top-to-Bottom Posting Sequence)")
        for r_start in range(0, len(mixed), 3):
            row_slice = mixed[r_start : r_start + 3]
            g_cols = st.columns(3, gap="medium")
            for c_i, (grp_tag, orig_lbl, im) in enumerate(row_slice):
                pos_num = r_start + c_i + 1
                with g_cols[c_i]:
                    st.image(
                        proportional_preview(im, 220),
                        caption=f"#{str(pos_num).zfill(pad_width)} [{grp_tag}] {orig_lbl}",
                    )
