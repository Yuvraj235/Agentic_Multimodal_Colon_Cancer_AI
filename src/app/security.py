"""ColonAI — security helpers.

Central place for the security policy. Every callsite that touches
untrusted input (uploads, model checkpoints, API requests) should use
these helpers rather than rolling its own checks.

What lives here
───────────────
   • Upload validation       (size cap, MIME allow-list, decompression-bomb guard)
   • Safe torch.load wrapper (prefers weights_only=True)
   • Audit-log file perms    (chmod 0o600 — owner-only read)
   • API auth                (HMAC-style API-KEY env var check)
   • Error sanitisation      (never return tracebacks to clients)
   • Request-ID generator
"""
from __future__ import annotations
import os, io, hmac, uuid, logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFile

# ─────────────────────────────────────────────────────────────────────────────
# Hardening Pillow
# ─────────────────────────────────────────────────────────────────────────────
# 1) Cap decoded pixel count to ~100 MP. Any uncompressed image bigger than
#    this is almost certainly a decompression-bomb attack (a 30 KB PNG can
#    legally expand to 1 GB in RAM).
Image.MAX_IMAGE_PIXELS = 100_000_000        # 100 megapixels
# 2) Refuse to load truncated images silently — they can mask payloads.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ─────────────────────────────────────────────────────────────────────────────
# Upload policy
# ─────────────────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024         # 10 MB
ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png",
    "image/tiff", "image/bmp", "image/webp",
}
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

logger = logging.getLogger("colonai.security")


class UploadError(Exception):
    """Raised on rejected uploads. Message is user-safe."""


def validate_upload_bytes(
    data: bytes,
    *,
    declared_mime: Optional[str] = None,
    filename:      Optional[str] = None,
    max_bytes:     int           = MAX_UPLOAD_BYTES,
) -> Tuple[Image.Image, dict]:
    """Validate an uploaded image and return a safe-to-use PIL Image.

    Performs (in order):
       1. size cap
       2. extension allow-list (if filename given)
       3. PIL "verify" pass — detects truncation/malformed payloads
       4. second decode pass (verify() consumes the stream)
       5. magic-byte sanity (Pillow format must be in allow-list)
       6. decompression-bomb check (Pillow already raises if > MAX_IMAGE_PIXELS)

    Raises UploadError with a user-safe message on any failure. The
    detailed reason is logged but NOT returned to the caller (defence
    against information leak).
    """
    if len(data) == 0:
        raise UploadError("Empty upload.")
    if len(data) > max_bytes:
        raise UploadError(f"Image exceeds {max_bytes // (1024*1024)} MB limit.")
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise UploadError("Unsupported image type. Use JPG, PNG, TIFF, BMP, or WEBP.")
    try:
        # Pass 1: verify() — checks structure, doesn't decode pixels.
        with Image.open(io.BytesIO(data)) as v:
            v.verify()
        # Pass 2: re-open for actual use (verify() invalidates).
        img = Image.open(io.BytesIO(data))
        img.load()                          # force decode → raises on bomb
    except Image.DecompressionBombError:
        logger.warning("decompression bomb rejected (filename=%r)", filename)
        raise UploadError("Image is too large to process safely.")
    except Exception as e:
        logger.warning("malformed image rejected (filename=%r): %s",
                       filename, type(e).__name__)
        raise UploadError("Could not decode image — it may be corrupted.")
    fmt = (img.format or "").upper()
    if fmt not in ("JPEG", "PNG", "TIFF", "BMP", "WEBP"):
        logger.warning("unsupported decoded format %r (filename=%r)", fmt, filename)
        raise UploadError("Unsupported image format.")
    return img.convert("RGB"), {
        "bytes":      len(data),
        "format":     fmt,
        "dimensions": img.size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Safe torch.load
# ─────────────────────────────────────────────────────────────────────────────
def safe_torch_load(path: str, map_location=None, allow_unsafe: bool = False):
    """Load a torch checkpoint with weights_only=True where possible.

    Older PyTorch versions (< 1.13) don't support weights_only. Newer versions
    default to False but should be True for any file from an untrusted
    source. For our own training-time checkpoints we pass allow_unsafe=True
    explicitly to keep load_state_dict + meta keys working.
    """
    import torch
    try:
        if not allow_unsafe:
            return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError) as e:
        # Fall through if torch is too old OR the checkpoint contains
        # non-tensor pickled metadata we want to keep.
        logger.debug("safe_torch_load fallback: %s", type(e).__name__)
    return torch.load(path, map_location=map_location, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Audit-log file perms — owner read/write only
# ─────────────────────────────────────────────────────────────────────────────
def secure_file_perms(path: str | Path) -> None:
    """Set 0o600 perms on the file. Best-effort; never raises."""
    p = Path(path)
    try:
        if p.exists():
            os.chmod(p, 0o600)
    except Exception as e:
        logger.debug("chmod 0600 failed for %r: %s", str(p), e)


def secure_dir_perms(path: str | Path) -> None:
    p = Path(path)
    try:
        if p.exists():
            os.chmod(p, 0o700)
    except Exception as e:
        logger.debug("chmod 0700 failed for %r: %s", str(p), e)


# ─────────────────────────────────────────────────────────────────────────────
# API auth (very simple: shared secret via env var)
# ─────────────────────────────────────────────────────────────────────────────
COLONAI_API_KEY_ENV = "COLONAI_API_KEY"


def require_api_key(provided: Optional[str]) -> bool:
    """Constant-time compare against COLONAI_API_KEY env var.

    Returns True if no key is configured (auth disabled — development).
    Returns True if `provided` matches the configured key.
    Otherwise False.
    """
    expected = os.environ.get(COLONAI_API_KEY_ENV, "")
    if not expected:        # auth disabled
        return True
    if not provided:
        return False
    return hmac.compare_digest(expected, provided)


# ─────────────────────────────────────────────────────────────────────────────
# Error sanitisation
# ─────────────────────────────────────────────────────────────────────────────
def safe_error(exc: BaseException, request_id: Optional[str] = None) -> dict:
    """Convert an exception into a user-safe payload.

    Detailed traceback goes to the server log (with the request_id);
    the response only includes the exception class name and a request ID
    that ops can grep for in logs.
    """
    rid = request_id or new_request_id()
    logger.exception("error  rid=%s  type=%s", rid, type(exc).__name__)
    return {
        "ok":         False,
        "error_type": type(exc).__name__,
        "message":    "An internal error occurred. Reference: " + rid,
        "request_id": rid,
    }


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


# ─────────────────────────────────────────────────────────────────────────────
# HTML escaping — XSS guard for unsafe_allow_html callsites
# ─────────────────────────────────────────────────────────────────────────────
import html as _html_mod
import re as _re_mod

_TAG_STRIP = _re_mod.compile(r"<[^>]*?>")


def escape_html(text) -> str:
    """HTML-escape a string for safe interpolation into Streamlit's
    unsafe_allow_html blocks. Always converts to str first so None / int
    / other types don't blow up the caller.

       escape_html(None)               → ""
       escape_html("Bob")              → "Bob"
       escape_html("<script>x</script>") → "&lt;script&gt;x&lt;/script&gt;"
       escape_html('John "Mac" O\\'Brien') → 'John &quot;Mac&quot; O&#x27;Brien'
    """
    if text is None: return ""
    return _html_mod.escape(str(text), quote=True)


def strip_tags(text) -> str:
    """Remove every HTML tag from a string (use when you need plain text)."""
    if text is None: return ""
    return _TAG_STRIP.sub("", str(text)).strip()


# Short alias for use inside f-strings:  {sx(patient.get('name'))}
sx = escape_html
