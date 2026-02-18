"""Image encoding helpers for multimodal API providers."""

from pathlib import Path
import base64
import mimetypes
from typing import Tuple


def image_mime_type(image_path: str) -> str:
    """Infer MIME type from an image path."""
    mime, _ = mimetypes.guess_type(str(image_path))
    return mime or "image/png"


def encode_image_base64(image_path: str) -> str:
    """Encode an image file as base64 text."""
    path = Path(image_path)
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def image_bytes(image_path: str) -> Tuple[bytes, str]:
    """Return raw bytes and MIME type for an image path."""
    path = Path(image_path)
    with path.open("rb") as handle:
        return handle.read(), image_mime_type(image_path)


def as_data_url(image_path: str) -> str:
    """Return an RFC2397 data URL for the image."""
    mime = image_mime_type(image_path)
    payload = encode_image_base64(image_path)
    return f"data:{mime};base64,{payload}"

