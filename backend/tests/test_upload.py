"""
Tests for POST /upload/.

The deep validator's heavy deps (rembg / cv2 / numpy) are absent in CI, so
ImageValidator degrades gracefully and the route still saves the file. These
tests therefore assert the *route* behaviour (HTTP status / success flag), not
the optional deep-validation outcome.
"""

import io

from PIL import Image


def _png_bytes(size=(256, 256), color=(120, 80, 200)) -> bytes:
    # >= 256x256 and roughly square so the validator's pure-PIL format check
    # (the only stage that runs without cv2/rembg) accepts it.
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_png_upload_succeeds(client):
    files = {"file": ("good.png", _png_bytes(), "image/png")}
    resp = client.post("/upload/", files=files)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["filename"] == "good.png"
    assert body["status"] == "uploaded"
    assert body["size"] > 0


def test_empty_file_is_rejected(client):
    files = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/upload/", files=files)

    # Route raises HTTPException(400) for empty content.
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_oversize_file_returns_413(client):
    # MAX_FILE_SIZE is 20 MB; send 21 MB of bytes (header magic irrelevant once
    # size check trips first).
    oversize = b"\x89PNG\r\n\x1a\n" + b"0" * (21 * 1024 * 1024)
    files = {"file": ("huge.png", oversize, "image/png")}
    resp = client.post("/upload/", files=files)

    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


def test_corrupt_bytes_are_rejected(client):
    # Non-empty, under the size cap, but not a decodable image.
    files = {"file": ("corrupt.png", b"this is definitely not a png", "image/png")}
    resp = client.post("/upload/", files=files)

    # PIL verify() fails -> HTTPException(400).
    assert resp.status_code == 400
    assert "invalid image" in resp.json()["detail"].lower()
