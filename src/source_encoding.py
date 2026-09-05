"""Lossless, deterministic decoding for cached public source files."""

from __future__ import annotations

import hashlib
from typing import Mapping


_BOM_CODECS: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xfe\x00\x00", "utf-32-le-bom", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be-bom", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8-sig", "utf-8"),
    (b"\xff\xfe", "utf-16-le-bom", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be-bom", "utf-16-be"),
)


def decode_source_bytes(content: bytes) -> tuple[str, str]:
    """Decode source bytes without replacement and return an invertible label."""

    for bom, label, codec in _BOM_CODECS:
        if content.startswith(bom):
            return content[len(bom) :].decode(codec), label

    # NUL bytes are valid UTF-8 code points, so BOM-less UTF-16/32 must be
    # recognized before the strict UTF-8 branch.
    if len(content) >= 8:
        lanes = [content[offset::4] for offset in range(4)]
        zero_shares = [lane.count(0) / len(lane) for lane in lanes]
        if zero_shares[0] <= 0.10 and all(value >= 0.70 for value in zero_shares[1:]):
            return content.decode("utf-32-le"), "utf-32-le"
        if zero_shares[3] <= 0.10 and all(value >= 0.70 for value in zero_shares[:3]):
            return content.decode("utf-32-be"), "utf-32-be"
    if len(content) >= 4:
        even = content[0::2]
        odd = content[1::2]
        even_zero_share = even.count(0) / len(even)
        odd_zero_share = odd.count(0) / len(odd)
        if odd_zero_share >= 0.30 and even_zero_share <= 0.10:
            return content.decode("utf-16-le"), "utf-16-le"
        if even_zero_share >= 0.30 and odd_zero_share <= 0.10:
            return content.decode("utf-16-be"), "utf-16-be"

    try:
        return content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        return content.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        pass

    # Latin-1 is the final lossless byte-to-code-point fallback. It avoids
    # silently inserting replacement characters when CP1252 is undefined.
    return content.decode("latin-1"), "latin-1"


def encode_source_text(text: str, source_encoding: str) -> bytes:
    """Invert :func:`decode_source_bytes` for cache-integrity checks."""

    if source_encoding == "utf-8-sig":
        return b"\xef\xbb\xbf" + text.encode("utf-8")
    for bom, label, codec in _BOM_CODECS:
        if source_encoding == label:
            return bom + text.encode(codec)
    if source_encoding in {
        "utf-8",
        "utf-16-le",
        "utf-16-be",
        "utf-32-le",
        "utf-32-be",
        "cp1252",
        "latin-1",
    }:
        return text.encode(source_encoding)
    raise ValueError(f"unsupported cached source encoding: {source_encoding}")


def cached_text_matches(payload: Mapping[str, object]) -> bool:
    """Return whether cached text reconstructs the recorded response bytes."""

    text = payload.get("text")
    if text is None:
        return True
    if not isinstance(text, str):
        return False
    if not payload.get("source_encoding") and text.startswith("\ufeff"):
        # Legacy UTF-8-SIG caches retained the BOM in the parsed text. Force a
        # pinned refetch so the BOM cannot contaminate the first token.
        return False
    if (
        payload.get("source_encoding") == "latin-1"
        and not payload.get("decoder_version")
        and any(0x80 <= ord(character) <= 0x9F for character in text)
    ):
        # Re-evaluate legacy Latin-1 caches under the stricter CP1252 branch;
        # this converts punctuation bytes such as 0x97 to their intended text.
        return False
    try:
        body = encode_source_text(
            text,
            str(payload.get("source_encoding") or "utf-8"),
        )
        expected_bytes = int(payload["bytes"])
        expected_sha = str(payload["sha256"])
    except (KeyError, TypeError, ValueError, UnicodeError):
        return False
    return len(body) == expected_bytes and hashlib.sha256(body).hexdigest() == expected_sha
