import hashlib

from source_encoding import (
    cached_text_matches,
    decode_source_bytes,
    encode_source_text,
)


def test_source_decoding_is_lossless_for_utf8_utf16_and_latin1() -> None:
    bodies = (
        "openai\n".encode("utf-8"),
        "openai\r\n".encode("utf-16"),
        "openai\r\n".encode("utf-16-le"),
        "openai\r\n".encode("utf-32-be"),
        "quiz\u2014generated\n".encode("cp1252"),
        b"opaque\x81byte\n",
    )
    for body in bodies:
        text, encoding = decode_source_bytes(body)
        assert encode_source_text(text, encoding) == body


def test_cp1252_punctuation_is_not_decoded_as_a_control_character() -> None:
    body = b"quiz\x97generated\n"
    text, encoding = decode_source_bytes(body)
    assert text == "quiz\u2014generated\n"
    assert encoding == "cp1252"
    assert encode_source_text(text, encoding) == body


def test_cached_text_integrity_detects_legacy_replacement_decoding() -> None:
    body = "openai\r\n".encode("utf-16")
    valid_text, valid_encoding = decode_source_bytes(body)
    valid = {
        "text": valid_text,
        "source_encoding": valid_encoding,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    legacy = {
        **valid,
        "text": body.decode("utf-8", errors="replace"),
        "source_encoding": "utf-8",
    }
    assert cached_text_matches(valid)
    assert not cached_text_matches(legacy)


def test_legacy_utf8_sig_cache_is_repaired_before_tokenization() -> None:
    body = "\ufeffopenai\n".encode("utf-8")
    legacy = {
        "text": body.decode("utf-8"),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    text, encoding = decode_source_bytes(body)
    repaired = {**legacy, "text": text, "source_encoding": encoding}
    assert not cached_text_matches(legacy)
    assert text == "openai\n"
    assert cached_text_matches(repaired)
