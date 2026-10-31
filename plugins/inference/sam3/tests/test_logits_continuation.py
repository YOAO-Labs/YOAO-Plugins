"""SAM3 native logits retain the shared continuation wire contract."""

import numpy as np
import pytest

from aap_protocol_v2 import decode_low_res_mask
from predictor import _maybe_encode_low_res, _resolve_mask_input


def test_native_logits_round_trip_preserves_prompt_geometry() -> None:
    native = np.full((1, 288, 288), -4.0, dtype=np.float32)
    native[:, 72:216, 36:180] = 8.0

    token = _maybe_encode_low_res(native, enable=True)
    decoded = decode_low_res_mask(token)
    expected = np.full((1, 256, 256), -4.0, dtype=np.float32)
    expected[:, 64:192, 32:160] = 8.0
    np.testing.assert_array_equal(decoded, expected)

    restored = _resolve_mask_input(
        token,
        None,
        width=512,
        height=600,
        low_res_size=(288, 288),
    )
    np.testing.assert_array_equal(restored, native)


def test_invalid_single_candidate_logits_are_not_silently_dropped() -> None:
    logits = np.full((1, 256, 256), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        _maybe_encode_low_res(logits, enable=True)
