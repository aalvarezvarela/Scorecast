"""Lossless 16-bit storage of the vectors Nebius returns.

Measured on 2026-11-03: every value of a Qwen3-Embedding-8B vector from
Nebius is exactly a float32, and the vector equals, bit for bit,
``float32(q / d)`` with ``q`` a bfloat16 vector and ``d`` one float32 divisor
(the server computes in bfloat16 and L2-normalizes in float32). A vector of
4096 dimensions then has about 850 distinct values, and storing ``q``
(2 bytes per value) and ``d`` halves the size of float32 storage without
losing a bit.

``encode`` searches ``q`` and ``d`` and returns them only if ``decode``
reproduces the input exactly; otherwise it returns ``None`` and the caller
keeps the float32 vector. Nothing is ever rounded.
"""

from __future__ import annotations

import numpy as np

#: How many float32 steps around the estimated divisor are tried.
DIVISOR_STEPS = 8


def bf16_codes(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (to nearest, ties to even); the codes."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_values(codes: np.ndarray) -> np.ndarray:
    return (np.asarray(codes, np.uint16).astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )


def decode(codes: np.ndarray, divisor: np.float32 | float) -> np.ndarray:
    return bf16_values(codes) / np.float32(divisor)


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a, np.float32)
    b = np.ascontiguousarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def encode(vector: np.ndarray) -> tuple[np.ndarray, np.float32] | None:
    """``(codes, divisor)`` with ``decode(codes, divisor)`` bit-identical to
    ``vector``, or ``None`` when there is no such pair."""
    vector = np.ascontiguousarray(vector, np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        return None
    magnitude = np.abs(vector).astype(np.float64)
    largest = int(np.argmax(magnitude))
    if magnitude[largest] == 0:
        return None
    # The largest value is q_k / d with q_k a bfloat16 of mantissa m (128..255,
    # any power of two): each m gives one estimate of d.
    for mantissa in range(128, 256):
        estimate = mantissa / magnitude[largest]
        scaled = (vector.astype(np.float64) * estimate).astype(np.float32)
        codes = bf16_codes(scaled)
        if not np.allclose(bf16_values(codes), scaled, rtol=2.0**-12, atol=0.0):
            continue
        divisor = np.float32(estimate)
        candidates = [divisor]
        up = down = divisor
        for _ in range(DIVISOR_STEPS):
            up = np.nextafter(up, np.float32(np.inf))
            down = np.nextafter(down, np.float32(0))
            candidates += [up, down]
        for candidate in candidates:
            if same_bits(decode(codes, candidate), vector):
                return codes, np.float32(candidate)
    return None
