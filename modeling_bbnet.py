#!/usr/bin/env python3
"""
BB-Net: distance prediction for bivariate bicycle (BB) and perturbed BB codes.

Reference implementation of the model reported in "Distance Prediction for
Quantum Codes".  The file contains the full path from a code specification to
a predicted distance:

    code (l, m, k, A, B[, C, D])
      -> random distance-preserving symmetry, training only      (Sec. 4.3)
      -> shift canonicalisation                                  (Sec. 4.3)
      -> polynomial tokens and Tanner spectral tokens            (Sec. 4.1, App. C)
      -> polynomial encoder, FiLM-conditioned on (l, m, n, k)    (Sec. 4.2)
         Tanner spectral encoder
      -> gated fusion  h = h_poly + tanh(g) * h_tanner
      -> classification head over the distances D_MIN..D_MAX

Both encoders use the same depth L and width d_model.  These are the only
architectural parameters; the reported model uses L = 8, d_model = 256.
The model is trained with cross-entropy against the heuristic upper-bound
distance of each code (see `distance_loss`).
"""

from __future__ import annotations

import argparse
import functools
import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------------

D_MIN, D_MAX = 2, 25                  # distance range of the dataset
NUM_CLASSES = D_MAX - D_MIN + 1

NUM_POLYS = 4                         # A, B, C, D
FOURIER_ORDERS = 8                    # harmonics r = 1..8, Eq. (9)
POS_DIM = 4 * FOURIER_ORDERS          # cos/sin for each of the two coordinates
SPECTRAL_DIM = 20                     # character features, Eq. (21)
COND_SCALE = (30.0, 15.0, 360.0, 60.0)  # normalisers for (l, m, n, k)
COND_DIM = len(COND_SCALE)

NUM_HEADS = 8
HEAD_HIDDEN = 128
DROPOUT = 0.1


# ----------------------------------------------------------------------------
#  Code specification
# ----------------------------------------------------------------------------

Support = Sequence[tuple[int, int]]   # exponent pairs (a, b) of x^a y^b


@dataclass(frozen=True)
class BBCode:
    """A BB code (C = D = 0) or a PBB code (C, D non-zero) on Z_l x Z_m.

    Each polynomial is given by its support S_P: the exponent pairs (a, b),
    0 <= a < l, 0 <= b < m, of its monomials x^a y^b.  k is the number of
    logical qubits, which conditions the polynomial encoder.
    """
    ell: int
    m: int
    k: int
    A: Support
    B: Support
    C: Support = ()
    D: Support = ()

    @property
    def n(self) -> int:
        return 2 * self.ell * self.m

    @property
    def is_css(self) -> bool:
        return not self.C and not self.D

    @property
    def supports(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(tuple((int(a), int(b)) for a, b in S)
                     for S in (self.A, self.B, self.C, self.D))

# ----------------------------------------------------------------------------
#  Shift canonicalisation (Sec. 4.3, Eq. 12)
# ----------------------------------------------------------------------------

def _translate(S, a0: int, b0: int, ell: int, m: int):
    """tau_(a0, b0)(S): the support of x^-a0 y^-b0 * P, sorted."""
    return tuple(sorted(((a - a0) % ell, (b - b0) % m) for a, b in S))


def canonicalise(code: BBCode):
    """Deterministic representative of the code's cyclic-shift class.

    Every monomial is tried as the anchor that is moved to the origin, and the
    lexicographically smallest result is kept.  For a CSS code, A and B may be
    translated independently.  For a PBB code a translation is a symmetry only
    when it is applied to all four polynomials at once, so a single common
    translation, anchored on a monomial of A, is chosen.
    """
    ell, m, polys = code.ell, code.m, code.supports
    if code.is_css:
        return tuple(min(_translate(S, a, b, ell, m) for a, b in S) if S else ()
                     for S in polys)
    return min(tuple(_translate(S, a, b, ell, m) if S else () for S in polys)
               for a, b in polys[0])

# ----------------------------------------------------------------------------
#  Polynomial tokens (Sec. 4.1, Eqs. 8-9)
# ----------------------------------------------------------------------------

def _fourier(x: np.ndarray, period: int) -> np.ndarray:
    """[cos(2 pi r x / period)]_{r=1..8} || [sin(2 pi r x / period)]_{r=1..8}."""
    angle = 2 * np.pi * np.outer(x, np.arange(1, FOURIER_ORDERS + 1)) / period
    return np.concatenate([np.cos(angle), np.sin(angle)], axis=1)


def polynomial_tokens(polys, ell: int, m: int):
    """One token per monomial (P, a, b).

    Returns the polynomial index P in {0: A, 1: B, 2: C, 3: D}, embedded later
    as a one-hot vector, and the 32 cyclic Fourier coordinates of (a, b).
    """
    ids = np.array([p for p, S in enumerate(polys) for _ in S], dtype=np.int64)
    exps = np.array([ab for S in polys for ab in S], dtype=np.float64)
    pos = np.concatenate([_fourier(exps[:, 0], ell), _fourier(exps[:, 1], m)],
                         axis=1)
    return ids, pos.astype(np.float32)


# ----------------------------------------------------------------------------
#  Tanner spectral tokens (Sec. 4.1, Eqs. 10-11; App. C, Eqs. 13-21)
# ----------------------------------------------------------------------------

def _responses(S, ell: int, m: int) -> np.ndarray:
    """z_P(chi_{u,v}) for every character of Z_l x Z_m, shape (l, m).

    z_P(chi_{u,v}) = sum_{(a,b) in S_P} exp(-2 pi i (u a / l + v b / m)),
    i.e. the 2-D DFT of the indicator of S_P.
    """
    grid = np.zeros((ell, m))
    for a, b in S:
        grid[a, b] += 1.0
    return np.fft.fft2(grid)


def _character_orders(ell: int, m: int) -> np.ndarray:
    """o_chi = lcm(l / gcd(u, l), m / gcd(v, m)) for every chi_{u,v}, Eq. (16)."""
    u = np.arange(ell)[:, None]
    v = np.arange(m)[None, :]
    return np.lcm(ell // np.gcd(u, ell), m // np.gcd(v, m)).astype(np.float64)


def spectral_tokens(polys, ell: int, m: int) -> np.ndarray:
    """One 20-dimensional token per character chi_{u,v}; shape (l * m, 20)."""
    lm = ell * m
    z = [_responses(S, ell, m) for S in polys]
    w = [max(len(S), 1) for S in polys]
    aA, aB, aC, aD = (np.abs(zp) / wp for zp, wp in zip(z, w))   # Eq. (14)
    zA, zB = z[0], z[1]
    wA, wB = w[0], w[1]

    s2 = (np.abs(zA) ** 2 + np.abs(zB) ** 2) / (wA ** 2 + wB ** 2)  # Eq. (15)
    q = zA * np.conj(zB) / (wA * wB)
    delta = np.abs(zA - zB) / (wA + wB)

    order = _character_orders(ell, m)
    trivial = np.zeros((ell, m))
    trivial[0, 0] = 1.0
    zero = np.zeros((ell, m))

    features = np.stack([
        # s^(1): magnitudes and pairwise statistics                 (0-9)
        aA, aB, aC, aD, aA ** 2, aB ** 2, s2, q.real, q.imag, np.abs(q),
        # s^(2): near-degenerate regime                             (10-14)
        delta, np.exp(-8.0 * s2), np.exp(-8.0 * (1.0 - np.minimum(aA, aB))),
        zero, zero,   # constant zero in the reported model
        # s^(3): invariants of the character itself                 (15-19)
        order / lm, np.log1p(order) / math.log1p(lm), 1.0 / order,
        (order % 2 == 0).astype(np.float64), trivial,
    ], axis=-1)
    return features.reshape(lm, SPECTRAL_DIM).astype(np.float32)


# ----------------------------------------------------------------------------
#  Model inputs
# ----------------------------------------------------------------------------

def conditioning(code: BBCode) -> np.ndarray:
    """c = (l, m, n, k), scaled to O(1)."""
    c = np.array([code.ell, code.m, code.n, code.k], dtype=np.float64)
    return (c / np.array(COND_SCALE)).astype(np.float32)


def featurise(code: BBCode) -> dict[str, torch.Tensor]:
    """Both token sets are built from the same canonical representative.
    """
    polys = canonicalise(code)
    ids, pos = polynomial_tokens(polys, code.ell, code.m)
    return {
        "poly_ids": torch.from_numpy(ids),
        "poly_pos": torch.from_numpy(pos),
        "spectral": torch.from_numpy(spectral_tokens(polys, code.ell, code.m)),
        "cond": torch.from_numpy(conditioning(code)),
    }


def collate(examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad both token sequences to the longest in the batch.

    The masks are True on real tokens and False on padding.
    """
    def pad(key):
        seqs = [ex[key] for ex in examples]
        out = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        mask = torch.zeros(out.shape[:2], dtype=torch.bool)
        for i, s in enumerate(seqs):
            mask[i, :len(s)] = True
        return out, mask

    poly_ids, poly_mask = pad("poly_ids")
    poly_pos, _ = pad("poly_pos")
    spectral, spectral_mask = pad("spectral")
    return {
        "poly_ids": poly_ids, "poly_pos": poly_pos, "poly_mask": poly_mask,
        "spectral": spectral, "spectral_mask": spectral_mask,
        "cond": torch.stack([ex["cond"] for ex in examples]),
    }


# ----------------------------------------------------------------------------
#  Model (Sec. 4.2, Fig. 2)
# ----------------------------------------------------------------------------

class FiLM(nn.Module):
    """h -> h * (1 + gamma(c)) + beta(c).  Zero-initialised: the identity at start."""

    def __init__(self, width: int):
        super().__init__()
        self.to_gamma = nn.Linear(COND_DIM, width)
        self.to_beta = nn.Linear(COND_DIM, width)
        for layer in (self.to_gamma, self.to_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.to_gamma(c).unsqueeze(1)) + self.to_beta(c).unsqueeze(1)


class TransformerBlock(nn.Module):
    """Multi-head attention, Add & Norm, feed-forward, Add & Norm [, FiLM]."""

    def __init__(self, width: int, film: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, NUM_HEADS, dropout=DROPOUT,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Dropout(DROPOUT), nn.Linear(4 * width, width))
        self.norm2 = nn.LayerNorm(width)
        self.drop = nn.Dropout(DROPOUT)
        self.film = FiLM(width) if film else None

    def forward(self, x, padding, cond=None):
        a, _ = self.attn(x, x, x, key_padding_mask=padding, need_weights=False)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x if self.film is None else self.film(x, cond)


def _prepend_cls(cls: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
    """Prepend the CLS token; return the sequence and its key-padding mask."""
    x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
    keep = torch.cat([mask.new_ones(mask.shape[0], 1), mask], dim=1)
    return x, ~keep


class PolynomialEncoder(nn.Module):
    """Self-attention over monomial tokens, FiLM-conditioned on (l, m, n, k).

    Token input: one-hot(P) || Fourier(a, b), projected to d_model.
    """

    def __init__(self, depth: int, width: int):
        super().__init__()
        self.embed = nn.Linear(NUM_POLYS + POS_DIM, width)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, film=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, ids, pos, mask, cond):
        role = F.one_hot(ids, NUM_POLYS).to(pos.dtype)
        x, padding = _prepend_cls(self.cls, self.embed(torch.cat([role, pos], -1)), mask)
        for block in self.blocks:
            x = block(x, padding, cond)
        return self.norm(x[:, 0])


class SpectralEncoder(nn.Module):
    """Self-attention over the l*m character tokens of the Tanner graph lift."""

    def __init__(self, depth: int, width: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(SPECTRAL_DIM, width), nn.GELU(),
                                   nn.Linear(width, width))
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, film=False) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, mask):
        x, padding = _prepend_cls(self.cls, self.embed(tokens), mask)
        for block in self.blocks:
            x = block(x, padding)
        return self.norm(x[:, 0])


class BBNet(nn.Module):
    """Two encoders, a learned residual gate, and a distance classifier.

    h_fuse = h_polynomial + tanh(g) * h_tanner, with g initialised at 0.
    """

    def __init__(self, depth: int = 8, width: int = 256):
        super().__init__()
        if width % NUM_HEADS:
            raise ValueError(f"width must be divisible by {NUM_HEADS}")
        self.polynomial = PolynomialEncoder(depth, width)
        self.spectral = SpectralEncoder(depth, width)
        self.gate = nn.Parameter(torch.zeros(()))
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, HEAD_HIDDEN), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(HEAD_HIDDEN, NUM_CLASSES))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits over the distances D_MIN, ..., D_MAX; shape (batch, NUM_CLASSES)."""
        h_poly = self.polynomial(batch["poly_ids"], batch["poly_pos"],
                                 batch["poly_mask"], batch["cond"])
        h_tanner = self.spectral(batch["spectral"], batch["spectral_mask"])
        return self.head(h_poly + torch.tanh(self.gate) * h_tanner)

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted distance: the most probable class."""
        return self.forward(batch).argmax(dim=-1) + D_MIN


def distance_loss(logits: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Categorical cross-entropy against the reference (upper-bound) distance."""
    return F.cross_entropy(logits, distance - D_MIN)


# ----------------------------------------------------------------------------
#  Loading weights saved by the original training code
# ----------------------------------------------------------------------------

_RENAMES = (
    ("encoder.token.", "polynomial."),
    ("encoder.branches.0.", "spectral."),
    ("encoder.gate", "gate"),
    ("head.trunk.", "head."),
    ("head.classifier.", "head.4."),
    (".ln0.", ".norm1."),
    (".ln1.", ".norm2."),
    ("polynomial.ln_out.", "polynomial.norm."),
    ("spectral.ln.", "spectral.norm."),
)


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a training-code state dict to the parameter names used here."""
    out = {}
    for key, value in state.items():
        for old, new in _RENAMES:
            key = key.replace(old, new)
        out[key] = value.reshape(()) if key == "gate" else value
    return out


# ----------------------------------------------------------------------------
#  Example
# ----------------------------------------------------------------------------

EXAMPLES = {
    # [[72, 12, 6]], [[90, 8, 10]] and the [[144, 12, 12]] gross code
    # (Bravyi et al., 2024)
    "[[72,12,6]]": BBCode(6, 6, 12, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)]),
    "[[90,8,10]]": BBCode(15, 3, 8, A=[(9, 0), (0, 1), (0, 2)], B=[(0, 0), (2, 0), (7, 0)]),
    "[[144,12,12]]": BBCode(12, 6, 12, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--checkpoint", default=None,
                        help="optional weights saved by the training code")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = BBNet(depth=args.depth, width=args.width)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(convert_state_dict(ckpt.get("state_dict", ckpt)))
    model.eval()
    print(f"BB-Net  depth={args.depth}  width={args.width}  "
          f"parameters={sum(p.numel() for p in model.parameters()):,}  "
          f"weights={args.checkpoint or 'random (untrained)'}")
    if args.verbose > 0:
        print(model)
    batch = collate([featurise(code) for code in EXAMPLES.values()])
    for name, d in zip(EXAMPLES, model.predict(batch).tolist()):
        print(f"  {name:>14s}  predicted d = {d}")


if __name__ == "__main__":
    main()
