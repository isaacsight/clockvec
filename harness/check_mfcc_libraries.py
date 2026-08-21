"""Check shipped MFCC implementations against the clockvec vectors.

Kept out of the package and out of the test suite on purpose. clockvec
depends on numpy alone, and a suite that needed librosa and torch installed
to run would be un-runnable in exactly the CI environments these vectors are
meant to be vendored into. This is the optional adapter layer; run it when
you have the libraries.

    pip install librosa torchaudio
    python harness/check_mfcc_libraries.py

Every library is driven through its PUBLIC API only. No private functions and
no monkeypatching: a conformance result obtained by reaching inside an
implementation is a result about a program that nobody ships.

One consequence is visible in the output and is the point of the exercise.
The vectors declare db_clamp_top=None, because a function whose output
depends on the rest of the array is not checkable frame by frame. Neither
librosa nor torchaudio can be driven there -- librosa's feature.mfcc accepts
no top_db and forwards **kwargs to melspectrogram rather than power_to_db,
and torchaudio's MFCC hardcodes self.top_db = 80.0. The vectors are runnable
against both ONLY because their signals keep the log-mel dynamic range under
80 dB, where the clamp is inert. That constraint is enforced in
`mfcc_vectors.build_vectors`, not left to the author to remember.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from clockvec.mfcc_vectors import build_vectors, signal

N = 160 * 24 + 512

# Which signal each vector was built on, and which expectation each library's
# own convention set corresponds to.
PLAN = {
    "mfcc/mel-area-normalization": (
        "tone_plus_noise",
        [("torchaudio", "torchaudio", 0, False), ("librosa", "librosa", 1, True)],
    ),
    "mfcc/log-base": (
        "harmonic_stack",
        [("librosa", "librosa", 0, False), ("torchaudio", "torchaudio", 0, False)],
    ),
    "mfcc/window-placement": (
        "tone_plus_noise",
        [("librosa", "librosa", 0, False), ("torchaudio", "torchaudio", 0, False)],
    ),
}


def run_librosa(y, d):
    import librosa

    return librosa.feature.mfcc(
        y=y, sr=d.sample_rate, n_mfcc=d.n_mfcc, dct_type=2,
        norm="ortho" if d.dct_ortho else None, lifter=d.lifter,
        mel_norm="slaney" if d.mel_area_normalize else None,
        n_fft=d.n_fft, win_length=d.win_length, hop_length=d.hop_length,
        window="hann", center=d.center, n_mels=d.n_mels,
        htk=(d.mel_scale == "htk"), power=d.power, fmin=d.f_min, fmax=d.f_max,
    )


def run_torchaudio(y, d):
    import torch
    import torchaudio

    t = torchaudio.transforms.MFCC(
        sample_rate=int(d.sample_rate), n_mfcc=d.n_mfcc, dct_type=2,
        norm="ortho" if d.dct_ortho else None, log_mels=False,
        melkwargs=dict(
            n_fft=d.n_fft, win_length=d.win_length, hop_length=d.hop_length,
            center=d.center, n_mels=d.n_mels, mel_scale=d.mel_scale,
            norm="slaney" if d.mel_area_normalize else None, power=d.power,
            f_min=d.f_min, f_max=d.f_max, window_fn=torch.hann_window,
        ),
    )
    return t(torch.from_numpy(y).float()).numpy().astype(np.float64)


RUNNERS = {"librosa": run_librosa, "torchaudio": run_torchaudio}


def main() -> int:
    header = (f"{'vector':<32}{'library':<13}{'expectation':<44}"
              f"{'max|d|':>11}{'tol':>11}  verdict")
    print(header)
    print("-" * len(header))
    passes = fails = 0
    for v in build_vectors():
        if v.vector_id not in PLAN:
            continue
        kind, runs = PLAN[v.vector_id]
        y = signal(kind, N)
        for label, runner, idx, area in runs:
            exp = v.expectations[idx]
            decl = replace(v.declaration, mel_area_normalize=area)
            got = RUNNERS[runner](y, decl).ravel(order="C")
            want = np.asarray(exp.values)
            d = float(np.abs(got[: want.size] - want).max())
            ok = d <= exp.tolerance
            passes += ok
            fails += not ok
            print(f"{v.vector_id:<32}{label:<13}{exp.implementation[:43]:<44}"
                  f"{d:>11.3e}{exp.tolerance:>11.1e}  {'PASS' if ok else 'FAIL'}")
    print("-" * len(header))
    print(f"{passes} passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
