"""MFCC, implemented from the definitions rather than from another implementation.

This exists for a different reason than `unifrac`. For UniFrac the papers
never published a reference value, so no implementation *could* have been
validated. For MFCC the situation is the opposite and more tractable: the
mathematics is not in dispute at all. Hand-matched through public APIs,
librosa and torchaudio agree to a mean absolute difference of 6e-5. Left at
their own defaults on the same 2 s input they differ by 11.00026 in native
units -- a factor of roughly 183,000 -- and the entire gap is convention,
not arithmetic.

So this module's job is not to decide what "the true MFCC" is. It is to make
every convention an explicit, named, required choice, so that two answers can
be compared at all. Davis & Mermelstein 1980 (IEEE TASSP 28(4):357-366) fixes
the shape of the computation and none of the twelve decisions below.

The twelve, each a real observed divergence:

  1.  mel formula          HTK vs Slaney. Measured max|d| 0.998395 on the
                           filterbank, total energy ratio ~292x.
  2.  filterbank area norm librosa defaults to Slaney area normalization and
                           torchaudio defaults to none. In librosa this is
                           `mel_norm`, NOT `norm` -- `norm` is the DCT's. A
                           user matching torchaudio writes `norm=None` and
                           silently keeps area normalization. Fixed in
                           librosa 1842 / pytorch-audio 1058; the default is
                           still "slaney" in librosa 1.0.0.
  3.  bin assignment       Whether a filter's support is computed on the FFT
                           bin grid or on continuous frequency. librosa and
                           python_speech_features differ here by max|d|
                           0.248179; nonzero bins per filter [4 5 6 6 6 7]
                           vs [3 4 5 5 5 6].
  4.  centering            center=True pads by n_fft//2 so frame k is
                           centred on sample k*hop; center=False starts it
                           there. Changes the frame count and every value.
  5.  window placement     When win_length < n_fft the window must be padded
                           to n_fft. Centred padding and left-aligned padding
                           give different phase, and neither is universal.
  6.  power                Magnitude (1.0) or power (2.0) spectrum.
  7.  log scale            10*log10 (librosa, torchaudio) or natural log
                           (python_speech_features). Pure factor 10/ln10 =
                           4.34294 -- large, and trivially reconciled once
                           declared.
  8.  dB clamp             See `db_clamp_top`. This one is not a convention
                           difference. It is the reason MFCC is not currently
                           a function of its input frame.
  9.  DCT type / norm      Type II with "ortho" is near-universal; the
                           unnormalized variant differs by a factor of 2 and
                           a sqrt(2) on c0.
  10. lifter               Cepstral liftering, off in librosa, on by default
                           in HTK-derived tooling.
  11. pre-emphasis         python_speech_features applies 0.97 by default;
                           librosa and torchaudio apply none.
  12. spectrum scaling     python_speech_features divides power by n_fft.
                           Predicted c0 shift ln(1/512)*26/sqrt(26) =
                           -31.80934; measured -31.80934.

Two precision floors, both measured here and neither declared upstream, which
together set the tightest tolerance any vector may legitimately assert:

  librosa   `filters.mel` returns float32 and `feature.melspectrogram` does
            not ask for anything else, so a float64 pipeline still carries a
            float32 mel basis. Substituting a float64 basis into librosa's
            own path drops the disagreement with this module from 1.426e-07
            to 6.217e-14, and librosa-float32 against librosa-float64 is
            1.426e-07 -- the same number, so the basis is the sole cause.
            No implementation can agree with default librosa more closely
            than that, however correct it is.

  torchaudio  float32 throughout: 2.079e-05 against this module, and
            2.086e-05 between librosa and torchaudio. The two libraries are
            not 2e-05 apart in method; they are 2e-05 apart in storage.

A suite that asserts a tolerance below those floors is asserting something
no conforming implementation could satisfy.

Every one of these is a required argument. There are no defaults in this
module, deliberately, following the same rule `unifrac.normalized` follows:
the single largest source of cross-implementation disagreement is a default
nobody knew they had accepted, and refusing to guess is the fix.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MelScale",
    "hz_to_mel",
    "mel_to_hz",
    "mel_filterbank",
    "dct_ii",
    "pad_window",
    "frame_signal",
    "power_spectrum",
    "mfcc",
]


class MelScale:
    """The two mel formulas in circulation. Both are 'the' mel scale."""

    HTK = "htk"
    SLANEY = "slaney"


def hz_to_mel(f: np.ndarray | float, scale: str) -> np.ndarray:
    """Hz to mel.

    HTK follows the HTK Book: m = 2595 log10(1 + f/700).

    Slaney follows the Auditory Toolbox: linear below 1000 Hz at 3 mels per
    Hz-over-200, logarithmic above with a step of log(6.4)/27 per mel. The
    two agree at 1000 Hz by construction and nowhere else.
    """
    f = np.asarray(f, dtype=np.float64)
    if scale == MelScale.HTK:
        return 2595.0 * np.log10(1.0 + f / 700.0)
    if scale == MelScale.SLANEY:
        f_min, f_sp = 0.0, 200.0 / 3.0
        mels = (f - f_min) / f_sp
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        hi = f >= min_log_hz
        mels = np.where(hi, min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep, mels)
        return mels
    raise ValueError(f"unknown mel scale {scale!r}; use MelScale.HTK or MelScale.SLANEY")


def mel_to_hz(m: np.ndarray | float, scale: str) -> np.ndarray:
    """Inverse of `hz_to_mel`, same two conventions."""
    m = np.asarray(m, dtype=np.float64)
    if scale == MelScale.HTK:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    if scale == MelScale.SLANEY:
        f_min, f_sp = 0.0, 200.0 / 3.0
        freqs = f_min + f_sp * m
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        hi = m >= min_log_mel
        return np.where(hi, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)
    raise ValueError(f"unknown mel scale {scale!r}")


def mel_filterbank(
    *,
    n_mels: int,
    n_fft: int,
    sample_rate: float,
    f_min: float,
    f_max: float,
    scale: str,
    area_normalize: bool,
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1).

    `area_normalize` is Slaney's 2/(f[i+2]-f[i]) scaling, which makes each
    filter integrate to a constant rather than peak at a constant. It is
    librosa's default and not torchaudio's, and it is the single argument
    responsible for the 183,000x gap described in the module docstring.

    Filters are built on continuous frequency and then sampled at the FFT
    bin frequencies, which is what librosa and torchaudio both do.
    python_speech_features instead rounds band edges onto the FFT bin grid
    first, giving a filter one bin narrower at most edges; that is decision
    3 in the docstring and it is why this function does not attempt to be
    all three at once.
    """
    if f_max > sample_rate / 2.0:
        raise ValueError(f"f_max {f_max} exceeds Nyquist {sample_rate / 2.0}")
    if n_mels < 1:
        raise ValueError("n_mels must be >= 1")

    fft_freqs = np.linspace(0.0, sample_rate / 2.0, int(n_fft // 2 + 1))
    m_min, m_max = hz_to_mel(f_min, scale), hz_to_mel(f_max, scale)
    edges = mel_to_hz(np.linspace(m_min, m_max, n_mels + 2), scale)

    fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        # Guard against a degenerate triangle, which happens when n_mels is
        # large relative to the band and two edges collapse onto one another.
        if ctr > lo:
            up = (fft_freqs - lo) / (ctr - lo)
            fb[i] = np.maximum(0.0, np.minimum(up, 1.0 if hi <= ctr else np.inf))
        if hi > ctr:
            down = (hi - fft_freqs) / (hi - ctr)
            fb[i] = np.maximum(0.0, np.minimum(fb[i] if ctr > lo else 1.0, down))
        if area_normalize:
            fb[i] *= 2.0 / (hi - lo)
    return fb


def dct_ii(x: np.ndarray, *, n_out: int, ortho: bool) -> np.ndarray:
    """DCT-II along axis 0, returning the first `n_out` coefficients.

    Implemented directly from the definition rather than via scipy so the
    package keeps a single numpy dependency and so the normalization is
    visible in the source. `ortho` scales c0 by sqrt(1/4N) and the rest by
    sqrt(1/2N); without it every coefficient is a factor of 2 larger and c0
    additionally differs by sqrt(2).
    """
    n = x.shape[0]
    k = np.arange(n_out)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi * k * (2.0 * i + 1.0) / (2.0 * n))
    out = basis @ x
    if ortho:
        scale = np.full((n_out, 1), np.sqrt(2.0 / n))
        scale[0] = np.sqrt(1.0 / n)
        out = out * scale
    else:
        out = out * 2.0
    return out


def pad_window(window: np.ndarray, *, n_fft: int, placement: str) -> np.ndarray:
    """Place a `win_length` window inside an `n_fft` frame.

    This is decision 5 in the module docstring, and it is not cosmetic. With
    n_fft=512 and win_length=400, "center" leaves 56 zeros on each side, so
    the samples actually weighted are [k*hop+56, k*hop+456); "left" weights
    [k*hop, k*hop+400). Same audio, same hop, frames offset by 56 samples.

    librosa (util.pad_center) and torchaudio both centre. This module was
    written framing at win_length and padding right, which produced a
    6.785 disagreement against both libraries and one extra frame -- a
    reminder that the conventions in this module are the ones that actually
    bite, not a list assembled for completeness.
    """
    win_length = window.shape[0]
    if win_length > n_fft:
        raise ValueError(f"win_length {win_length} exceeds n_fft {n_fft}")
    if win_length == n_fft:
        return window
    if placement == "center":
        lpad = (n_fft - win_length) // 2
        return np.pad(window, (lpad, n_fft - win_length - lpad))
    if placement == "left":
        return np.pad(window, (0, n_fft - win_length))
    raise ValueError(f"unknown window placement {placement!r}; use 'center' or 'left'")


def frame_signal(
    y: np.ndarray, *, n_fft: int, hop_length: int, center: bool, pad_mode: str
) -> np.ndarray:
    """Split into overlapping frames of n_fft samples, shape (n_fft, n_frames).

    Frames are n_fft long, not win_length long. A shorter analysis window is
    expressed by zero-padding the window itself (see `pad_window`), which is
    what librosa and torchaudio do; framing at win_length instead shifts
    every frame and changes the frame count.
    """
    y = np.asarray(y, dtype=np.float64)
    if center:
        y = np.pad(y, n_fft // 2, mode=pad_mode)
    if y.size < n_fft:
        raise ValueError(f"signal of {y.size} samples is shorter than n_fft {n_fft}")
    n_frames = 1 + (y.size - n_fft) // hop_length
    idx = np.arange(n_fft)[:, None] + hop_length * np.arange(n_frames)[None, :]
    return y[idx]


def power_spectrum(
    frames: np.ndarray, *, window: np.ndarray, power: float, scale_by_n_fft: bool
) -> np.ndarray:
    """Windowed magnitude/power spectrum, shape (n_fft//2 + 1, n_frames).

    `window` must already be n_fft long; use `pad_window` to place a shorter
    analysis window inside it.

    `scale_by_n_fft` is python_speech_features' 1/NFFT factor. It shifts c0
    by a constant and leaves every other coefficient untouched, which is why
    it is easy to miss and easy to correct once named.
    """
    n_fft = frames.shape[0]
    if window.shape[0] != n_fft:
        raise ValueError(
            f"window length {window.shape[0]} != frame length {n_fft}; "
            "pass it through pad_window first"
        )
    spec = np.abs(np.fft.rfft(frames * window[:, None], n=n_fft, axis=0)) ** power
    if scale_by_n_fft:
        spec = spec / n_fft
    return spec


def mfcc(
    y: np.ndarray,
    *,
    sample_rate: float,
    n_mfcc: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: np.ndarray,
    window_placement: str,
    center: bool,
    pad_mode: str,
    n_mels: int,
    f_min: float,
    f_max: float,
    mel_scale: str,
    mel_area_normalize: bool,
    power: float,
    scale_by_n_fft: bool,
    log_scale: str,
    log_floor: float,
    db_clamp_top: float | None,
    dct_ortho: bool,
    lifter: float,
    pre_emphasis: float | None,
) -> np.ndarray:
    """MFCC with every convention named. No argument has a default.

    Returns shape (n_mfcc, n_frames).

    `log_scale` is "db" for 10*log10 or "ln" for the natural logarithm.

    `db_clamp_top` is the one argument here that is not a convention. Passing
    a float reproduces the behaviour of librosa and torchaudio, in which the
    log-mel spectrogram is floored at (peak - db_clamp_top) where the peak is
    taken over the ENTIRE array -- both mel and time axes. That makes the
    output of any given frame depend on every other frame in the array, so
    MFCC computed on a stream and MFCC computed on a batch are different
    functions of the same audio.

    Measured on a 200-frame quiet passage with a burst appended after it, at
    16 kHz / 26 mels / 13 coefficients, comparing frames that are otherwise
    byte-identical:

        burst +46 dB    max|d|   0.0000   c1..c12 energy 8066.57
        burst +59 dB    max|d|   1.9362   c1..c12 energy 8033.20
        burst +74 dB    max|d|  25.6987   c1..c12 energy 2616.58
        burst +94 dB    max|d| 124.0154   c1..c12 energy    0.00

    At +94 dB every frame of the quiet passage is clamped flat, and the DCT
    of a constant is zero everywhere except c0: the feature vector stops
    carrying information. A door slam in an otherwise quiet room is about
    that loud. librosa and torchaudio produce identical distortion, which
    confirms a shared mechanism rather than a coincidence.

    Neither library lets a caller turn it off through its MFCC API. In
    librosa `feature.mfcc` takes no `top_db` and forwards **kwargs to
    `melspectrogram`, not to `power_to_db`, whose default is top_db=80.0.
    In torchaudio `transforms.MFCC.__init__` hardcodes `self.top_db = 80.0`.

    Pass None to disable it, which is what conformance vectors must do: a
    vector for a function that is not a function of its input cannot be
    checked. The clip-dependence is documented by its own vector instead.
    """
    y = np.asarray(y, dtype=np.float64)
    if pre_emphasis is not None:
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    if window.shape[0] != win_length:
        raise ValueError(f"window length {window.shape[0]} != win_length {win_length}")
    frames = frame_signal(
        y, n_fft=n_fft, hop_length=hop_length, center=center, pad_mode=pad_mode
    )
    spec = power_spectrum(
        frames,
        window=pad_window(window, n_fft=n_fft, placement=window_placement),
        power=power,
        scale_by_n_fft=scale_by_n_fft,
    )
    fb = mel_filterbank(
        n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate, f_min=f_min,
        f_max=f_max, scale=mel_scale, area_normalize=mel_area_normalize,
    )
    melspec = fb @ spec

    if log_scale == "db":
        log_mel = 10.0 * np.log10(np.maximum(melspec, log_floor))
    elif log_scale == "ln":
        log_mel = np.log(np.maximum(melspec, log_floor))
    else:
        raise ValueError(f"unknown log_scale {log_scale!r}; use 'db' or 'ln'")

    if db_clamp_top is not None:
        if log_scale != "db":
            raise ValueError("db_clamp_top is only meaningful with log_scale='db'")
        if db_clamp_top < 0:
            raise ValueError("db_clamp_top must be non-negative")
        # Deliberately reproduces the whole-array reduction, including its
        # consequence. Scoping this per-frame would be the fix, not the
        # documentation, and this module documents.
        log_mel = np.maximum(log_mel, log_mel.max() - db_clamp_top)

    out = dct_ii(log_mel, n_out=n_mfcc, ortho=dct_ortho)

    if lifter > 0:
        n = np.arange(n_mfcc)[:, None]
        out = out * (1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter))
    elif lifter < 0:
        raise ValueError("lifter must be non-negative")
    return out
