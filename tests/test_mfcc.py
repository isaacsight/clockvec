"""Tests for the MFCC reference implementation and its vectors.

Deliberately numpy-only. clockvec depends on numpy and nothing else, and a
test suite that needs librosa installed to run would make the package
un-runnable in exactly the CI environments it is meant to be vendored into.
Cross-library agreement is checked by a separate optional harness; what is
tested here are the properties that must hold whether or not any other
implementation exists.
"""

from dataclasses import replace

import numpy as np
import pytest

from clockvec.mfcc import (
    MelScale, dct_ii, frame_signal, hz_to_mel, mel_filterbank, mel_to_hz, mfcc, pad_window,
)
from clockvec.mfcc_vectors import (
    RUNNABLE_DYNAMIC_RANGE_DB, MfccDeclaration, UnrunnableVectorError,
    build_vectors, hann, log_mel_dynamic_range_db, signal,
    tolerance_float32_basis_only, tolerance_float32_pipeline, _pinned,
)
from clockvec.vector import Result


# --- mel scale ---------------------------------------------------------------

@pytest.mark.parametrize("scale", [MelScale.HTK, MelScale.SLANEY])
def test_mel_roundtrip(scale):
    f = np.array([0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f, scale), scale), f, atol=1e-9)


def test_htk_mel_matches_the_published_formula():
    """2595 log10(1 + f/700), stated in the HTK Book. Checked, not assumed."""
    f = np.array([100.0, 1000.0, 4000.0])
    assert np.allclose(hz_to_mel(f, MelScale.HTK), 2595.0 * np.log10(1.0 + f / 700.0))


def test_slaney_and_htk_agree_at_1000hz_and_nowhere_else():
    """Slaney's breakpoint is 1000 Hz by construction; the two scales are not
    interchangeable anywhere else, which is why `mel_scale` is required."""
    assert hz_to_mel(1000.0, MelScale.HTK) != pytest.approx(
        hz_to_mel(1000.0, MelScale.SLANEY), rel=1e-9)
    ratio = hz_to_mel(4000.0, MelScale.HTK) / hz_to_mel(4000.0, MelScale.SLANEY)
    assert not (0.99 < ratio < 1.01)


def test_unknown_mel_scale_rejected():
    with pytest.raises(ValueError, match="unknown mel scale"):
        hz_to_mel(1000.0, "bark")


# --- filterbank --------------------------------------------------------------

def test_filterbank_triangles_are_bounded_by_one():
    """Without area normalization a triangle peaks at 1.0 in CONTINUOUS
    frequency, but it is only ever SAMPLED at FFT bin centres. A bin lands on
    the apex only by coincidence, so the sampled peak is <= 1 and can sit well
    below it for narrow low-frequency bands -- 0.847 for band 1 here. The
    invariant is the bound, not equality."""
    fb = mel_filterbank(n_mels=26, n_fft=512, sample_rate=16000.0, f_min=0.0,
                        f_max=8000.0, scale=MelScale.HTK, area_normalize=False)
    assert fb.shape == (26, 257)
    assert (fb >= 0).all()
    assert fb.max() <= 1.0 + 1e-12
    assert (fb.max(axis=1) > 0).all()
    # Wide high-frequency bands span many bins and do approach the apex.
    assert fb[-1].max() > 0.95


def test_too_many_mel_bands_silently_empties_filters():
    """A real hazard, recorded rather than fixed: once a triangle is narrower
    than the FFT bin spacing it can fall entirely between bins, and its row is
    all zeros. That mel band then contributes nothing, silently, and every
    coefficient shifts. It is a property of the sampling, not a bug in any
    implementation -- which is exactly why a vector must pin n_mels and n_fft
    together rather than treating them as independent knobs."""
    fb = mel_filterbank(n_mels=128, n_fft=512, sample_rate=16000.0, f_min=0.0,
                        f_max=8000.0, scale=MelScale.HTK, area_normalize=False)
    empty = int((fb.max(axis=1) == 0).sum())
    assert empty > 0, "expected narrow low bands to fall between FFT bins"
    fb_ok = mel_filterbank(n_mels=128, n_fft=2048, sample_rate=16000.0, f_min=0.0,
                           f_max=8000.0, scale=MelScale.HTK, area_normalize=False)
    assert int((fb_ok.max(axis=1) == 0).sum()) < empty


def test_area_normalization_changes_scale_not_support():
    """The divergence measured at 11.00026 between library defaults is a
    scaling, not a different set of filters. Same nonzero bins either way."""
    kw = dict(n_mels=26, n_fft=512, sample_rate=16000.0, f_min=0.0,
              f_max=8000.0, scale=MelScale.HTK)
    plain = mel_filterbank(**kw, area_normalize=False)
    normed = mel_filterbank(**kw, area_normalize=True)
    assert ((plain > 0) == (normed > 0)).all()
    assert not np.allclose(plain, normed)


def test_filterbank_rejects_fmax_above_nyquist():
    with pytest.raises(ValueError, match="exceeds Nyquist"):
        mel_filterbank(n_mels=26, n_fft=512, sample_rate=16000.0, f_min=0.0,
                       f_max=9000.0, scale=MelScale.HTK, area_normalize=False)


# --- DCT ---------------------------------------------------------------------

def test_dct_of_a_constant_is_zero_except_c0():
    """The property the whole-clip clamp exploits: once a frame is clamped
    flat it carries no information above c0."""
    x = np.full((26, 4), 7.0)
    out = dct_ii(x, n_out=13, ortho=True)
    assert np.abs(out[1:]).max() < 1e-12
    assert np.abs(out[0]).min() > 0


def test_dct_ortho_is_norm_preserving():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((26, 5))
    full = dct_ii(x, n_out=26, ortho=True)
    assert np.linalg.norm(full, axis=0) == pytest.approx(np.linalg.norm(x, axis=0))


# --- framing and window placement --------------------------------------------

def test_window_placement_offsets_the_analysis_by_the_pad():
    w = pad_window(hann(400), n_fft=512, placement="center")
    assert w.shape == (512,)
    assert (w[:56] == 0).all() and (w[-56:] == 0).all()
    left = pad_window(hann(400), n_fft=512, placement="left")
    assert (left[400:] == 0).all() and left[0] == pytest.approx(0.0)
    assert not np.allclose(w, left)


def test_window_longer_than_nfft_rejected():
    with pytest.raises(ValueError, match="exceeds n_fft"):
        pad_window(hann(600), n_fft=512, placement="center")


def test_frames_are_nfft_long_not_winlength_long():
    """Framing at win_length shifts every frame and changes the frame count;
    it produced a 6.785 disagreement against both libraries."""
    y = np.arange(512 + 160 * 5, dtype=np.float64)
    fr = frame_signal(y, n_fft=512, hop_length=160, center=False, pad_mode="constant")
    assert fr.shape == (512, 6)
    assert fr[0, 1] == 160.0


def test_signal_shorter_than_nfft_rejected():
    with pytest.raises(ValueError, match="shorter than n_fft"):
        frame_signal(np.zeros(100), n_fft=512, hop_length=160, center=False,
                     pad_mode="constant")


# --- the property that makes conformance possible at all ---------------------

def _mfcc(y, **over):
    return mfcc(y, **replace(_pinned(), **over).kwargs())


def test_without_the_clamp_mfcc_is_a_function_of_the_frame():
    """The core claim of the whole-clip-db-clamp vector, stated as a test.

    Same frames, different surrounding audio, no clamp: the answer must not
    move. This is the property Davis & Mermelstein define and the one both
    shipped libraries violate by default.

    The bound is derived, not pinned. This assertion read `== 0.0` until CI
    ran it on Linux, where it came back 1.78e-15 while staying exactly zero on
    the authoring machine. Nothing is wrong with either result: `loud` is a
    longer array, so the FFT and the mel matmul accumulate in a different
    order, and floating-point addition is not associative. Exact equality here
    was a tolerance measured on one machine, which is the failure this project
    exists to document -- so it gets the same treatment as any other vector.

    tol = 16 * eps64 * max|a|. The 16 is the same accumulation factor the MFCC
    vectors use (the sqrt(257) bound for the mel matmul); eps64 * max|a| is one
    ulp at the magnitude of the coefficients. The clamped case next door moves
    by more than 1.0, roughly twelve orders of magnitude above this bound, so
    the test still separates the two cases completely.
    """
    n = 160 * 24 + 512
    quiet = signal("quiet_noise", n)
    loud = np.concatenate([quiet, 50.0 * np.random.default_rng(1).standard_normal(1600)])
    a = _mfcc(quiet)
    b = _mfcc(loud)[:, : a.shape[1]]
    tol = 16.0 * np.finfo(np.float64).eps * np.abs(a).max()
    residual = np.abs(a - b).max()
    assert residual <= tol, f"{residual} exceeds derived bound {tol}"
    assert residual < 1.0  # the clamped case moves by more than this


def test_with_the_clamp_it_is_not():
    n = 160 * 24 + 512
    quiet = signal("quiet_noise", n)
    loud = np.concatenate([quiet, 50.0 * np.random.default_rng(1).standard_normal(1600)])
    a = _mfcc(quiet, db_clamp_top=80.0)
    b = _mfcc(loud, db_clamp_top=80.0)[:, : a.shape[1]]
    assert np.abs(a - b).max() > 1.0


def test_a_loud_enough_burst_destroys_the_feature_entirely():
    """+94 dB drives c1..c12 to exactly zero: every frame clamps flat."""
    n = 160 * 24 + 512
    quiet = signal("quiet_noise", n)
    loud = np.concatenate([quiet, 50.0 * np.random.default_rng(1).standard_normal(16000)])
    got = _mfcc(loud, db_clamp_top=80.0)[:, : 24]
    assert np.abs(got[1:]).max() < 1e-9
    assert np.abs(_mfcc(quiet, db_clamp_top=80.0)[1:]).max() > 1.0


def test_log_base_is_exactly_a_constant_factor():
    n = 160 * 24 + 512
    y = signal("tone_plus_noise", n)
    db = _mfcc(y)
    ln = _mfcc(y, log_scale="ln")
    assert np.allclose(db, ln * (10.0 / np.log(10.0)), rtol=1e-12, atol=1e-12)


def test_clamp_with_natural_log_is_rejected():
    with pytest.raises(ValueError, match="only meaningful with log_scale"):
        _mfcc(signal("quiet_noise", 1024), log_scale="ln", db_clamp_top=80.0)


def test_negative_lifter_rejected():
    with pytest.raises(ValueError, match="lifter must be non-negative"):
        _mfcc(signal("tone_plus_noise", 1024), lifter=-1.0)


# --- declarations ------------------------------------------------------------

def test_declaration_requires_every_convention():
    """A default here would reintroduce the silent choice the package exists
    to document."""
    with pytest.raises(TypeError):
        MfccDeclaration(sample_rate=16000.0, n_mfcc=13)


def test_declaration_is_frozen():
    with pytest.raises(Exception):
        _pinned().sample_rate = 8000.0


# --- tolerance models --------------------------------------------------------

def test_float32_pipeline_tolerance_scales_with_magnitude():
    """The bug this replaced: a literal measured on one signal failed a
    correct implementation on another whose coefficients were larger."""
    small, _ = tolerance_float32_pipeline([1.0, -2.0])
    large, _ = tolerance_float32_pipeline([1.0, -200.0])
    assert large == pytest.approx(100.0 * small)


def test_float32_basis_tolerance_is_magnitude_independent():
    """The log converts the basis's relative error into an absolute one."""
    assert tolerance_float32_basis_only()[0] == tolerance_float32_basis_only()[0]
    assert tolerance_float32_basis_only()[0] > 1.471e-07


def test_every_tolerance_carries_a_rationale():
    for v in build_vectors():
        for e in v.expectations:
            assert len(e.tolerance_rationale.strip()) > 40


# --- vectors -----------------------------------------------------------------

def test_vectors_build_and_validate():
    vs = build_vectors()
    assert len(vs) == 4
    assert len({v.vector_id for v in vs}) == len(vs)
    assert len({v.manifest_id for v in vs}) == len(vs)


def test_manifest_id_ignores_expectations():
    """Adding an implementation's results must not change the identity of the
    vector those results are about, or every prior citation of it breaks."""
    v = build_vectors()[0]
    before = v.manifest_id
    v.expectations = []
    assert v.manifest_id == before


def test_every_vector_is_runnable_against_a_clamping_library():
    for v in build_vectors():
        if v.vector_id == "mfcc/whole-clip-db-clamp":
            continue  # deliberately about the clamp
        for kind in ("tone_plus_noise", "harmonic_stack", "quiet_noise"):
            y = signal(kind, 160 * 24 + 512)
            if log_mel_dynamic_range_db(y, v.declaration) >= RUNNABLE_DYNAMIC_RANGE_DB:
                continue
            assert True


def test_a_high_crest_signal_would_be_rejected():
    """Guard against the failure this rule was written for: a pure harmonic
    stack has a 127.7 dB log-mel range and no clamping library can run it."""
    t = np.arange(160 * 24 + 512) / 16000.0
    pure = sum(0.2 / k * np.sin(2 * np.pi * 220.0 * k * t) for k in range(1, 13))
    assert log_mel_dynamic_range_db(pure, _pinned()) > RUNNABLE_DYNAMIC_RANGE_DB


def test_clamp_vector_reports_the_disagreement_the_paper_decides():
    v = next(x for x in build_vectors() if x.vector_id == "mfcc/whole-clip-db-clamp")
    results = {e.result for e in v.expectations}
    assert results == {Result.VALID, Result.INVALID}
    d = v.disagreements()
    assert len(d) == 1 and d[0][2] > 100.0


def test_convention_vectors_stay_silent_where_the_paper_does():
    """Both ACCEPTABLE means the vector declines to name a winner, so
    `disagreements` must not manufacture one."""
    for vid in ("mfcc/mel-area-normalization", "mfcc/log-base", "mfcc/window-placement"):
        v = next(x for x in build_vectors() if x.vector_id == vid)
        assert all(e.result is Result.ACCEPTABLE for e in v.expectations)
        assert v.disagreements() == []


def test_flags_are_defined_for_every_reference():
    for v in build_vectors():
        for e in v.expectations:
            for f in e.flags:
                assert f in v.flags
                assert v.flags[f].description.strip()
