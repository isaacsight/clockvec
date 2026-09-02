"""Tests for the Promax reference implementation and its vector.

numpy-only, like test_mfcc. Agreement with R is checked by the optional
harness in harness/check_promax_r.py; what is tested here are the
properties that hold whether or not R is installed.
"""

import numpy as np
import pytest

from clockvec.promax import canonicalize, promax, varimax
from clockvec.promax_vectors import (
    FLAGS, PromaxDeclaration, build_vectors, loadings, tolerance_convergence, _pinned,
)
from clockvec.vector import Result

EPS = 1e-5
IT = 1000


def _L():
    return loadings(12, 3, 20260901)


# --- varimax ------------------------------------------------------------------

@pytest.mark.parametrize("norm", [True, False])
def test_varimax_rotation_is_orthogonal(norm):
    z, rot = varimax(_L(), kaiser_normalize=norm, eps=EPS, max_iter=IT)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.allclose(z, (_L() @ rot) if not norm else z)


@pytest.mark.parametrize("norm", [True, False])
def test_varimax_preserves_communalities(norm):
    """An orthogonal rotation cannot change how much of each variable the
    factors explain. With normalization the scaling is undone afterwards, so
    the invariant holds in both modes."""
    L = _L()
    z, _ = varimax(L, kaiser_normalize=norm, eps=EPS, max_iter=IT)
    assert np.allclose((z**2).sum(1), (L**2).sum(1), atol=1e-12)


def test_varimax_increases_the_criterion():
    """Kaiser's criterion: sum over factors of the variance of squared
    loadings. Rotating must not lower it."""
    L = _L()
    def crit(x):
        s = x**2
        return float((s.var(axis=0)).sum())
    z, _ = varimax(L, kaiser_normalize=False, eps=EPS, max_iter=IT)
    assert crit(z) > crit(L)


def test_normalization_matters_only_when_communalities_differ():
    """The reason the input has spread communalities. With every row the
    same length the two conventions coincide exactly, so a vector built on
    such a matrix would document nothing."""
    L = _L()
    equal = L / np.linalg.norm(L, axis=1, keepdims=True) * 0.7
    a, _ = varimax(equal, kaiser_normalize=True, eps=EPS, max_iter=IT)
    b, _ = varimax(equal, kaiser_normalize=False, eps=EPS, max_iter=IT)
    assert np.abs(canonicalize(a) - canonicalize(b)).max() < 1e-9
    a, _ = varimax(L, kaiser_normalize=True, eps=EPS, max_iter=IT)
    b, _ = varimax(L, kaiser_normalize=False, eps=EPS, max_iter=IT)
    assert np.abs(canonicalize(a) - canonicalize(b)).max() > 1e-4


def test_varimax_rejects_one_factor_and_bad_eps():
    with pytest.raises(ValueError, match="at least two"):
        varimax(np.ones((5, 1)), kaiser_normalize=True, eps=EPS, max_iter=IT)
    with pytest.raises(ValueError, match="eps must be positive"):
        varimax(_L(), kaiser_normalize=True, eps=0.0, max_iter=IT)


def test_zero_communality_cannot_be_normalized():
    L = _L()
    L[0] = 0.0
    with pytest.raises(ValueError, match="zero communality"):
        varimax(L, kaiser_normalize=True, eps=EPS, max_iter=IT)


# --- promax -------------------------------------------------------------------

def test_promax_factors_have_unit_variance():
    """The column normalization in Hendrickson & White keeps diag(Phi) = 1."""
    out = promax(_L(), m=4, kaiser_normalize=True, eps=EPS, max_iter=IT)
    assert np.allclose(np.diag(out["phi"]), 1.0, atol=1e-12)


def test_promax_reproduces_the_pattern_from_the_rotation():
    L = _L()
    out = promax(L, m=4, kaiser_normalize=False, eps=EPS, max_iter=IT)
    assert np.allclose(L @ out["rotmat"], out["loadings"], atol=1e-12)


def test_promax_with_m_one_is_varimax():
    """Raising to the first power fits the loadings to themselves, so the
    transform is the identity and Promax collapses to its varimax stage."""
    L = _L()
    z, _ = varimax(L, kaiser_normalize=True, eps=EPS, max_iter=IT)
    out = promax(L, m=1, kaiser_normalize=True, eps=EPS, max_iter=IT)
    assert np.allclose(out["loadings"], z, atol=1e-10)


def test_promax_rejects_power_below_one():
    with pytest.raises(ValueError, match="m must be"):
        promax(_L(), m=0.5, kaiser_normalize=True, eps=EPS, max_iter=IT)


# --- canonicalization ---------------------------------------------------------

def test_canonicalize_is_invariant_to_column_order_and_sign():
    z = _L()
    scrambled = z[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    assert np.allclose(canonicalize(z), canonicalize(scrambled))


def test_canonicalize_is_idempotent():
    z = canonicalize(_L())
    assert np.array_equal(canonicalize(z), z)


def test_canonicalize_makes_the_dominant_entry_positive():
    z = canonicalize(_L())
    idx = np.abs(z).argmax(axis=0)
    assert (z[idx, np.arange(3)] > 0).all()


# --- declarations and tolerance -----------------------------------------------

def test_declaration_requires_every_convention():
    with pytest.raises(TypeError):
        PromaxDeclaration(n_variables=12, n_factors=3)


def test_declaration_is_frozen():
    with pytest.raises(Exception):
        _pinned().power_m = 3.0


def test_declaration_carries_the_input_itself():
    """An R user with no numpy must be able to run the vector from the JSON."""
    d = _pinned()
    assert len(d.input_loadings) == 36
    assert np.allclose(np.asarray(d.input_loadings).reshape(12, 3), loadings(12, 3, 20260901))


def test_tolerance_is_ten_times_the_stopping_eps():
    tol, why = tolerance_convergence(1e-5)
    assert tol == pytest.approx(1e-4)
    assert len(why.strip()) > 40


# --- the vector ---------------------------------------------------------------

def test_vector_builds_and_validates():
    (v,) = build_vectors()
    v.validate()
    assert v.vector_id == "promax/kaiser-normalization"
    assert len(v.expectations) == 2
    assert all(len(e.values) == 36 for e in v.expectations)


def test_manifest_id_ignores_expectations():
    (v,) = build_vectors()
    before = v.manifest_id
    v.expectations = []
    assert v.manifest_id == before


def test_the_two_expectations_are_separated_by_more_than_the_tolerance():
    """If they were not, a dead `normalize` argument would be undetectable
    and the vector would be a no-op."""
    (v,) = build_vectors()
    a, b = (np.asarray(e.values) for e in v.expectations)
    gap = float(np.abs(a - b).max())
    assert gap > 10 * v.expectations[0].tolerance
    assert gap == pytest.approx(1.38e-3, rel=0.05)


def test_vector_stays_silent_where_the_paper_does():
    (v,) = build_vectors()
    assert all(e.result is Result.ACCEPTABLE for e in v.expectations)
    assert v.disagreements() == []


def test_flags_are_defined_and_cited():
    (v,) = build_vectors()
    for e in v.expectations:
        for f in e.flags:
            assert f in v.flags
            assert v.flags[f].citations
    assert "DeadNormalizeArgument" in FLAGS
    assert FLAGS["DeadNormalizeArgument"].bug_type == "DeadArgument"
