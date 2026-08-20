import numpy as np
import pytest

from clockvec.generate import bimodal_betas, replicate_pair, with_missing


def test_same_seed_is_byte_identical():
    """The whole format rests on this. If a seed does not reproduce a
    matrix exactly, a vector id identifies nothing."""
    a = bimodal_betas(500, 8, seed=429)
    b = bimodal_betas(500, 8, seed=429)
    assert a.tobytes() == b.tobytes()


def test_different_seed_differs():
    a = bimodal_betas(500, 8, seed=429)
    b = bimodal_betas(500, 8, seed=430)
    assert not np.array_equal(a, b)


def test_betas_are_in_unit_interval():
    betas = bimodal_betas(1000, 10, seed=1, noise=0.05)
    assert betas.min() >= 0.0
    assert betas.max() <= 1.0
    assert not np.isnan(betas).any()


def test_shape_is_exact_despite_integer_split():
    """The three mode blocks are sized by truncating float fractions, so
    the remainder has to absorb the rounding or the matrix comes out short."""
    for n in (7, 99, 100, 173, 353):
        assert bimodal_betas(n, 3, seed=2).shape == (n, 3)


def test_distribution_is_bimodal_not_uniform():
    """Guards the reason this generator exists. A uniform draw would put
    roughly a third of the mass in the middle third; a real methylation
    array puts very little there."""
    betas = bimodal_betas(5000, 4, seed=3).ravel()
    middle = np.mean((betas > 0.35) & (betas < 0.65))
    assert middle < 0.25, f"middle mass {middle:.3f} looks uniform, not bimodal"

    low = np.mean(betas < 0.2)
    high = np.mean(betas > 0.8)
    assert low > 0.25 and high > 0.25, f"modes too weak: low={low:.3f} high={high:.3f}"


def test_noise_must_be_in_range():
    with pytest.raises(ValueError):
        bimodal_betas(10, 2, seed=1, noise=0.9)
    with pytest.raises(ValueError):
        bimodal_betas(10, 2, seed=1, noise=-0.1)


def test_replicate_pair_differs_but_tracks():
    """A replicate is the same biology measured twice: not identical, but
    far more similar to its partner than two unrelated samples would be."""
    a, b = replicate_pair(2000, 6, seed=21, noise=0.02)
    assert not np.array_equal(a, b)

    paired = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    shuffled = np.corrcoef(a.ravel(), np.random.default_rng(0).permutation(b.ravel()))[0, 1]
    assert paired > shuffled


def test_replicate_pair_is_reproducible():
    a1, b1 = replicate_pair(300, 4, seed=7, noise=0.01)
    a2, b2 = replicate_pair(300, 4, seed=7, noise=0.01)
    assert a1.tobytes() == a2.tobytes()
    assert b1.tobytes() == b2.tobytes()


def test_with_missing_punches_roughly_the_right_fraction():
    betas = bimodal_betas(1000, 10, seed=4)
    holed = with_missing(betas, fraction=0.1, seed=4)
    observed = np.isnan(holed).mean()
    assert 0.08 < observed < 0.12, f"expected ~0.10 NaN, got {observed:.3f}"


def test_with_missing_does_not_mutate_input():
    betas = bimodal_betas(100, 5, seed=5)
    before = betas.tobytes()
    with_missing(betas, fraction=0.2, seed=5)
    assert betas.tobytes() == before


def test_with_missing_rejects_bad_fraction():
    betas = bimodal_betas(10, 2, seed=1)
    with pytest.raises(ValueError):
        with_missing(betas, fraction=1.0, seed=1)
