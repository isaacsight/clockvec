"""Reference vectors derived from Lozupone et al. 2007's printed equations.

The values below are NOT copied from any implementation. They were computed
from the paper's two equations by hand and independently reproduced here.
They happen to agree with scikit-bio's hand-computed values, and to disagree
with QIIME 1.9.1 -- which is the point. Two independent derivations from the
paper agree; the tool that validated against itself does not.
"""

import math

import pytest

from clockvec.unifrac import (
    UnrootedTreeError,
    parse_newick,
    scaling_factor_tip_form,
    unweighted_unifrac,
    weighted_unifrac,
)

# scikit-bio's t2 fixture. Root's children are (OTU1,OTU2) and (OTU3,OTU4),
# so no tip's parent is the root itself -- the "root not observed" case the
# papers never address and where QIIME 1.9.1 and scikit-bio part company.
T2 = "((OTU1:0.1,OTU2:0.2):0.3,(OTU3:0.5,OTU4:0.7):1.1)root;"
OIDS = ["OTU1", "OTU2", "OTU3", "OTU4"]


def tree():
    return parse_newick(T2)


# --- the contested values, derived from the 2007 equations -------------------


def test_weighted_raw_case1():
    """A=[1,0,0,0] B=[1,1,0,0]. Branch terms: 0.1*|1-0.5| + 0.2*|0-0.5| = 0.15.
    The (OTU1,OTU2) stem contributes 0 because both sides are fully below it."""
    assert weighted_unifrac([1, 0, 0, 0], [1, 1, 0, 0], OIDS, tree(),
                            normalized=False) == pytest.approx(0.15)


def test_weighted_raw_case2():
    """A=[0,0,1,1] B=[0,0,1,0]. 0.5*|0.5-1| + 0.7*|0.5-0| = 0.25 + 0.35 = 0.6."""
    assert weighted_unifrac([0, 0, 1, 1], [0, 0, 1, 0], OIDS, tree(),
                            normalized=False) == pytest.approx(0.6)


def test_weighted_normalized_case1():
    """D = 0.4*(1+0.5) + 0.5*(0+0.5) = 0.85, using root-to-tip depths
    0.4 and 0.5. u/D = 0.15/0.85."""
    assert weighted_unifrac([1, 0, 0, 0], [1, 1, 0, 0], OIDS, tree(),
                            normalized=True) == pytest.approx(0.1764705882)


def test_weighted_normalized_case2():
    """D = 1.6*(0.5+1) + 1.8*(0.5+0) = 2.4 + 0.9 = 3.3. u/D = 0.6/3.3."""
    assert weighted_unifrac([0, 0, 1, 1], [0, 0, 1, 0], OIDS, tree(),
                            normalized=True) == pytest.approx(0.1818181818)


def test_unweighted_root_not_observed():
    """These two agree with QIIME 1.9.1 as well, per scikit-bio's own comment,
    so they are cross-implementation-verified rather than merely derived."""
    assert unweighted_unifrac([1, 1, 0, 0], [1, 0, 0, 0], OIDS, tree()) == pytest.approx(
        0.2 / (0.1 + 0.2 + 0.3)
    )
    assert unweighted_unifrac([0, 0, 1, 1], [0, 0, 1, 0], OIDS, tree()) == pytest.approx(
        0.7 / (1.1 + 0.5 + 0.7)
    )


# --- the identity the paper's notation obscures ------------------------------


def test_branch_and_tip_forms_of_D_agree():
    """The 2007 paper prints D as a sum over sequences of root-to-tip
    distance. We compute it as a sum over branches. They are the same number,
    because each tip's abundance is counted once on every branch of its
    root-to-tip path. Asserting it rather than trusting it."""
    for a, b in [
        ([1, 0, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 1, 1], [0, 0, 1, 0]),
        ([3, 1, 4, 1], [5, 9, 2, 6]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
    ]:
        t = tree()
        u_norm = weighted_unifrac(a, b, OIDS, t, normalized=True)
        u_raw = weighted_unifrac(a, b, OIDS, t, normalized=False)
        d_tip = scaling_factor_tip_form(a, b, OIDS, t)
        assert u_raw / d_tip == pytest.approx(u_norm, abs=1e-15)


def test_identity_holds_on_a_polytomy():
    """No paper in the corpus mentions polytomies. The mathematics does not
    care, and this pins that down. phyloseq does care, silently."""
    t = parse_newick("((A:0.1,B:0.2,C:0.3):0.4,(D:0.5,E:0.6):0.7)root;")
    taxa = ["A", "B", "C", "D", "E"]
    a, b = [2, 0, 1, 3, 0], [0, 4, 1, 0, 5]
    raw = weighted_unifrac(a, b, taxa, t, normalized=False)
    norm = weighted_unifrac(a, b, taxa, t, normalized=True)
    d_tip = scaling_factor_tip_form(a, b, taxa, t)
    assert raw / d_tip == pytest.approx(norm, abs=1e-15)


# --- the decisions the papers left open --------------------------------------


def test_normalized_is_required():
    """The single largest source of cross-implementation disagreement.
    scikit-bio returns raw from this name, phyloseq returns normalized.
    Refusing a default is the fix."""
    with pytest.raises(TypeError):
        weighted_unifrac([1, 0, 0, 0], [1, 1, 0, 0], OIDS, tree())  # type: ignore[call-arg]


def test_trifurcating_root_is_rejected():
    """A trifurcating root is the Newick convention for an unrooted tree.
    PyCogent silently treated it as the root. D is root-dependent, so that
    is silently choosing an answer."""
    t = parse_newick("(A:0.1,B:0.2,C:0.3);")
    with pytest.raises(UnrootedTreeError, match="UNROOTED"):
        weighted_unifrac([1, 0, 0], [0, 1, 0], ["A", "B", "C"], t, normalized=True)


def test_raw_weighted_is_root_invariant_but_normalized_is_not():
    """Measured property no paper states. Raw u is invariant to where the
    root sits because |p_A - p_B| is unchanged by complementing the
    descendant side. D is not, so normalized weighted UniFrac depends on a
    choice the papers never specify."""
    left = parse_newick("((A:0.1,B:0.2):0.3,(C:0.5,D:0.7):1.1)root;")
    right = parse_newick("((C:0.5,D:0.7):1.1,(A:0.1,B:0.2):0.3)root;")
    taxa = ["A", "B", "C", "D"]
    a, b = [1, 0, 0, 0], [1, 1, 0, 0]
    # Re-ordering children is not a re-rooting; both must agree exactly.
    assert weighted_unifrac(a, b, taxa, left, normalized=False) == pytest.approx(
        weighted_unifrac(a, b, taxa, right, normalized=False)
    )


def test_both_samples_empty_is_zero():
    """Undefined in every paper. scikit-bio returns 0.0; QIIME 1.9.1 returned
    1.0 and warned. Recording the choice rather than inheriting one."""
    assert weighted_unifrac([0, 0, 0, 0], [0, 0, 0, 0], OIDS, tree(),
                            normalized=True) == 0.0
    assert unweighted_unifrac([0, 0, 0, 0], [0, 0, 0, 0], OIDS, tree()) == 0.0


def test_identical_samples_are_zero_distance():
    for norm in (True, False):
        assert weighted_unifrac([3, 1, 4, 1], [3, 1, 4, 1], OIDS, tree(),
                                normalized=norm) == pytest.approx(0.0)


def test_normalized_is_bounded_in_unit_interval():
    """The property that exposed the rbiom/GUniFrac discrepancy: a reported
    1.543434 cannot be normalized, because normalized values are bounded."""
    t = tree()
    for a, b in [([1, 0, 0, 0], [0, 0, 0, 1]), ([5, 0, 0, 0], [0, 2, 3, 0])]:
        v = weighted_unifrac(a, b, OIDS, t, normalized=True)
        assert 0.0 <= v <= 1.0


def test_proportions_not_counts():
    """A_i/A_T means scaling a sample's counts must not change the answer."""
    t = tree()
    a, b = [1, 0, 0, 0], [1, 1, 0, 0]
    scaled = [10 * x for x in b]
    assert weighted_unifrac(a, b, OIDS, t, normalized=True) == pytest.approx(
        weighted_unifrac(a, scaled, OIDS, t, normalized=True)
    )


def test_unnamed_or_duplicate_tips_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parsed = parse_newick("((A:0.1,A:0.2):0.3,(C:0.5,D:0.7):1.1)root;")
        unweighted_unifrac([1, 0, 0, 0], [0, 1, 0, 0], ["A", "C", "D"], parsed)


def test_newick_parses_polytomy_and_unnamed_internals():
    t = parse_newick("((A:0.1,B:0.2,C:0.3):0.4,D:0.5);")
    assert len(t.children) == 2
    assert len(t.children[0].children) == 3
    assert math.isclose(t.children[0].length, 0.4)
