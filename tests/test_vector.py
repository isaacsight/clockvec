import math

import pytest

from clockvec.vector import (
    Authority,
    Declaration,
    Expectation,
    Flag,
    Result,
    Vector,
    canonical_json,
    content_id,
)


def _decl(**over):
    base = dict(
        missing_cpg_policy="mean_impute",
        imputation_source="GSE55763",
        normalization="quantile_to_target",
        normalization_reference="gold_standard_means",
        notes="",
    )
    base.update(over)
    return Declaration(**base)


def _auth(**over):
    base = dict(
        claims="the scoring step: betas and coefficients to a score",
        does_not_claim="preprocessing, imputation, or normalization agreement",
        basis="Belsky et al. 2022 eLife, PoAmProjector.R",
    )
    base.update(over)
    return Authority(**base)


def _vec(**over):
    base = dict(
        vector_id="v001",
        clock="DunedinPACE",
        source_paper="Belsky et al. 2022 eLife",
        seed=21,
        shape=(173, 12),
        declaration=_decl(),
        authority=_auth(),
    )
    base.update(over)
    return Vector(**base)


def _exp(impl, values, tol=1e-9, result=Result.VALID, flags=()):
    return Expectation(
        implementation=impl,
        version="1.0",
        values=tuple(values),
        tolerance=tol,
        tolerance_rationale="floating-point summation order",
        result=result,
        flags=flags,
    )


# --- canonical serialization -------------------------------------------------


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_rejects_nan():
    """A NaN reaching the hash would bake an unreproducible value into an id."""
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_content_id_is_stable():
    assert content_id({"a": 1}) == content_id({"a": 1})
    assert content_id({"a": 1}) != content_id({"a": 2})


# --- vector identity ---------------------------------------------------------


def test_manifest_id_ignores_expectations():
    """The id names the question, not the answers. Adding a seventh
    implementation must not renumber a vector that papers already cite."""
    v = _vec()
    before = v.manifest_id
    v.expectations.append(_exp("dnaMethyAge", [1.0, 2.0]))
    assert v.manifest_id == before


def test_manifest_id_changes_when_the_question_changes():
    a = _vec()
    assert a.manifest_id != _vec(seed=22).manifest_id
    assert a.manifest_id != _vec(declaration=_decl(missing_cpg_policy="zero")).manifest_id


def test_manifest_id_changes_when_authority_changes():
    """Narrowing or widening what a vector claims makes it a different
    vector. Silently restating the scope under a stable id would let a
    citation mean something it did not originally mean."""
    a = _vec()
    b = _vec(authority=_auth(does_not_claim="nothing"))
    assert a.manifest_id != b.manifest_id


def test_schema_version_is_part_of_identity():
    a = _vec()
    b = _vec(schema="clockvec/v2")
    assert a.manifest_id != b.manifest_id


# --- disagreement logic ------------------------------------------------------


def test_agreement_within_tolerance_is_not_a_disagreement():
    v = _vec()
    v.expectations = [_exp("dnaMethyAge", [1.0, 2.0]), _exp("DunedinPACE", [1.0 + 4e-16, 2.0])]
    assert v.disagreements() == []


def test_disagreement_beyond_tolerance_is_reported():
    v = _vec()
    v.expectations = [_exp("biolearn", [1.0]), _exp("methylclock", [1.5])]
    (a, b, diff, tol) = v.disagreements()[0]
    assert {a, b} == {"biolearn", "methylclock"}
    assert diff == pytest.approx(0.5)
    assert tol == 1e-9


def test_looser_tolerance_wins():
    """Judging a pair under a tolerance one side never accepted would
    manufacture a disagreement nobody asserted."""
    v = _vec()
    v.expectations = [_exp("strict", [1.0], tol=1e-12), _exp("loose", [1.0 + 1e-9], tol=1e-6)]
    assert v.disagreements() == []


def test_length_mismatch_is_infinite_disagreement():
    v = _vec()
    v.expectations = [_exp("a", [1.0, 2.0]), _exp("b", [1.0])]
    (_, _, diff, _) = v.disagreements()[0]
    assert math.isinf(diff)


def test_all_pairs_are_compared():
    v = _vec()
    v.expectations = [_exp(n, [float(i)]) for i, n in enumerate(["a", "b", "c"])]
    assert len(v.disagreements()) == 3  # ab, ac, bc


# --- the three-valued result -------------------------------------------------


def test_acceptable_suppresses_disagreement():
    """Where the paper is silent the vector has already declared it does not
    decide. Reporting a difference as a conflict would contradict its own
    authority statement."""
    v = _vec()
    v.expectations = [
        _exp("a", [1.0], result=Result.ACCEPTABLE),
        _exp("b", [9.0]),
    ]
    assert v.disagreements() == []


def test_acceptable_on_either_side_suppresses():
    v = _vec()
    v.expectations = [_exp("a", [1.0]), _exp("b", [9.0], result=Result.ACCEPTABLE)]
    assert v.disagreements() == []


def test_invalid_results_still_disagree():
    """INVALID is a verdict, not an exemption. A known-wrong implementation
    still has to show up in the matrix or the matrix understates the damage."""
    v = _vec()
    v.expectations = [_exp("a", [1.0]), _exp("b", [9.0], result=Result.INVALID)]
    assert len(v.disagreements()) == 1


# --- validation --------------------------------------------------------------


def test_undefined_flag_reference_is_rejected():
    v = _vec()
    v.expectations = [_exp("a", [1.0], flags=("NoSuchFlag",))]
    with pytest.raises(ValueError, match="undefined flags"):
        v.validate()


def test_defined_flag_reference_passes():
    v = _vec()
    v.flags = {
        "ImputationDivergence": Flag(
            name="ImputationDivergence",
            bug_type="ImputationPolicy",
            description="Implementations fill missing CpGs from different references.",
            citations=("https://github.com/bio-learn/biolearn/issues/101",),
        )
    }
    v.expectations = [_exp("a", [1.0], flags=("ImputationDivergence",))]
    v.validate()  # must not raise


def test_tolerance_without_rationale_is_rejected():
    """An undefended tolerance is how a conformance suite becomes a
    marketing number."""
    v = _vec()
    v.expectations = [
        Expectation("a", "1.0", (1.0,), 1e-9, "   ", Result.VALID, ())
    ]
    with pytest.raises(ValueError, match="no rationale"):
        v.validate()


def test_negative_tolerance_is_rejected():
    v = _vec()
    v.expectations = [_exp("a", [1.0], tol=-1.0)]
    with pytest.raises(ValueError, match="negative tolerance"):
        v.validate()


def test_flag_carries_citations():
    """A failing vector that cites the issue where researchers hit this
    divergence tells a maintainer something a bare assertion cannot."""
    f = Flag(
        name="ScoringFormEquivalence",
        bug_type="SpecAmbiguity",
        description="diag/rowSums spelling versus plain matrix multiply.",
        citations=("https://github.com/yiluyucheng/dnaMethyAge/issues/21",),
    )
    assert f.citations[0].endswith("/21")
    assert f.bug_type == "SpecAmbiguity"
