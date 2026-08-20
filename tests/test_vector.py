import math

import pytest

from clockvec.vector import Declaration, Expectation, Vector, canonical_json, content_id


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


def _vec(**over):
    base = dict(
        vector_id="v001",
        clock="DunedinPACE",
        source_paper="Belsky et al. 2022 eLife",
        seed=21,
        shape=(173, 12),
        declaration=_decl(),
    )
    base.update(over)
    return Vector(**base)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_rejects_nan():
    """A NaN reaching the hash would bake an unreproducible value into an id."""
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_content_id_is_stable():
    assert content_id({"a": 1}) == content_id({"a": 1})
    assert content_id({"a": 1}) != content_id({"a": 2})


def test_manifest_id_ignores_expectations():
    """The id names the question, not the answers. Adding a seventh
    implementation must not renumber a vector that papers already cite."""
    v = _vec()
    before = v.manifest_id
    v.expectations.append(
        Expectation("dnaMethyAge", "0.6.0", [1.0, 2.0], 1e-9, "float summation order")
    )
    assert v.manifest_id == before


def test_manifest_id_changes_when_the_question_changes():
    a = _vec()
    b = _vec(seed=22)
    assert a.manifest_id != b.manifest_id

    c = _vec(declaration=_decl(missing_cpg_policy="zero"))
    assert a.manifest_id != c.manifest_id


def test_agreement_within_tolerance_is_not_a_disagreement():
    v = _vec()
    v.expectations = [
        Expectation("dnaMethyAge", "0.6.0", [1.0, 2.0], 1e-9, "x"),
        Expectation("DunedinPACE", "0.99", [1.0 + 4e-16, 2.0], 1e-9, "y"),
    ]
    assert v.disagreements() == []


def test_disagreement_beyond_tolerance_is_reported():
    v = _vec()
    v.expectations = [
        Expectation("biolearn", "0.9.1", [1.0], 1e-9, "x"),
        Expectation("methylclock", "1.0", [1.5], 1e-9, "y"),
    ]
    (a, b, diff, tol) = v.disagreements()[0]
    assert {a, b} == {"biolearn", "methylclock"}
    assert diff == pytest.approx(0.5)
    assert tol == 1e-9


def test_looser_tolerance_wins():
    """Judging a pair under a tolerance one side never accepted would
    manufacture a disagreement nobody asserted."""
    v = _vec()
    v.expectations = [
        Expectation("strict", "1", [1.0], 1e-12, "tight"),
        Expectation("loose", "1", [1.0 + 1e-9], 1e-6, "wide"),
    ]
    assert v.disagreements() == []


def test_length_mismatch_is_infinite_disagreement():
    v = _vec()
    v.expectations = [
        Expectation("a", "1", [1.0, 2.0], 1e-9, "x"),
        Expectation("b", "1", [1.0], 1e-9, "y"),
    ]
    (_, _, diff, _) = v.disagreements()[0]
    assert math.isinf(diff)


def test_all_pairs_are_compared():
    v = _vec()
    v.expectations = [
        Expectation(name, "1", [float(i)], 1e-9, "x")
        for i, name in enumerate(["a", "b", "c"])
    ]
    assert len(v.disagreements()) == 3  # ab, ac, bc


def test_underspecified_vectors_are_still_valid():
    """Where the paper is silent the disagreement is the finding, so the
    vector has to be expressible rather than rejected."""
    v = _vec(underspecified=True)
    v.expectations = [
        Expectation("a", "1", [1.0], 1e-9, "x"),
        Expectation("b", "1", [9.0], 1e-9, "y"),
    ]
    assert v.underspecified
    assert len(v.disagreements()) == 1
