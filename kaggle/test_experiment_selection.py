"""Targeted no-ML checks for selection and exact runtime provenance."""
from itertools import permutations
from unittest.mock import patch

import run_experiments as runner


def _record(candidate_id, average_precision, mean_hit_at_3=0.0, best_round=10):
    return {
        "status": "complete",
        "candidate_id": candidate_id,
        "average_precision": average_precision,
        "mean_hit_at_3": mean_hit_at_3,
        "best_round": best_round,
    }


def test_selection_is_permutation_invariant_for_all_ties():
    records = [
        _record("c0", 0.9, 0.2, 30),
        _record("c1", 0.9, 0.2, 30),
        _record("c2", 0.9, 0.2, 30),
        _record("c3", 0.9, 0.2, 30),
    ]
    for ordering in permutations(records):
        assert runner.choose_candidate(list(ordering), 1e-6)["candidate_id"] == "c0"


def test_selection_filters_from_global_ap_before_tie_breaking():
    records = [
        _record("c0", 0.900, 0.10, 20),
        _record("c1", 0.805, 0.20, 10),
        # c2 is within tolerance of c1, but outside tolerance of the global AP maximum.
        _record("c2", 0.706, 0.99, 1),
        _record("c3", 0.500, 0.99, 1),
    ]
    assert runner.choose_candidate(records, 0.1)["candidate_id"] == "c1"


def test_selection_tie_keys_are_hit_round_then_ascending_id():
    records = [
        _record("c0", 0.9, 0.2, 10),
        _record("c1", 0.9, 0.3, 20),
        _record("c2", 0.9, 0.3, 10),
        _record("c3", 0.9, 0.3, 10),
    ]
    assert runner.choose_candidate(records, 1e-6)["candidate_id"] == "c2"


def _runtime_pins():
    return {
        "python": runner.sys.version.split()[0],
        "numpy": "1.0",
        "scipy": "2.0",
        "pandas": "3.0",
        "scikit_learn": "4.0",
        "xgboost": "5.0",
        "lightgbm": "6.0",
        "catboost": "7.0",
    }


def test_environment_reads_exact_runtime_keys_and_distribution_names():
    expected = _runtime_pins()
    installed = {
        "numpy": "1.0",
        "scipy": "2.0",
        "pandas": "3.0",
        "scikit-learn": "4.0",
        "xgboost": "5.0",
        "lightgbm": "6.0",
        "catboost": "7.0",
    }
    with patch.object(runner.importlib_metadata, "version", side_effect=installed.__getitem__) as version:
        frozen, digest = runner._environment({"runtime": expected})
    assert "python==" + expected["python"] in frozen
    assert "scikit-learn==4.0" in frozen
    assert digest == runner.hashlib.sha256(frozen.encode()).hexdigest()
    assert [call.args[0] for call in version.call_args_list] == list(installed)


def test_environment_fails_closed_for_missing_or_mismatched_pins():
    missing = _runtime_pins()
    del missing["numpy"]
    try:
        runner._environment({"runtime": missing})
    except RuntimeError as error:
        assert "missing: numpy" in str(error)
    else:
        raise AssertionError("missing runtime pins must fail closed")

    mismatched = _runtime_pins()
    installed = {key.replace("scikit_learn", "scikit-learn"): value for key, value in _runtime_pins().items()
                 if key != "python"}
    installed["pandas"] = "wrong"
    with patch.object(runner.importlib_metadata, "version", side_effect=installed.__getitem__):
        try:
            runner._environment({"runtime": mismatched})
        except RuntimeError as error:
            assert "package pin mismatch for pandas" in str(error)
        else:
            raise AssertionError("mismatched runtime pins must fail closed")


if __name__ == "__main__":
    for test in (test_selection_is_permutation_invariant_for_all_ties,
                 test_selection_filters_from_global_ap_before_tie_breaking,
                 test_selection_tie_keys_are_hit_round_then_ascending_id,
                 test_environment_reads_exact_runtime_keys_and_distribution_names,
                 test_environment_fails_closed_for_missing_or_mismatched_pins):
        test()
    print("PASS: candidate selection and exact runtime provenance")
