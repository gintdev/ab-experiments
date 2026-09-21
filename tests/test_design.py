import pytest

from abkit.design import SampleSizeResult, sample_size_proportion
from abkit.design import sample_size_continuous

def test_matches_known_reference_value():
    result = sample_size_proportion(0.10, 0.02, alpha = 0.05, power = 0.8)
    assert result == SampleSizeResult(3841, 3841, 7682)

def test_equal_split_gives_equal_groups():
    result = sample_size_proportion(0.10, 0.02, ratio = 1.0)
    assert result.n_control == result.n_treatment

@pytest.mark.parametrize("ratio", [1/9, 1, 9])
def test_treatment_size_matches_ratio(ratio):
    result = sample_size_proportion(0.10, 0.02, ratio = ratio)
    assert result.n_treatment == pytest.approx(result.n_control * ratio, rel = 0.01)

def test_smaller_mde_requires_more_samples():
    small_effect = sample_size_proportion(0.10, 0.01)
    large_effect = sample_size_proportion(0.10, 0.05)
    assert small_effect.n_total > large_effect.n_total

def test_higher_power_requires_more_samples():
    low_power = sample_size_proportion(0.10, 0.02, power = 0.7)
    high_power = sample_size_proportion(0.10, 0.02, power = 0.95)
    assert high_power.n_total > low_power.n_total

def test_stricter_alpha_requires_more_samples():
    lenient = sample_size_proportion(0.10, 0.02, alpha = 0.10)
    strict = sample_size_proportion(0.10, 0.02, alpha = 0.01)
    assert strict.n_total > lenient.n_total

@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_baseline": 0.0, "mde": 0.02},
        {"p_baseline": 1.0, "mde": 0.02},
        {"p_baseline": 0.95, "mde": 0.10},
        {"p_baseline": 0.10, "mde": 0.02, "alpha": 0.0},
        {"p_baseline": 0.10, "mde": 0.02, "power": 1.0},
        {"p_baseline": 0.10, "mde": 0.02, "ratio": 0},
        {"p_baseline": 0.10, "mde": 0.02, "ratio": -1},
    ],
)
def test_invalid_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        sample_size_proportion(**kwargs)

@pytest.mark.parametrize(
    ("cohens_d", "expected_n"),
    [(0.2, 393), (0.5, 63), (0.8, 25)]
)
def test_continuous_matches_cohen_benchmarks(cohens_d, expected_n):
    result = sample_size_continuous(mean = 0.0, std = 1.0, mde = cohens_d)
    assert result.n_control == expected_n

def test_continuous_scales_with_variance():
    small_std = sample_size_continuous(mean = 100.0, std = 10.0, mde = 5.0)
    large_std = sample_size_continuous(mean = 100.0, std = 20.0, mde = 5.0)
    assert large_std.n_control == pytest.approx(4 * small_std.n_control, rel = 0.01)

def test_continuous_ignores_when_absolute():
    low = sample_size_continuous(mean= 10.0, std = 5.0, mde = 1.0)
    high = sample_size_continuous(mean = 10000.0, std = 5.0, mde = 1.0)
    assert low == high

def test_continuous_relative_mde_matches_absolute():
    relative = sample_size_continuous(mean = 200.0, std = 50.0, mde = 0.05, relative = True)
    absolute = sample_size_continuous(mean = 200.0, std = 50.0, mde = 10.0)
    assert relative == absolute

def test_continuous_unequal_variance_needs_more_samples():
    equal = sample_size_continuous(mean = 0.0, std = 1.0, mde = 0.5)
    unequal = sample_size_continuous(mean = 0.0, std = 1.0, mde = 0.5, std_treatment = 2.0)
    assert unequal.n_control > equal.n_control

@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean": 0.0, "std": 0.0, "mde": 0.5},
        {"mean": 0.0, "std": -1.0, "mde": 0.5},
        {"mean": 0.0, "std": 1.0, "mde": 0.0},
        {"mean": 0.0, "std": 1.0, "mde": 0.5, "std_treatment": 0.0},
        {"mean": 0.0, "std": 1.0, "mde": 0.5, "alpha": 1.0},
        {"mean": 0.0, "std": 1.0, "mde": 0.5, "power": 0.0},
        {"mean": 0.0, "std": 1.0, "mde": 0.5, "ratio": -1},
        {"mean": 0.0, "std": 1.0, "mde": 0.05, "relative": True},
    ],
)
def test_continuous_invalid_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        sample_size_continuous(**kwargs)