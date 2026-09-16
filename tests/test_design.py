import pytest

from abkit.design import SampleSizeResult, sample_size_proportion

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
