from dataclasses import dataclass
from math import ceil, sqrt
from scipy.stats import norm

@dataclass(frozen=True)
class SampleSizeResult:
    n_control: int
    n_treatment: int
    n_total: int

def sample_size_proportion(
    p_baseline: float, # Expected conversion rate in the control group, in (0, 1).
    mde: float, # Minimum detectable effect.
    alpha: float = 0.05, # Significance level
    power: float = 0.8, # Statistical power
    ratio: float = 1.0 # Allocation ratio n_treatment / n_control.
) -> SampleSizeResult:
    
    if not (0 < p_baseline < 1):
        raise ValueError(f"Control group conversion must be in (0,1), got {p_baseline}.")
    p_treatment = p_baseline + mde
    if not (0 < p_baseline < 1):
        raise ValueError(f"Experimental group conversion must be in (0,1), got {p_treatment}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1), got {alpha}.")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0,1), got {power}.")
    if (ratio <= 0):
        raise ValueError(f"ratio must be postive, got {ratio}.")

    delta = abs(mde)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    p1, p2 = p_baseline, p_treatment
    p_pooled = (p1 + ratio * p2) / (1 + ratio)
    var_null = p_pooled * (1 - p_pooled) * (1 + 1 / ratio)
    var_alt = p1 * (1 - p1) + p2 * (1 - p2) / ratio
    n_control = (z_alpha * sqrt(var_null) + z_power * sqrt(var_alt)) ** 2 / delta ** 2
    n_control = ceil(n_control)
    n_treatment = ceil(n_control * ratio)

    return SampleSizeResult(
        n_control,
        n_treatment,
        n_control + n_treatment
    )