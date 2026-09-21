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
    alpha: float = 0.05, # Significance level.
    power: float = 0.8, # Statistical power.
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

    return SampleSizeResult(n_control, n_treatment, n_control + n_treatment)

def sample_size_continuous(
        mean: float, # Expected control group mean. Only used when relative=True.
        std: float, # Control group standard deviation (must be positive).
        mde: float, # Minimum detectable effect.
        alpha: float = 0.05, # Significance level.
        power: float = 0.8, # Statistical power.
        ratio: float = 1.0, # Allocation ratio n_treatment / n_control.
        std_treatment: float | None = None, # Treatment group std if it differs from control.
        relative: bool = False, # Interpret `mde` as a fraction of `mean` instead of absolute units.
) -> SampleSizeResult:
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    if std_treatment is not None and std_treatment <= 0:
            raise ValueError(f"std_treatment must be positive, got {std_treatment}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1), got {alpha}.")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0,1), got {power}.")
    if (ratio <= 0):
        raise ValueError(f"ratio must be postive, got {ratio}.")
    if (mde == 0):
        raise ValueError(f"mde must be non zero, got {mde}.")
    if (relative and mean == 0):
        raise ValueError("relative mde requires a non-zero mean")
    delta = abs(mde*mean) if relative else abs(mde)
    std_1 = std
    std_2 = std if std_treatment is None else std_treatment
    z_alpha = norm.ppf(1 - alpha/2)
    z_power = norm.ppf(power)
    variance_term = std_1**2 + std_2**2 / ratio
    n_control = ceil((z_alpha + z_power)**2 * variance_term / delta**2)
    n_treatment = ceil(n_control * ratio)

    return SampleSizeResult(n_control, n_treatment, n_control + n_treatment)