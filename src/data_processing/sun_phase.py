"""Sun et al. (2017) fuzzy flight-phase identification, independently expressed.

Reference: https://doi.org/10.2514/1.I010520
Author implementation: https://github.com/junzis/flight-data-processor/blob/master/flightphase.py

Membership grids, fuzzy rules, spline smoothing and largest-of-maximum
defuzzification follow the author implementation. This implementation includes
the final window, lowers spline degree for short flights, and retains uncovered
windows as NA. Inputs are seconds, feet, knots and feet/minute. No input values
are modified in place, and this module performs no synthetic-noise injection.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.interpolate import UnivariateSpline


PHASES = ('GND', 'CL', 'DE', 'CR', 'LVL', 'NA')


def _s_membership(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    fraction = np.clip((values - lower) / (upper - lower), 0., 1.)
    return np.where(fraction <= .5, 2. * fraction ** 2, 1. - 2. * (1. - fraction) ** 2)


@lru_cache(maxsize=1)
def _membership_tables() -> tuple:
    altitude = np.arange(0., 40000., 1.)
    rate = np.arange(-4000., 4000., .1)
    speed = np.arange(0., 600., 1.)
    states = np.arange(0., 6., .01)
    gaussian = lambda x, center, sigma: np.exp(-((x - center) ** 2) / (2. * sigma ** 2))
    return (
        altitude,
        (1. - _s_membership(altitude, 0., 200.), gaussian(altitude, 10000., 5000.), gaussian(altitude, 35000., 20000.)),
        rate,
        (gaussian(rate, 0., 100.), _s_membership(rate, 10., 1000.), 1. - _s_membership(rate, -1000., -10.)),
        speed,
        (gaussian(speed, 600., 100.), gaussian(speed, 200., 100.), gaussian(speed, 0., 50.)),
        states,
        np.stack([gaussian(states, center, .1) for center in range(1, 6)]),
    )


def identify_sun_phases(
    times_seconds, altitude_ft, speed_knots, vertical_rate_ft_min=None,
    *, window_seconds: float = 60., smooth: bool = True,
) -> np.ndarray:
    """Return one categorical phase for each strictly time-ordered observation.

    If climb rate is absent, derive it from altitude using actual time intervals.
    Spline smoothing affects only temporary classification arrays. Consecutive
    equal labels can subsequently be merged into a phase segment.
    """
    times, altitude, speed = (np.asarray(x, dtype=np.float64) for x in (times_seconds, altitude_ft, speed_knots))
    if times.ndim != 1 or not len(times) or altitude.shape != times.shape or speed.shape != times.shape:
        raise ValueError('Sun inputs must be nonempty matching one-dimensional arrays')
    if not all(np.isfinite(x).all() for x in (times, altitude, speed)):
        raise ValueError('Sun inputs contain nonfinite observations')
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('window_seconds must be positive')
    times = times - times[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError('Sun identification requires strictly increasing timestamps')
    if vertical_rate_ft_min is None:
        rate = np.gradient(altitude, times) * 60. if len(times) > 1 else np.zeros(1)
    else:
        rate = np.asarray(vertical_rate_ft_min, dtype=np.float64)
        if rate.shape != times.shape or not np.isfinite(rate).all():
            raise ValueError('Vertical rate must be finite and match the observations')
    if smooth and len(times) > 1:
        degree = min(3, len(times) - 1)
        altitude, speed, rate = (UnivariateSpline(times, values, k=degree)(times) for values in (altitude, speed, rate))
    if not all(np.isfinite(x).all() for x in (altitude, speed, rate)):
        raise ValueError('Spline or derived vertical rate produced nonfinite values')

    windows = np.floor(times / window_seconds).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(windows)) + 1]
    counts = np.diff(np.r_[starts, len(times)])
    altitude, speed, rate = (np.add.reduceat(values, starts) / counts for values in (altitude, speed, rate))
    ag, am, rg, rm, sg, sm, states, output_memberships = _membership_tables()
    alt_ground, alt_low, alt_high = (np.interp(np.clip(altitude, ag[0], ag[-1]), ag, curve) for curve in am)
    rate_zero, rate_plus, rate_minus = (np.interp(np.clip(rate, rg[0], rg[-1]), rg, curve) for curve in rm)
    speed_high, speed_medium, speed_low = (np.interp(np.clip(speed, sg[0], sg[-1]), sg, curve) for curve in sm)
    strengths = np.stack([
        np.minimum.reduce((alt_ground, rate_zero, speed_low)),
        np.minimum.reduce((alt_low, rate_plus, speed_medium)),
        np.minimum.reduce((alt_low, rate_minus, speed_medium)),
        np.minimum.reduce((alt_high, rate_zero, speed_high)),
        np.minimum.reduce((alt_low, rate_zero, speed_medium)),
    ], axis=1)
    labels = np.full(len(starts), 'NA', dtype='<U3')
    phase_codes = np.asarray(PHASES)
    # Bound temporary fuzzy-aggregation memory independently of flight duration.
    for first in range(0, len(starts), 256):
        block = strengths[first:first + 256]
        aggregate = np.minimum(block[:, :, None], output_memberships[None, :, :]).max(axis=1)
        peaks = aggregate.max(axis=1)
        supported = peaks > 0
        lom = np.where(aggregate == peaks[:, None], states[None, :], -np.inf).max(axis=1)
        codes = np.clip(np.rint(lom[supported]).astype(np.int64), 1, 6)
        labels[first + np.flatnonzero(supported)] = phase_codes[codes - 1]
    return np.repeat(labels, counts)
