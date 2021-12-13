"""
Reference "Guide to Sentinel-1 Geocoding" UZH-S1-GC-AD 1.10 26.03.2019
https://sentinel.esa.int/documents/247904/0/Guide-to-Sentinel-1-Geocoding.pdf/e0450150-b4e9-4b2d-9b32-dadf989d3bd3
"""
import typing as T

import numpy as np
import numpy.typing as npt

SPEED_OF_LIGHT = 299_792_458  # m / s

TimedeltaArrayLike = T.TypeVar("TimedeltaArrayLike", bound=npt.ArrayLike)
FloatArrayLike = T.TypeVar("FloatArrayLike", bound=npt.ArrayLike)


def secant_method(
    ufunc: T.Callable[[TimedeltaArrayLike], T.Tuple[FloatArrayLike, FloatArrayLike]],
    t_prev: TimedeltaArrayLike,
    t_curr: TimedeltaArrayLike,
    diff_ufunc: float = 1,
    diff_t: np.timedelta64 = np.timedelta64(0, "ns"),
) -> T.Tuple[TimedeltaArrayLike, TimedeltaArrayLike, FloatArrayLike, FloatArrayLike]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    while True:
        f_curr, g_curr = ufunc(t_curr)

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff: TimedeltaArrayLike
        p: TimedeltaArrayLike
        q: FloatArrayLike

        t_diff = t_curr - t_prev  # type: ignore
        p = f_curr * t_diff  # type: ignore
        q = f_curr - f_prev  # type: ignore

        # t_prev, t_curr = t_curr, t_curr - f_curr * np.timedelta64(-148_000, "ns")
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, p / q, 0)  # type: ignore
        f_prev = f_curr

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

    return t_curr, t_prev, f_curr, g_curr
