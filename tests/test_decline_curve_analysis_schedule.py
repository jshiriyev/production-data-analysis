# tests/test_schedule.py
import datetime as dt
import numpy as np
import pandas as pd
import pytest

# 🔁 Adjust this import to your project structure:
# from prodpy.decline.schedule import Schedule
from prodpy import Schedule  # e.g., if Schedule is in schedule.py next to tests


@pytest.fixture
def sched_simple():
    # Unsorted input; Schedule must sort but preserve original df index
    s = pd.Series(
        pd.to_datetime([
            "2021-01-03", "2021-01-01", "2021-01-05", "2021-01-02", "2021-01-04"
        ]),
        name="date"
    )
    return Schedule(s)

def test_basic_properties_and_sorting(sched_simple):
    s = sched_simple.series
    # Sorted ascending
    assert s.is_monotonic_increasing
    # Size and bounds
    assert len(sched_simple) == 5
    assert sched_simple.mindate == dt.date(2021, 1, 1)
    assert sched_simple.maxdate == dt.date(2021, 1, 5)
    assert sched_simple.limit == (dt.date(2021, 1, 1), dt.date(2021, 1, 5))

def test_days_since_start_and_subtract(sched_simple):
    t0 = sched_simple.days_since_start()
    assert t0.dtype == np.float64
    assert np.allclose(t0, [0, 1, 2, 3, 4])

    # subtract (aka days_since) from 2020-12-31 → 1..5
    t_ref = sched_simple.subtract("2020-12-31")
    assert np.allclose(t_ref, [1, 2, 3, 4, 5])

def test_month_lengths_shift(sched_simple):
    # All dates are Jan 2021 → 31 days
    ml0 = sched_simple.month_lengths(shift=0)
    assert (ml0 == 31).all()

    # Next month (Feb 2021) → 28 days (non-leap year)
    ml1 = sched_simple.month_lengths(shift=1)
    assert (ml1 == 28).all()

    # Previous month (Dec 2020) → 31 days
    mlm1 = sched_simple.month_lengths(shift=-1)
    assert (mlm1 == 31).all()

def test_predicates_prior_later_between_with_inclusive(sched_simple):
    # isprior inclusive/exclusive
    mask_prior_inc = sched_simple.isprior("2021-01-03", inclusive=True)
    assert mask_prior_inc.sum() == 3
    mask_prior_exc = sched_simple.isprior("2021-01-03", inclusive=False)
    assert mask_prior_exc.sum() == 2

    # islater inclusive/exclusive
    mask_later_inc = sched_simple.islater("2021-01-03", inclusive=True)
    assert mask_later_inc.sum() == 3
    mask_later_exc = sched_simple.islater("2021-01-03", inclusive=False)
    assert mask_later_exc.sum() == 2

    # isbetween with inclusive variants on 1..5 over [2,4]
    both = sched_simple.isbetween("2021-01-02", "2021-01-04", inclusive="both")
    left = sched_simple.isbetween("2021-01-02", "2021-01-04", inclusive="left")
    right = sched_simple.isbetween("2021-01-02", "2021-01-04", inclusive="right")
    neither = sched_simple.isbetween("2021-01-02", "2021-01-04", inclusive="neither")
    assert both.sum() == 3
    assert left.sum() == 2
    assert right.sum() == 2
    assert neither.sum() == 1

def test_iswithin_union_and_returning_schedules(sched_simple):
    # Union of [2,2] and [4,4] on dates 1..5 → two positions
    mask = sched_simple.iswithin(("2021-01-02", "2021-01-02"),
                                 ("2021-01-04", "2021-01-04"))
    assert mask.sum() == 2

    # within/prior/later should return Schedule objects
    w = sched_simple.within(("2021-01-02", "2021-01-04"))
    p = sched_simple.prior("2021-01-03")
    l = sched_simple.later("2021-01-04", inclusive=False)
    assert isinstance(w, Schedule) and len(w.series) == 3
    assert isinstance(p, Schedule) and len(p.series) == 3
    assert isinstance(l, Schedule) and len(l.series) == 1

def test_get_with_freq_and_periods_and_dedup():
    # Two ranges that overlap on 2021-01-01; ensure dedup + sorting
    g = Schedule.get(("2021-01-01", "2021-01-03"),
                     ("2020-12-31", "2021-01-01"),
                     freq="D")
    vals = g.series.dt.date.tolist()
    assert vals == [
        dt.date(2020, 12, 31),
        dt.date(2021, 1, 1),
        dt.date(2021, 1, 2),
        dt.date(2021, 1, 3),
    ]

    # periods-based: start + 5 periods of 7D → 5 stamps
    g2 = Schedule.get(("2021-01-01", "2021-06-01"), periods=5, freq="7D")
    assert len(g2.series) == 5
    # Expect: 2021-01-01, 01-08, 01-15, 01-22, 01-29
    expected = pd.date_range("2021-01-01", periods=5, freq="7D")
    assert (g2.series.values == expected.values).all()

def test_get_raises_on_bad_args():
    with pytest.raises(ValueError):
        Schedule.get()  # no ranges
    with pytest.raises(ValueError):
        Schedule.get(("2021-01-01", "2021-01-10"))  # freq required
    with pytest.raises(ValueError):
        Schedule.get(("2021-01-01", "2021-06-01"), periods=5)  # freq required with periods
    with pytest.raises(ValueError):
        Schedule.get(("2021-01-01", "2021-01-10", "extra"), freq="D")  # bad tuple

def test_clamp_and_slice_return_schedule(sched_simple):
    clamped = sched_simple.clamp("2021-01-02", "2021-01-04")
    assert isinstance(clamped, Schedule)
    assert clamped.mindate == dt.date(2021, 1, 2)
    assert clamped.maxdate == dt.date(2021, 1, 4)

    # Boolean mask slicing returns Schedule
    mask = sched_simple.isbetween("2021-01-02", "2021-01-03")
    sub = sched_simple[mask]
    assert isinstance(sub, Schedule)
    assert len(sub.series) == 2

def test_nearest_ties_choose_left(sched_simple):
    # 2021-01-02 12:00 is equidistant to 2021-01-02 and 2021-01-03 → choose left (02)
    ts = pd.Timestamp("2021-01-02 12:00")
    nearest = sched_simple.nearest(ts)
    assert nearest == pd.Timestamp("2021-01-02")

def test_align_to_preserves_original_index_alignment():
    # Create an unsorted DataFrame, then build Schedule from its 'date' column
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-03", "2021-01-01", "2021-01-05", "2021-01-02"]),
        "q": [30, 10, 50, 20],
    })
    sched = Schedule(df["date"])

    # Index returned by align_to should be the original df index in sorted-by-date order
    idx = sched.align_to(df, on="date")
    sorted_dates = df.loc[idx, "date"]
    assert sorted_dates.is_monotonic_increasing

    # Alignment: use the same mask/ordering to align q with t
    t = sched.days_since_start()
    q_sorted = df.loc[idx, "q"].to_numpy()
    # First day corresponds to min date's q (which is 10)
    assert q_sorted[0] == 10
    assert np.all(np.diff(t) > 0)  # strictly increasing in this fixture