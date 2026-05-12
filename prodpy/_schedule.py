import datetime as _dt
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

DateLike = Union[pd.Timestamp, np.datetime64, _dt.datetime, _dt.date, str]

def _to_timestamp(x: DateLike) -> pd.Timestamp:
	"""Robust datetime parser to pd.Timestamp."""
	if isinstance(x, pd.Timestamp):
		return x
	if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
		# convert date -> datetime (midnight) to be consistent
		return pd.Timestamp(x)
	return pd.to_datetime(x)

class Schedule:
	"""
	A thin helper for date/time indexing that:
	  - enforces sorting and monotonicity
	  - aligns cleanly with pandas Series/DataFrames
	  - produces elapsed times (days) for decline-curve analysis

	Typical use:
	------------
	>>> sched = Schedule(df['date'])			 # sorted copy; keeps df index
	>>> t = sched.days_since_start()			 # float days since first date
	>>> mask = sched.iswithin(('2021-01-01','2021-06-30'))
	>>> q_fit = df.loc[mask, 'qoil'].to_numpy()
	>>> t_fit = t[mask]						  # same mask aligns perfectly

	"""
	def __init__(self, series: pd.Series):
		"""
		Parameters
		----------
		series : pd.Series
			A 1D datetime-like Series. Values are converted to datetime64[ns].
			Original index is preserved for alignment; values are sorted.
		"""
		s = pd.to_datetime(series)
		# sort but keep original index for alignment with the original df
		order = np.argsort(s.values)
		self._series = s.iloc[order]
		self._series = self._series.astype("datetime64[ns]")
		self._series = self._series.rename(series.name)

	@property
	def series(self) -> pd.Series:
		"""Sorted datetime Series (datetime64[ns]) with original index."""
		return self._series

	def __len__(self) -> int:
		return self._series.size

	@property
	def size(self) -> int:
		return self._series.size

	@property
	def mindate(self) -> _dt.date:
		"""First date (date object)."""
		return self._series.iloc[0].date()

	@property
	def maxdate(self) -> _dt.date:
		"""Last date (date object)."""
		return self._series.iloc[-1].date()

	@property
	def limit(self) -> Tuple[_dt.date, _dt.date]:
		"""(min_date, max_date)"""
		return (self.mindate, self.maxdate)

	def copy(self) -> "Schedule":
		return Schedule(self._series.copy())

	# slicing
	def __getitem__(self, key) -> "Schedule":
		"""Subselect by any valid pandas indexer (bool mask, slice, positions)."""
		return Schedule(self._series[key])

	def _days_between(self, ref):
		"""Vectorized (series - ref) in days as float64."""
		ref_ts = pd.to_datetime(ref)  # robust parse
		# ns since epoch for both sides (avoid Series.view deprecation)
		ref_ns = pd.Timestamp(ref_ts).value
		s_ns = self._series.astype("int64").to_numpy()
		return (s_ns - ref_ns).astype("float64") / 86_400_000_000_000.0 # 24*60*60*1e9

	# def _days_between(self, ref: DateLike) -> np.ndarray:
	# 	"""Vectorized (series - ref) in **days** as float64 (preserves index order)."""
	# 	ref_ts = _to_timestamp(ref).tz_localize(None) if pd.Timestamp(ref).tz else _to_timestamp(ref)
	# 	# use int ns for max speed
	# 	ns = self._series.view("i8") - pd.Timestamp(ref_ts).value
	# 	return ns.astype("float64") / 86_400_000_000.0  # 24*60*60*1e9

	def subtract(self, date: DateLike) -> np.ndarray:
		"""
		Alias for `days_since(date)`. Kept for backward compatibility.
		Returns float days (np.ndarray).
		"""
		return self._days_between(date)

	def days_since(self, date: DateLike) -> np.ndarray:
		"""Days since an arbitrary reference datetime (float64)."""
		return self._days_between(date)

	def days_since_start(self) -> np.ndarray:
		"""Elapsed days from the **first** timestamp (t0)."""
		return self._days_between(self._series.iloc[0])

	# --------------------------- month info --------------------------- #
	def month_lengths(self, shift: int = 0) -> pd.Series:
		"""
		Number of days in the month of each timestamp (optionally shifted by N months).
		If shift is zero, it is the number of days in the current month.

		Examples
		--------
		>>> sched.month_lengths()		# days in current month of each date
		>>> sched.month_lengths(shift=1) # days in next month for each date
		>>> sched.month_lengths(shift=-1)# days in previous month
		"""
		# shift months using PeriodIndex to avoid day overflow
		next_month = (self._series.dt.to_period('M')+shift+1).dt.to_timestamp()

		# Calculate the start of the given month
		prev_month = (self._series.dt.to_period('M')+shift+0).dt.to_timestamp()

		# Return the number of days in the month
		return (next_month-prev_month).dt.days

	# --------------------------- predicates --------------------------- #
	def isprior(self, date: DateLike, inclusive: bool = True) -> np.ndarray:
		"""Bool array: timestamps <= date (inclusive) by default."""
		d = _to_timestamp(date).date()
		cmp = self._series.dt.date <= d if inclusive else self._series.dt.date < d
		return cmp.to_numpy()

	def islater(self, date: DateLike, inclusive: bool = True) -> np.ndarray:
		"""Bool array: timestamps >= date (inclusive) by default."""
		d = _to_timestamp(date).date()
		cmp = self._series.dt.date >= d if inclusive else self._series.dt.date > d
		return cmp.to_numpy()

	def isbetween(
		self,
		start: DateLike,
		end: DateLike,
		inclusive: str = "both",
	) -> np.ndarray:
		"""
		Bool array for start..end.

		inclusive: 'both' (default), 'left', 'right', 'neither'
		"""
		s = _to_timestamp(start)
		e = _to_timestamp(end)
		return self._series.between(s, e, inclusive=inclusive).to_numpy()

	def iswithin(self, *ranges: Sequence[DateLike], inclusive: str = "both") -> np.ndarray:
		"""
		Union of multiple (start, end) ranges as a bool mask.

		Example
		-------
		>>> mask = sched.iswithin(('2021-01-01','2021-02-01'),
		...						('2024-01-01','2024-02-01'))
		"""
		if not ranges:
			return np.zeros(self.size, dtype=bool)
		out = np.zeros(self.size, dtype=bool)
		for start, end in ranges:
			out |= self.isbetween(start, end, inclusive=inclusive)
		return out

	# convenience returning Schedule
	def prior(self, date: DateLike, inclusive: bool = True) -> "Schedule":
		return self[self.isprior(date, inclusive=inclusive)]

	def later(self, date: DateLike, inclusive: bool = True) -> "Schedule":
		return self[self.islater(date, inclusive=inclusive)]

	def within(self, *ranges: Sequence[DateLike], inclusive: str = "both") -> "Schedule":
		return self[self.iswithin(*ranges, inclusive=inclusive)]

	def align_to(self, df: pd.DataFrame, on: Optional[str] = None) -> pd.Index:
		"""
		Return an indexer that aligns this schedule to rows of `df`.

		Parameters
		----------
		df : DataFrame that has the same time column
		on : column name in df if the schedule was built from df[on];
			 if None, tries to infer by matching Series name.

		Returns
		-------
		pd.Index (of df) corresponding to the sorted schedule order.
		"""
		col = on or self._series.name
		if col is None:
			raise ValueError("Provide `on` or ensure the source Series had a name.")
		# Match by original df index values (we preserved them through sorting)
		return self._series.index

	@staticmethod
	def get(
		*ranges: Sequence[DateLike],
		freq: Optional[str] = None,
		periods: Optional[int] = None,
		inclusive: Optional[str] = 'both',
		normalize: bool = False,
	) -> "Schedule":
		"""
		Build a Schedule from one or more (start, end) ranges.

		Rules:
		- If `periods` is given, each range uses `start` and `periods` (ignores `end`).
		- Else uses `start`..`end` inclusive (pandas default) with `freq` (required).
		- All pieces are concatenated, de-duplicated, and sorted.

		Examples
		--------
		>>> Schedule.get(('2021-01-01','2021-02-01'), freq='D')
		>>> Schedule.get(('2021-01-01','2021-06-01'), periods=5)
		"""
		if not ranges:
			raise ValueError("Provide at least one (start, end) tuple.")

		idx = []
		for r in ranges:
			if len(r) != 2:
				raise ValueError("Each range must be a (start, end) tuple.")
			start, end = map(_to_timestamp, r)
			if periods is not None:
				if freq is None:
					raise ValueError("When using `periods`, also provide a `freq`.")
				idx.append(pd.date_range(start=start, periods=periods, freq=freq, normalize=normalize))
			else:
				if freq is None:
					raise ValueError("Provide `freq` when using start..end ranges.")
				idx.append(pd.date_range(start=start, end=end, freq=freq, inclusive=inclusive, normalize=normalize))

		combined = pd.DatetimeIndex([]).append(idx)
		combined = combined.unique().sort_values()

		return Schedule(pd.Series(combined))

	 # ---------------------------- utilities --------------------------- #
	def clamp(self, start: Optional[DateLike] = None, end: Optional[DateLike] = None) -> "Schedule":
		"""
		Trim to [start, end]. If start/end is None, uses current bounds.
		"""
		s = _to_timestamp(start) if start is not None else self._series.iloc[0]
		e = _to_timestamp(end)   if end   is not None else self._series.iloc[-1]
		return self[self.isbetween(s, e, inclusive="both")]

	def nearest(self, date: DateLike) -> pd.Timestamp:
		"""Return the nearest timestamp in the schedule."""
		ts = _to_timestamp(date)
		pos = np.searchsorted(self._series.values, ts.to_datetime64())
		pos = np.clip(pos, 1, len(self._series) - 1)
		left = self._series.iloc[pos - 1]
		right = self._series.iloc[pos]
		return left if abs(ts - left) <= abs(right - ts) else right

if __name__ == "__main__":
	# Example: two disjoint monthly blocks sampled every 7 days
	span = Schedule.get(
		('2021-01-01', '2021-02-01'),
		('2024-01-01', '2024-02-01'),
		freq='7D'
	)

	print(span.series)

	print("First / last:", span.mindate, span.maxdate)
	print("Month lengths:", span.month_lengths().tolist())
	t = span.days_since_start()
	print("Elapsed days:", np.round(t, 2))