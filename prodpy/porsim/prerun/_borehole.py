import math

from typing import Iterable

from ._constraint import Constraint

class Borehole:
    """
    Borehole object with location, radius, skin, and scheduled constraints.

    The class stores multiple `Constraint` objects and guarantees that their
    active time intervals do not overlap. Constraint intervals are treated as
    half-open intervals:

        start <= t < stop

    Therefore, the following is allowed:

        Constraint(start=0, stop=10, orate=1000)
        Constraint(start=10, stop=20, press=3000)

    but the following is not allowed:

        Constraint(start=0, stop=10, orate=1000)
        Constraint(start=9, stop=20, press=3000)

    Parameters
    ----------
    index : tuple[int, ...], optional
        Grid indices containing the borehole.
    axis    : str, optional
        Borehole orientation. Accepted values are:

        - "x" : borehole aligned with x-direction
        - "y" : borehole aligned with y-direction
        - "z" : borehole aligned with z-direction

        Default is "z", representing a vertical well in most grid conventions.
    radius  : float, optional
        Borehole radius in ft. Default is 0.5 ft.
    skin    : float, optional
        Dimensionless mechanical/completion skin. Default is 0.
    constraints : iterable of Constraint, optional
        Initial iterable of non-overlapping `Constraint` objects.

    Notes
    -----
    Public radius is exposed in ft to stay consistent with oilfield units.
    Internally, radius is stored in meters.
    """
    FT_TO_METER = 0.3048
    AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}
    ID_TO_AXIS = {0: "x", 1: "y", 2: "z"}

    def __init__(
        self,
        *,
        index: tuple[int, ...] | None = None,
        axis: str = "z",
        radius: float = 0.5,
        skin: float = 0.,
        constraints: Iterable[Constraint] | None = None,
    ):
        self._index = None
        self._axis = None
        self._radius = None
        self._skin = None
        self._constraints: list[Constraint] = []

        self.index = index
        self.axis = axis
        self.radius = radius
        self.skin = skin

        if constraints is not None:
            for constraint in constraints:
                self.add_constraint(constraint)

    @property
    def index(self):
        """Getter for well index."""
        return self._index
    
    @index.setter
    def index(self, value: tuple[int, ...] | None):
        if value is None:
            self._index = None
            return

        if not isinstance(value, tuple):
            raise TypeError(f"index must be a tuple or None; got {type(value).__name__}.")

        if len(value) == 0:
            raise ValueError("index cannot be an empty tuple.")

        if not all(isinstance(item, int) for item in value):
            raise TypeError("all index values must be integers.")

        self._index = value
        
    @property
    def axis(self):
        """Getter for well axis in grids."""
        return self.ID_TO_AXIS[self._axis]

    @axis.setter
    def axis(self, value):
        """Setter for well axis in grids."""

        if not isinstance(value, str):
            raise TypeError("axis must be a string.")
        
        if value not in self.AXIS_TO_ID:
            raise ValueError(f"axis must be one of {list(self.AXIS_TO_ID.keys())}; got {value!r}.")
        
        self._axis = self.AXIS_TO_ID[value]

    @property
    def radius(self):
        """Getter for well radius."""
        return self._radius/self.FT_TO_METER

    @radius.setter
    def radius(self, value: float):
        try:
            value = float(value)
        except Exception as exc:
            raise ValueError(f"radius must be convertible to float; got {value!r}.") from exc

        if value <= 0:
            raise ValueError("radius must be positive.")

        self._radius = value * self.FT_TO_METER

    @property
    def area(self):
        """Getter for borehole area in ft2."""
        return self._area / (self.FT_TO_METER**2)

    @property
    def _area(self):
        """Getter for borehole area in m2."""
        return math.pi*self._radius**2
    
    @property
    def circumference(self):
        """Getter for borehole circumference in ft."""
        return self._circumference / self.FT_TO_METER
    
    @property
    def _circumference(self):
        """Getter for borehole circumference in m."""
        return 2*math.pi*self._radius

    @property
    def skin(self):
        """Getter for well skin."""
        return self._skin
    
    @skin.setter
    def skin(self, value: float):
        try:
            self._skin = float(value)
        except Exception as exc:
            raise ValueError(f"skin must be convertible to float; got {value!r}.") from exc
        
    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """
        Constraints sorted by start time.

        A tuple is returned to prevent accidental list-level modification.
        """
        return tuple(self._constraints)

    # ---------------------------------------------------------------------
    # Constraint management
    # ---------------------------------------------------------------------
    def add_constraint(self, constraint: Constraint | None = None, **kwargs) -> Constraint:
        """
        Add a new borehole constraint.

        Parameters
        ----------
        constraint : Constraint, optional
            Existing Constraint object.
        **kwargs
            Keyword arguments used to create a Constraint if `constraint`
            is not provided. For example:

                add_constraint(start=0, stop=30, orate=1000)

        Returns
        -------
        Constraint
            The added Constraint object.

        Raises
        ------
        TypeError
            If `constraint` is not a Constraint object.
        ValueError
            If the new constraint overlaps with an existing constraint.
        """
        if constraint is None:
            constraint = Constraint(**kwargs)
        elif kwargs:
            raise ValueError(
                "Provide either an existing Constraint object or keyword "
                "arguments to create one, not both."
            )

        self._check_constraint_type(constraint)
        self._check_no_overlap(constraint)

        self._constraints.append(constraint)
        self._constraints.sort(key=lambda item: item.start)

        return constraint

    def remove_constraint(self, constraint: Constraint) -> None:
        """
        Remove a constraint from the borehole.

        Parameters
        ----------
        constraint : Constraint
            Constraint object to remove.
        """
        self._constraints.remove(constraint)

    def clear_constraints(self) -> None:
        """Remove all constraints from the borehole."""
        self._constraints.clear()

    def active_constraint(self, time_days: float) -> Constraint | None:
        """
        Return the active constraint at a given time.

        Parameters
        ----------
        time_days : float
            Time in days.

        Returns
        -------
        Constraint or None
            Active constraint at `time_days`, or None if no constraint is active.
        """
        try:
            time_days = float(time_days)
        except Exception as exc:
            raise ValueError(f"time_days must be convertible to float; got {time_days!r}.") from exc

        active = [constraint for constraint in self._constraints if constraint.is_active(time_days)]

        if len(active) > 1:
            raise RuntimeError(
                "More than one active constraint was found. "
                "The borehole schedule is internally inconsistent."
            )

        return active[0] if active else None

    def validate_schedule(self) -> bool:
        """
        Validate that all stored constraints are non-overlapping.

        Returns
        -------
        bool
            True if the schedule is valid.

        Raises
        ------
        ValueError
            If any two constraints overlap.
        """
        sorted_constraints = sorted(self._constraints, key=lambda item: item.start)

        for previous, current in zip(sorted_constraints[:-1], sorted_constraints[1:]):
            if self._constraints_overlap(previous, current):
                raise ValueError(
                    "Overlapping constraints detected:\n"
                    f"  Existing: {self._format_constraint(previous)}\n"
                    f"  Current : {self._format_constraint(current)}"
                )

        return True

    # ---------------------------------------------------------------------
    # Schedule helpers
    # ---------------------------------------------------------------------
    def schedule(self):
        """
        Return the borehole constraint schedule.

        Returns
        -------
        pandas.DataFrame or list[dict]
            If pandas is available, returns a DataFrame. Otherwise, returns
            a list of dictionaries.
        """
        rows = []

        for i, constraint in enumerate(self._constraints):
            rows.append(
                {
                    "index": i,
                    "start_days": constraint.start,
                    "stop_days": constraint.stop,
                    "mode": constraint.mode,
                    "limit": constraint.limit,
                    "unit": self._unit_for_mode(constraint.mode),
                }
            )

        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def change_times(self) -> list[float]:
        """
        Return all finite start and stop times where the schedule changes.

        Returns
        -------
        list of float
            Sorted unique times in days.
        """
        times = []

        for constraint in self._constraints:
            times.append(constraint.start)
            if constraint.stop is not None:
                times.append(constraint.stop)

        return sorted(set(times))

    # ---------------------------------------------------------------------
    # Plotting methods
    # ---------------------------------------------------------------------
    def plot_schedule(self, ax=None, max_time: float | None = None, show_labels: bool = True):
        """
        Plot constraint intervals as a Gantt-style schedule.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing matplotlib axes. If None, a new figure and axes are created.
        max_time : float, optional
            Plot horizon in days. Useful when the last constraint has stop=None.
        show_labels : bool, optional
            If True, write constraint mode and limit on each bar.

        Returns
        -------
        tuple
            Matplotlib `(fig, ax)`.
        """
        if not self._constraints:
            raise ValueError("No constraints available to plot.")

        import matplotlib.pyplot as plt

        if ax is None:
            height = max(2.5, 0.55 * len(self._constraints) + 1.5)
            fig, ax = plt.subplots(figsize=(9, height))
        else:
            fig = ax.figure

        x_max = self._plot_horizon(max_time=max_time)

        for i, constraint in enumerate(self._constraints):
            start = constraint.start
            stop = x_max if constraint.stop is None else constraint.stop
            width = stop - start

            if width <= 0:
                continue

            ax.broken_barh([(start, width)], (i - 0.35, 0.7))

            if show_labels:
                label = (
                    f"{constraint.mode} = {constraint.limit:g} "
                    f"{self._unit_for_mode(constraint.mode)}"
                )
                ax.text(start + 0.01 * max(x_max, 1.0), i, label, va="center")

        ax.set_xlabel("Time, days")
        ax.set_ylabel("Constraint")
        ax.set_yticks(range(len(self._constraints)))
        ax.set_yticklabels([f"{i}: {c.mode}" for i, c in enumerate(self._constraints)])
        ax.set_xlim(left=0, right=x_max)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(f"Borehole constraint schedule: {self.index}")

        return fig, ax

    def plot_limits(self, mode: str | None = None, ax=None, max_time: float | None = None):
        """
        Plot constraint limit versus time for one constraint mode.

        This method is best used when plotting one mode at a time because
        pressure, liquid rate, oil rate, water rate, and gas rate have different
        physical units.

        Parameters
        ----------
        mode : str, optional
            Constraint mode to plot. If None, the method only works when all
            constraints have the same mode.
        ax : matplotlib.axes.Axes, optional
            Existing matplotlib axes. If None, a new figure and axes are created.
        max_time : float, optional
            Plot horizon in days. Useful when the last constraint has stop=None.

        Returns
        -------
        tuple
            Matplotlib `(fig, ax)`.
        """
        if not self._constraints:
            raise ValueError("No constraints available to plot.")

        modes = {constraint.mode for constraint in self._constraints}

        if mode is None:
            if len(modes) != 1:
                raise ValueError(
                    f"Multiple constraint modes exist: {sorted(modes)}. "
                    "Pass mode='press', 'orate', etc. to plot one mode."
                )
            mode = next(iter(modes))

        if mode not in Constraint.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of {Constraint.VALID_MODES}.")

        selected = [constraint for constraint in self._constraints if constraint.mode == mode]

        if not selected:
            raise ValueError(f"No constraints found for mode={mode!r}.")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        else:
            fig = ax.figure

        x_max = self._plot_horizon(max_time=max_time)

        for constraint in selected:
            start = constraint.start
            stop = x_max if constraint.stop is None else constraint.stop

            ax.hlines(
                y=constraint.limit,
                xmin=start,
                xmax=stop,
                linewidth=2,
            )

            ax.plot([start, stop], [constraint.limit, constraint.limit], marker="o")

        ax.set_xlabel("Time, days")
        ax.set_ylabel(f"{mode} limit, {self._unit_for_mode(mode)}")
        ax.set_xlim(left=0, right=x_max)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{mode} constraint limits: {self.index}")

        return fig, ax

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _check_constraint_type(constraint: Constraint) -> None:
        if not isinstance(constraint, Constraint):
            raise TypeError(
                "constraint must be a Constraint object. "
                f"Got {type(constraint).__name__}."
            )

    def _check_no_overlap(self, new_constraint: Constraint) -> None:
        for existing in self._constraints:
            if self._constraints_overlap(existing, new_constraint):
                raise ValueError(
                    "New constraint overlaps with an existing constraint:\n"
                    f"  Existing: {self._format_constraint(existing)}\n"
                    f"  New     : {self._format_constraint(new_constraint)}"
                )

    @staticmethod
    def _constraints_overlap(first: Constraint, second: Constraint) -> bool:
        """
        Check whether two half-open intervals overlap.

        Intervals are:

            [first.start, first.stop)
            [second.start, second.stop)

        A stop value of None is treated as infinity.
        """
        first_start = first.start
        first_stop = math.inf if first.stop is None else first.stop

        second_start = second.start
        second_stop = math.inf if second.stop is None else second.stop

        return first_start < second_stop and second_start < first_stop

    @staticmethod
    def _format_constraint(constraint: Constraint) -> str:
        stop = "None" if constraint.stop is None else f"{constraint.stop:g}"

        return (
            f"[{constraint.start:g}, {stop}) "
            f"{constraint.mode}={constraint.limit:g}"
        )

    @staticmethod
    def _unit_for_mode(mode: str) -> str:
        units = {
            "press": "psi",
            "lrate": "bbl/day",
            "orate": "bbl/day",
            "wrate": "bbl/day",
            "grate": "ft3/day",
        }
        return units.get(mode, "")

    def _plot_horizon(self, max_time: float | None = None) -> float:
        if max_time is not None:
            try:
                max_time = float(max_time)
            except Exception as exc:
                raise ValueError(f"max_time must be convertible to float; got {max_time!r}.") from exc

            if max_time <= 0:
                raise ValueError("max_time must be positive.")

            return max_time

        finite_stops = [
            constraint.stop for constraint in self._constraints if constraint.stop is not None
        ]
        max_start = max(constraint.start for constraint in self._constraints)

        if finite_stops:
            x_max = max(max(finite_stops), max_start)
        else:
            x_max = max_start

        # Add a small visual horizon for open-ended final constraints.
        if any(constraint.stop is None for constraint in self._constraints):
            x_max = max(x_max, max_start + max(1.0, 0.10 * max(max_start, 1.0)))

        return max(1.0, 1.05 * x_max)

    def __repr__(self) -> str:
        return (
            f"Borehole(index={self.index!r}, "
            f"radius={self.radius:g} ft, "
            f"skin={self.skin:g}, "
            f"constraints={len(self._constraints)})"
        )
