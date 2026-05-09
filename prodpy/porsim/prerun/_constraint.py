import numpy as np

class Constraint():
    """
    Base class for pressure and rate constraints.

    A constraint defines one operating control over a time interval.

    Exactly one control keyword must be provided.

    Public values are supplied and returned in oilfield units; values are stored
    internally in SI units.

    Productivity values exposed through `prod` are in ft3/day/psi, not bbl/day/psi.

    Raises
    ------
    ValueError
        If no control is provided, more than one control is provided, an
        unsupported control mode is provided, or the time interval is invalid.

    """
    VALID_MODES = {"press", "lrate", "orate", "wrate", "grate"}

    DAY_TO_SEC = 24*60*60
    FT_TO_METER = 0.3048
    PSI_TO_PA = 6894.757293168
    BBL_TO_FT3 = 5.61458

    def __init__(self,*,start:float=0,stop:float|None=None,**kwargs):
        """
        Parameters
        ----------
        start   : float, optional
            Start time of the constraint (days), default is 0
        stop    : float, optional
            End time for the constraint (days), default is None

        **kwargs : dict
        Specifies the constraint. Only one of the following should be provided:
        - 'press' : Constant pressure limit (psi)
        - 'lrate' : Constant liquid rate limit, (bbl/day)
        - 'orate' : Constant oil rate limit, (bbl/day)
        - 'wrate' : Constant water rate limit, (bbl/day)
        - 'grate' : Constant gas rate limit, (ft3/day)
        
        """
        self._start = 0.0
        self._stop = None
        self._prod = None

        self.start  = start
        self.stop   = stop
        self.set_limit(**kwargs)

    def set_limit(self, **kwargs):

        constraint  = {key:value for key,value in kwargs.items() if value is not None}

        if len(constraint) == 0:
            raise ValueError(f"No condition is provided.")
        elif len(constraint) == 1:
            mode_key, limit_value = next(iter(constraint.items()))
        elif len(constraint) > 1:
            raise ValueError(f"Multiple conditions are provided: {list(constraint.keys())}. Assign only one.")
        
        if mode_key not in self.VALID_MODES:
            raise ValueError(f"Invalid constraint type: {mode_key}. Must be one of {self.VALID_MODES}.")

        try:
            limit_value = float(limit_value)
        except Exception as exc:
            raise ValueError(f"Limit value must be convertible to float; got {limit_value!r}.") from exc
        
        self._mode = mode_key

        if self._mode == "press":
            self._limit = limit_value*self.PSI_TO_PA
        elif self._mode in ("lrate","orate","wrate"):
            self._limit = limit_value*(self.FT_TO_METER**3)/self.DAY_TO_SEC*self.BBL_TO_FT3
        elif self._mode == "grate":
            self._limit = limit_value*(self.FT_TO_METER**3)/self.DAY_TO_SEC

    @property
    def start(self):
        """Getter for the constraint start time."""
        return self._start/self.DAY_TO_SEC

    @start.setter
    def start(self, value):
        """Setter for the constraint start time."""
        try:
            value = float(value)
        except Exception as exc:
            raise ValueError(f"start must be convertible to float; got {value!r}.") from exc
        
        if value < 0:
            raise ValueError("start must be non-negative.")
        
        if self.stop is not None and value >= self.stop:
            raise ValueError("start must be less than stop.")
        
        self._start = value*self.DAY_TO_SEC

    @property
    def stop(self):
        """Getter for the constraint end time."""
        return None if self._stop is None else self._stop/self.DAY_TO_SEC

    @stop.setter
    def stop(self, value):
        """Setter for the constraint end time."""
        if value is None:
            self._stop = None
            return

        try:
            value = float(value)
        except Exception as exc:
            raise ValueError(f"stop must be convertible to float; got {value!r}.") from exc
        
        if value < 0:
            raise  ValueError("stop must be non-negative.")
        elif value <= self.start:
            raise ValueError("stop must be greater than start.")
        
        self._stop = value*self.DAY_TO_SEC

    def is_active(self, time_days):
        """Check whether the constraint is active at a given time: start <= time_days < stop."""
        return self.start <= time_days and (self.stop is None or time_days < self.stop)

    @property
    def mode(self):
        """Getter for the constraint type."""
        return self._mode
    
    @property
    def limit(self):
        """Getter for the constraint limit value."""
        if self._mode == "press":
            return self._limit/self.PSI_TO_PA
        elif self._mode in ("lrate","orate","wrate"):
            return self._limit*self.DAY_TO_SEC/(self.FT_TO_METER**3)/self.BBL_TO_FT3
        elif self._mode == "grate":
            return self._limit*self.DAY_TO_SEC/(self.FT_TO_METER**3)

    @property
    def prod(self):
        """Productivity or boundary transmissibility in ft3/day/psi."""
        if self._prod is None:
            raise ValueError("productivity has not been assigned yet.")
        
        return self._prod*(self.DAY_TO_SEC*self.PSI_TO_PA)/(self.FT_TO_METER**3)

    @prod.setter
    def prod(self,value:np.ndarray):
        """Set productivity or boundary transmissibility in ft3/day/psi."""
        self._prod = value*(self.FT_TO_METER**3)/(self.DAY_TO_SEC*self.PSI_TO_PA)

