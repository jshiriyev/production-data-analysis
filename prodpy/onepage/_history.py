import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns

class History():

    @staticmethod
    def wells(
        frame: pd.DataFrame,
        *,
        date_col: str = "date",
        oil_col: str = "oil_rate",
        water_col: str = "water_rate",
        gas_col: str = "gas_rate",
        choke_col: str = "choke",
        prod_days_col: str = "prod_days",
        lift_col: str = "lift_method",
        fit: pd.DataFrame | None = None,         # expects columns: x (datetime-like), y (float)
        forecast: pd.DataFrame | None = None,    # expects columns: x (datetime-like), y (float)
        figsize=(10, 16),
        show_today_line: bool = True,
        xlimit_to_year_plus_one: bool = True,
    ):
        """
        Plot a 4-panel production dashboard:
          1) Oil rate with optional fit and forecast
          2) Water rate (left axis) and Gas rate (right axis)
          3) Choke (left axis) and Production days (right axis)
          4) Lift method (categorical)

        Parameters
        ----------
        frame : DataFrame
            Must contain the columns specified by the *_col args.
        date_col, oil_col, water_col, gas_col, choke_col, prod_days_col, lift_col : str
            Column names in `frame`.
        fit, forecast : DataFrame or None
            Optional lines to overlay on subplot 1. Each should have columns ['x','y'].
        figsize : tuple
            Figure size (inches).
        show_today_line : bool
            Draw a vertical line at today's date on all subplots.
        xlimit_to_year_plus_one : bool
            Set x-limit max to Jan 1 of next year to leave room for forecast.

        Returns
        -------
        fig, axes : (Figure, ndarray[Axes])
        """

        # --- Ensure datetime index/columns are correct ---
        df = frame.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        if fit is not None:
            fit = fit.copy()
            fit["x"] = pd.to_datetime(fit["x"])
        if forecast is not None:
            forecast = forecast.copy()
            forecast["x"] = pd.to_datetime(forecast["x"])

        # --- Figure and axes ---
        fig, axes = plt.subplots(nrows=4, figsize=figsize, sharex=True)

        # === 1) Oil rate with optional fit/forecast ===
        axes[0].plot(
            df[date_col], df[oil_col],
            linestyle="", marker="o", markersize=2, color="#6B5B95", label="Oil rate"
        )

        if fit is not None:
            l_fit, = axes[0].plot(fit["x"], fit["y"], color="red", linewidth=1.2, label="Fit")
            l_fit.set_path_effects([patheffects.withStroke(linewidth=3, foreground="black")])

        if forecast is not None:
            l_fc, = axes[0].plot(forecast["x"], forecast["y"], color="blue", linewidth=1.2, label="Forecast")
            l_fc.set_path_effects([patheffects.withStroke(linewidth=3, foreground="black")])

        axes[0].set_facecolor("#f4f4f4")
        axes[0].spines["left"].set_color("#6B5B95")
        axes[0].spines["bottom"].set_color("#6B5B95")
        axes[0].grid(True, linestyle="--", color="gray", alpha=0.5)
        axes[0].tick_params(colors="#6B5B95")
        axes[0].set_ylabel("Daily Oil, t/d", fontsize=12, color="#88B04B", weight="bold")
        axes[0].legend(loc="upper left", fontsize=9)

        # === 2) Water (left) & Gas (right) ===
        axes[1].plot(
            df[date_col], df[water_col],
            linestyle="", marker="o", markersize=2, color="tab:blue", label="Water"
        )
        axes[1].set_ylabel("Water, m³/d", color="tab:blue")
        ax12 = axes[1].twinx()
        ax12.plot(
            df[date_col], df[gas_col],
            linestyle="", marker="o", markersize=3, color="tab:red", label="Gas"
        )
        ax12.set_ylabel("Gas, km³/d", color="tab:red")

        # === 3) Choke (left) & Production days (right) ===
        axes[2].plot(
            df[date_col], df[choke_col],
            linestyle="", marker="o", markersize=2, color="tab:blue", label="Choke"
        )
        axes[2].set_ylabel("Choke, %", color="tab:blue")
        ax22 = axes[2].twinx()
        ax22.plot(
            df[date_col], df[prod_days_col],
            linestyle="", marker="o", markersize=3, color="tab:red", label="Prod days"
        )
        ax22.set_ylabel("Prod days", color="tab:red")

        # === 4) Lift method (categorical -> codes + legend) ===
        # Map categories to integers for plotting and keep a legend mapping
        lift_codes, uniques = pd.factorize(df[lift_col])
        axes[3].scatter(df[date_col], lift_codes, s=10, color="tab:purple")
        axes[3].set_yticks(range(len(uniques)))
        axes[3].set_yticklabels(list(uniques))
        axes[3].set_ylabel("Lift method")

        # --- Shared decorations ---
        today = pd.Timestamp.today().normalize()
        if show_today_line:
            for ax in (axes.tolist() + [ax12, ax22]):
                ax.axvline(x=today, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

        if xlimit_to_year_plus_one:
            # Max of data/fit/forecast to choose a reasonable x max
            max_x = df[date_col].max()
            if fit is not None:
                max_x = max(max_x, fit["x"].max())
            if forecast is not None:
                max_x = max(max_x, forecast["x"].max())
            # extend to Jan 1 of next year relative to max_x
            x_end = pd.Timestamp(year=max_x.year + 1, month=1, day=1)
            for ax in axes:
                ax.set_xlim(right=x_end)

        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate()
        plt.tight_layout()

        return fig, axes

    @staticmethod
    def groups(
        df,
        *,
        figsize=(14, 6),
        seaborn_style="dark",
        seaborn_context="poster",
        seaborn_font_scale=1,
        seaborn_rc=None,  # e.g., {"grid.linewidth": 5}
        kind="bar",
        stacked=True,
        width=1.0,
        colormap="tab20",
        edgecolor="black",
        linewidth=1,
        title=None,
        xlabel="",
        ylabel="",
        legend_title="",
        legend_title_fontsize=12,
        legend_fontsize=12,
        xtick_rotation=90,
        xtick_fontsize=12,
        ytick_fontsize=12,
        grid=True,
        grid_which="both",
        grid_linestyle="--",
        grid_linewidth=0.5,
        tight_layout=True,
        ax=None,
        show=True,
    ):
        """
        Plot a stacked bar chart from a wide-form DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide-form table where index are x labels (e.g., years) and columns are categories to stack.
        figsize : tuple
            Figure size in inches.
        seaborn_style, seaborn_context, seaborn_font_scale, seaborn_rc : Seaborn styling controls.
        kind : str
            Pandas plot kind ("bar" recommended here).
        stacked : bool
            Whether to stack bars.
        width : float
            Bar width.
        colormap : str or Colormap
            Matplotlib colormap name or object (e.g., "tab20").
        edgecolor : str
            Bar edge color.
        linewidth : float
            Bar edge line width.
        title, xlabel, ylabel : str
            Plot text labels.
        legend_title, legend_title_fontsize, legend_fontsize : legend styling.
        xtick_rotation, xtick_fontsize, ytick_fontsize : tick styling.
        grid : bool
            Show grid lines if True.
        grid_which : {"both", "major", "minor"}
            Which grid lines to show.
        grid_linestyle : str
            Grid line style (e.g., "--").
        grid_linewidth : float
            Grid line width.
        tight_layout : bool
            Apply plt.tight_layout() if True.
        ax : matplotlib.axes.Axes or None
            Existing axes to draw on. If None, a new figure/axes are created.
        show : bool
            Call plt.show() at the end.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        # --- Styling ---
        if seaborn_rc is None:
            seaborn_rc = {"grid.linewidth": 5}
        sns.set(style=seaborn_style)
        sns.set_context(seaborn_context, font_scale=seaborn_font_scale, rc=seaborn_rc)

        # --- Axes / Figure ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            # If ax provided, still ensure figure has desired size
            ax.figure.set_size_inches(*figsize)

        # --- Main plot (pandas plotting API) ---
        df.plot(
            kind=kind,
            stacked=stacked,
            width=width,
            colormap=colormap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
        )

        # --- Labels & Title ---
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)

        # --- Legend ---
        leg = ax.legend(title=legend_title, fontsize=str(legend_fontsize))
        if leg and leg.get_title() is not None:
            leg.get_title().set_fontsize(str(legend_title_fontsize))

        # --- Ticks ---
        ax.tick_params(axis='x', labelrotation=xtick_rotation, labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)

        # --- Grid ---
        if grid:
            ax.grid(True, which=grid_which, linestyle=grid_linestyle, linewidth=grid_linewidth)

        # --- Layout ---
        if tight_layout:
            plt.tight_layout()

        if show:
            plt.show()

        return ax

if __name__ == "__main__":

	# --- create a small reproducible dataset ---
	rng = pd.date_range("2024-01-01", periods=200, freq="D")
	np.random.seed(42)

	frame = pd.DataFrame({
	    "date": rng,
	    "oil_rate": 300 + 20*np.sin(np.linspace(0, 6*np.pi, len(rng))) + np.random.normal(0, 8, len(rng)),
	    "water_rate": 120 + np.random.normal(0, 5, len(rng)),
	    "gas_rate": 0.12 + 0.02*np.sin(np.linspace(0, 2*np.pi, len(rng))) + np.random.normal(0, 0.005, len(rng)),
	    "choke": 45 + 5*np.sin(np.linspace(0, 3*np.pi, len(rng))) + np.random.normal(0, 1, len(rng)),
	    "prod_days": np.clip(np.random.normal(0.95, 0.08, len(rng))*30, 0, 30).round(0),
	    "lift_method": np.random.choice(["ESP", "Gas Lift", "Rod Pump"], size=len(rng), p=[0.5, 0.3, 0.2])
	})

	# Optional "fit": rolling mean of oil
	fit = (frame[["date","oil_rate"]]
	       .rename(columns={"date":"x","oil_rate":"y"})
	       .assign(y=lambda d: d["y"].rolling(14, min_periods=1).mean()))

	# Optional "forecast": simple flat forecast for the next 60 days
	future = pd.date_range(frame["date"].max() + pd.Timedelta(days=1), periods=60, freq="D")
	forecast = pd.DataFrame({
	    "x": future,
	    "y": np.full(len(future), frame["oil_rate"].iloc[-14:].mean())  # naïve average of last 2 weeks
	})

	# --- plot ---
	fig, axes = well_timeseries(
	    frame,
	    fit=fit,
	    forecast=forecast,
	    # if your columns have different names, adjust the *_col parameters here
	)

	plt.show()
