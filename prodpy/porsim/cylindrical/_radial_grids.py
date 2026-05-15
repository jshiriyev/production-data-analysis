import numpy as np

from ._base_class import BaseClass

class RadialGrids(BaseClass):
    """
    Cylindrical grid geometry with logarithmically spaced radial boundaries.

    The radial domain starts at ``well_radius`` and extends through one or more
    concentric outer boundary radii. Every radial region receives at least one
    cell. Any remaining radial cells are distributed in proportion to each
    region's logarithmic radial thickness, and boundaries within each region are
    geometrically spaced.

    Parameters
    ----------
    well_radius : float
        Wellbore radius, ft. Must be finite and positive.
    *outer_radii : float or 1-D array-like of float
        Additional outer radial boundaries, ft. Pass them as separate positional
        values or as a one-dimensional sequence. Together with
        ``well_radius``, values must be finite, positive, and strictly
        increasing.
    zdelta : float or 1-D array-like of float
        Cell thicknesses in the z direction, ft. Values must be positive.
    tdelta : float or 1-D array-like of float, optional
        Cell angles in the theta direction, rad. Values must be positive and
        sum to ``2*pi``. Defaults to one full-circle theta cell.
    depth : float, default 1000.0
        Depth of the reservoir-domain top surface, ft.
    num : int, default 50
        Total number of radial cells across all regions. Must be a positive
        integer and at least the number of radial regions.
    """

    def __init__(
        self,
        well_radius: float,
        *outer_radii: float | np.typing.ArrayLike,
        zdelta: np.typing.ArrayLike,
        tdelta: np.typing.ArrayLike | None = None,
        depth: float = 1000.0,
        num: int = 50,
    ):
        super().__init__(
            radii=self.get_radii(well_radius, *outer_radii, num=num),
            zdelta=zdelta,
            tdelta=tdelta,
            depth=depth,
        )

    @property
    def rnums(self) -> int:
        """Return the number of radial cells."""
        return self.radii.size - 1

    @property
    def tnums(self) -> int:
        """Return the number of theta-direction cells."""
        return self.tdelta.size

    @property
    def znums(self) -> int:
        """Return the number of z-direction cells."""
        return self.zdelta.size

    @property
    def nums_tuple(self) -> tuple[int, int, int]:
        """Return cell counts as ``(rnums, tnums, znums)``."""
        return (self.rnums, self.tnums, self.znums)

    @staticmethod
    def _to_outer_radii_array(
        outer_radii: tuple[float | np.typing.ArrayLike, ...],
    ) -> np.ndarray:
        """Normalize positional outer-radii inputs to a one-dimensional array."""
        value = outer_radii[0] if len(outer_radii) == 1 else outer_radii

        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("outer_radii must be convertible to a numeric array.") from error

        if array.ndim == 0:
            array = array.reshape(1)
        elif array.ndim != 1:
            raise ValueError("outer_radii must be scalar values or one-dimensional.")

        return array

    @staticmethod
    def _validate_well_radius(well_radius: float) -> float:
        """Return a validated scalar wellbore radius in feet."""
        try:
            array = np.asarray(well_radius, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("well_radius must be convertible to a numeric scalar.") from error

        if array.ndim != 0:
            raise ValueError("well_radius must be scalar.")

        if not np.isfinite(array):
            raise ValueError("well_radius must be finite.")

        if array <= 0:
            raise ValueError("well_radius must be positive.")

        return float(array)

    @staticmethod
    def _validate_num(num: int) -> int:
        """Return a validated positive integer radial-cell count."""
        try:
            array = np.asarray(num, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("num must be convertible to a numeric scalar.") from error

        if array.ndim != 0:
            raise ValueError("num must be scalar.")

        if not np.isfinite(array):
            raise ValueError("num must be finite.")

        if array <= 0 or array != np.floor(array):
            raise ValueError("num must be a positive integer.")

        return int(array)

    @staticmethod
    def _allocate_region_cells(log_lengths: np.ndarray, num: int) -> np.ndarray:
        """
        Allocate radial cells while keeping at least one cell in each region.

        After assigning one mandatory cell to every region, remaining cells are
        distributed by largest remainder in proportion to logarithmic radial
        thickness.
        """
        n_regions = log_lengths.size
        cells = np.ones(n_regions, dtype=int)
        remaining = num - n_regions

        if remaining == 0:
            return cells

        raw_extra_cells = remaining * log_lengths / log_lengths.sum()
        extra_cells = np.floor(raw_extra_cells).astype(int)
        cells += extra_cells

        remainder = remaining - extra_cells.sum()
        if remainder:
            fractions = raw_extra_cells - extra_cells
            order = np.argsort(-fractions, kind="stable")
            cells[order[:remainder]] += 1

        return cells

    @classmethod
    def get_radii(
        cls,
        well_radius: float,
        *outer_radii: float | np.typing.ArrayLike,
        num: int = 50,
    ) -> np.ndarray:
        """
        Generate logarithmically spaced radial boundaries for concentric regions.

        Parameters
        ----------
        well_radius : float
            Wellbore radius, ft. Must be finite and positive.
        *outer_radii : float or 1-D array-like of float
            Additional outer radial boundaries, ft. Pass them as separate
            positional values, for example ``rd, re``, or as a
            one-dimensional sequence, for example ``[rd, re]``. Together with
            ``well_radius``, values must be finite, positive, and strictly
            increasing.
        num : int, default 50
            Total number of radial cells across all regions. Must be a positive
            integer and at least the number of radial regions.

        Returns
        -------
        numpy.ndarray
            Radial boundary coordinates in feet. Length is ``num + 1``.

        Notes
        -----
        Each radial region receives one cell first. Remaining cells are
        allocated in proportion to logarithmic radial thickness using a
        largest-remainder rule. Boundaries inside each region are then spaced
        geometrically.
        """
        well_radius = cls._validate_well_radius(well_radius)
        outer_radii_array = cls._to_outer_radii_array(outer_radii)
        num = cls._validate_num(num)

        radii = np.concatenate(([well_radius], outer_radii_array))

        if radii.size < 2:
            raise ValueError("At least two radii are required.")

        if not np.all(np.isfinite(radii)):
            raise ValueError("All radii must be finite.")

        if np.any(radii <= 0):
            raise ValueError("All radii must be positive.")

        if np.any(np.diff(radii) <= 0):
            raise ValueError("Radii must be strictly increasing.")

        n_regions = radii.size - 1
        if num < n_regions:
            raise ValueError(
                f"num must be at least {n_regions}, so each region gets at least one cell."
            )

        log_lengths = np.log(radii[1:] / radii[:-1])
        cells = cls._allocate_region_cells(log_lengths, num)

        grid = []
        for i in range(n_regions):
            r_in = radii[i]
            r_out = radii[i + 1]
            n = cells[i]

            gamma = (r_out / r_in) ** (1 / n)
            region_grid = r_in * gamma ** np.arange(n + 1)
            region_grid[0] = r_in
            region_grid[-1] = r_out

            if i > 0:
                region_grid = region_grid[1:]

            grid.extend(region_grid)

        return np.asarray(grid)

    @staticmethod
    def _validate_angular_resolution(angular_resolution: int) -> int:
        """Return a validated angular sample count for smooth circular surfaces."""
        try:
            array = np.asarray(angular_resolution, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("angular_resolution must be convertible to a numeric scalar.") from error

        if array.ndim != 0:
            raise ValueError("angular_resolution must be scalar.")

        if not np.isfinite(array):
            raise ValueError("angular_resolution must be finite.")

        if array != np.floor(array):
            raise ValueError("angular_resolution must be an integer.")

        if array < 8:
            raise ValueError("angular_resolution must be at least 8.")

        return int(array)

    def _plot3d_geometry(
        self,
        angular_resolution: int = 96,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return radius, theta, depth, and smooth-theta arrays for 3-D plotting."""
        angular_resolution = self._validate_angular_resolution(angular_resolution)

        radii = self.radii
        theta_edges = np.insert(np.cumsum(self.tdelta), 0, 0.0)
        theta_edges[-1] = 2 * np.pi

        depth_edges = self.depth.item() + np.insert(np.cumsum(self.zdelta), 0, 0.0)
        theta_curve = np.linspace(0.0, 2 * np.pi, angular_resolution + 1)

        return radii, theta_edges, depth_edges, theta_curve

    def plot3d(
        self,
        title: str | None = None,
        *,
        show: bool = False,
        show_surfaces: bool = True,
        show_edges: bool = True,
        surface_opacity: float = 0.32,
        edge_color: str = "rgba(15, 23, 42, 0.62)",
        colorscale: str = "Viridis",
        vertical_exaggeration: float = 5.0,
        angular_resolution: int = 96,
        html_path: str | None = None,
        include_plotlyjs: bool | str = True,
        width: int | None = None,
        height: int = 720,
    ):
        """
        Build an interactive 3-D Plotly figure of the cylindrical grid.

        The plot uses oilfield units on all public axes: x and y are in feet,
        and the vertical axis is depth in feet with increasing depth downward.
        Exterior top, bottom, inner-radius, and outer-radius surfaces may be
        shown together with wireframe grid boundaries for the radial, theta,
        and z subdivisions.

        Parameters
        ----------
        title : str, optional
            Figure title. Defaults to a grid-size summary.
        show : bool, default False
            If True, immediately display the figure.
        show_surfaces : bool, default True
            If True, draw translucent top, bottom, inner, and outer surfaces.
        show_edges : bool, default True
            If True, draw wireframe edges for all cell boundaries.
        surface_opacity : float, default 0.32
            Opacity applied to translucent surfaces.
        edge_color : str, default "rgba(15, 23, 42, 0.62)"
            Plotly color used for grid-line edges.
        colorscale : str, default "Viridis"
            Plotly colorscale used for depth-colored surfaces.
        vertical_exaggeration : float, default 5.0
            Visual z-axis exaggeration used only for the scene aspect ratio.
        angular_resolution : int, default 96
            Number of angular samples used to render smooth circular surfaces.
            Must be an integer of at least 8.
        html_path : str, optional
            Path to write a standalone HTML plot. If omitted and ``show=True``,
            a temporary standalone HTML file is created and opened.
        include_plotlyjs : bool or str, default True
            Passed to Plotly's HTML writer. The default embeds Plotly so the
            browser page works offline.
        width : int, optional
            Figure width in pixels.
        height : int, default 720
            Figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 3-D cylindrical-grid figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as error:
            raise ImportError("3-D grid plotting requires plotly. Install prodpy[plots].") from error

        if vertical_exaggeration <= 0:
            raise ValueError("vertical_exaggeration must be positive.")

        radii, theta_edges, depth_edges, theta_curve = self._plot3d_geometry(
            angular_resolution=angular_resolution
        )
        fig = go.Figure()

        if show_surfaces:
            surface_kwargs = {
                "colorscale": colorscale,
                "opacity": surface_opacity,
                "hovertemplate": "X %{x:.2f} ft<br>Y %{y:.2f} ft<br>Depth %{z:.2f} ft<extra></extra>",
                "contours": {"z": {"show": False}},
            }

            theta_mesh, radius_mesh = np.meshgrid(theta_curve, radii, indexing="ij")
            x_annulus = radius_mesh * np.cos(theta_mesh)
            y_annulus = radius_mesh * np.sin(theta_mesh)

            top_depth = np.full_like(x_annulus, depth_edges[0], dtype=float)
            base_depth = np.full_like(x_annulus, depth_edges[-1], dtype=float)

            fig.add_trace(
                go.Surface(
                    x=x_annulus,
                    y=y_annulus,
                    z=top_depth,
                    surfacecolor=top_depth,
                    name="Top depth",
                    showscale=False,
                    **surface_kwargs,
                )
            )
            fig.add_trace(
                go.Surface(
                    x=x_annulus,
                    y=y_annulus,
                    z=base_depth,
                    surfacecolor=base_depth,
                    name="Base depth",
                    showscale=False,
                    **surface_kwargs,
                )
            )

            theta_shell, depth_shell = np.meshgrid(theta_curve, depth_edges, indexing="ij")
            shell_depth_colors = depth_shell

            for radius, name, showscale in (
                (radii[0], "Inner radius", False),
                (radii[-1], "Outer radius", True),
            ):
                fig.add_trace(
                    go.Surface(
                        x=radius * np.cos(theta_shell),
                        y=radius * np.sin(theta_shell),
                        z=depth_shell,
                        surfacecolor=shell_depth_colors,
                        name=name,
                        showscale=showscale,
                        colorbar={"title": "Depth, ft", "len": 0.72} if showscale else None,
                        **surface_kwargs,
                    )
                )

        if show_edges:
            xline = []
            yline = []
            zline = []

            for depth in depth_edges:
                for radius in radii:
                    xline.extend([*(radius * np.cos(theta_curve)), None])
                    yline.extend([*(radius * np.sin(theta_curve)), None])
                    zline.extend([*np.full(theta_curve.size, depth), None])

            theta_boundaries = theta_edges[:-1] if self.tnums > 1 else np.array([])
            for depth in depth_edges:
                for theta in theta_boundaries:
                    xline.extend([*(radii * np.cos(theta)), None])
                    yline.extend([*(radii * np.sin(theta)), None])
                    zline.extend([*np.full(radii.size, depth), None])

            for theta in theta_boundaries:
                for radius in radii:
                    xline.extend([*np.full(depth_edges.size, radius * np.cos(theta)), None])
                    yline.extend([*np.full(depth_edges.size, radius * np.sin(theta)), None])
                    zline.extend([*depth_edges, None])

            fig.add_trace(
                go.Scatter3d(
                    x=xline,
                    y=yline,
                    z=zline,
                    mode="lines",
                    line={"color": edge_color, "width": 3},
                    name="Grid lines",
                    hoverinfo="skip",
                )
            )

        horizontal_span = max(float(2 * radii[-1]), 1.0)
        zspan = max(float(np.ptp(depth_edges)) * vertical_exaggeration, 1.0)
        maxspan = max(horizontal_span, zspan)

        fig.update_layout(
            title=(
                title
                or f"{self.__class__.__name__} 3-D Grid "
                f"({self.rnums} x {self.tnums} x {self.znums})"
            ),
            template="plotly_white",
            width=width,
            height=height,
            margin={"l": 0, "r": 0, "t": 58, "b": 0},
            showlegend=False,
            scene={
                "xaxis": {
                    "title": "X, ft",
                    "backgroundcolor": "rgba(248, 250, 252, 0.72)",
                    "gridcolor": "#d7dee8",
                },
                "yaxis": {
                    "title": "Y, ft",
                    "backgroundcolor": "rgba(248, 250, 252, 0.72)",
                    "gridcolor": "#d7dee8",
                },
                "zaxis": {
                    "title": "Depth, ft",
                    "autorange": "reversed",
                    "backgroundcolor": "rgba(248, 250, 252, 0.72)",
                    "gridcolor": "#d7dee8",
                },
                "aspectmode": "manual",
                "aspectratio": {
                    "x": horizontal_span / maxspan,
                    "y": horizontal_span / maxspan,
                    "z": zspan / maxspan,
                },
                "camera": {"eye": {"x": 1.55, "y": -1.7, "z": 1.05}},
            },
        )

        if show or html_path is not None:
            if html_path is None:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    prefix="prodpy-radial-grid-",
                    suffix=".html",
                    delete=False,
                ) as html_file:
                    html_path = html_file.name

            fig.write_html(
                html_path,
                include_plotlyjs=include_plotlyjs,
                full_html=True,
                auto_open=show,
                config={"responsive": True, "displaylogo": False},
            )

        return fig
