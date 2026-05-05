import pandas as pd
import numpy as np
import plotly.graph_objects as go

def add_horizontal_reservoir_cuboid(
    fig,
    data,
    units="ft",
    section_name="Horizontal Lateral",
    horizontal_inclination_min=80,
    length_padding=200,
    width=800,
    height=120,
    opacity=0.18,
    color="sandybrown",
    name="Reservoir",
    show_edges=True
):
    """
    Add a rectangular cuboid around the horizontal well section.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing Plotly figure to add the reservoir cuboid to.

    data : str or pandas.DataFrame
        Either a CSV path or DataFrame containing the trajectory data.

    units : str, default="ft"
        Units to use for plotting. Options:
        - "ft"
        - "m"

    section_name : str, default="Horizontal Lateral"
        Section value used to identify the horizontal section when the
        DataFrame contains a 'Section' column. Matching is case-insensitive
        and also accepts partial matches such as "Lateral".

    horizontal_inclination_min : float, default=80
        Fallback inclination cutoff in degrees when the requested section is
        not found. Points with inclination greater than or equal to this value
        are treated as horizontal section points.

    length_padding : float, default=200
        Extra distance added before the heel and after the toe along both
        easting and northing bounds.

    width : float, default=800
        Reservoir block width added around the lateral in map view.

    height : float, default=120
        Reservoir block thickness centered on the horizontal section depth.

    opacity : float, default=0.18
        Cuboid transparency.

    color : str, default="sandybrown"
        Cuboid fill color.

    name : str, default="Reservoir"
        Trace name for the cuboid.

    show_edges : bool, default=True
        If True, draws black outline edges around the cuboid.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The same figure object with reservoir cuboid traces added.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    units = units.lower()
    if units not in ["ft", "m"]:
        raise ValueError("units must be either 'ft' or 'm'.")

    md_col = f"MD_{units}"
    tvd_col = f"TVD_{units}"
    north_col = f"Northing_offset_{units}"
    east_col = f"Easting_offset_{units}"

    required_cols = [md_col, tvd_col, north_col, east_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(md_col).reset_index(drop=True)

    horizontal_df = pd.DataFrame()
    if "Section" in df.columns:
        section_values = df["Section"].astype(str).str.lower()
        section_match = section_values == section_name.lower()
        if not section_match.any():
            section_match = section_values.str.contains(section_name.lower(), regex=False)
        horizontal_df = df.loc[section_match]

    if horizontal_df.empty and "Inclination_deg" in df.columns:
        horizontal_df = df.loc[df["Inclination_deg"] >= horizontal_inclination_min]

    if horizontal_df.empty:
        horizontal_df = df.tail(max(2, len(df) // 4))

    if len(horizontal_df) < 2:
        raise ValueError("At least two horizontal section points are required.")

    x_min = horizontal_df[east_col].min() - length_padding
    x_max = horizontal_df[east_col].max() + length_padding
    y_min = horizontal_df[north_col].min() - width / 2
    y_max = horizontal_df[north_col].max() + width / 2
    z_center = -horizontal_df[tvd_col].mean()
    z_min = z_center - height / 2
    z_max = z_center + height / 2

    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])

    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            hoverinfo="skip",
            showlegend=True
        )
    )

    if show_edges:
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        edge_x = []
        edge_y = []
        edge_z = []
        for start, end in edge_pairs:
            edge_x.extend([vertices[start, 0], vertices[end, 0], None])
            edge_y.extend([vertices[start, 1], vertices[end, 1], None])
            edge_z.extend([vertices[start, 2], vertices[end, 2], None])

        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                name=f"{name} outline",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False
            )
        )

    return fig

def add_fracture_cluster(
    fig,
    data,
    units="ft",
    cluster_location="toe",
    fracture_count=4,
    offset_from_anchor=300,
    fracture_spacing=75,
    fracture_half_length=350,
    fracture_height=250,
    section_name="Horizontal Lateral",
    horizontal_inclination_min=80,
    opacity=0.35,
    color="royalblue",
    edge_color="navy",
    name="Fracture cluster",
    show_edges=True
):
    """
    Add one cluster of rectangular hydraulic fractures orthogonal to the well.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing Plotly figure to add the fracture cluster to.

    data : str or pandas.DataFrame
        Either a CSV path or DataFrame containing the trajectory data.

    units : str, default="ft"
        Units to use for plotting. Options:
        - "ft"
        - "m"

    cluster_location : str, default="toe"
        Anchor end of the horizontal section. Options:
        - "toe"
        - "heel"

    fracture_count : int, default=4
        Number of fractures in the cluster. Must be between 3 and 5.

    offset_from_anchor : float, default=300
        Distance from the toe or heel anchor to the center of the cluster.
        Toe clusters are placed back toward the heel; heel clusters are placed
        forward toward the toe.

    fracture_spacing : float, default=75
        Distance between adjacent fractures along the wellbore.

    fracture_half_length : float, default=350
        Half-length of each rectangular fracture wing measured away from the
        wellbore in the fracture plane.

    fracture_height : float, default=250
        Total near-vertical height of each rectangular fracture.

    section_name : str, default="Horizontal Lateral"
        Section value used to identify the horizontal section when the
        DataFrame contains a 'Section' column. Matching is case-insensitive
        and also accepts partial matches such as "Lateral".

    horizontal_inclination_min : float, default=80
        Fallback inclination cutoff in degrees when the requested section is
        not found. Points with inclination greater than or equal to this value
        are treated as horizontal section points.

    opacity : float, default=0.35
        Fracture transparency.

    color : str, default="royalblue"
        Fracture fill color.

    edge_color : str, default="navy"
        Fracture outline color.

    name : str, default="Fracture cluster"
        Trace name for the fracture cluster.

    show_edges : bool, default=True
        If True, draws outlines around each rectangular fracture.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The same figure object with fracture cluster traces added.
    """
    if not 3 <= fracture_count <= 5:
        raise ValueError("fracture_count must be between 3 and 5.")

    cluster_location = cluster_location.lower()
    if cluster_location not in ["toe", "heel"]:
        raise ValueError("cluster_location must be either 'toe' or 'heel'.")

    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    units = units.lower()
    if units not in ["ft", "m"]:
        raise ValueError("units must be either 'ft' or 'm'.")

    md_col = f"MD_{units}"
    tvd_col = f"TVD_{units}"
    north_col = f"Northing_offset_{units}"
    east_col = f"Easting_offset_{units}"

    required_cols = [md_col, tvd_col, north_col, east_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(md_col).reset_index(drop=True)

    horizontal_df = pd.DataFrame()
    if "Section" in df.columns:
        section_values = df["Section"].astype(str).str.lower()
        section_match = section_values == section_name.lower()
        if not section_match.any():
            section_match = section_values.str.contains(section_name.lower(), regex=False)
        horizontal_df = df.loc[section_match]

    if horizontal_df.empty and "Inclination_deg" in df.columns:
        horizontal_df = df.loc[df["Inclination_deg"] >= horizontal_inclination_min]

    if horizontal_df.empty:
        horizontal_df = df.tail(max(2, len(df) // 4))

    if len(horizontal_df) < 2:
        raise ValueError("At least two horizontal section points are required.")

    heel_point = np.array([
        horizontal_df[east_col].iloc[0],
        horizontal_df[north_col].iloc[0],
        -horizontal_df[tvd_col].iloc[0],
    ], dtype=float)
    toe_point = np.array([
        horizontal_df[east_col].iloc[-1],
        horizontal_df[north_col].iloc[-1],
        -horizontal_df[tvd_col].iloc[-1],
    ], dtype=float)

    well_direction = toe_point - heel_point
    well_direction_norm = np.linalg.norm(well_direction)
    if well_direction_norm == 0:
        raise ValueError("Horizontal section has zero length.")
    well_direction = well_direction / well_direction_norm

    if cluster_location == "toe":
        cluster_center = toe_point - well_direction * offset_from_anchor
    else:
        cluster_center = heel_point + well_direction * offset_from_anchor

    vertical_reference = np.array([0.0, 0.0, 1.0])
    fracture_width_axis = np.cross(vertical_reference, well_direction)
    fracture_width_norm = np.linalg.norm(fracture_width_axis)
    if fracture_width_norm < 1e-9:
        fracture_width_axis = np.array([1.0, 0.0, 0.0])
    else:
        fracture_width_axis = fracture_width_axis / fracture_width_norm
    fracture_height_axis = np.cross(well_direction, fracture_width_axis)
    fracture_height_axis = fracture_height_axis / np.linalg.norm(fracture_height_axis)

    vertices = []
    i = []
    j = []
    k = []
    edge_x = []
    edge_y = []
    edge_z = []
    center_offsets = (
        np.arange(fracture_count) - (fracture_count - 1) / 2
    ) * fracture_spacing

    for center_offset in center_offsets:
        center = cluster_center + well_direction * center_offset
        base_index = len(vertices)
        rectangle = [
            center - fracture_width_axis * fracture_half_length - fracture_height_axis * fracture_height / 2,
            center + fracture_width_axis * fracture_half_length - fracture_height_axis * fracture_height / 2,
            center + fracture_width_axis * fracture_half_length + fracture_height_axis * fracture_height / 2,
            center - fracture_width_axis * fracture_half_length + fracture_height_axis * fracture_height / 2,
        ]
        vertices.extend(rectangle)
        i.extend([base_index, base_index])
        j.extend([base_index + 1, base_index + 2])
        k.extend([base_index + 2, base_index + 3])

        edge_pairs = [
            (base_index, base_index + 1),
            (base_index + 1, base_index + 2),
            (base_index + 2, base_index + 3),
            (base_index + 3, base_index),
        ]
        for start, end in edge_pairs:
            edge_x.extend([vertices[start][0], vertices[end][0], None])
            edge_y.extend([vertices[start][1], vertices[end][1], None])
            edge_z.extend([vertices[start][2], vertices[end][2], None])

    vertices = np.array(vertices)

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=f"{name} ({cluster_location})",
            hoverinfo="skip",
            showlegend=True
        )
    )

    if show_edges:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                name=f"{name} outline ({cluster_location})",
                line=dict(color=edge_color, width=3),
                hoverinfo="skip",
                showlegend=False
            )
        )

    return fig

def plot_well_trajectory_3d(
    data,
    units="ft",
    color_by_section=True,
    show_stations=False,
    title="3D Well Trajectory"
):
    """
    Plot a well trajectory in 3D using Plotly.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Either:
        - path to a CSV file containing trajectory data, or
        - a pandas DataFrame.

    units : str, default="ft"
        Units to use for plotting. Options:
        - "ft"
        - "m"

    color_by_section : bool, default=True
        If True and the DataFrame contains a 'Section' column, the trajectory
        is colored by section (e.g., Vertical, Build, Lateral).

    show_stations : bool, default=False
        If True, shows markers at each survey station.

    title : str, default="3D Well Trajectory"
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object.

    Notes
    -----
    Expected columns for feet:
        - MD_ft
        - TVD_ft
        - Northing_offset_ft
        - Easting_offset_ft

    Expected columns for meters:
        - MD_m
        - TVD_m
        - Northing_offset_m
        - Easting_offset_m

    Optional columns for hover information:
        - Inclination_deg
        - Azimuth_deg
        - Section

    The z-axis is plotted as negative TVD so that deeper points appear lower.
    """
    # Load data if a file path is provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    units = units.lower()
    if units not in ["ft", "m"]:
        raise ValueError("units must be either 'ft' or 'm'.")

    # Column selection based on units
    md_col = f"MD_{units}"
    tvd_col = f"TVD_{units}"
    north_col = f"Northing_offset_{units}"
    east_col = f"Easting_offset_{units}"

    required_cols = [md_col, tvd_col, north_col, east_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by measured depth
    df = df.sort_values(md_col).reset_index(drop=True)

    # Plot z downward
    df["Z_plot"] = -df[tvd_col]

    # Hover data
    hover_cols = []
    for col in [md_col, tvd_col, "Inclination_deg", "Azimuth_deg", "Section"]:
        if col in df.columns:
            hover_cols.append(col)

    def make_hover_text(subdf):
        texts = []
        for _, row in subdf.iterrows():
            parts = []
            if md_col in row:
                parts.append(f"MD: {row[md_col]:.2f} {units}")
            if tvd_col in row:
                parts.append(f"TVD: {row[tvd_col]:.2f} {units}")
            if "Inclination_deg" in subdf.columns:
                parts.append(f"Inc: {row['Inclination_deg']:.2f}°")
            if "Azimuth_deg" in subdf.columns:
                parts.append(f"Azi: {row['Azimuth_deg']:.2f}°")
            if "Section" in subdf.columns:
                parts.append(f"Section: {row['Section']}")
            texts.append("<br>".join(parts))
        return texts

    fig = go.Figure()

    mode = "lines+markers" if show_stations else "lines"

    # Plot by section if available
    if color_by_section and "Section" in df.columns:
        for section, subdf in df.groupby("Section", sort=False):
            fig.add_trace(
                go.Scatter3d(
                    x=subdf[east_col],
                    y=subdf[north_col],
                    z=subdf["Z_plot"],
                    mode=mode,
                    name=str(section),
                    text=make_hover_text(subdf),
                    hovertemplate="%{text}<extra></extra>",
                    line=dict(width=6),
                    marker=dict(size=3)
                )
            )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=df[east_col],
                y=df[north_col],
                z=df["Z_plot"],
                mode=mode,
                name="Trajectory",
                text=make_hover_text(df),
                hovertemplate="%{text}<extra></extra>",
                line=dict(width=6),
                marker=dict(size=3)
            )
        )

    # Add start marker
    fig.add_trace(
        go.Scatter3d(
            x=[df[east_col].iloc[0]],
            y=[df[north_col].iloc[0]],
            z=[df["Z_plot"].iloc[0]],
            mode="markers+text",
            name="Surface",
            text=["Surface"],
            textposition="top center",
            marker=dict(size=6, symbol="circle")
        )
    )

    # Add toe marker
    fig.add_trace(
        go.Scatter3d(
            x=[df[east_col].iloc[-1]],
            y=[df[north_col].iloc[-1]],
            z=[df["Z_plot"].iloc[-1]],
            mode="markers+text",
            name="Toe",
            text=["Toe"],
            textposition="top center",
            marker=dict(size=6, symbol="diamond")
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"Easting Offset ({units})",
            yaxis_title=f"Northing Offset ({units})",
            zaxis_title=f"Depth ({units}, downward)",
            # aspectmode="data"
        ),
        legend_title="Trajectory",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

if __name__ == "__main__":

    import pandas as pd

    # data = pd.read_csv("horizontal_well_trajectory.csv")

    fig = plot_well_trajectory_3d(
        "c:/Users/lapto/Documents/GitHub/production-data-analysis/docs/transient_analysis/horizontal_well_trajectory.csv",
        units="ft",
        color_by_section=True,
        show_stations=True,
        title="Horizontal Well Trajectory"
    )
    fig = add_horizontal_reservoir_cuboid(
        fig,
        "c:/Users/lapto/Documents/GitHub/production-data-analysis/docs/transient_analysis/horizontal_well_trajectory.csv",
        units="ft",
        height = 500,
    )
    fig = add_fracture_cluster(
        fig,
        "c:/Users/lapto/Documents/GitHub/production-data-analysis/docs/transient_analysis/horizontal_well_trajectory.csv",
        units="ft",
        cluster_location="toe",
        fracture_count=4,
        offset_from_anchor=300,
        fracture_spacing=75,
        fracture_half_length=350,
        fracture_height=250,
        name="Toe fracture cluster"
    )
    # Add a second cluster near the heel by calling the same function again:
    fig = add_fracture_cluster(
        fig,
        "c:/Users/lapto/Documents/GitHub/production-data-analysis/docs/transient_analysis/horizontal_well_trajectory.csv",
        units="ft",
        cluster_location="heel",
        fracture_count=3,
        name="Heel fracture cluster"
    )

    fig.show()
