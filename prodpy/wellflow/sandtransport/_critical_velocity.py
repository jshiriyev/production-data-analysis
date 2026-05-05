import math

def critical_velocity(
    vs: float,
    theta: float = 0.0,
    alpha: float = 0.81,
    k: float = 1.0
) -> float:
    """
    Calculate critical transport velocity for sand in an annulus,
    including inclination effects.

    The model is based on a simple empirical relationship:

        u_crit,vertical = alpha * vs
        u_crit,inclined = f(theta) * u_crit,vertical
        f(theta) = 1 + k * sin(theta)

    Parameters
    ----------
    vs : float
        Particle terminal settling velocity [m/s].
    theta : float, optional
        Well inclination angle from vertical [degrees].
        - 0°   = vertical well
        - 90°  = horizontal well
        Default is 0°.
    alpha : float, optional
        Empirical factor relating settling velocity to critical velocity.
        Default is 0.81 (typical experimental value).
    k : float, optional
        Inclination sensitivity factor (empirical).
        Higher values increase the impact of deviation.
        Default is 1.0.

    Returns
    -------
    float
        Critical transport velocity [m/s].

    Raises
    ------
    ValueError
        If inputs are non-physical.

    Notes
    -----
    - This is a simplified engineering model suitable for early-stage screening.
    - Does NOT account for:
        * multiphase flow
        * bed formation
        * turbulent dispersion
        * particle interactions
    - k should ideally be calibrated using experimental or field data.

    Interpretation
    --------------
    If:
        u_a >= u_crit → particles are transportable
        u_a <  u_crit → settling risk exists

    Examples
    --------
    >>> critical_velocity(0.02)  # vertical well
    0.0162

    >>> critical_velocity(0.02, theta=60)
    0.0162 * (1 + sin(60°)) ≈ 0.0303
    """

    if vs < 0:
        raise ValueError("Settling velocity 'vs' must be non-negative.")
    if not (0 <= theta <= 90):
        raise ValueError("Theta must be between 0 and 90 degrees.")
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")
    if k < 0:
        raise ValueError("k must be non-negative.")

    # Vertical critical velocity
    v_c_vertical = alpha * vs

    # Inclination correction
    f_theta = 1.0 + k * math.sin(math.radians(theta))

    return f_theta * v_c_vertical