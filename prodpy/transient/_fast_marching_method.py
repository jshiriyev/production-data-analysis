"""
Fast Marching Method / DTOF demonstration
1D homogeneous linear reservoir

Finite-volume Eikonal equation:

    sum_m T_nm * (s_n - s_m)^2 = Vp_n * mu * ct

For a 1D homogeneous grid with one accepted upstream neighbor:

    T * (s_n - s_m)^2 = Vp * mu * ct

so:

    s_n = s_m + sqrt(Vp * mu * ct / T)

Analytical DTOF for homogeneous linear flow:

    s(x) = x / sqrt(alpha)

where:

    alpha = k / (phi * mu * ct)
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt


def build_1d_grid(
    n_cells: int = 31,
    dx: float = 1.0,
    area: float = 1.0,
    permeability: float = 1.0,
    porosity: float = 1.0,
):
    """
    Build a simple 1D homogeneous finite-volume grid.

    Parameters
    ----------
    n_cells : int
        Number of grid cells.
    dx : float
        Cell length.
    area : float
        Cross-sectional area.
    permeability : float
        Permeability in consistent units.
    porosity : float
        Porosity.

    Returns
    -------
    x : np.ndarray
        Cell-center locations. Here cell 0 is treated as x = 0.
    pore_volume : np.ndarray
        Pore volume of each cell.
    transmissibility : np.ndarray
        Transmissibility between neighboring cells.
        transmissibility[i] connects cell i and i+1.
    """
    x = np.arange(n_cells, dtype=float) * dx

    bulk_volume = area * dx
    pore_volume = np.full(n_cells, porosity * bulk_volume)

    # Transmissibility between neighboring cell centers:
    # T = k A / dx
    transmissibility = np.full(n_cells - 1, permeability * area / dx)

    return x, pore_volume, transmissibility

def neighbors_1d(cell: int, n_cells: int):
    """Return valid 1D neighboring cell indices."""
    if cell - 1 >= 0:
        yield cell - 1
    if cell + 1 < n_cells:
        yield cell + 1

def get_transmissibility_between(i: int, j: int, transmissibility: np.ndarray) -> float:
    """
    Return transmissibility between two neighboring cells i and j.

    transmissibility[m] connects m and m+1.
    """
    if abs(i - j) != 1:
        raise ValueError("Cells are not direct 1D neighbors.")

    return transmissibility[min(i, j)]

def solve_local_eikonal_update(
    cell: int,
    tau: np.ndarray,
    accepted: np.ndarray,
    pore_volume: np.ndarray,
    transmissibility: np.ndarray,
    mu: float,
    ct: float,
) -> float:
    """
    Compute tentative DTOF value for one cell using accepted neighbors.

    General finite-volume Eikonal update:

        sum_m T_nm * (s_n - s_m)^2 = Vp_n * mu * ct

    This becomes a quadratic equation in s_n.
    """
    accepted_neighbors = []
    for nb in neighbors_1d(cell, len(tau)):
        if accepted[nb]:
            T = get_transmissibility_between(cell, nb, transmissibility)
            accepted_neighbors.append((nb, T))

    if not accepted_neighbors:
        return np.inf

    rhs = pore_volume[cell] * mu * ct

    # Build quadratic:
    # A*s^2 + B*s + C = 0
    # sum T_j * (s - s_j)^2 = rhs
    A = sum(T for _, T in accepted_neighbors)
    B = -2.0 * sum(T * tau[nb] for nb, T in accepted_neighbors)
    C = sum(T * tau[nb] ** 2 for nb, T in accepted_neighbors) - rhs

    discriminant = B**2 - 4.0 * A * C
    if discriminant < 0:
        raise RuntimeError("Negative discriminant in local Eikonal update.")

    roots = [
        (-B + np.sqrt(discriminant)) / (2.0 * A),
        (-B - np.sqrt(discriminant)) / (2.0 * A),
    ]

    # Upwind/causal root must be larger than accepted neighbor DTOFs.
    min_valid = max(tau[nb] for nb, _ in accepted_neighbors)

    valid_roots = [r for r in roots if r >= min_valid]

    if not valid_roots:
        # Fallback for robustness: use one-neighbor updates and take minimum.
        candidates = []
        for nb, T in accepted_neighbors:
            candidates.append(tau[nb] + np.sqrt(rhs / T))
        return min(candidates)

    return min(valid_roots)

def fast_marching_1d(
    pore_volume: np.ndarray,
    transmissibility: np.ndarray,
    mu: float = 1.0,
    ct: float = 1.0,
    source_cell: int = 0,
):
    """
    Compute DTOF using a simple 1D Fast Marching Method.

    Parameters
    ----------
    pore_volume : np.ndarray
        Pore volume of each cell.
    transmissibility : np.ndarray
        Transmissibility between neighboring cells.
    mu : float
        Fluid viscosity.
    ct : float
        Total compressibility.
    source_cell : int
        Cell where pressure disturbance starts. DTOF = 0 here.

    Returns
    -------
    tau : np.ndarray
        DTOF value for each cell.
    accepted_order : list[int]
        Order in which cells were accepted by FMM.
    """
    n_cells = len(pore_volume)

    tau = np.full(n_cells, np.inf)
    accepted = np.zeros(n_cells, dtype=bool)

    tau[source_cell] = 0.0

    # Priority queue stores (tentative_tau, cell_index)
    heap = []

    # Accept source cell first
    accepted[source_cell] = True
    accepted_order = [source_cell]

    # Initialize neighbors of source
    for nb in neighbors_1d(source_cell, n_cells):
        tau_nb = solve_local_eikonal_update(
            nb, tau, accepted, pore_volume, transmissibility, mu, ct
        )
        tau[nb] = tau_nb
        heapq.heappush(heap, (tau_nb, nb))

    # March outward
    while heap:
        tau_current, cell = heapq.heappop(heap)

        if accepted[cell]:
            continue

        # Skip old heap entries
        if tau_current > tau[cell]:
            continue

        accepted[cell] = True
        accepted_order.append(cell)

        # Update neighbors
        for nb in neighbors_1d(cell, n_cells):
            if accepted[nb]:
                continue

            tau_candidate = solve_local_eikonal_update(
                nb, tau, accepted, pore_volume, transmissibility, mu, ct
            )

            if tau_candidate < tau[nb]:
                tau[nb] = tau_candidate
                heapq.heappush(heap, (tau_candidate, nb))

    return tau, accepted_order

def main():
    # -----------------------------
    # 1. Define simple homogeneous case
    # -----------------------------
    n_cells = 31
    dx = 1.0
    area = 1.0
    k = 1.0
    phi = 1.0
    mu = 1.0
    ct = 1.0

    x, pore_volume, transmissibility = build_1d_grid(
        n_cells=n_cells,
        dx=dx,
        area=area,
        permeability=k,
        porosity=phi,
    )

    # -----------------------------
    # 2. Run FMM
    # -----------------------------
    tau_fmm, accepted_order = fast_marching_1d(
        pore_volume=pore_volume,
        transmissibility=transmissibility,
        mu=mu,
        ct=ct,
        source_cell=0,
    )

    # -----------------------------
    # 3. Analytical DTOF
    # -----------------------------
    alpha = k / (phi * mu * ct)
    tau_analytical = x / np.sqrt(alpha)

    error = tau_fmm - tau_analytical

    print("Accepted order:")
    print(accepted_order)

    print("\nFirst 10 DTOF values:")
    for i in range(10):
        print(
            f"cell={i:2d}, x={x[i]:6.2f}, "
            f"tau_fmm={tau_fmm[i]:8.4f}, "
            f"tau_analytical={tau_analytical[i]:8.4f}, "
            f"error={error[i]: .2e}"
        )

    print(f"\nMaximum absolute error = {np.max(np.abs(error)):.3e}")

    # -----------------------------
    # 4. Compute Vp(s) and w(s)
    # -----------------------------
    # In this simple 1D case, Vp(s) should be linear in s.
    sort_idx = np.argsort(tau_fmm)
    tau_sorted = tau_fmm[sort_idx]
    vp_sorted = pore_volume[sort_idx]

    cumulative_vp = np.cumsum(vp_sorted)

    # Numerical derivative w(s) = dVp/ds
    # For this simple case, w(s) should be approximately constant.
    w_numeric = np.gradient(cumulative_vp, tau_sorted)

    # -----------------------------
    # 5. Plot results
    # -----------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x, tau_fmm, "o", label="FMM DTOF")
    plt.plot(x, tau_analytical, "-", label="Analytical DTOF")
    plt.xlabel("Distance x")
    plt.ylabel("DTOF, s")
    plt.title("1D Homogeneous Linear DTOF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tau_sorted, cumulative_vp, "o-")
    plt.xlabel("DTOF, s")
    plt.ylabel("Cumulative pore volume, Vp(s)")
    plt.title("Drainage Pore Volume vs DTOF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(tau_sorted, w_numeric, "o-")
    plt.xlabel("DTOF, s")
    plt.ylabel("w(s) = dVp/ds")
    plt.title("w(s) Function")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()