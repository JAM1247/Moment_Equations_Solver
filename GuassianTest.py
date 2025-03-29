import jax
import jax.numpy as jnp 
from jax import lax  # For parallelization
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

# Potential Function
def potential_V(x, y):
    a = 2.0 
    theta1 = 5.0
    theta2 = -5.0
    return (x**4 + y**4 +
            x**3 -
            2.0 * x * y**2 +
            a * (x**2 + y**2) +
            theta1 * x +
            theta2 * y)

def partial_deriv_of_V(x, y):
    a = 2.0 
    theta1 = 5.0
    theta2 = -5.0
    dVdx = 4 * x**3 + 3.0 * x**2 - 2.0 * y**2 + 2.0 * a * x + theta1
    dVdy = 4 * y**3 + 3.0 * y**2 - 2.0 * x**2 + 2.0 * a * y + theta2
    return dVdx, dVdy

def control(t, params=None):
    # Control is kept zero for now.
    return 0.0, 0.0

# Fokker-Planck
@partial(jax.jit, static_argnames=('control_fn', 'control_params'))
def fp_pde_update(P, dx, dy, dt, D, x_grid, y_grid, t, control_fn, control_params):
    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    dVdx, dVdy = partial_deriv_of_V(XX, YY)

    U1, U2 = control_fn(t, control_params)
    fx = -dVdx + U1
    fy = -dVdy + U2

    flux_x = fx * P
    flux_y = fy * P

    # Enforce boundary conditions
    flux_x = flux_x.at[:, 0].set(0.0)
    flux_x = flux_x.at[:, -1].set(0.0)
    flux_y = flux_y.at[0, :].set(0.0)
    flux_y = flux_y.at[-1, :].set(0.0)

    flux_x_left = jnp.concatenate([flux_x[:, :1], flux_x[:, :-1]], axis=1)
    div_x = (flux_x - flux_x_left) / dx

    flux_y_up = jnp.concatenate([flux_y[:1, :], flux_y[:-1, :]], axis=0)
    div_y = (flux_y - flux_y_up) / dy

    div_fP = div_x + div_y

    # Laplacian for diffusion
    P_left  = jnp.concatenate([P[:, :1], P[:, :-1]], axis=1)
    P_right = jnp.concatenate([P[:, 1:], P[:, -1:]], axis=1)
    P_up    = jnp.concatenate([P[:1, :], P[:-1, :]], axis=0)
    P_down  = jnp.concatenate([P[1:, :], P[-1:, :]], axis=0)
    lap_x = (P_left - 2.0 * P + P_right) / (dx**2)
    lap_y = (P_up   - 2.0 * P + P_down) / (dy**2)
    lap_P = lap_x + lap_y

    dPdt = -div_fP + D * lap_P
    P_new = P + dt * dPdt
    P_new = jnp.clip(P_new, 0.0, None)
    norm = jnp.sum(P_new)
    P_new = jnp.where(norm > 0, P_new / norm, P_new)
    return P_new

def solve_fokker_planck_2d(D=0.2, x_min=-2.0, x_max=2.0, Nx=50,
                           y_min=-2.0, y_max=2.0, Ny=50,
                           dt=1e-5, T=0.005,
                           control_fn=control, control_params=None,
                           store_every=100, max_order=2):
    Nx, Ny = int(Nx), int(Ny)
    x_grid = jnp.linspace(x_min, x_max, Nx)
    y_grid = jnp.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    n_steps = int(T / dt)

    def init_cond(x, y):
        sigma0 = 0.2
        return jnp.exp(-(x**2 + y**2) / (2 * sigma0**2))
    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    P0 = jax.vmap(jax.vmap(init_cond))(XX, YY)
    P0 = P0 / jnp.sum(P0)

    init_carry = (P0, 0.0)
    def body_fn(carry, i):
        P_current, t_current = carry
        P_next = fp_pde_update(P_current, dx, dy, dt, D, x_grid, y_grid,
                               t_current, control_fn, control_params)
        moment = jnp.array([
            jnp.sum(XX * P_next),
            jnp.sum(YY * P_next),
            jnp.sum((XX**2) * P_next),
            jnp.sum((YY**2) * P_next),
            jnp.sum((XX * YY) * P_next)
        ])
        diff = jnp.sum(jnp.abs(P_next - P_current))
        norm_error = jnp.abs(jnp.sum(P_next) - 1.0)
        min_val = jnp.min(P_next)
        diagnostics = jnp.concatenate([moment, jnp.array([diff, norm_error, min_val])])
        new_carry = (P_next, t_current + dt)
        return new_carry, diagnostics

    final_carry, diagnostics_history = lax.scan(body_fn, init_carry, jnp.arange(n_steps))
    P_final, t_final = final_carry
    return x_grid, y_grid, P_final, diagnostics_history, {}

# Naive Closure ODE
def naive_controlled_rhs(M, t, D, control_fn, control_params):
    Ex, Ey, _, _, _ = M
    U1, U2 = control_fn(t, control_params)
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    dEx_dt = - (4.0 * Ey**3 + 3.0 * Ex**2 - 2.0 * Ey**2 + 2.0 * a * Ex + theta1) + U1
    dEy_dt = - (4.0 * Ey**3 - 4.0 * Ex + Ey + 2.0 * a * Ey + theta2) + U2
    return jnp.array([dEx_dt, dEy_dt, 0.0, 0.0, 0.0], dtype=jnp.float32)

def solve_naive_closure(D=0.2, dt=1e-5, T=0.005,
                        control_fn=control, control_params=None,
                        Ex0=0.0, Ey0=0.0):
    n_steps = int(T / dt)
    t_array = jnp.linspace(0, T, n_steps + 1)
    M0 = jnp.array([Ex0, Ey0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    def body_fn(M, t):
        rhs = naive_controlled_rhs(M, t, D, control_fn, control_params)
        M_next = M + dt * rhs
        return M_next, M_next
    M_final, M_scan = lax.scan(body_fn, M0, t_array[:-1])
    M_scan = jnp.concatenate([M_scan, M_final[jnp.newaxis, :]], axis=0)
    return t_array, M_scan

# Gaussian Closure ODE
def gaussian_controlled_rhs_fixed(M, t, D, control_fn, control_params):
    # Moments
    Ex, Ey, Vx, Vy, Cxy = M

    # Paramaters
    U1, U2 = control_fn(t, control_params)

    a      = 2.0
    theta1 = 5.0
    theta2 = -5.0

    # Lower order moments
    Ex2 = Ex**2 + Vx       # E[x^2]
    Ey2 = Ey**2 + Vy       # E[y^2]
    Exy = Ex * Ey + Cxy    # E[x y]

    # Cubic and Quadrtic Moments 
    Ex3 = Ex**3 + 3.0 * Ex * Vx                   # E[x^3]
    Ey3 = Ey**3 + 3.0 * Ey * Vy                   # E[y^3]
    Ex4 = Ex**4 + 6.0 * Ex**2 * Vx + 3.0 * Vx**2  # E[x^4]
    Ey4 = Ey**4 + 6.0 * Ey**2 * Vy + 3.0 * Vy**2  # E[y^4]

    # Mixed Moments
    xy2 = Ex * Ey2 + 2.0 * Ey * Cxy
    x2y = Ey * Ex2 + 2.0 * Ex * Cxy

    # FirstM oment Derivatives
    dEx_dt = - (4.0 * Ex3 + 3.0 * (Ex**2) - 2.0 * Ey2 + 2.0 * a * Ex + theta1) + U1
    dEy_dt = - (4.0 * Ey3 + 3.0 * (Ey**2) - 2.0 * Ex2 + 2.0 * a * Ey + theta2) + U2

    #Second Moment E[x^2]
    term_x_dVdx = (4.0 * Ex3 +
                   3.0 * (Ex**2) -
                   2.0 * xy2 +       
                   2.0 * a * Ex2 +
                   theta1 * Ex)
    dEx2_dt = -2.0 * term_x_dVdx + 2.0 * U1 * Ex + 2.0 * D
    dVx_dt = dEx2_dt - 2.0 * Ex * dEx_dt 

    # Second Moment E[y^2]
    term_y_dVdy = (4.0 * Ey4 -
                   4.0 * x2y +      
                   2.0 * a * Ey2 +
                   theta2 * Ey)
    dEy2_dt = 2.0 * term_y_dVdy + 2.0 * U2 * Ey + 2.0 * D
    dVy_dt = dEy2_dt - 2.0 * Ey * dEy_dt  

    # Mixed Moment E[x y]
    x_dVdy = 4.0 * (Ex * Ey3) + 3.0 * xy2 - 2.0 * Ex3 + 2.0 * a * Exy + theta2 * Ex
    y_dVdx = 4.0 * (Ey * Ex3) + 3.0 * x2y - 2.0 * Ey3 + 2.0 * a * Exy + theta1 * Ey

    E_xy_dt = - (x_dVdy + y_dVdx) + U2 * Ex + U1 * Ey
    dCxy_dt = E_xy_dt - (Ex * dEy_dt + Ey * dEx_dt)

    return jnp.array([dEx_dt, dEy_dt, dVx_dt, dVy_dt, dCxy_dt], dtype=jnp.float32)

def solve_gaussian_closure_fixed(D=0.2, dt=1e-5, T=0.005,
                                 control_fn=control, control_params=None,
                                 Ex0=0.0, Ey0=0.0,
                                 Vx0=0.04, Vy0=0.04, Cxy0=0.0):
    n_steps = int(T / dt)
    t_array = jnp.linspace(0, T, n_steps + 1)
    M0 = jnp.array([Ex0, Ey0, Vx0, Vy0, Cxy0], dtype=jnp.float32)
    def body_fn(M, t):
        rhs = gaussian_controlled_rhs_fixed(M, t, D, control_fn, control_params)
        M_next = M + dt * rhs
        return M_next, M_next
    M_final, M_scan = lax.scan(body_fn, M0, t_array[:-1])
    M_scan = jnp.concatenate([M_scan, M_final[jnp.newaxis, :]], axis=0)
    return t_array, M_scan

def solve_all_closure_and_compare():
    D = 0.2
    dt_val = 1e-5
    Tfinal = 0.01 
    Nx, Ny = 50, 50

    print("SOLVING PDE ...")
    xg, yg, P_final, diagnostic_history, _ = solve_fokker_planck_2d(
        D=D, x_min=-2, x_max=2, Nx=Nx,
        y_min=-2, y_max=2, Ny=Ny,
        dt=dt_val, T=Tfinal,
        control_fn=control, control_params=None,
        store_every=200
    )
    diag_hist = np.array(diagnostic_history)
    pde_moments = diag_hist[:, :5]
    n_pde_steps = pde_moments.shape[0]
    t_pde = np.linspace(0, Tfinal, n_pde_steps)

    print("Solving Naive Closure ODEs... ")
    t_naive_jax, M_naive_jax = solve_naive_closure(
        D=D, dt=dt_val, T=Tfinal,
        control_fn=control, control_params=None,
        Ex0=0.0, Ey0=0.0
    )
    t_naive = np.array(t_naive_jax)
    M_naive = np.array(M_naive_jax)
    naive_moments = np.column_stack([
        M_naive[:, 0],
        M_naive[:, 1],
        M_naive[:, 0]**2,
        M_naive[:, 1]**2,
        M_naive[:, 0]*M_naive[:, 1]
    ])

    print("Solving Gaussian Closure ODE ...")
    t_gauss_jax, M_gauss_jax = solve_gaussian_closure_fixed(
        D=D, dt=dt_val, T=Tfinal,
        control_fn=control, control_params=None,
        Ex0=0.0, Ey0=0.0,
        Vx0=0.04, Vy0=0.04, Cxy0=0.0
    )
    t_gauss = np.array(t_gauss_jax)
    M_gauss = np.array(M_gauss_jax)
    gauss_moments = np.column_stack([
        M_gauss[:, 0],
        M_gauss[:, 1],
        M_gauss[:, 0]**2 + M_gauss[:, 2],
        M_gauss[:, 1]**2 + M_gauss[:, 3],
        M_gauss[:, 0]*M_gauss[:, 1] + M_gauss[:, 4]
    ])

    common_length = min(pde_moments.shape[0], naive_moments.shape[0], gauss_moments.shape[0])
    pde_moments = pde_moments[:common_length, :]
    naive_moments = naive_moments[:common_length, :]
    gauss_moments = gauss_moments[:common_length, :]
    t_common = np.linspace(0, Tfinal, common_length)

    plt.figure(figsize=(15, 12))
    moment_titles = ["E[x](t)", "E[y](t)", "E[x^2](t)", "E[y^2](t)", "E[xy](t)"]
    for i in range(5):
        plt.subplot(3, 2, i+1)
        plt.plot(t_common, pde_moments[:, i], 'b-', label="PDE")
        plt.plot(t_common, naive_moments[:, i], 'r--', label="Naive Closure")
        plt.plot(t_common, gauss_moments[:, i], 'g-.', label="Gaussian Closure")
        plt.title(moment_titles[i])
        plt.xlabel("Time")
        plt.ylabel("Moment")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("closure_comparison.png", dpi=300)
    plt.close()
    print("Saved figure: closure_comparison.png")

    eps = 1e-6
    rel_err_naive = np.abs(pde_moments - naive_moments) / (np.abs(pde_moments) + eps)
    rel_err_gauss = np.abs(pde_moments - gauss_moments) / (np.abs(pde_moments) + eps)

    plt.figure(figsize=(15, 12))
    for i in range(5):
        plt.subplot(3, 2, i+1)
        plt.plot(t_common, rel_err_naive[:, i], 'r-', label="Rel. Err. Naive")
        plt.plot(t_common, rel_err_gauss[:, i], 'g-', label="Rel. Err. Gaussian")
        plt.title("Relative Error for " + moment_titles[i])
        plt.xlabel("Time")
        plt.ylabel("Relative Error")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("relative_errors_comparison.png", dpi=300)
    plt.close()
    print("Saved figure: relative_errors_comparison.png")

    print("\n=== Data for Analysis ===")
    print("\nTime Vector (t_common):")
    print(t_common)
    print("\nPDE Moments (columns: E[x], E[y], E[x^2], E[y^2], E[xy]):")
    print(pde_moments)
    print("\nNaive Closure Moments (columns: E[x], E[y], E[x^2], E[y^2], E[xy]):")
    print(naive_moments)
    print("\nGaussian Closure Moments (columns: E[x], E[y], E[x^2], E[y^2], E[xy]):")
    print(gauss_moments)
    print("\nRelative Errors - Naive Closure:")
    print(rel_err_naive)
    print("\nRelative Errors - Gaussian Closure:")
    print(rel_err_gauss)

def compare_all_closures_vs_pde():
    solve_all_closure_and_compare()

def main():
    compare_all_closures_vs_pde()

if __name__ == "__main__":
    main()
