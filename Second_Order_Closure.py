import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time


# Same FokkerPlanck PDE Solver as Before
# Once you have the updated one, it should be pretty straightforward replacing it so that the code is functional 
def potential_V(x, y):
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    return (x**4 + y**4
            + x**3
            - 2.0*x*y**2
            + a*(x**2 + y**2)
            + theta1*x
            + theta2*y)

@jax.jit
def partial_derivs_of_V(x, y):
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    dVdx = 4.0*x**3 + 3.0*x**2 - 2.0*y**2 + 2.0*a*x + theta1
    dVdy = 4.0*y**3 - 4.0*x*y + 2.0*a*y + theta2
    return dVdx, dVdy

@jax.jit
def update_p_with_control(P, dx, dy, dt, D, x_grid, y_grid, t):
    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    dVdx, dVdy = partial_derivs_of_V(XX, YY)
    fx = -dVdx
    fy = -dVdy
    flux_x = fx * P
    flux_y = fy * P
    flux_x = flux_x.at[:, 0].set(0.0)
    flux_x = flux_x.at[:, -1].set(0.0)
    flux_y = flux_y.at[0, :].set(0.0)
    flux_y = flux_y.at[-1, :].set(0.0)
    flux_x_left = jnp.concatenate([flux_x[:, 0:1], flux_x[:, :-1]], axis=1)
    div_x = (flux_x - flux_x_left) / dx
    flux_y_up = jnp.concatenate([flux_y[0:1, :], flux_y[:-1, :]], axis=0)
    div_y = (flux_y - flux_y_up) / dy
    div_fP = div_x + div_y
    P_left  = jnp.concatenate([P[:, 0:1], P[:, :-1]], axis=1)
    P_right = jnp.concatenate([P[:, 1:], P[:, -1:]], axis=1)
    P_up    = jnp.concatenate([P[0:1, :], P[:-1, :]], axis=0)
    P_down  = jnp.concatenate([P[1:, :], P[-1:, :]], axis=0)
    lap_x = (P_left - 2.0*P + P_right) / (dx*dx)
    lap_y = (P_up   - 2.0*P + P_down ) / (dy*dy)
    lap_P = lap_x + lap_y
    dPdt = -div_fP + D*lap_P
    P_new = P + dt*dPdt
    P_new = jnp.clip(P_new, 0.0, None)
    s = jnp.sum(P_new)
    P_new = jnp.where(s > 0, P_new/s, P_new)
    return P_new

def compute_moments_up_to_order_n(P, x_grid, y_grid, max_order=10):
    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    powers_x = [XX**i for i in range(max_order+1)]
    powers_y = [YY**j for j in range(max_order+1)]
    moments_dict = {}
    for i in range(max_order+1):
        for j in range(max_order+1):
            if i == 0 and j == 0:
                continue
            M_ij = jnp.sum(powers_x[i] * powers_y[j] * P)
            moments_dict[(i, j)] = M_ij
    return moments_dict

def solve_fokker_planck_moments(D=0.2,
                                x_min=-2.0, x_max=2.0, Nx=50,
                                y_min=-2.0, y_max=2.0, Ny=50,
                                dt=1e-5,
                                T=0.005,
                                max_order=10):
    Nx, Ny = int(Nx), int(Ny)
    x_grid = jnp.linspace(x_min, x_max, Nx)
    y_grid = jnp.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    # Initial condition is Gaussian near (0,0)
    def init_cond(x, y):
        sigma0 = 0.2
        return jnp.exp(-(x**2 + y**2) / (2*sigma0**2))
    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    P0 = jax.vmap(jax.vmap(init_cond))(XX, YY)
    P0 = P0 / jnp.sum(P0)  
    n_steps = int(T / dt)
    P_current = P0
    moments_list = []
    diffs = []
    for step_idx in range(n_steps):
        moment_dict = compute_moments_up_to_order_n(P_current, x_grid, y_grid, max_order)
        moments_list.append(moment_dict)
        t_val = step_idx * dt
        P_next = update_p_with_control(P_current, dx, dy, dt, D, x_grid, y_grid, t_val)
        diff_val = jnp.sum(jnp.abs(P_next - P_current))
        diffs.append(diff_val)
        P_current = P_next
    diffs = np.array(diffs)
    return x_grid, y_grid, P_current, moments_list, diffs









# This is the part where we are analyzing the moments and comparing the computed moments to the derived ones 
@jax.jit
def approximate_moment_equations(M, a=2.0, theta1=5.0, theta2=-5.0, D=0.2):
   


    Ex, Ey, Ex2, Ey2, Exy = M
    Ex3   = Ex**3
    Ey3   = Ey**3
    Ex4   = Ex**4
    Ey4   = Ey**4
    Ex_y2 = Ex * Ey2

    # Derived Moment Equations 
    dEx_dt = -(4.0*Ex3 + 3.0*Ex**2 - 2.0*Ey**2 + 2.0*a*Ex + theta1)
    dEy_dt = -(4.0*Ey3 - 4.0*Ex*Ey + 2.0*a*Ey + theta2)
    closureX = (4.0*Ex4 + 3.0*Ex3 - 2.0*Ex_y2 + 2.0*a*Ex2 + theta1*Ex)
    dEx2_dt = -2.0 * closureX + Ex + 2.0*D
    closureY = (4.0*Ey4 - 4.0*Ex_y2 + 2.0*a*Ey2 + theta2*Ey)
    dEy2_dt = -2.0 * closureY + 2.0*D
    closureXY_1 = (4.0*Ex*Ey3 - 4.0*(Ex2*Ey) + 2.0*a*(Ex*Ey) + theta2*Exy)
    closureXY_2 = (4.0*Ex3*Ey + 3.0*(Ex**2)*Exy - 2.0*Ey2*Ey + 2.0*a*(Ex*Exy))
    closureXY   = closureXY_1 + closureXY_2
    dExy_dt   = -closureXY + Ex
    return jnp.array([dEx_dt, dEy_dt, dEx2_dt, dEy2_dt, dExy_dt])

def solve_approximate_moment_equations(D=0.2, dt=1e-5, T=0.005,
                                       Ex0=0.0, Ey0=0.0,
                                       Ex2_0=0.04, Ey2_0=0.04, Exy0=0.0):
    n_steps = int(T / dt)
    M = jnp.array([Ex0, Ey0, Ex2_0, Ey2_0, Exy0], dtype=jnp.float32)
    arr_t = np.zeros(n_steps+1)
    arr_M = np.zeros((n_steps+1, 5))
    arr_M[0] = np.array(M)
    for i in range(n_steps):
        dM = approximate_moment_equations(M, D=D)
        M = M + dt*dM
        arr_M[i+1] = np.array(M)
        arr_t[i+1] = (i+1)*dt
    return arr_t, arr_M





def compare_moment_methods():
    print("=== Comparing PDE-based Moments vs. 2nd-Order Closure ===")
    D = 0.2
    dt_val = 1e-5
    T_val  = 0.005
    Nx, Ny = 50, 50

    # Solve PDE and store final distribution
    x_grid, y_grid, P_final, moments_list, diffs = solve_fokker_planck_moments(
        D=D, x_min=-2, x_max=2, Nx=Nx,
        y_min=-2, y_max=2, Ny=Ny,
        dt=dt_val, T=T_val, max_order=3
    )

    # Final distribution from PDE
    plt.figure()
    plt.imshow(jnp.log(P_final + 1e-12),
               origin="lower", extent=[-2,2,-2,2],
               cmap="viridis", aspect="auto")
    plt.colorbar(label="log P(x,y)")
    plt.title("Final PDE Distribution (log-scale)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("final_PDE_distribution.png", dpi=300)
    plt.close()
    print("Saved final_PDE_distribution.png")

    t_array_pde = np.linspace(0, T_val, len(moments_list))
    pde_Ex, pde_Ey = [], []
    pde_Ex2, pde_Ey2, pde_Exy = [], [], []

    for mdict in moments_list:
        pde_Ex.append(mdict.get((1,0), 0.0))
        pde_Ey.append(mdict.get((0,1), 0.0))
        pde_Ex2.append(mdict.get((2,0), 0.0))
        pde_Ey2.append(mdict.get((0,2), 0.0))
        pde_Exy.append(mdict.get((1,1), 0.0))

    pde_Ex  = np.array(pde_Ex)
    pde_Ey  = np.array(pde_Ey)
    pde_Ex2 = np.array(pde_Ex2)
    pde_Ey2 = np.array(pde_Ey2)
    pde_Exy = np.array(pde_Exy)

    # Solve moment-closure ODE
    t_ode, M_ode = solve_approximate_moment_equations(
        D=D, dt=dt_val, T=T_val,
        Ex0=0.0, Ey0=0.0,
        Ex2_0=0.04, Ey2_0=0.04,
        Exy0=0.0
    )
    if len(M_ode) > len(t_array_pde):
        M_ode = M_ode[:len(t_array_pde)]
        t_ode = t_ode[:len(t_array_pde)]

    ode_Ex  = M_ode[:, 0]
    ode_Ey  = M_ode[:, 1]
    ode_Ex2 = M_ode[:, 2]
    ode_Ey2 = M_ode[:, 3]
    ode_Exy = M_ode[:, 4]

    # Raw moments PDE vs. Closure
    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.plot(t_array_pde, pde_Ex, 'b-', label='PDE E[x]')
    plt.plot(t_ode, ode_Ex, 'r--', label='Closure')
    plt.title("E[x]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(t_array_pde, pde_Ey, 'b-')
    plt.plot(t_ode, ode_Ey, 'r--')
    plt.title("E[y]")
    plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(t_array_pde, pde_Ex2, 'b-')
    plt.plot(t_ode, ode_Ex2, 'r--')
    plt.title("E[x^2]")
    plt.grid(True)

    plt.subplot(2,3,4)
    plt.plot(t_array_pde, pde_Ey2, 'b-')
    plt.plot(t_ode, ode_Ey2, 'r--')
    plt.title("E[y^2]")
    plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(t_array_pde, pde_Exy, 'b-')
    plt.plot(t_ode, ode_Exy, 'r--')
    plt.title("E[x y]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("compare_pde_vs_closure.png", dpi=300)
    plt.close()
    print("Saved compare_pde_vs_closure.png")

    # Relative errors of each moment
    eps = 1e-12
    rel_err_Ex  = np.abs(pde_Ex  - ode_Ex ) /(np.abs(pde_Ex ) + eps)
    rel_err_Ey  = np.abs(pde_Ey  - ode_Ey ) /(np.abs(pde_Ey ) + eps)
    rel_err_Ex2 = np.abs(pde_Ex2 - ode_Ex2)/(np.abs(pde_Ex2) + eps)
    rel_err_Ey2 = np.abs(pde_Ey2 - ode_Ey2)/(np.abs(pde_Ey2) + eps)
    rel_err_Exy = np.abs(pde_Exy - ode_Exy)/(np.abs(pde_Exy) + eps)

    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.plot(t_array_pde, rel_err_Ex, 'k-')
    plt.yscale('log')
    plt.title("Rel. Err in E[x]")
    plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(t_array_pde, rel_err_Ey, 'k-')
    plt.yscale('log')
    plt.title("Rel. Err in E[y]")
    plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(t_array_pde, rel_err_Ex2, 'k-')
    plt.yscale('log')
    plt.title("Rel. Err in E[x^2]")
    plt.grid(True)

    plt.subplot(2,3,4)
    plt.plot(t_array_pde, rel_err_Ey2, 'k-')
    plt.yscale('log')
    plt.title("Rel. Err in E[y^2]")
    plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(t_array_pde, rel_err_Exy, 'k-')
    plt.yscale('log')
    plt.title("Rel. Err in E[x y]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("closure_relative_errors.png", dpi=300)
    plt.close()
    print("Saved closure_relative_errors.png")

    # Plot PDE L1 diffs over time (convergence measure)
    steps = np.arange(len(diffs))
    plt.figure()
    plt.plot(steps, diffs, 'b-')
    plt.yscale('log')
    plt.xlabel("Time Step")
    plt.ylabel("L1 Diff (P_new - P_old)")
    plt.title("PDE Convergence")
    plt.grid(True)
    plt.savefig("pde_convergence.png", dpi=300)
    plt.close()
    print("Saved pde_convergence.png")

def main():
    start_time = time.time()
    print("Fokker–Planck with n-th Moments")
    D = 0.2
    dt = 1e-5
    T = 0.005
    Nx, Ny = 50, 50
    max_order = 10

    x_grid, y_grid, P_final, moments_list, diffs = solve_fokker_planck_moments(
        D=D,
        x_min=-2.0, x_max=2.0, Nx=Nx,
        y_min=-2.0, y_max=2.0, Ny=Ny,
        dt=dt, T=T,
        max_order=max_order
    )

    end_time = time.time()
    print(f"Completed PDE in ~{end_time - start_time:.2f}s.")
    print(f"Final L1 difference: {diffs[-1]:.4e}")

    final_moments = moments_list[-1]
    print("Some final moments:")
    print(f" E[x]    = {final_moments.get((1,0),0.):.4f}")
    print(f" E[y]    = {final_moments.get((0,1),0.):.4f}")
    print(f" E[x^2]  = {final_moments.get((2,0),0.):.4f}")
    print(f" E[y^2]  = {final_moments.get((0,2),0.):.4f}")
    print(f" E[x^3]  = {final_moments.get((3,0),0.):.4f}")
    print(f" E[xy]   = {final_moments.get((1,1),0.):.4f}")

    # Plotting final distribution in log-scale
    plt.figure(figsize=(7,6))
    plt.imshow(jnp.log(P_final + 1e-12),
               origin="lower", extent=[-2,2,-2,2],
               cmap="viridis", aspect="auto")
    plt.colorbar(label="log P(x,y)")
    plt.title("Final Fokker–Planck Dist (log-scale)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("final_distribution_with_control_nth_moments.png", dpi=300)
    plt.close()
    print("Saved final_distribution_with_nth_moments.png")

    # Plotting the PDE convergence
    steps = np.arange(len(diffs))
    plt.figure(figsize=(7,5))
    plt.plot(steps, diffs, '-')
    plt.yscale('log')
    plt.xlabel("Time Step")
    plt.ylabel("L1 Diff (P_new - P_old)")
    plt.title("Convergence (Fokker–Planck PDE w/ control)")
    plt.grid(True)
    plt.savefig("pde_convergence_nth_moments.png", dpi=300)
    plt.close()
    print("Saved pde_convergence_nth_moments.png")

    # Extracting the moments so we can plot special moments over time
    arr_Ex = []
    arr_Ey = []
    arr_Ex2 = []
    arr_Ey2 = []
    arr_Ex3 = []
    arr_Exy = []

    for mdict in moments_list:
        arr_Ex.append(mdict.get((1,0), 0.0))
        arr_Ey.append(mdict.get((0,1), 0.0))
        arr_Ex2.append(mdict.get((2,0), 0.0))
        arr_Ey2.append(mdict.get((0,2), 0.0))
        arr_Ex3.append(mdict.get((3,0), 0.0))
        arr_Exy.append(mdict.get((1,1), 0.0))

    arr_Ex  = np.array(arr_Ex)
    arr_Ey  = np.array(arr_Ey)
    arr_Ex2 = np.array(arr_Ex2)
    arr_Ey2 = np.array(arr_Ey2)
    arr_Ex3 = np.array(arr_Ex3)
    arr_Exy = np.array(arr_Exy)

    # Plotting the moments
    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.plot(steps, arr_Ex, '-')
    plt.title("E[x]")
    plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(steps, arr_Ey, '-')
    plt.title("E[y]")
    plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(steps, arr_Ex2, '-')
    plt.title("E[x^2]")
    plt.grid(True)

    plt.subplot(2,3,4)
    plt.plot(steps, arr_Ey2, '-')
    plt.title("E[y^2]")
    plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(steps, arr_Ex3, '-')
    plt.title("E[x^3]")
    plt.grid(True)

    plt.subplot(2,3,6)
    plt.plot(steps, arr_Exy, '-')
    plt.title("E[x y]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("moments_up_to_n.png", dpi=300)
    plt.close()
    print("Saved moments_up_to_n.png")
    print("Done. Computed all moments up to order n = %d." % max_order)

if __name__ == "__main__":
    main()
    compare_moment_methods()
