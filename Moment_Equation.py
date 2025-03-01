import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

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
    dVdy = 4.0*y**3 - 4.0*x*y   + 2.0*a*y + theta2
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

def main():
    start_time = time.time()
    print("=== Fokker–Planck with n-th Moments ===")
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

    plt.figure(figsize=(7,6))
    plt.imshow(jnp.log(P_final + 1e-12),
               origin="lower", extent=[-2,2,-2,2],
               cmap="viridis", aspect="auto")
    plt.colorbar(label="log P(x,y)")
    plt.title("Final Fokker–Planck Dist(log-scale)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("final_distribution_with_control_nth_moments.png", dpi=300)
    plt.close()


    print("Saved final_distribution_with_nth_moments.png")
    steps = np.arange(len(diffs))
    plt.figure(figsize=(7,5))
    plt.plot(steps, diffs, 'b-')
    plt.yscale('log')
    plt.xlabel("Time Step")
    plt.ylabel("L1 Diff (P_new - P_old)")
    plt.title("Convergence (Fokker–Planck PDE w/ control)")
    plt.grid(True)
    plt.savefig("pde_convergence_nth_moments.png", dpi=300)
    plt.close()
    print("Saved pde_convergence_nth_moments.png")


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

    arr_Ex = np.array(arr_Ex)
    arr_Ey = np.array(arr_Ey)
    arr_Ex2 = np.array(arr_Ex2)
    arr_Ey2 = np.array(arr_Ey2)
    arr_Ex3 = np.array(arr_Ex3)
    arr_Exy = np.array(arr_Exy)


    plt.figure(figsize=(10,6))

    plt.subplot(2,3,1)
    plt.plot(steps, arr_Ex, 'b-')
    plt.title("E[x]")
    plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(steps, arr_Ey, 'r-')
    plt.title("E[y]")
    plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(steps, arr_Ex2, 'g-')
    plt.title("E[x^2]")
    plt.grid(True)

    plt.subplot(2,3,4)
    plt.plot(steps, arr_Ey2, 'c-')
    plt.title("E[y^2]")
    plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(steps, arr_Ex3, 'm-')
    plt.title("E[x^3]")
    plt.grid(True)

    plt.subplot(2,3,6)
    plt.plot(steps, arr_Exy, 'k-')
    plt.title("E[x y]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("moments_up_to_n.png", dpi=300)
    plt.close()

    
    print("Saved moments_up_to_n.png")
    print("Done. Computed all moments up to order n = %d." % max_order)

if __name__ == "__main__":
    main()
