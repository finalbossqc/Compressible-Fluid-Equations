import numpy as np
import matplotlib.pyplot as plt
from ns_solver import ns_solver
from initialize_a import initialize_a
from fluids import FLUIDS

def calculate_fluid(fluid_tuple, dt, t_fin):
    rho = fluid_tuple[0]
    mu = fluid_tuple[1]
    tn = int(np.floor(t_fin / dt))

    # Set up grid
    L = 1
    N = 100
    h = L / N
    x = np.arange(0, L, h)
    y = np.arange(0, L, h)

    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Initialize velocity (make sure that ux + vy = 0)
    U = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    V = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    U1 = U.copy()
    V1 = V.copy()

    # Define derivatives
    Ux1 = 2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    Uy1 = -2 * np.pi * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    LU1 = -2 * (2 * np.pi) ** 2 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    Vx1 = 2 * np.pi * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    Vy1 = -2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    LV1 = 2 * (2 * np.pi) ** 2 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    Px1 = -2 * np.pi * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    Py1 = -2 * np.pi * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # Initialize a1 and a2
    a1, a2 = initialize_a(L, h, N, dt, rho, mu)

    # Time-stepping loop
    for ind in range(tn):
        # Evaluate f at t^{n+1/2}
        t = (ind + 0.5) * dt

        # Setting up f
        ft = np.cos(t)
        dft = -np.sin(t)
        Ut = dft * U1
        Vt = dft * V1

        U11 = U1 * ft
        V11 = V1 * ft

        Ux = Ux1 * ft
        Vx = Vx1 * ft
        Uy = Uy1 * ft
        Vy = Vy1 * ft
        Px = Px1 * ft
        Py = Py1 * ft

        LU = LU1 * ft
        LV = LV1 * ft

        f1 = (rho * (Ut + U11 * Ux + V11 * Uy) + Px - mu * LU)
        f2 = (rho * (Vt + U11 * Vx + V11 * Vy) + Py - mu * LV)

        # Solve the NS equations
        U, V = ns_solver(U, V, f1, f2, dt, h, rho, mu, a1, a2)

    # True solution
    U_true = U1 * np.cos(t_fin)
    V_true = V1 * np.cos(t_fin)

    # Calculate the maximum absolute difference
    Eu1 = np.max(np.abs(U - U_true))
    Ev1 = np.max(np.abs(V - V_true))

    # Display the results
    print("Maximum absolute difference in U:", Eu1)
    print("Maximum absolute difference in V:", Ev1)

    return U, U_true, V, V_true, X, Y


def plot_waves(U, U_true, V, V_true, X, Y):
    # Plot the results in 3-D
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("U")
    ax1.plot_surface(X, Y, U, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("U True")
    ax2.plot_surface(X, Y, U_true, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("V")
    ax1.plot_surface(X, Y, V, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("V True")
    ax2.plot_surface(X, Y, V_true, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)
    plt.show()


def main():
    # Time step parameters
    dt = 0.1
    t_fin = 1
    while True:
        fluid = input("fluid: ")
        dt = float(input("time_step: "))
        t_fin = float(input("final time: "))
        if fluid == "q" or dt == "q" or t_fin == "q":
            break
        if fluid not in FLUIDS:
            continue
        U, U_true, V, V_true, X, Y = calculate_fluid(FLUIDS[fluid], dt, t_fin)
        plot_waves(U, U_true, V, V_true, X, Y)


if __name__ == "__main__":
    main()