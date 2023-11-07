import numpy as np
import scipy.fftpack as fft

def ns_solver(U, V, f1, f2, dt, dx, rho, mu, a1, a2):
    # Periodic boundary conditions
    U_bc = np.column_stack((U[:, -1], U, U[:, 0]))
    U_bc = np.row_stack((U_bc[-1, :], U_bc, U_bc[0, :]))

    V_bc = np.column_stack((V[:, -1], V, V[:, 0]))
    V_bc = np.row_stack((V_bc[-1, :], V_bc, V_bc[0, :]))

    # Nonlinear terms
    UU = U_bc * U_bc
    UV = U_bc * V_bc
    VV = V_bc * V_bc

    U_nl = 1/2 * (U * (U_bc[2:, 1:-1] - U_bc[:-2, 1:-1]) / (2 * dx) +
                 V * (U_bc[1:-1, 2:] - U_bc[1:-1, :-2]) / (2 * dx) +
                 (UU[2:, 1:-1] - UU[:-2, 1:-1]) / (2 * dx) +
                 (UV[1:-1, 2:] - UV[1:-1, :-2]) / (2 * dx))
    V_nl = 1/2 * (U * (V_bc[2:, 1:-1] - V_bc[:-2, 1:-1]) / (2 * dx) +
                 V * (V_bc[1:-1, 2:] - V_bc[1:-1, :-2]) / (2 * dx) +
                 (UV[2:, 1:-1] - UV[:-2, 1:-1]) / (2 * dx) +
                 (VV[1:-1, 2:] - VV[1:-1, :-2]) / (2 * dx))

    # Laplacian
    LU = (U_bc[2:, 1:-1] - 2 * U_bc[1:-1, 1:-1] + U_bc[:-2, 1:-1]) / (dx**2) + \
         (U_bc[1:-1, 2:] - 2 * U_bc[1:-1, 1:-1] + U_bc[1:-1, :-2]) / (dx**2)

    LV = (V_bc[2:, 1:-1] - 2 * V_bc[1:-1, 1:-1] + V_bc[:-2, 1:-1]) / (dx**2) + \
         (V_bc[1:-1, 2:] - 2 * V_bc[1:-1, 1:-1] + V_bc[1:-1, :-2]) / (dx**2)

    # Right-hand side
    w1_h = U - (dt/2) * U_nl + (dt/(2*rho)) * f1
    w2_h = V - (dt/2) * V_nl + (dt/(2*rho)) * f2

    # Discrete fast Fourier transform
    w1_h = fft.fft(w1_h, axis=0)
    w1_h = fft.fft(w1_h, axis=1)

    w2_h = fft.fft(w2_h, axis=0)
    w2_h = fft.fft(w2_h, axis=1)

    # Half step
    U_h = a1[:, :, 0] * w1_h + a1[:, :, 1] * w2_h
    V_h = a2[:, :, 0] * w1_h + a2[:, :, 1] * w2_h

    # Inverse fast Fourier transform
    U_h = fft.ifft(U_h, axis=1)
    U_h = np.real(fft.ifft(U_h, axis=0))

    V_h = fft.ifft(V_h, axis=1)
    V_h = np.real(fft.ifft(V_h, axis=0))

    # Half step nonlinear terms with periodic boundary conditions
    U_bc_h = np.column_stack((U_h[:, -1], U_h, U_h[:, 0]))
    U_bc_h = np.row_stack((U_bc_h[-1, :], U_bc_h, U_bc_h[0, :]))

    V_bc_h = np.column_stack((V_h[:, -1], V_h, V_h[:, 0]))
    V_bc_h = np.row_stack((V_bc_h[-1, :], V_bc_h, V_bc_h[0, :]))

    UU_h = U_bc_h * U_bc_h
    UV_h = U_bc_h * V_bc_h
    VV_h = V_bc_h * V_bc_h

    U_nl_h = 1/2 * (U_h * (U_bc_h[2:, 1:-1] - U_bc_h[:-2, 1:-1]) / (2 * dx) +
                    V_h * (U_bc_h[1:-1, 2:] - U_bc_h[1:-1, :-2]) / (2 * dx) +
                    (UU_h[2:, 1:-1] - UU_h[:-2, 1:-1]) / (2 * dx) +
                    (UV_h[1:-1, 2:] - UV_h[1:-1, :-2]) / (2 * dx))
    V_nl_h = 1/2 * (U_h * (V_bc_h[2:, 1:-1] - V_bc_h[:-2, 1:-1]) / (2 * dx) +
                    V_h * (V_bc_h[1:-1, 2:] - V_bc_h[1:-1, :-2]) / (2 * dx) +
                    (UV_h[2:, 1:-1] - UV_h[:-2, 1:-1]) / (2 * dx) +
                    (VV_h[1:-1, 2:] - VV_h[1:-1, :-2]) / (2 * dx))

    w1 = U - dt * U_nl_h + (dt/rho) * f1 + (dt/2) * (mu/rho) * LU
    w2 = V - dt * V_nl_h + (dt/rho) * f2 + (dt/2) * (mu/rho) * LV

    # Discrete fast Fourier transform
    w1 = fft.fft(w1, axis=0)
    w1 = fft.fft(w1, axis=1)

    w2 = fft.fft(w2, axis=0)
    w2 = fft.fft(w2, axis=1)

    # Next velocity
    U_n = a1[:, :, 0] * w1 + a1[:, :, 1] * w2
    V_n = a2[:, :, 0] * w1 + a2[:, :, 1] * w2

    # Inverse fast Fourier transform
    U_n = fft.ifft(U_n, axis=1)
    U_n = np.real(fft.ifft(U_n, axis=0))

    V_n = fft.ifft(V_n, axis=1)
    V_n = np.real(fft.ifft(V_n, axis=0))

    return U_n, V_n
