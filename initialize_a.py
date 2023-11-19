import numpy as np

def initialize_a(L, h, N, dt, rho, mu):
    ind = np.arange(N)

    # defining m1 and m2, m1 changing in rows, m2 changing in columns
    m1, m2 = np.meshgrid(ind, ind)

    # defining D1 and D2 fourier transforms
    D1_hat = 1j / h * np.sin(2 * np.pi / L * m1 * h)
    D2_hat = 1j / h * np.sin(2 * np.pi / L * m2 * h)

    # defining Laplacian fourier transform
    L_hat = -4 / h**2 * (np.sin(np.pi / L * m1 * h)**2 + np.sin(np.pi / L * m2 * h)**2)

    a1 = np.empty((N, N, 2), dtype=np.complex128)
    a2 = np.empty((N, N, 2), dtype=np.complex128)
    
    a1[:, :, 0] = 1 - (D1_hat**2) / (D1_hat**2 + D2_hat**2)
    a1[:, :, 1] = -(D1_hat * D2_hat) / (D1_hat**2 + D2_hat**2)

    a2[:, :, 0] = -(D1_hat * D2_hat) / (D1_hat**2 + D2_hat**2)
    a2[:, :, 1] = 1 - (D2_hat**2) / (D1_hat**2 + D2_hat**2)

    # Special cases
    in_half = N // 2
    a1[0, 0, 0] = 1
    a1[0, 0, 1] = 0
    a1[0, in_half, 0] = 1
    a1[0, in_half, 1] = 0
    a1[in_half, 0, 0] = 1
    a1[in_half, 0, 1] = 0
    a1[in_half, in_half, 0] = 1
    a1[in_half, in_half, 1] = 0

    a2[0, 0, 0] = 0
    a2[0, 0, 1] = 1
    a2[0, in_half, 0] = 0
    a2[0, in_half, 1] = 1
    a2[in_half, 0, 0] = 0
    a2[in_half, 0, 1] = 1
    a2[in_half, in_half, 0] = 0
    a2[in_half, in_half, 1] = 1

    # Perform the final updates with size considerations
    denominator = 1 - dt * mu / (2 * rho) * L_hat
    a1[:, :, 0] /= denominator
    a1[:, :, 1] /= denominator
    a2[:, :, 0] /= denominator
    a2[:, :, 1] /= denominator

    return a1, a2