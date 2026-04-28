# %% [markdown]
# # Multi-Coil MRI Reconstruction via Conjugate Gradient (SENSE)
#
# This demo implements the full pipeline for reconstructing an MR image
# from undersampled multi-coil k-space data using the **Conjugate Gradient**
# algorithm to solve the **SENSE** (SENSitivity Encoding) inverse problem.
#
# ## Pipeline Overview
# 1. Create a phantom image (Shepp-Logan)
# 2. Generate simulated multi-coil sensitivity maps
# 3. Simulate undersampled k-space acquisition for each coil
# 4. Formulate the SENSE inverse problem as a linear system
# 5. Solve with hand-written Conjugate Gradient
# 6. Compare against zero-filled coil-combined reconstruction

# %% [markdown]
# ### Notation
# - $N \times N$: image dimensions (pixels)
# - $C$: number of receiver coils
# - $R$: undersampling factor (acceleration rate)
# - $x \in \mathbb{C}^{N \times N}$: unknown image
# - $s_i \in \mathbb{C}^{N \times N}$: sensitivity map for coil $i$
# - $y_i \in \mathbb{C}^{N/R \times N}$: acquired k-space data for coil $i$
# - $M$: binary undersampling mask

# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import os, sys, time

# Ensure figures directory exists
fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'slides', 'figures')
os.makedirs(fig_dir, exist_ok=True)
print(f"Figures will be saved to: {fig_dir}")

# %% [markdown]
# ## 1. Create the Phantom Image
#
# We generate a Shepp-Logan phantom — a standard test image in medical imaging
# that mimics the cross-section of a human head.

# %%
def shepp_logan_phantom(N=128):
    """
    Generate a Shepp-Logan phantom of size N×N.

    The Shepp-Logan phantom is composed of 10 ellipses with different
    intensities, representing different tissue types in the brain.
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    phantom = np.zeros((N, N), dtype=np.float64)

    # Each ellipse: (cx, cy, rx, ry, rotation_degrees, intensity)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0.0, 2.0),       # outer skull
        (0.0, -0.0184, 0.6624, 0.874, 0.0, -0.98), # inner skull
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.02),   # right eye
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.02),   # left eye
        (0.0, 0.35, 0.21, 0.25, 0.0, 0.01),      # bright spot 1
        (0.0, 0.1, 0.046, 0.046, 0.0, 0.01),     # bright spot 2
        (0.0, -0.1, 0.046, 0.046, 0.0, 0.01),    # bright spot 3
        (-0.08, -0.605, 0.046, 0.023, 0.0, 0.01), # left small
        (0.0, -0.605, 0.023, 0.023, 0.0, 0.01),   # bottom small
        (0.06, -0.605, 0.046, 0.023, 0.0, 0.01),  # right small
    ]

    for cx, cy, rx, ry, angle_deg, intensity in ellipses:
        angle = np.radians(angle_deg)
        X_rot = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        Y_rot = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)
        mask = (X_rot**2 / rx**2 + Y_rot**2 / ry**2) <= 1.0
        phantom[mask] += intensity

    return phantom

# %% [markdown]
# ## 2. Generate Coil Sensitivity Maps
#
# In MRI, each receiver coil has a spatially-varying sensitivity profile
# $s_i(\mathbf{r})$. Coils closer to a given region pick up stronger signal
# from that region. We model sensitivity maps as offset 2D Gaussians with
# complex phase variations (Biot-Savart-inspired profiles).

# %%
def generate_coil_sensitivity_maps(N, n_coils=6):
    """
    Generate C simulated coil sensitivity maps of size N×N.

    Each sensitivity map is a complex-valued array. The magnitude follows
    a 2D Gaussian centered at a position around the circle, and the phase
    varies smoothly across the FOV to simulate realistic B1 field variations.
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    maps = []

    # Evenly space coils around a circle
    angles = np.linspace(0, 2 * np.pi, n_coils, endpoint=False)
    radius = 0.55  # distance from center

    for i, theta in enumerate(angles):
        cx = radius * np.cos(theta)
        cy = radius * np.sin(theta)

        # Magnitude: 2D Gaussian centered at (cx, cy)
        sigma = 0.65
        magnitude = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

        # Phase: linear ramp in the direction of the coil center
        # plus some quadratic variation for realism
        phase = np.pi * (X * cx + Y * cy)
        # Add a small quadratic phase for realism
        phase += 0.15 * np.pi * ((X - cx)**2 + (Y - cy)**2)

        s = magnitude * np.exp(1j * phase)
        maps.append(s)

    return maps

# %% [markdown]
# ## 3. Simulate Undersampled k-Space Acquisition
#
# In MRI, we don't measure the image directly. We measure samples of its
# Fourier transform (k-space). To accelerate the scan, we skip some
# phase-encoding lines (Cartesian undersampling).
#
# For each coil $i$, the forward model is:
# $$y_i^{\text{acquired}} = \text{downsample}[\mathcal{F}\{s_i \odot x\}, R] + \eta_i$$

# %%
def cartesian_undersampling_mask(N, R):
    """
    Create a Cartesian undersampling mask.

    Parameters:
    - N: image size (N×N)
    - R: acceleration factor (keep every R-th line)

    Returns binary mask of shape (N, N) where 1 = acquired, 0 = skipped.
    """
    mask = np.zeros((N, N), dtype=np.float64)
    mask[::R, :] = 1.0  # keep every R-th phase-encoding line
    return mask

def simulate_acquisition(phantom, sensitivity_maps, mask, noise_std=0.02):
    """
    Simulate multi-coil MRI acquisition.

    For each coil i:
    1. Apply sensitivity: coil_img = s_i * phantom
    2. Compute k-space: k_full = FFT(coil_img)
    3. Undersample: k_undersampled = mask * k_full
    4. Add complex Gaussian noise

    Returns:
    - kspace_data: list of undersampled k-spaces (complex arrays)
    - kspace_full: list of full k-spaces (for reference)
    """
    kspace_data = []
    kspace_full = []

    for s in sensitivity_maps:
        # Modulate image by coil sensitivity
        coil_image = s * phantom

        # 2D Fourier transform → k-space
        k_full = fftshift(fft2(ifftshift(coil_image)))

        # Undersample
        k_undersampled = mask * k_full

        # Add complex Gaussian noise
        noise_real = np.random.randn(*k_undersampled.shape) * noise_std
        noise_imag = np.random.randn(*k_undersampled.shape) * noise_std
        k_noisy = k_undersampled + (noise_real + 1j * noise_imag)

        kspace_data.append(k_noisy)
        kspace_full.append(k_full)

    return kspace_data, kspace_full

# %% [markdown]
# ## 4. Zero-Filled Coil-Combined Reconstruction (Baseline)
#
# The simplest reconstruction: zero-fill unacquired k-space locations,
# inverse FFT, then combine coils using the sum-of-squares method.
# This produces aliasing artifacts at acceleration factor R > 1.

# %%
def zero_filled_reconstruction(kspace_data, sensitivity_maps):
    """
    Simple zero-filled IFFT reconstruction with sum-of-squares coil combination.

    For each coil i:
    1. Inverse FFT of zero-filled k-space
    2. Divide by coil sensitivity (approximate)
    3. Sum-of-squares combination
    """
    C = len(kspace_data)
    N = kspace_data[0].shape[0]
    recon = np.zeros((N, N), dtype=np.complex128)
    sos_weight = np.zeros((N, N), dtype=np.float64)

    for c in range(C):
        # IFFT back to image domain
        coil_img = fftshift(ifft2(ifftshift(kspace_data[c])))

        # Sum-of-squares coil combination:
        # We weight each coil by its conjugate sensitivity
        s = sensitivity_maps[c]
        recon += np.conj(s) * coil_img
        sos_weight += np.abs(s)**2

    # Normalize by sum-of-squares weights (avoid division by zero)
    eps = np.max(sos_weight) * 1e-6
    recon = recon / (sos_weight + eps)

    return recon

# %% [markdown]
# ## 5. SENSE Normal Equations
#
# The SENSE reconstruction solves:
#
# $$\min_x \sum_{i=1}^C \|M F S_i x - y_i\|_2^2$$
#
# This is a linear least-squares problem. The **normal equations** are:
#
# $$\underbrace{\left(\sum_{i=1}^C S_i^H F^H M F S_i\right)}_{A_{\text{norm}}} x
#   \;=\;
#   \underbrace{\sum_{i=1}^C S_i^H F^H y_i}_{b}$$
#
# where:
# - $S_i$ = multiplication by $s_i$ (diagonal operator)
# - $F$ = 2D Fourier transform (unitary)
# - $M$ = undersampling mask (zeros out skipped k-space lines)
# - $S_i^H$ = multiplication by $\overline{s_i}$ (complex conjugate)
# - $F^H = F^{-1}$ = inverse 2D Fourier transform
#
# The matrix $A_{\text{norm}}$ is Hermitian positive semi-definite.
# We apply it implicitly (never form it explicitly) via FFT operations.

# %%
def apply_sense_normal_operator(x, sensitivity_maps, mask):
    """
    Apply the SENSE normal equations matrix A_norm to vector x.

    A_norm(x) = sum_i conj(s_i) * IFFT(mask * FFT(s_i * x))

    This computes A_norm @ x without explicitly forming A_norm.
    """
    result = np.zeros_like(x)
    for s in sensitivity_maps:
        # Forward: apply sensitivity, then FFT, then mask
        coil_img = s * x                        # S_i @ x
        kspace = fftshift(fft2(ifftshift(coil_img)))  # F @ S_i @ x
        kspace_masked = mask * kspace           # M @ F @ S_i @ x
        # Adjoint: IFFT, then conjugate sensitivity
        back_img = fftshift(ifft2(ifftshift(kspace_masked)))  # F^H @ M @ F @ S_i @ x
        result += np.conj(s) * back_img          # S_i^H @ F^H @ M @ F @ S_i @ x
    return result

def compute_sense_rhs(kspace_data, sensitivity_maps):
    """
    Compute the right-hand side of the SENSE normal equations:

    b = sum_i S_i^H F^H y_i

    For each coil: b += conj(s_i) * IFFT(y_i_zero_filled)
    """
    rhs = np.zeros_like(kspace_data[0], dtype=np.complex128)
    for c, y in enumerate(kspace_data):
        s = sensitivity_maps[c]
        coil_back = fftshift(ifft2(ifftshift(y)))  # F^H @ y_i
        rhs += np.conj(s) * coil_back               # S_i^H @ F^H @ y_i
    return rhs

# %% [markdown]
# ## 6. Conjugate Gradient Implementation
#
# The CG method solves $A x = b$ when $A$ is Hermitian positive definite.
# It generates a sequence of search directions $\{p_k\}$ that are
# $A$-orthogonal: $p_i^H A p_j = 0$ for $i \neq j$.
#
# ### Algorithm (Hestenes & Stiefel, 1952):
#
# ```
# x₀ = 0, r₀ = b, p₀ = r₀
# for k = 0, 1, 2, ...:
#     αₖ = (rₖ^H rₖ) / (pₖ^H A pₖ)
#     x_{k+1} = xₖ + αₖ pₖ
#     r_{k+1} = rₖ - αₖ A pₖ
#     βₖ = (r_{k+1}^H r_{k+1}) / (rₖ^H rₖ)
#     p_{k+1} = r_{k+1} + βₖ pₖ
# ```
#
# The inner product for complex vectors is $\langle u, v \rangle = u^H v = \sum \overline{u_j} v_j$.

# %%
def conjugate_gradient_sense(kspace_data, sensitivity_maps, mask,
                              x0=None, max_iter=50, tol=1e-6,
                              lambda_reg=0.0):
    """
    Solve the SENSE inverse problem using Conjugate Gradient.

    Solves: (A_norm + λ I) x = b

    where:
    - A_norm = sum_i S_i^H F^H M F S_i  (the SENSE normal operator)
    - b = sum_i S_i^H F^H y_i           (the right-hand side)
    - λ ≥ 0 is an optional Tikhonov regularization parameter

    This is a hand-written CG implementation that uses matrix-free
    operator application via FFTs.

    Returns:
    - x: reconstructed image
    - residual_norms: list of residual norms at each iteration
    """
    N = kspace_data[0].shape[0]

    # Compute right-hand side
    b = compute_sense_rhs(kspace_data, sensitivity_maps)

    # Initial guess
    if x0 is None:
        x = np.zeros((N, N), dtype=np.complex128)
    else:
        x = x0.copy()

    # Define the operator A_λ = A_norm + λ I
    def apply_A(v):
        """Apply the regularized normal operator to v."""
        Av = apply_sense_normal_operator(v, sensitivity_maps, mask)
        if lambda_reg > 0:
            Av += lambda_reg * v
        return Av

    # CG initialization
    r = b - apply_A(x)   # initial residual
    p = r.copy()          # first search direction
    residual_norms = [np.sqrt(np.real(np.sum(np.conj(r) * r)))]

    for k in range(max_iter):
        Ap = apply_A(p)

        # Step size α = (r^H r) / (p^H A p)
        r_norm_sq = np.real(np.sum(np.conj(r) * r))
        pAp = np.real(np.sum(np.conj(p) * Ap))

        if pAp < 1e-16:
            print(f"  CG: p^H A p too small at iteration {k}, stopping.")
            break

        alpha = r_norm_sq / pAp

        # Update solution and residual
        x = x + alpha * p
        r_new = r - alpha * Ap

        # Compute residual norm
        res_norm = np.sqrt(np.real(np.sum(np.conj(r_new) * r_new)))
        residual_norms.append(res_norm)

        if res_norm / residual_norms[0] < tol:
            print(f"  CG converged at iteration {k+1}, "
                  f"residual relative = {res_norm/residual_norms[0]:.2e}")
            break

        # β = (r_{k+1}^H r_{k+1}) / (r_k^H r_k)
        r_new_norm_sq = np.real(np.sum(np.conj(r_new) * r_new))
        beta = r_new_norm_sq / r_norm_sq

        # Update search direction: p_{k+1} = r_{k+1} + β p_k
        p = r_new + beta * p
        r = r_new

    else:
        print(f"  CG reached max_iter={max_iter}, "
              f"residual relative = {residual_norms[-1]/residual_norms[0]:.2e}")

    return x, np.array(residual_norms)

# %% [markdown]
# ## 7. Run the Full Pipeline

# %%
# Parameters
N = 128          # image size (N×N pixels)
n_coils = 6      # number of receiver coils
R = 3            # acceleration factor (1=fully sampled, >1=undersampled)
noise_std = 0.01  # complex Gaussian noise standard deviation
max_cg_iter = 40
lambda_reg = 1e-3  # small Tikhonov regularization

print("=" * 60)
print("Multi-Coil MRI Reconstruction via SENSE + Conjugate Gradient")
print("=" * 60)
print(f"  Image size: {N}×{N}")
print(f"  Number of coils: {n_coils}")
print(f"  Acceleration factor: R = {R}")
print(f"  CG max iterations: {max_cg_iter}")
print()

# Step 1: Create phantom
print("1. Creating Shepp-Logan phantom...")
phantom = shepp_logan_phantom(N)
print(f"   Phantom range: [{phantom.min():.2f}, {phantom.max():.2f}]")

# Step 2: Generate coil sensitivity maps
print("2. Generating coil sensitivity maps...")
sens_maps = generate_coil_sensitivity_maps(N, n_coils)
print(f"   Each map shape: {sens_maps[0].shape}")

# Step 3: Simulate acquisition
print("3. Simulating undersampled k-space acquisition...")
mask = cartesian_undersampling_mask(N, R)
kspace_data, kspace_full = simulate_acquisition(phantom, sens_maps, mask,
                                                  noise_std=noise_std)
undersampling_ratio = np.sum(mask) / mask.size
print(f"   Undersampling: {undersampling_ratio:.1%} of k-space acquired")

# Step 4: Zero-filled reconstruction (baseline)
print("4. Computing zero-filled reconstruction...")
t0 = time.time()
recon_zf = zero_filled_reconstruction(kspace_data, sens_maps)
t_zf = time.time() - t0
error_zf = np.linalg.norm(np.abs(recon_zf) - phantom) / np.linalg.norm(phantom)
print(f"   Time: {t_zf:.3f}s, NRMSE: {error_zf:.4f}")

# Step 5: CG reconstruction
print("5. Running Conjugate Gradient reconstruction...")
x0 = recon_zf.copy()  # start from zero-filled solution
t0 = time.time()
recon_cg, cg_residuals = conjugate_gradient_sense(
    kspace_data, sens_maps, mask,
    x0=x0, max_iter=max_cg_iter, tol=1e-6, lambda_reg=lambda_reg
)
t_cg = time.time() - t0

# Get magnitude for display
recon_cg_mag = np.abs(recon_cg)
error_cg = np.linalg.norm(recon_cg_mag - phantom) / np.linalg.norm(phantom)
print(f"   Time: {t_cg:.3f}s, NRMSE: {error_cg:.4f}")
print(f"   CG iterations: {len(cg_residuals)-1}")

# %% [markdown]
# ## 8. Visualize Results

# %%
# --- Figure 1: Reconstruction Comparison ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Original phantom
ax = axes[0, 0]
im = ax.imshow(phantom, cmap='gray', origin='lower')
ax.set_title('Original Phantom\n(Ground Truth)', fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Show sensitivity maps (first 4)
for c in range(min(4, n_coils)):
    row = 0 if c < 2 else 1
    col = 1 + (c % 2)
    ax = axes[row, col]
    im = ax.imshow(np.abs(sens_maps[c]), cmap='viridis', origin='lower')
    ax.set_title(f'Coil {c+1} Sensitivity\n$|s_{c+1}(r)|$', fontsize=11)
    ax.axis('off')

# Zero-filled reconstruction
ax = axes[1, 0]
im = ax.imshow(np.abs(recon_zf), cmap='gray', origin='lower')
ax.set_title(f'Zero-Filled IFFT\n(R={R}, NRMSE={error_zf:.3f})\nAliasing artifacts',
             fontsize=11)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# CG reconstruction
ax = axes[1, 1]
im = ax.imshow(recon_cg_mag, cmap='gray', origin='lower')
ax.set_title(f'CG Reconstruction\n(R={R}, NRMSE={error_cg:.3f})\nAliasing removed',
             fontsize=11)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Undersampling mask
ax = axes[1, 2]
ax.imshow(mask, cmap='gray', origin='lower')
ax.set_title(f'Undersampling Mask\n(R={R}, {undersampling_ratio:.0%} acquired)',
             fontsize=11)
ax.axis('off')

# Error map (difference between CG and original)
ax = axes[1, 3]
error_map = np.abs(recon_cg_mag - phantom)
im = ax.imshow(error_map, cmap='hot', origin='lower')
ax.set_title(f'Error Map\n$\\|x_{{\\rm CG}}\\| - x_{{\\rm true}}$',
             fontsize=11)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Multi-Coil MRI Reconstruction: SENSE + Conjugate Gradient\n'
             f'{N}×{N} pixels, {n_coils} coils, R={R}, CG iter={len(cg_residuals)-1}',
             fontsize=15, y=1.01)
plt.tight_layout()

outpath1 = os.path.join(fig_dir, 'mri_reconstruction_results.pdf')
plt.savefig(outpath1, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath1}")
plt.show()

# --- Figure 2: CG Convergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Residual norm vs iteration
ax1.semilogy(cg_residuals, 'o-', color='#3498db', linewidth=2,
             markersize=5, markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Iteration $k$', fontsize=12)
ax1.set_ylabel(r'Residual $\|r_k\|_2$', fontsize=12)
ax1.set_title('CG Residual Norm vs Iteration', fontsize=13)
ax1.grid(True, alpha=0.3, which='both')

# Relative residual
rel_res = cg_residuals / cg_residuals[0]
ax2.semilogy(rel_res, 'o-', color='#e74c3c', linewidth=2,
             markersize=5, markerfacecolor='white', markeredgewidth=2)
ax2.set_xlabel('Iteration $k$', fontsize=12)
ax2.set_ylabel(r'Relative Residual $\|r_k\|_2 / \|r_0\|_2$', fontsize=12)
ax2.set_title('CG Relative Residual (Semilog)', fontsize=13)
ax2.grid(True, alpha=0.3, which='both')

plt.suptitle('Conjugate Gradient Convergence for SENSE Reconstruction',
             fontsize=14, y=1.02)
plt.tight_layout()

outpath2 = os.path.join(fig_dir, 'mri_cg_convergence.pdf')
plt.savefig(outpath2, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath2}")
plt.show()

# --- Figure 3: Coil Sensitivities Grid ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for c in range(n_coils):
    ax = axes[c // 3, c % 3]
    im = ax.imshow(np.abs(sens_maps[c]), cmap='viridis', origin='lower')
    ax.set_title(f'Coil {c+1} Magnitude', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Simulated Multi-Coil Sensitivity Maps (Magnitude)',
             fontsize=14, y=1.01)
plt.tight_layout()

outpath3 = os.path.join(fig_dir, 'coil_sensitivity_maps.pdf')
plt.savefig(outpath3, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath3}")
plt.show()

print("\n" + "=" * 60)
print("Demo complete! All figures saved to slides/figures/")
print("=" * 60)
