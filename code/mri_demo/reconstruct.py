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

def simulate_acquisition(phantom, sensitivity_maps, mask, noise_std=0.02, seed=42):
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
    rng = np.random.default_rng(seed)
    kspace_data = []
    kspace_full = []

    for s in sensitivity_maps:
        # Modulate image by coil sensitivity
        coil_image = s * phantom

        # 2D Fourier transform → k-space
        k_full = fftshift(fft2(ifftshift(coil_image)))

        # Undersample
        k_undersampled = mask * k_full

        # Add complex Gaussian noise (with seeded RNG for reproducibility)
        noise_real = rng.standard_normal(k_undersampled.shape) * noise_std
        noise_imag = rng.standard_normal(k_undersampled.shape) * noise_std
        k_noisy = k_undersampled + (noise_real + 1j * noise_imag)

        kspace_data.append(k_noisy)
        kspace_full.append(k_full)

    return kspace_data, kspace_full

# %% [markdown]
# ## 4. Baseline Reconstructions
#
# We compute three baseline reconstructions for pedagogical comparison:
# 1. **Naive sum-of-squares (SoS):** Pure per-coil IFFT without any
#    sensitivity weighting — the simplest possible reconstruction.
# 2. **Zero-filled with sensitivity weighting:** IFFT of zero-filled k-space,
#    then combine coils using conjugate-sensitivity weighting and
#    sum-of-squares normalization. Already uses some sensitivity information
#    for coil combination, making it better than naive SoS but still showing
#    aliasing artifacts.
# 3. **CG (SENSE):** Full least-squares solution using conjugate gradient,
#    which properly inverts the SENSE model.

# %%
def naive_sos_reconstruction(kspace_data):
    """
    Pure sum-of-squares coil combination — the simplest possible reconstruction.

    For each coil: IFFT the zero-filled k-space, then compute sqrt(sum |IFFT_i|^2).
    This does NOT use any coil sensitivity information for the combination,
    so it shows both aliasing AND coil shading artifacts.
    """
    sos = np.zeros(kspace_data[0].shape, dtype=np.float64)
    for y in kspace_data:
        coil_img = fftshift(ifft2(ifftshift(y)))
        sos += np.abs(coil_img)**2
    return np.sqrt(sos)

def zero_filled_reconstruction(kspace_data, sensitivity_maps):
    """
    Zero-filled IFFT reconstruction with sensitivity-weighted coil combination.

    For each coil i:
    1. Inverse FFT of zero-filled k-space
    2. Weight by conjugate coil sensitivity
    3. Normalize by sum-of-squares of sensitivity magnitudes

    NOTE: This reconstruction already uses coil sensitivity information for
    combining the per-coil images (via conj(s_i) weighting and SoS normalization),
    which makes it better than a purely naive IFFT. However, it does NOT solve
    the full SENSE inverse problem — the zero-filling in k-space still produces
    aliasing artifacts from the undersampling.
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
# Tikhonov regularization (adding $\lambda I$) ensures positive definiteness
# for CG.

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
# x_0 = 0, r_0 = b, p_0 = r_0
# for k = 0, 1, 2, ...:
#     alpha_k = (r_k^H r_k) / (p_k^H A p_k)
#     x_{k+1} = x_k + alpha_k p_k
#     r_{k+1} = r_k - alpha_k A p_k
#     beta_k = (r_{k+1}^H r_{k+1}) / (r_k^H r_k)
#     p_{k+1} = r_{k+1} + beta_k p_k
# ```
#
# The inner product for complex vectors is $\langle u, v \rangle = u^H v = \sum \overline{u_j} v_j$.

# %%
def conjugate_gradient_sense(kspace_data, sensitivity_maps, mask,
                              x0=None, max_iter=50, tol=1e-6,
                              lambda_reg=0.0, x_true=None):
    """
    Solve the SENSE inverse problem using Conjugate Gradient.

    Solves: (A_norm + lambda I) x = b

    where:
    - A_norm = sum_i S_i^H F^H M F S_i  (the SENSE normal operator)
    - b = sum_i S_i^H F^H y_i           (the right-hand side)
    - lambda >= 0 is an optional Tikhonov regularization parameter

    This is a hand-written CG implementation that uses matrix-free
    operator application via FFTs.

    Parameters:
    - x_true: optional ground truth for tracking true reconstruction error

    Returns:
    - x: reconstructed image
    - residual_norms: list of residual norms at each iteration
    - true_errors: list of relative true errors (NRMSE) at each iteration
                   (only if x_true is provided)
    """
    N = kspace_data[0].shape[0]

    # Compute right-hand side
    b = compute_sense_rhs(kspace_data, sensitivity_maps)

    # Initial guess
    if x0 is None:
        x = np.zeros((N, N), dtype=np.complex128)
    else:
        x = x0.copy()

    # Define the operator A_lambda = A_norm + lambda I
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

    # Track true reconstruction error if ground truth is provided
    if x_true is not None:
        true_norm = np.linalg.norm(x_true)
        true_errors = [np.linalg.norm(np.abs(x) - x_true) / true_norm]
    else:
        true_errors = []

    for k in range(max_iter):
        Ap = apply_A(p)

        # Step size alpha = (r^H r) / (p^H A p)
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

        # Track true error if ground truth available
        if x_true is not None:
            err = np.linalg.norm(np.abs(x) - x_true) / true_norm
            true_errors.append(err)

        if res_norm / residual_norms[0] < tol:
            print(f"  CG converged at iteration {k+1}, "
                  f"residual relative = {res_norm/residual_norms[0]:.2e}")
            break

        # beta = (r_{k+1}^H r_{k+1}) / (r_k^H r_k)
        r_new_norm_sq = np.real(np.sum(np.conj(r_new) * r_new))
        beta = r_new_norm_sq / r_norm_sq

        # Update search direction: p_{k+1} = r_{k+1} + beta p_k
        p = r_new + beta * p
        r = r_new

    else:
        print(f"  CG reached max_iter={max_iter}, "
              f"residual relative = {residual_norms[-1]/residual_norms[0]:.2e}")

    return x, np.array(residual_norms), np.array(true_errors)

# %% [markdown]
# ## 7. Run the Full Pipeline

# %%
# Parameters
N = 128          # image size (N×N pixels)
n_coils = 6      # number of receiver coils
R = 3            # acceleration factor (1=fully sampled, >1=undersampled)
noise_std = 0.01  # complex Gaussian noise standard deviation
max_cg_iter = 60  # increased from 40 to allow convergence
lambda_reg = 1e-3  # small Tikhonov regularization
random_seed = 42   # for reproducibility

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

# Step 3: Simulate acquisition (seeded for reproducibility)
print("3. Simulating undersampled k-space acquisition...")
mask = cartesian_undersampling_mask(N, R)
kspace_data, kspace_full = simulate_acquisition(phantom, sens_maps, mask,
                                                  noise_std=noise_std,
                                                  seed=random_seed)
undersampling_ratio = np.sum(mask) / mask.size
print(f"   Undersampling: {undersampling_ratio:.1%} of k-space acquired")

# Step 4a: Naive SoS reconstruction (simplest baseline)
print("4a. Computing naive sum-of-squares reconstruction...")
t0 = time.time()
recon_sos = naive_sos_reconstruction(kspace_data)
t_sos = time.time() - t0
error_sos = np.linalg.norm(recon_sos - phantom) / np.linalg.norm(phantom)
print(f"   Time: {t_sos:.3f}s, NRMSE: {error_sos:.4f}")

# Step 4b: Zero-filled with sensitivity weighting (baseline)
print("4b. Computing zero-filled reconstruction (with sensitivity weighting)...")
t0 = time.time()
recon_zf = zero_filled_reconstruction(kspace_data, sens_maps)
t_zf = time.time() - t0
error_zf = np.linalg.norm(np.abs(recon_zf) - phantom) / np.linalg.norm(phantom)
print(f"   Time: {t_zf:.3f}s, NRMSE: {error_zf:.4f}")

# Step 5: CG reconstruction
print("5. Running Conjugate Gradient reconstruction...")
x0 = recon_zf.copy()  # start from zero-filled solution
t0 = time.time()
recon_cg, cg_residuals, cg_true_errors = conjugate_gradient_sense(
    kspace_data, sens_maps, mask,
    x0=x0, max_iter=max_cg_iter, tol=1e-4, lambda_reg=lambda_reg,
    x_true=phantom
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
fig, axes = plt.subplots(2, 5, figsize=(24, 10))

# Original phantom
ax = axes[0, 0]
im = ax.imshow(phantom, cmap='gray', origin='lower')
ax.set_title('Original Phantom\n(Ground Truth)', fontsize=11, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Show sensitivity maps (first 4)
for c in range(min(4, n_coils)):
    row = 0 if c < 2 else 1
    col = 1 + (c % 2)
    ax = axes[row, col]
    im = ax.imshow(np.abs(sens_maps[c]), cmap='viridis', origin='lower')
    ax.set_title(r'Coil %d Sensitivity\n$|s_{%d}(r)|$' % (c+1, c+1), fontsize=10)
    ax.axis('off')

# Naive SoS reconstruction
ax = axes[1, 0]
im = ax.imshow(recon_sos, cmap='gray', origin='lower')
ax.set_title(r'Naive SoS IFFT\\NRMSE=%.3f\\No sens. info, heavy shading' % error_sos,
             fontsize=10)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Zero-filled reconstruction (sensitivity-weighted)
ax = axes[1, 1]
im = ax.imshow(np.abs(recon_zf), cmap='gray', origin='lower')
ax.set_title(r'Zero-Filled IFFT\\R=%d, NRMSE=%.3f\\Sens.-weighted, aliasing' % (R, error_zf),
             fontsize=10)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# CG reconstruction
ax = axes[1, 2]
im = ax.imshow(recon_cg_mag, cmap='gray', origin='lower')
ax.set_title(r'CG Reconstruction\\R=%d, NRMSE=%.3f\\Aliasing removed' % (R, error_cg),
             fontsize=10)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Undersampling mask
ax = axes[1, 3]
ax.imshow(mask, cmap='gray', origin='lower')
ax.set_title(r'Undersampling Mask\\R=%d, %.0f%% acquired' % (R, undersampling_ratio*100),
             fontsize=10)
ax.axis('off')

# Error map (difference between CG and original)
ax = axes[1, 4]
error_map = np.abs(recon_cg_mag - phantom)
im = ax.imshow(error_map, cmap='hot', origin='lower')
ax.set_title(r'Error Map\\$|x_{\rm CG}| - x_{\rm true}$',
             fontsize=10)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Multi-Coil MRI Reconstruction: SENSE + Conjugate Gradient\n'
             r'%d×%d pixels, %d coils, R=%d, CG iter=%d' % (
                 N, N, n_coils, R, len(cg_residuals)-1),
             fontsize=14, y=1.01)
plt.tight_layout()

outpath1 = os.path.join(fig_dir, 'mri_reconstruction_results.pdf')
plt.savefig(outpath1, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath1}")
plt.show()

# --- Figure 2: CG Convergence AND Semiconvergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Residual norm vs iteration (always decreasing)
ax1.semilogy(cg_residuals, 'o-', color='#3498db', linewidth=2,
             markersize=5, markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel(r'Iteration $k$', fontsize=12)
ax1.set_ylabel(r'Residual $\|r_k\|_2$', fontsize=12)
ax1.set_title('CG Residual Norm\n(always decreases)', fontsize=13)
ax1.grid(True, alpha=0.3, which='both')

# Right: True reconstruction error vs iteration (semiconvergence!)
# Plot both residual and true error on the same axes for comparison
if len(cg_true_errors) > 0:
    # Relative residual (for comparison with true error)
    rel_res = cg_residuals / cg_residuals[0]

    color_err = '#e74c3c'
    color_res = '#3498db'

    ax2.semilogy(cg_true_errors, 's-', color=color_err, linewidth=2,
                 markersize=5, markerfacecolor='white', markeredgewidth=2,
                 label=r'True error $\| |x_k| - x_{\rm true}\|_2 / \|x_{\rm true}\|_2$')
    ax2.semilogy(rel_res, 'o-', color=color_res, linewidth=2,
                 markersize=5, markerfacecolor='white', markeredgewidth=2,
                 label=r'Relative residual $\|r_k\|_2 / \|r_0\|_2$')

    # Highlight the minimum-error iteration
    if len(cg_true_errors) > 0:
        best_iter = np.argmin(cg_true_errors)
        ax2.axvline(x=best_iter, color='gray', linestyle='--', alpha=0.7,
                    label=f'Min error at k={best_iter}')

    ax2.set_xlabel(r'Iteration $k$', fontsize=12)
    ax2.set_ylabel('Relative Norm', fontsize=12)
    ax2.set_title(r'Semiconvergence: Residual $\neq$ Error',
                  fontsize=13)
    ax2.legend(fontsize=8, loc='center left')
    ax2.grid(True, alpha=0.3, which='both')

plt.suptitle('Conjugate Gradient Convergence for SENSE Reconstruction\n'
             'The residual always decreases, but the true error can plateau or increase!',
             fontsize=13, y=1.02)
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
    ax.set_title('Coil %d Magnitude' % (c+1), fontsize=11)
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
