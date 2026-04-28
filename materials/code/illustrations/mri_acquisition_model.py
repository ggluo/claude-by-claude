# %% [markdown]
# # Multi-Coil MRI Acquisition Model Diagram
#
# This script creates a visual diagram of the multi-coil MRI forward model:
#
# $$y_i = M \cdot \mathcal{F}\{S_i \odot x\} + \eta_i$$
#
# The diagram shows the pipeline from the unknown image $x$ through
# coil sensitivity modulation, Fourier transform, k-space undersampling,
# to the per-coil measurements.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
import os

fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'slides', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# # =========================================================================
# ## Part 1: Data Generation — Phantom, Coil Sensitivities, and k-Space
# # =========================================================================
#
# All simulation functions are defined first, then called once to produce
# the data arrays. This separation makes it easy to reuse the functions
# or understand the data-generation logic independently from the diagram.

# %% [markdown]
# ### Generate a Shepp-Logan Phantom

# %%
def shepp_logan_small(N=128):
    """Create a simple geometric phantom."""
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    img = np.zeros((N, N))
    ellipses = [
        (0, 0, 0.9, 0.85, 0, 1.0),          # outer skull
        (0, 0, 0.8, 0.75, 0, 0.6),           # brain
        (0.25, 0.35, 0.22, 0.18, 0, 0.8),    # bright spot 1
        (-0.25, 0.35, 0.22, 0.18, 0, 0.8),   # bright spot 2
        (0, -0.1, 0.25, 0.15, 0, 0.4),       # center
        (0, -0.55, 0.22, 0.1, 0, 0.3),       # bottom
    ]
    for cx, cy, rx, ry, _, val in ellipses:
        mask = ((X - cx) / rx)**2 + ((Y - cy) / ry)**2 <= 1
        img[mask] = val
    return img

# %% [markdown]
# ### Generate Coil Sensitivity Maps

# %%
def generate_sensitivity_maps(N, n_coils=4):
    """Generate coil sensitivity maps as offset 2D Gaussians."""
    maps = []
    angles = np.linspace(0, 2 * np.pi, n_coils, endpoint=False)
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    for theta in angles:
        cx, cy = 0.5 * np.cos(theta), 0.5 * np.sin(theta)
        gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.5)
        maps.append(gauss)
    return maps

# %% [markdown]
# ### Generate All Simulation Data at Once

# %%
# Parameters
N = 64
n_coils = 4
noise_std = 0.03  # standard deviation of complex Gaussian noise

# Generate phantom and sensitivity maps
phantom = shepp_logan_small(N)
sens_maps = generate_sensitivity_maps(N, n_coils)

# Per-coil images: s_i * x (element-wise modulation by sensitivity)
coil_images = [s * phantom for s in sens_maps]

# Per-coil k-space (full, before undersampling):
# FFT of each coil image, with shift to center low frequencies
kspaces = [np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) for img in coil_images]

# Undersampling mask (R=2 Cartesian, skip every other line)
mask = np.zeros((N, N))
mask[::2, :] = 1  # keep every 2nd phase-encoding line

# Apply mask to k-space data to simulate undersampled acquisition
undersampled_kspaces = [k * mask for k in kspaces]

# Add complex Gaussian noise to undersampled k-space for the "measured" row
rng = np.random.default_rng(42)
noisy_kspaces = []
for k_us in undersampled_kspaces:
    noise = rng.standard_normal(k_us.shape) * noise_std + \
            1j * rng.standard_normal(k_us.shape) * noise_std
    noisy_kspaces.append(k_us + noise)

# %% [markdown]
# # =========================================================================
# ## Part 2: Diagram Creation — Visualizing the MRI Forward Model
# # =========================================================================
#
# The diagram is organized as a pipeline flowing from top to bottom:
# - Row 0: Unknown image + coil sensitivity maps
# - Row 1: Per-coil modulated images (s_i ⊙ x)
# - Row 2: Full k-space (Fourier transform)
# - Row 3: Measured undersampled k-space with noise (M · F + η)
# - Row 4: SENSE inverse problem formulation (solve with CG)
#
# Row 3 now shows noisy k-space data, making it visually distinct from
# the clean undersampled k-space implied by the "Undersample" label above.

# %%
fig = plt.figure(figsize=(18, 10))

# === Row 0: Unknown image + coil sensitivities ===
ax_main = plt.subplot2grid((5, 6), (0, 0), rowspan=1, colspan=1)
ax_main.imshow(phantom, cmap='gray', origin='lower')
ax_main.set_title(r'Unknown Image\\$x$', fontsize=11, fontweight='bold')
ax_main.axis('off')

# Show sensitivity maps in a row
for c in range(n_coils):
    ax = plt.subplot2grid((5, 6), (0, c + 1), rowspan=1, colspan=1)
    ax.imshow(sens_maps[c], cmap='viridis', origin='lower')
    ax.set_title(r'Coil %d\\$s_{%d}(r)$' % (c+1, c+1), fontsize=10)
    ax.axis('off')

# Add arrow from image+coils to modulated images
fig.text(0.07, 0.82, r'$\times$', fontsize=24, ha='center')

# === Row 1: Per-coil images (s_i * x) ===
for c in range(n_coils):
    ax = plt.subplot2grid((5, 6), (1, c + 1), rowspan=1, colspan=1)
    ax.imshow(coil_images[c], cmap='gray', origin='lower')
    ax.set_title(r'$s_{%d} \odot x$' % (c+1), fontsize=10)
    ax.axis('off')

# Arrow from coil images to k-space
fig.text(0.38, 0.68, 'FFT', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))

# === Row 2: K-space (full) ===
for c in range(n_coils):
    ax = plt.subplot2grid((5, 6), (2, c + 1), rowspan=1, colspan=1)
    ax.imshow(np.log(np.abs(kspaces[c]) + 1e-6), cmap='gray', origin='lower')
    ax.set_title(r'K-space $\mathcal{F}$', fontsize=10)
    ax.axis('off')

# Arrow from full k-space to undersampled + noisy
fig.text(0.38, 0.52, 'Undersample\n+ noise', fontsize=10, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2))

# === Row 3: Measured data (undersampled k-space + noise) ===
for c in range(n_coils):
    ax = plt.subplot2grid((5, 6), (3, c + 1), rowspan=1, colspan=1)
    disp = np.log(np.abs(noisy_kspaces[c]) + 1e-6)
    ax.imshow(disp, cmap='gray', origin='lower')
    ax.set_title(r'Measured $y_{%d}$' % (c+1) + '\n' + r'$M\mathcal{F}\!+\!\eta$', fontsize=9)
    ax.axis('off')

# === Bottom: Inverse problem box ===
ax_inv = plt.subplot2grid((5, 6), (4, 1), rowspan=1, colspan=4)
ax_inv.text(0.5, 0.7, 'SENSE Inverse Problem', fontsize=14, fontweight='bold',
            ha='center', transform=ax_inv.transAxes)
ax_inv.text(0.5, 0.35,
            r'$\min_x \; \sum_{i=1}^C \|M \mathcal{F} S_i x - y_i\|_2^2$',
            fontsize=16, ha='center', transform=ax_inv.transAxes)
ax_inv.text(0.5, 0.1, 'Solve with Conjugate Gradient',
            fontsize=11, ha='center', style='italic',
            transform=ax_inv.transAxes)
ax_inv.axis('off')
rect = FancyBboxPatch((0.05, 0.0), 0.9, 1.0,
                      boxstyle="round,pad=0.15",
                      facecolor='#2ecc71', edgecolor='#27ae60',
                      linewidth=2, alpha=0.25, transform=ax_inv.transAxes)
ax_inv.add_patch(rect)

# === Annotations ===
# Show the undersampling mask (what was acquired vs skipped)
ax_mask = plt.subplot2grid((5, 6), (2, 0), rowspan=1, colspan=1)
ax_mask.imshow(mask, cmap='gray', origin='lower')
ax_mask.set_title(r'Mask $M$\\(1=acquired)', fontsize=10)
ax_mask.axis('off')

# Big overarching title
fig.suptitle(
    r'Multi-Coil MRI Forward Model: $y_i = M\,\mathcal{F}\,\{(S_i \odot x)\} + \eta_i$',
    fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

outpath = os.path.join(fig_dir, 'mri_acquisition_model.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Figure saved to {outpath}")
plt.show()
