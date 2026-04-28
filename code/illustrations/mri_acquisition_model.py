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
# ## Create Block Diagram

# %%
# Generate a small Shepp-Logan-like phantom for illustration
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

# Coil sensitivity maps (simulated as Gaussian profiles)
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

# Generate a full coil-combined image (sum of S_i x)
def coil_combined_image(phantom, sensitivity_maps):
    return sum(s * phantom for s in sensitivity_maps)

N = 64
phantom = shepp_logan_small(N)
n_coils = 4
sens_maps = generate_sensitivity_maps(N, n_coils)

# Per-coil k-space (full, before undersampling)
coil_images = [s * phantom for s in sens_maps]
kspaces = [np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) for img in coil_images]

# Undersampling mask (R=2 Cartesian, skip every other line)
mask = np.zeros((N, N))
mask[::2, :] = 1  # keep every 2nd line

undersampled_kspaces = [k * mask for k in kspaces]

# %% [markdown]
# ## Create the Diagram

# %%
fig = plt.figure(figsize=(18, 10))

# Define positions for the block diagram items
# Top row: the pipeline
# We'll use a mix of images and text

# Helper: add a colored box with text
def add_text_box(ax, x, y, w, h, text, color='#ecf0f1', fontsize=9, edgecolor='gray'):
    """Add a text box to the axis in data coordinates."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=edgecolor,
                          linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold')

def add_arrow(ax, x1, y1, x2, y2, color='black'):
    """Add an arrow from (x1, y1) to (x2, y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2, connectionstyle='arc3,rad=0'))

# === Row 1: Unknown image + coil sensitivities ===
ax_main = plt.subplot2grid((6, 6), (0, 0), rowspan=1, colspan=1)
ax_main.imshow(phantom, cmap='gray', origin='lower')
ax_main.set_title('Unknown Image\n$x$', fontsize=11, fontweight='bold')
ax_main.axis('off')

# Show sensitivity maps in a row
for c in range(n_coils):
    ax = plt.subplot2grid((6, 6), (0, c + 1), rowspan=1, colspan=1)
    ax.imshow(sens_maps[c], cmap='viridis', origin='lower')
    ax.set_title(f'Coil {c+1}\n$s_{c+1}(r)$', fontsize=10)
    ax.axis('off')

# Add arrow
fig.text(0.07, 0.74, '×', fontsize=24, ha='center')

# === Row 2: Per-coil images ===
for c in range(n_coils):
    ax = plt.subplot2grid((6, 6), (1, c + 1), rowspan=1, colspan=1)
    ax.imshow(coil_images[c], cmap='gray', origin='lower')
    ax.set_title(f'$s_{c+1} \odot x$', fontsize=10)
    ax.axis('off')

# Arrow from sensitivities+image to coil images
fig.text(0.38, 0.62, 'FFT', fontsize=14, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))

# === Row 3: K-space (full) ===
for c in range(n_coils):
    ax = plt.subplot2grid((6, 6), (2, c + 1), rowspan=1, colspan=1)
    ax.imshow(np.log(np.abs(kspaces[c]) + 1e-6), cmap='gray', origin='lower')
    ax.set_title('K-space $\\mathcal{F}$', fontsize=10)
    ax.axis('off')

# Arrow
fig.text(0.38, 0.48, 'Undersample', fontsize=12, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2))

# === Row 4: Undersampled k-space ===
for c in range(n_coils):
    ax = plt.subplot2grid((6, 6), (3, c + 1), rowspan=1, colspan=1)
    # Show mask overlaid
    disp = np.log(np.abs(undersampled_kspaces[c]) + 1e-6)
    ax.imshow(disp, cmap='gray', origin='lower')
    ax.set_title(f'$M \\cdot \\mathcal{{F}}$ (R=2)', fontsize=10)
    ax.axis('off')

# Arrow
fig.text(0.38, 0.34, '+ noise', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='#95a5a6', alpha=0.3))

# === Row 5: Measurements ===
for c in range(n_coils):
    ax = plt.subplot2grid((6, 6), (4, c + 1), rowspan=1, colspan=1)
    k_disp = np.log(np.abs(undersampled_kspaces[c]) + 1e-6)
    ax.imshow(k_disp, cmap='gray', origin='lower')
    ax.set_title(f'Measured $y_{c+1}$', fontsize=10)
    ax.axis('off')

# === Bottom: Inverse problem box ===
ax_inv = plt.subplot2grid((6, 6), (5, 1), rowspan=1, colspan=4)
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
# Add a note about the undersampling mask
ax_mask = plt.subplot2grid((6, 6), (2, 0), rowspan=1, colspan=1)
ax_mask.imshow(mask, cmap='gray', origin='lower')
ax_mask.set_title('Mask $M$\n(1=acquired)', fontsize=10)
ax_mask.axis('off')

# Big overarching title
fig.suptitle(
    'Multi-Coil MRI Forward Model: $y_i = M\\,\\mathcal{F}\\,\\{(S_i \\odot x)\\} + \\eta_i$',
    fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()

outpath = os.path.join(fig_dir, 'mri_acquisition_model.pdf')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Figure saved to {outpath}")
plt.show()
