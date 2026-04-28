# Multi-Coil MRI Reconstruction Demo

## Overview

This demo shows how to reconstruct an MR image from **undersampled multi-coil k-space data**
using the **Conjugate Gradient (CG) algorithm**. It implements the **SENSE (SENSitivity Encoding)**
reconstruction model from scratch.

### The SENSE Model

In multi-coil MRI, an array of receiver coils simultaneously acquires the MR signal.
Each coil $i$ has a spatially-varying sensitivity profile $s_i(\mathbf{r})$.
The signal measured by coil $i$ in k-space is:

$$y_i(\mathbf{k}) = \int s_i(\mathbf{r})\, x(\mathbf{r})\, e^{-j2\pi\mathbf{k}\cdot\mathbf{r}} \, d\mathbf{r} + \text{noise}$$

In matrix-vector form:

$$y_i = M F S_i x + \eta_i$$

where:
- $x \in \mathbb{C}^N$ is the unknown image (flattened),
- $S_i = \operatorname{diag}(s_i)$ applies coil sensitivity $i$,
- $F$ is the 2D Fourier transform matrix,
- $M$ is the Cartesian undersampling mask (skips phase-encoding lines),
- $y_i$ is the acquired k-space data for coil $i$,
- $\eta_i$ is complex Gaussian noise.

The SENSE reconstruction solves the least-squares problem:

$$\min_x \sum_{i=1}^C \|M F S_i x - y_i\|_2^2$$

This leads to the **normal equations**:

$$\left(\sum_{i=1}^C S_i^H F^H M F S_i\right) x \;=\; \sum_{i=1}^C S_i^H F^H y_i$$

which is a linear system $A x = b$ with $A$ Hermitian positive semi-definite.
Tikhonov regularization ($+\lambda I$) is added to ensure positive definiteness,
making the system suitable for **Conjugate Gradient**.

## Installation

```bash
pip install -r ../../requirements.txt
```

## Running the Demo

```bash
python reconstruct.py
```

This will:
1. Generate a Shepp-Logan phantom
2. Create 6 simulated coil sensitivity maps
3. Simulate undersampled k-space acquisition (R=3 acceleration)
4. Compute three reconstructions:
   - Naive sum-of-squares (pure IFFT, no sensitivity info)
   - Zero-filled with sensitivity-weighted coil combination
   - CG (SENSE) — full least-squares solution with Tikhonov regularization
5. Display and save the results

## Output Files

- `../../slides/figures/mri_reconstruction_results.pdf` — Side-by-side comparison of all reconstructions
- `../../slides/figures/mri_cg_convergence.pdf` — Residual and true error vs iteration (semiconvergence plot)
- `../../slides/figures/coil_sensitivity_maps.pdf` — Grid of all coil sensitivity map magnitudes

## Key Observations

- **Naive SoS reconstruction** shows both aliasing artifacts and coil shading,
  since it uses no sensitivity information at all.
- **Zero-filled reconstruction** (with sensitivity-weighted combination) reduces
  shading but still shows aliasing artifacts from undersampling.

### Why Zero-Filled Can Be Worse Than Naive SoS

A **counterintuitive but instructive** result: the sensitivity-weighted
zero-filled reconstruction often has *worse* NRMSE than the completely naive
sum-of-squares (e.g., 0.77 vs 0.58). This happens because:

1. The sensitivity-weighted combination divides each coil image by a
   sum-of-squares normalizer $\sum_i |s_i(\mathbf{r})|^2$,
2. In regions where this normalizer is small (e.g., far from all coils),
   the division amplifies aliasing artifacts from the undersampled k-space,
3. The naive SoS avoids this amplification — it simply adds squared magnitudes
   without any spatial normalization, so aliasing isn't concentrated at
   low-sensitivity locations.

This is a great teaching moment about inverse problems: **"Using more physics
doesn't always guarantee better results if you don't fully invert the model."**
Only CG, which properly solves the full SENSE linear system, correctly separates
the aliased signals without amplifying noise at low-sensitivity regions.

- **CG reconstruction** removes aliasing by using coil sensitivity information to
  separate overlapping signals (the SENSE principle).
- CG converges quickly (typically 10-20 iterations) because the normal equations
  matrix is well-conditioned when the coil sensitivities are sufficiently distinct.

### Semiconvergence

An important phenomenon in inverse problems is **semiconvergence**:

- The **residual** $\|r_k\|_2$ always decreases monotonically in CG.
- The **true reconstruction error** $\| |x_k| - x_{\text{true}} \|_2$ may reach
  a minimum before the residual is fully converged, then start to increase as
  the algorithm begins to fit the noise.
- This means we should stop iterating **before** the residual gets too small!
- Regularization ($\lambda > 0$) helps by biasing the solution away from
  noise-fitting and making the system strictly positive definite.

The convergence plot (`mri_cg_convergence.pdf`) shows both the residual and the
true error on the same axes to illustrate this effect.

### CG Implementation Note

The CG solver is implemented as a generic `conjugate_gradient(A_operator, b, ...)`
function that takes an abstract linear operator. The SENSE-specific setup
(computing the RHS via coil sensitivities, constructing the normal operator)
is separated into `conjugate_gradient_sense()`, which delegates to the generic
CG solver. This separation helps students understand:
1. What CG does (the generic iterative algorithm)
2. How SENSE uses CG (the problem-specific setup)

## References

- Pruessmann, K. P., et al. "SENSE: Sensitivity Encoding for Fast MRI."
  *Magnetic Resonance in Medicine*, 42(5):952-962, 1999.
- Hestenes, M. R. and Stiefel, E. "Methods of Conjugate Gradients for Solving Linear Systems."
  *Journal of Research of the National Bureau of Standards*, 49(6):409-436, 1952.
