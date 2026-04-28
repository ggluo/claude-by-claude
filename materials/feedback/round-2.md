# Professor's Review — Round 2

## Summary
Excellent progress. Nearly all Round 1 issues are resolved: the LaTeX compiles cleanly (zero errors), all figures generate without crashes, the semiconvergence plot is added, naive SoS baseline is included, and the premature CG reveal on the "Geometric Intuition" frame is fixed with a clean contour-only plot. The SENSE formulation and CG implementation remain correct. Remaining issues are mostly about fine-tuning the pedagogical sequencing and visual presentation for undergraduate clarity.

---

## Slides (LaTeX Beamer)

### Strengths
- LaTeX compiles with **zero errors** on pdflatex — all figures are found, all commands resolve correctly.
- The new `paraboloid_contours.pdf` on frame "Geometric Intuition" is exactly right: students see the quadratic bowl and contours *before* any optimization algorithms are introduced. This fixes the worst sequencing problem from Round 1.
- The A-norm is now defined on the same frame where it's first used ("Why Steepest Descent is Slow", line 286: `\|v\|_A = \sqrt{v^T A v}`). Good.
- The Tikhonov regularization discussion on frame "SENSE Normal Equations for CG" (lines 633-641) correctly explains *why* regularization is needed for CG: positive definiteness. The bias-variance mention is appropriate.
- The "Further Reading and Code" frame connects students directly to the companion Python code — excellent for self-study.
- Overall slide count (49 pages) is appropriate for a full lecture.

### Issues to Fix

- [ ] **Important — CG trajectories still shown before CG is introduced** (frame "Why Steepest Descent is Slow", line 275): This frame displays `steepest_descent_vs_cg.pdf`, which plots *both* SD (zigzag, red) and CG (2-step direct path, blue) trajectories. CG is not formally introduced until Section 4 (frame 13, line 320). An undergraduate seeing this figure will wonder "what is that blue path?" and the lecture doesn't answer until several frames later. Two options: (a) generate a separate figure showing *only* SD's zigzag trajectory on the contours (no CG), or (b) add a brief "sneak preview" label on the CG trajectory like "(we'll build this next!)" and add a `\pause` so it appears only after discussing SD's problem. Option (a) is cleaner pedagogically.

- [ ] **Important — "The Zigzag Problem Visualized" frame title doesn't match content** (line 301): This frame displays `sd_vs_cg_error.pdf`, which is a semilogy plot of convergence curves (error vs iteration), not a visualization of zigzag paths. The zigzag visualization is in the *previous* frame. Rename this frame to something like "Convergence Speed: SD vs CG" or "Quantifying the Zigzag Cost" so the title matches what students actually see.

- [ ] **Minor — Overfull vbox warnings on multiple frames**: The log shows ~30 overfull vbox warnings, mostly from frames with large algorithm blocks or figures. Specific offenders:
  - Frame "Steepest Descent: The Natural First Attempt" (line 94): the `algorithm` block is too tall. Reduce `\small` to `\footnotesize` or set `\begin{algorithm}[H]` with a scale option.
  - Frame "CG Algorithm" (line 405): same issue — the algorithm overflows the frame. Reduce font or trim comments.
  - Frames with large figures (lines 201, 315, 475): figures are slightly too tall. Reduce `width=` by 5-10%.
  - These are cosmetic but distracting in presentation mode (content gets clipped at the bottom). A quick `\resizebox` or slightly smaller font resolves them all.

- [ ] **Minor — Missing coil sensitivity visualization on multi-coil slides**: Frame "Multi-Coil Acquisition: The Key to Acceleration" (line 571) shows `mri_acquisition_model.pdf`, which is a process-flow diagram. This is useful but doesn't show what coil sensitivity maps actually *look like*. Consider also including a small inset or a separate thumbnail of the actual sensitivity maps (e.g., from `coil_sensitivity_maps.pdf`) so students can connect the abstract "sensitivity profile" concept to a concrete image. This is the first time undergraduates encounter coil sensitivities — a picture is worth a thousand words.

- [ ] **Minor — Frame "Least Squares and the Normal Equations" uses `\onslide` inconsistently**: The equations on lines 215-219 use a mix of `\pause` (line 212) and `\onslide` (lines 217-218) for progressive disclosure. The two `\pause` commands on lines 214-215 (i.e., `\pause\pause`) result in two sequential reveals, which is correct but fragile — if slide order changes, the count breaks. Replace `\pause\pause` with explicit `\onslide<3->` and `\onslide<4->` for robustness, or use numbered `\pause` throughout.

---

## Illustration Code

### Strengths
- All three scripts run without errors and produce clean vector PDFs.
- The `plt.setp` fix for stem plot colors in `eigenvalue_illustration.py` is correct and works.
- Raw string literals (`r'...'`) are used throughout — no more Python escape sequence warnings.
- The new `paraboloid_contours.pdf` generation at the end of `steepest_descent_vs_cg.py` is clean and well-annotated.
- The CG and SD implementations are correct and match the lecture slides precisely.

### Issues to Fix

- [ ] **Important — `eigenvalue_illustration.py` compares residual norm to A-norm error bound** (lines 112, 148-153): The theoretical CG bound `2((sqrt(κ)-1)/(sqrt(κ)+1))^k` bounds the **A-norm error** `\|x_k - x^*\|_A`, not the residual norm `\|r_k\|`. Your plot draws this bound as a gray dashed line overlaid on a semilogy plot of *relative residuals* `\|r_k\|/\|r_0\|`. These are different quantities — the bound doesn't apply to residuals. For the pedagogical purpose of showing "CG is faster than the bound," this is a minor inaccuracy, but a sharp undergraduate might notice. Fix: either (a) compute and plot `\|x_k - x^*\|_A` (you have `A1`, `A2`, and `x^*` = `np.linalg.solve(A1, b1)`), or (b) add an explanatory comment that the bound is for A-norm error and the residual comparison is illustrative only. Option (a) is better for rigor.

- [ ] **Minor — `mri_acquisition_model.py` simulation and plotting still interleaved** (lines 71-189): The phantom generation, k-space computation, and diagram creation are in one flat script with minimal separation. For a student who wants to understand *just* the diagram logic, the simulation code is noise. Split into clearly marked sections (the `# %%` cells help but the overall structure is still flat), or move the data-generation functions (`shepp_logan_small`, `generate_sensitivity_maps`) to a shared utility module.

- [ ] **Minor — `steepest_descent_vs_cg.py` contour labels show shifted values** (lines 130-136, 221-222): The `Z_shifted = Z - Z.min() + eps` transform shifts the function values before taking logs. The contour labels (`clabel` on line 222) display these shifted values, which don't correspond to actual `f(x)` values. Since the contours are there for geometric intuition (shape, not exact values), this isn't critical, but labeling with true function values (by computing levels directly on `Z`) would be cleaner. The simplest fix: use `np.linspace(Z.min(), Z.max(), 20)` for linear levels instead of log-spacing on shifted data, which is also more intuitive for undergraduates.

- [ ] **Minor — `mri_acquisition_model.py` figure layout could be clearer** (lines 96-190): The 6×6 `subplot2grid` layout is clever but hard to follow from code alone. Row 4 and Row 5 both show undersampled k-space — are they meant to be different? Row 4 is labeled "M · F (R=2)" and Row 5 is "Measured y_i", but they display the same data (the undersampled k-space). Adding simulated noise in Row 5 would make the distinction meaningful. Alternatively, collapse rows 4 and 5 into one row labeled "Undersampled k-space (measured)" to reduce visual repetition.

---

## MRI Demo Code

### Strengths
- The full pipeline runs successfully: **CG converges in 35 iterations**, reducing NRMSE from ~0.77 (zero-filled) to **0.029** (CG SENSE). The reconstruction quality is excellent.
- The semiconvergence plot (`mri_cg_convergence.pdf`) now shows both residual and true error on the same axes, with the minimum-error iteration highlighted — this directly supports the teaching point and is exactly what Round 1 asked for.
- The naive SoS reconstruction is a valuable third baseline — students can see three levels of quality (naive SoS → sensitivity-weighted ZF → CG SENSE).
- The CG implementation correctly handles complex-valued data with Hermitian inner products: `np.real(np.sum(np.conj(u) * v))`.
- The matrix-free `apply_sense_normal_operator` is clean, efficient, and uses FFTs correctly.
- The seeded RNG (`np.random.default_rng(seed)`) ensures reproducibility.
- `coil_sensitivity_maps.pdf` is now generated and listed in the README.

### Issues to Fix

- [ ] **Important — Zero-filled NRMSE (0.77) is worse than naive SoS (0.58) — this deserves explanation**: Students running the demo will see that the "smarter" sensitivity-weighted zero-filled reconstruction produces *worse* NRMSE than the completely naive sum-of-squares. This is counterintuitive. The reason: sensitivity-weighted combination of undersampled data can amplify aliasing patterns at locations where the sum-of-squares normalization is small, producing bright artifacts. This is actually a great teaching moment about inverse problems — "using more physics doesn't always guarantee better results if you don't fully invert the model." Add a brief explanatory comment in the code (around line 489) and mention it in the README's "Key Observations" section.

- [ ] **Minor — `conjugate_gradient_sense` uses `np.abs(x)` for error computation but x is complex** (line 413): `np.linalg.norm(np.abs(x) - x_true)` compares the *magnitude* image to the real-valued phantom. Since the phantom is real and nonnegative, this is reasonable, but it discards phase information in the reconstruction. If the reconstructed image has nonzero phase (it does, from the complex sensitivity maps), taking `np.abs(x)` is correct for display but the error computation is measuring magnitude error, not complex error. Add a comment noting that magnitude error is the clinically relevant metric. Alternatively, modify the ground truth to include phase (e.g., `phantom_complex = phantom * np.exp(1j * slow_phase_variation)`) for a more complete error analysis.

- [ ] **Minor — `lambda_reg` default is 0.0 but may cause CG breakdown** (line 336): The parameter `lambda_reg` defaults to 0.0, which means no regularization. If a student calls `conjugate_gradient_sense(..., lambda_reg=0.0)` with aggressive undersampling (R ≥ 4), the normal equations matrix may become singular and CG will break down (division by zero or near-zero `pAp`). Since the slides and README emphasize that regularization ensures positive definiteness, the default should be a small positive value (e.g., `lambda_reg=1e-3`) to match the lecture's teaching. The current main-block call correctly passes `lambda_reg=1e-3`, but the function default is misleading.

- [ ] **Minor — `conjugate_gradient_sense` is doing too many things** (lines 334-433): This single function computes the RHS, initializes CG, runs the iteration loop, tracks both residual and true error, and handles early termination. For a teaching demo, consider extracting a pure `conjugate_gradient(A_operator, b, x0, max_iter, tol)` function that takes an abstract linear operator, then wrap it with the SENSE-specific setup. This would make the CG implementation reusable and help students separate "what CG does" from "how SENSE uses CG." The illustration code already has a separate `conjugate_gradient` function — the MRI demo should reuse the same pattern.

- [ ] **Minor — Figure layout wastes space on reconstruction comparison** (lines 514-581): The 2×5 subplot grid has the original phantom and SoS reconstruction placed at `(0,0)` and `(1,0)`, while coil sensitivity magnitudes occupy `(0,1)` through `(1,1)`. The remaining 6 panels show mask, reconstructions, and error map. The layout is functional but has uneven spacing because the top-left panel (`(0,0)`) is the phantom and the bottom-left (`(1,0)`) is SoS, with half the top row dominated by small coil sensitivity thumbnails. For slide inclusion (`mri_reconstruction_results.pdf`), the figure is quite wide (24×10 inches at 150 dpi) and may not display well when scaled down. Consider a 2×4 or 3×3 layout with more even sizing.

---

## Overall Guidance for Next Round

1. **Fix the premature CG reveal**: The `steepest_descent_vs_cg.pdf` figure on the "Why Steepest Descent is Slow" frame shows CG trajectories before CG is taught. Generate a separate SD-only trajectory figure for that frame, or restructure so CG is introduced before students see its trajectory.

2. **Align the eigenvalue bound with what is plotted**: The theoretical CG bound in `eigenvalue_illustration.py` is for A-norm error, but the plot shows residual norms. Compute and plot the actual A-norm error to make the comparison rigorous, or clearly label the plot as illustrative.

3. **Explain the counterintuitive zero-filled NRMSE result**: The sensitivity-weighted zero-filled reconstruction has worse NRMSE than naive SoS — this is a valuable teaching moment about inverse problems that should be explicitly discussed in the code comments and README.
