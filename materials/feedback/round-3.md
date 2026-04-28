# Professor's Review — Round 3

## Summary
This is strong work approaching lecture-ready quality. All Round 2 critical issues have been addressed: the SD-only trajectory figure is used before CG is introduced, the A-norm error tracking in eigenvalue illustration is corrected, the generic CG solver is properly extracted from the SENSE wrapper, coil sensitivity maps are shown on the multi-coil slides, and the lambda_reg default is fixed. The LaTeX compiles cleanly (zero errors), all figures generate correctly, and the pedagogical sequencing is now well-structured. The remaining issues are subtle technical inaccuracies and visual polish items that separate "good" from "excellent" lecture materials.

---

## Slides (LaTeX Beamer)

### Strengths
- LaTeX compiles with zero errors (only 2 minor overfull vbox warnings on Summary and Further Reading frames, both cosmetic).
- The SD-only trajectory (`sd_zigzag_only.pdf`) on frame "Why Steepest Descent is Slow" is the correct sequencing fix — students now see only the zigzag before CG is introduced.
- The convergence bounds (SD and CG) on frame "Convergence Rate" are mathematically correct, including the factor of 2 in the CG bound and the correct per-iteration geometric factors in the comparison table.
- The SENSE normal equations formulation is correct: `(Σ_i S_i^H F^H M F S_i) x = Σ_i S_i^H F^H y_i`, with the implicit-operator application clearly explained.
- The CG algorithm pseudocode on frame "CG Algorithm" is correct (Hestenes & Stiefel, 1952), with proper initialization, α, β, and update steps.
- Good use of Beamer progressive disclosure (`\pause`, `\onslide`) throughout.
- The "Further Reading and Code" frame is excellent — it directly connects students to the companion Python code and to Shewchuk's classic CG tutorial.
- The coil sensitivity maps are now shown on frame "Multi-Coil Acquisition" (line 600) — students can now see what sensitivity profiles actually look like.
- All required sections are present and in logical order.

### Issues to Fix

- [ ] **Important — Krylov subspace definition uses `b` instead of `r_0`** (frame "Why CG Works: Krylov Subspace Interpretation", line 418): The Krylov subspace is defined as `K_k(A, b) = span{b, Ab, A^2b, ...}`. This is only correct when `x_0 = 0` (since `r_0 = b - A x_0 = b`). But the algorithm slide explicitly says "x_0 = 0 (**or any initial guess**)" — if a student picks `x_0 ≠ 0`, the Krylov subspace should be `K_k(A, r_0) = span{r_0, A r_0, ..., A^{k-1} r_0}`. Fix by changing to `K_k(A, r_0)` and noting that `r_0 = b - A x_0`, with the simplification to `b` when `x_0 = 0`. This is the fundamental CG theory and must be precise.

- [ ] **Important — k-space sampling diagram contradicts text description** (frame "MRI in One Slide", lines 556-567): The tikzpicture shows randomly scattered dots in k-space, but the text and the actual SENSE implementation use **Cartesian undersampling** (skipping entire phase-encoding lines, every R-th row). A student will see the random dots and think "compressed sensing with random sampling," when the SENSE method being taught uses regular line-skipping. Replace the scattered dots with horizontal lines showing a Cartesian pattern (e.g., every other row filled in solid, alternating with empty rows). This is a visual truth-in-advertising issue — the diagram should match what the algorithm actually does.

- [ ] **Minor — SENSE normal equations RHS drops M without explanation** (frame "SENSE Normal Equations for CG", line 640): The RHS is written as `Σ_i S_i^H F^H y_i`, but the correct gradient gives `Σ_i S_i^H F^H M y_i`. The `M` disappears because the code zero-fills `y_i` to full size (so `M y_i = y_i`). A careful student taking the gradient themselves will get `S_i^H F^H M^T y_i` and wonder where the `M` went. Add a brief note: "Since the measured data `y_i` is zero-padded to full k-space size, `M y_i = y_i`." Or use `M^T y_i` in the formula for rigor and note the simplification. Either way, the derivation should be transparent.

- [ ] **Minor — Overfull vbox on Summary frame** (line 741, 57pt overflow): The Summary frame has 7 enumerated items plus a block — it physically overflows the slide by 57pt. Split into two frames or trim the text. The last point about regularization can be shortened to one line.

- [ ] **Minor — Overfull vbox on Further Reading frame** (line 769, 5pt overflow): Slight overflow. Reduce font size slightly or trim the code listing to two items.

- [ ] **Minor — Frame "Least Squares and the Normal Equations" has a fragile `\pause`+`\onslide` mix** (lines 212-241): The frame uses `\pause` on line 212 and then `\onslide<3->`, `\onslide<4->`, etc. This works correctly (the `\pause` increments the beamer counter), but if anyone reorders the slide content or adds/removes a `\pause` before the align* environment, the overlay numbers will silently break. Replace `\onslide<N->` with `\pause` throughout for robustness, or use numbered `\pause` consistently. This is a maintainability concern, not a correctness bug.

- [ ] **Minor — No explicit definition of "well-conditioned" vs "ill-conditioned"**: The term "ill-conditioned" appears on frame "Two Kinds of Linear Systems" (line 149) without definition. Students with basic linear algebra may not know what condition number means or what values are considered "large." The condition number is formally defined on frame "Geometric Intuition" (line 199) but not framed as "κ ≫ 1 = ill-conditioned, κ ≈ 1 = well-conditioned." Add a one-line threshold explanation.

---

## Illustration Code

### Strengths
- All three scripts generate correct, well-labeled vector PDFs that match the lecture content.
- The A-norm error is now correctly computed and plotted in `eigenvalue_illustration.py` — the theoretical CG bound is compared against `‖x_k - x*‖_A`, which it actually bounds.
- The `steepest_descent_vs_cg.py` script now generates four distinct figures (comparison, error plot, contour-only, SD-only) — exactly the set needed for the lecture's progressive disclosure sequence.
- Both CG and SD implementations are mathematically correct and match the pseudocode in the slides.
- All plots use consistent color schemes (SD = red, CG = blue, optimum = green, start = black) — good visual consistency.
- The `mri_acquisition_model.py` pipeline diagram correctly shows the forward model flow: image → sensitivities → per-coil images → FFT → mask → measured data → SENSE inverse problem.
- Raw strings (`r'...'`) are used consistently for LaTeX in matplotlib labels — no escape sequence warnings.

### Issues to Fix

- [ ] **Important — eigenvalue_illustration.py stem plot titles show spurious backslash** (lines 159, 172): The titles use `r'Uniform Eigenvalue Distribution\\$(\kappa = 100)$'`. In a raw string, `\\` is two literal characters (backslash + backslash), which matplotlib renders as a visible backslash: "Uniform Eigenvalue Distribution\\(κ = 100)". The backslash before `$` is a visual artifact. Fix by using a regular (non-raw) string: `'Uniform Eigenvalue Distribution\n' + r'$(\kappa = 100)$'`, or use `\n` in the raw string pattern `r'Uniform Eigenvalue Distribution\n$(\kappa = 100)$'` (since `\n` in a raw string is backslash-n, which matplotlib still interprets as newline in its text renderer — but this relies on matplotlib behavior, not Python string escaping). The cleanest fix: use `'\n'.join([...])` with separate strings.

- [ ] **Minor — SD-only and paraboloid contour figures lack contour labels** (lines 216-294 vs 140-177): The `steepest_descent_vs_cg.py` comparison figure uses `ax.contour()` without labels, while the paraboloid-only figure adds `ax.clabel()` for readability. But the SD-only figure (`sd_zigzag_only.pdf`) does NOT add contour labels. Since this figure appears in the lecture before CG is introduced, and students are just learning to interpret the contour geometry, having labeled contour levels would help them connect the visualization to the math. Add `ax.clabel()` to the SD-only figure as well.

- [ ] **Minor — mri_acquisition_model.py uses a different, simpler phantom than the MRI demo** (line 35-52 vs reconstruct.py line 44-77): The illustration script defines `shepp_logan_small()` with 6 ellipses, while the MRI demo uses `shepp_logan_phantom()` with 10 ellipses (a proper Shepp-Logan). This inconsistency means the phantom in the acquisition model diagram doesn't match the phantom in the reconstruction results. A student comparing figures might notice. Either reconcile the phantoms or add a comment noting the simplification for the diagram.

- [ ] **Minor — FFT normalization convention not documented** (mri_acquisition_model.py lines 89, earnest_descent_vs_cg.py has no FFT): In `mri_acquisition_model.py`, the FFT uses `fftshift(fft2(ifftshift(...)))` without mentioning the normalization. The numpy FFT is not unitary by default (forward is unscaled, inverse has 1/N² factor), but since the product `F^H F = I` holds exactly, the SENSE operator is correct. This deserves a brief comment for students who might wonder about the `fftshift`/`ifftshift` wrapping pattern and why `1/N²` normalization isn't explicitly applied.

---

## MRI Demo Code

### Strengths
- The full pipeline runs correctly: phantom → coil sensitivities → undersampled acquisition → baseline reconstructions → CG SENSE → visualization.
- All three reconstructions (naive SoS, sensitivity-weighted ZF, CG SENSE) are implemented and compared — this three-way comparison is pedagogically excellent.
- The CG implementation correctly handles complex-valued data with Hermitian inner products (`np.real(np.sum(np.conj(u) * v))`).
- The generic `conjugate_gradient()` function is now extracted from `conjugate_gradient_sense()` — well done. This separation helps students understand CG independently from the SENSE application.
- The SENSE normal operator is applied implicitly via FFTs without ever forming the matrix — correct and efficient.
- The `lambda_reg` default is now `1e-3` (not `0.0`) — matches the lecture's teaching about needing positive definiteness.
- Error computation uses magnitude images with a clear pedagogical comment (lines 354-359, 379-381) explaining why magnitude is the clinically relevant metric and why complex comparison would overstate error.
- The seeded RNG ensures reproducibility.
- The "Why Zero-Filled Can Be Worse Than Naive SoS" explanation in the README (lines 76-92) is outstanding — this counterintuitive result is turned into a teaching moment with clear physical reasoning about sensitivity-weighting amplification.
- The semiconvergence plot (`mri_cg_convergence.pdf`) with the minimum-error iteration highlighted is pedagogically perfect.

### Issues to Fix

- [ ] **Minor — Runtime teaching note duplicates code comment** (lines 541-548 in code vs runtime print): The detailed explanation of why ZF NRMSE can exceed naive SoS NRMSE appears as both a formatted code comment and a runtime print statement. The runtime version has an f-string bug — `{error_zf:.4f}` and `{error_sos:.4f}` appear as literal text in the comment block (lines 541-542) because they're inside a regular string, not an f-string, but they're in a comment so it doesn't matter. However, the note is long enough that it clutters both the source code and the runtime output. Keep the detailed version in the README, trim the code comment to a one-line pointer, and keep the runtime print concise. The pedagogical content is excellent — just de-duplicate its placement.

- [ ] **Minor — CG tolerance (`tol=1e-4`) is looser than the zero-filled NRMSE delta**: The CG reconstruction uses `tol=1e-4` (line 556), but the ZF NRMSE is ~0.77 and CG achieves ~0.03. The tolerance of 1e-4 on relative residual is reasonable, but the CG often hits `max_iter` before the tolerance is reached for R=3. With `max_cg_iter=60` and `tol=1e-4`, CG converges in about 30-40 iterations (residual plateaus). This is fine for the demo but consider adding a comment that the tolerance should be tightened for higher-acceleration scenarios, or that hitting max_iter is expected behavior for ill-conditioned systems.

- [ ] **Minor — The `pAp < 1e-16` guard in CG** (line 395): This guard against near-zero `p^H A p` is correct but the message says "stopping" while the code uses `break`. In the MRI context, this would only trigger if the normal equations become singular (e.g., no regularization, high R). Since `lambda_reg=1e-3` is the default, this guard is now a safety net rather than expected behavior. Add a comment explaining that this catches the singular-case failure mode, or change the message to note that regularization likely needs to be increased.

---

## Generated Figures

### Strengths
- All 9 figures are present in `slides/figures/` as vector PDFs.
- The `mri_reconstruction_results.pdf` comparison figure shows clear improvement from Naive SoS → ZF → CG.
- The `mri_cg_convergence.pdf` plot effectively illustrates semiconvergence with the minimum-error iteration highlighted.
- The `coil_sensitivity_maps.pdf` grid shows all 6 coil sensitivity magnitudes clearly.
- The `eigenvalue_illustration.pdf` four-panel layout is well-organized and the clustering effect is visually apparent.

### Issues to Fix
- See the "K-space diagram" issue above (random dots vs Cartesian lines) — this is a figure content issue that affects `mri_acquisition_model.pdf` and the inline tikzpicture on "MRI in One Slide."
- The `eigenvalue_illustration.pdf` stem plot titles have the spurious backslash issue noted above.

---

## Overall Guidance for Next Round

1. **Fix the Krylov subspace definition**: Change `K_k(A, b)` to `K_k(A, r_0)` with the simplification noted for `x_0 = 0`. This is a core theoretical concept that must be precise — students who take numerical linear algebra later will be confused if they learned `K_k(A, b)` as the definition.

2. **Fix the k-space sampling diagrams**: Both the tikzpicture on "MRI in One Slide" and potentially the `mri_acquisition_model.pdf` diagram should show Cartesian line-skipping, not random dots. The visual must match what the SENSE method actually does (regular undersampling of phase-encoding lines).

3. **Clean up the stem plot titles in `eigenvalue_illustration.py`**: The spurious backslash in the figure titles looks sloppy. It's a one-line fix that significantly improves figure quality.
