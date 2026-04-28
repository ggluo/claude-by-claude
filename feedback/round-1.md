# Professor's Review — Round 1

## Summary
A solid first draft that gets the big ideas right: the CG algorithm is correctly stated, the SENSE formulation is accurate, and the code generally works. However, there are several **critical compilation errors** (LaTeX won't fully build without fixes), a **crashed figure script**, and a few **pedagogical sequencing issues** that would confuse undergraduates. The MRI demo is the strongest component — fix the issues below and this will be an excellent lecture.

---

## Slides (LaTeX Beamer)

### Strengths
- Good overall structure: motivation → optimization → SD → CG → convergence → MRI → summary. Logical flow.
- The SENSE model is correctly formulated with the right normal equations, and the connection between CG and multi-coil MRI is clearly explained.
- Effective use of Beamer features (`\pause`, `block`, `alertblock`, `columns`) for progressive disclosure.
- Good selection of references (Hestenes & Stiefel, Pruessmann, Trefethen & Bau, Shewchuk).
- The "Three equivalent views of CG" slide (Krylov subspace) is excellent — this is exactly the depth undergraduates need.

### Issues to Fix

- [ ] **Critical — Missing `\usetikzlibrary{positioning}`**: The TikZ code on frame "Two Kinds of Linear Systems" uses `left=1cm of A` syntax, which requires `\usetikzlibrary{positioning}`. Without it, pdflatex produces 8 PGF Math errors (`Unknown operator 'of'`). Add `\usetikzlibrary{positioning}` to the preamble.

- [ ] **Critical — Missing figures**: Frames reference `figures/eigenvalue_illustration.pdf`, `figures/mri_reconstruction_results.pdf`, and `figures/mri_cg_convergence.pdf`, but these are not in the repository. You must run `code/illustrations/eigenvalue_illustration.py` (after fixing its crash — see below) and `code/mri_demo/reconstruct.py` to generate them **before** compiling.

- [ ] **Important — Premature CG reveal**: Frame "Geometric Intuition" (the very first figure slide, right after "Linear Systems as Optimization") shows `steepest_descent_vs_cg.pdf`, which plots both SD and CG trajectories. At this point in the lecture, the student has never heard of CG — the algorithm isn't introduced until 4 sections later. This will confuse undergraduates. **Fix:** Either (a) replace the figure on this frame with a simple contour-only paraboloid plot that doesn't show any trajectories, or (b) move this frame to after CG is introduced and use a different geometric-intuition figure here.

- [ ] **Important — Same figure used twice**: Frames "Geometric Intuition" and "Why Steepest Descent is Slow" both display `steepest_descent_vs_cg.pdf`. If you keep the figure on only one of these frames (recommended: keep it on "Why Steepest Descent is Slow"), this issue resolves itself.

- [ ] **Important — A-norm used before definition**: Frame "Why Steepest Descent is Slow" writes the error bound using `\|x_k - x^*\|_A` (the A-norm / energy norm), but `\|·\|_A` is not defined until the later frame "Why CG Works: Krylov Subspace Interpretation". Define the A-norm before or at least on this frame: `\|v\|_A = \sqrt{v^T A v}`.

- [ ] **Important — CG requires positive definiteness; SENSE is only semidefinite**: Frame "SENSE Normal Equations for CG" states "A_norm is Hermitian positive semidefinite — CG applies directly." Standard CG **requires** the matrix to be positive definite, not just semidefinite. When the nullspace of A_norm contains nonzero vectors (which happens with aggressive undersampling), CG will break down. Your own code correctly adds Tikhonov regularization (`lambda_reg=1e-3`) to make the system positive definite. The slide should explain this: mention that a small regularization term λI is added to ensure positive definiteness, and note the bias-variance tradeoff.

- [ ] **Important — Oversimplified convergence rate presentation**: Frame "Convergence Rate: The Condition Number Story" presents the CG bound and then frame "Beyond the Condition Number" shows the eigenvalue clustering insight. This is good, but a crucial practical point is missing: the bound is a worst-case bound, and CG's actual convergence is often much faster (which you hint at with eigenvalue clustering but should say more explicitly). Also, the steepest descent convergence bound uses `\|·\|_A` while the CG bound uses `\|·\|_A` too — make the comparison explicit: CG replaces κ with √κ, which is dramatically better.

- [ ] **Minor — Frame "Two Kinds of Linear Systems"**: The inverse problem diagram labels the unknown operator as `A^{-1}?`. This is slightly misleading since A may not be square/invertible (especially for MRI). Consider labeling it "A⁺?" or "pseudoinverse?" or simply stating "we need to invert the forward model."

- [ ] **Minor — Frame "Least Squares and the Normal Equations"**: The expansion writes `2 b^T A x` but should be `(Ax)^T b + b^T (Ax) = 2 b^T A x`. While mathematically equivalent for real vectors, the jump from `(Ax - b)^T(Ax - b)` to `x^T A^T A x - 2 b^T A x + b^T b` is worth one intermediate line for undergraduates.

- [ ] **Minor — Overfull hbox in algorithm blocks**: The algorithm pseudocode environments are likely producing overfull box warnings. Consider reducing font size or adjusting line breaks in the comments column.

- [ ] **Minor — Section ordering**: "Least Squares and the Normal Equations" appears in Section 2 ("Linear Systems as Optimization") before CG is introduced. This is fine, but the transition from normal equations back to "we can use CG!" doesn't land yet because CG hasn't been introduced. Consider moving this frame to right before the SENSE derivation (Section 6), where it would naturally motivate the CG application.

- [ ] **Minor — `\xopt` macro name**: You define `\xopt` as `x^*`. In later sections dealing with complex vectors (MRI), `x^*` could be read as the conjugate transpose. This never actually causes confusion because you're consistent, but consider using `x^{\star}` instead of `x^*` for the optimal solution.

---

## Illustration Code

### Strengths
- `steepest_descent_vs_cg.py` is excellent: clean implementations of both algorithms, clear side-by-side visualization, well-labeled plots. It runs without errors.
- `mri_acquisition_model.py` correctly shows the full pipeline from image to k-space measurements. The diagram is conceptually sound.
- Good use of notebook-style code (`# %%`) for readability.
- Correct CG formulas in both illustration files.

### Issues to Fix

- [ ] **Critical — `eigenvalue_illustration.py` crashes at runtime**: `ax.stem(lambda1, np.ones(n), linefmt='#e74c3c-', ...)` fails because `stem()` does not accept hex color codes in the format string (unlike `plot()`). Use `markerline, stemlines, baseline = ax.stem(...)` and set colors via `plt.setp(stemlines, color='#e74c3c')` and `plt.setp(markerline, markerfacecolor='#e74c3c')`. This script currently cannot generate `eigenvalue_illustration.pdf`, which blocks LaTeX compilation.

- [ ] **Important — Invalid Python escape sequences**: Strings containing `$\lambda$`, `$\kappa$`, `$\odot$` use backslashes that Python interprets as escape sequences (e.g., `\l`, `\k`, `\o`). Fix by making these raw strings: `r'$\lambda$'` instead of `'$\lambda$'`. Affected lines in both `eigenvalue_illustration.py` (lines 123, 132, 148, 168) and `mri_acquisition_model.py` (line 126). Current Python 3.14 issues `SyntaxWarning`; future versions will raise errors.

- [ ] **Minor — Misleading variable name in `eigenvalue_illustration.py`**: The `errors` list in `conjugate_gradient()` stores residual norms (`\|r_k\|`), not errors (`\|x_k - x^*\|`). The plot labels correctly say "Relative Residual" but the variable name is confusing for a student reader. Rename `errors` to `residuals` or `residual_norms`.

- [ ] **Minor — `steepest_descent_vs_cg.py` contour levels calculation**: Lines 129–130 use `np.log10(Z.min() - Z.min() + 1e-10)` which simplifies to `np.log10(1e-10) = -10` — the `Z.min() - Z.min()` term is always 0. Consider using `np.linspace(Z.min(), Z.max(), 20)` for linear spacing, or `np.logspace(np.log10(Z.min() + eps), np.log10(Z.max()), 20)` for log spacing.

- [ ] **Minor — `mri_acquisition_model.py` is dense**: The diagram-generation code is mixed with simulation code (phantom creation, k-space computation) in one monolithic script. For a student reader, separate the data generation from the plotting, or add more section-comment headers.

---

## MRI Demo Code

### Strengths
- This is the highlight of the submission. The SENSE+CGG pipeline is correctly implemented from scratch.
- The Shepp-Logan phantom is well-parameterized with rotating ellipses (correct handling of rotated coordinate frames).
- Coil sensitivity maps include both magnitude (Gaussian) and phase variation — realistic and pedagogically appropriate.
- The matrix-free `apply_sense_normal_operator` is correctly implemented with FFTs.
- The CG implementation correctly handles complex-valued data with Hermitian inner products (`np.conj(u) * v`).
- Excellent output: reconstruction comparison, convergence plot, and coil sensitivity maps grid — all saved to the slides directory.
- The README is clear and well-structured.

### Issues to Fix

- [ ] **Important — CG doesn't converge within tolerance**: With default parameters (R=3, tol=1e-6), CG reaches max_iter=40 with relative residual ~4.4e-5 but doesn't hit the tolerance. This is a teaching opportunity: either (a) increase max_iter to ~60, (b) relax tol to ~1e-4, or (c) add a discussion in the README about why semiconvergence means we should stop before fitting the noise anyway (you mention this in the slides — connect it here).

- [ ] **Important — Add a semiconvergence plot**: The slides mention that in inverse problems we should "stop before fitting the noise." The demo currently only shows residual vs iteration. Add a plot of **reconstruction error to ground truth vs iteration** alongside the residual plot to illustrate semiconvergence clearly. Students seeing the residual monotonically decrease won't understand why early stopping matters unless they also see the error curve.

- [ ] **Minor — README output files list is incomplete**: The README lists only `mri_reconstruction_results.pdf` and `mri_cg_convergence.pdf` as outputs, but the script also generates `coil_sensitivity_maps.pdf`. Update the README.

- [ ] **Minor — Zero-filled reconstruction includes sensitivity weighting**: Your `zero_filled_reconstruction()` divides by coil sensitivities (via conj(s_i) weighting and SoS normalization), which is actually closer to a simple SENSE-like approach than a truly "naive" reconstruction. For pedagogical contrast, consider adding a second baseline: pure sum-of-squares of per-coil IFFT images (without sensitivity weighting), so students can see three levels of quality: (1) naive SoS → (2) sensitivity-weighted → (3) full CG. Or at minimum, add a comment explaining that your zero-filled baseline already uses sensitivity information for coil combination, which makes it better than the simplest possible reconstruction but still shows aliasing from undersampling.

- [ ] **Minor — Hardcoded random seed**: The noise in `simulate_acquisition()` uses unseeded `np.random.randn()`. For reproducibility (so students get the same results), set a seed at the top of the main block.

- [ ] **Minor — Missing `requirements.txt`**: The README says `pip install -r ../../requirements.txt` but no such file was found at the project root. If the dependencies are just numpy, matplotlib, scipy, create a minimal requirements.txt.

---

## Overall Guidance for Next Round

1. **Fix compilation blockers first**: Add `\usetikzlibrary{positioning}` to the LaTeX preamble, fix `eigenvalue_illustration.py` so it runs, and ensure ALL referenced figures exist in `slides/figures/` before the next pdflatex run. The LaTeX must compile with zero errors.

2. **Fix the pedagogical sequencing**: The premature reveal of CG on the "Geometric Intuition" frame is the most significant teaching issue. Create a simple contour-only paraboloid plot for that frame, and save the SD-vs-CG comparison figure for after CG is introduced.

3. **Add the semiconvergence plot to the MRI demo**: This directly supports your own teaching point about stopping before fitting the noise. Plot both residual and true error vs iteration on the same axes (or side-by-side) so students can see the disconnection between residual decrease and reconstruction quality.
