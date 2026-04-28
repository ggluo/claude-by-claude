# Conjugate Gradient Lecture Project

## Goal
Prepare a complete lecture (slides + course materials) on the **Conjugate Gradient (CG) algorithm** for solving **linear inverse problems**, with application to **multi-coil MRI reconstruction**.

## Target Audience
Undergraduate students with basic linear algebra and calculus background. No prior knowledge of inverse problems or MRI assumed.

## Deliverables
All deliverables live under the project root:

1. **`slides/`** — Lecture slides (LaTeX Beamer)
   - Main file: `slides/lecture.tex`
   - Use the Beamer document class with a clean theme (e.g., `metropolis`, `Madrid`, or `CambridgeUS`)
   - Structure: title slide → motivation → linear systems review → inverse problems intro → steepest descent → conjugate gradient derivation → convergence properties → multi-coil MRI reconstruction case study → summary
   - Each slide (frame) should be self-contained and visually clear
   - Use `\includegraphics` for figures from `slides/figures/`
   - Use `\texttt`, `\begin{align*}`, `\item` etc. for clear formatting
   - Compile with `pdflatex` (or `lualatex`) to verify it builds without errors
   - Output PDF: `slides/lecture.pdf`

2. **`code/illustrations/`** — Python code for generating lecture figures
   - 2D contour plots comparing steepest descent vs CG convergence
   - Eigenvalue distribution illustration explaining CG convergence rate
   - A clean, labeled diagram of the multi-coil MRI acquisition model
   - All figures saved to `slides/figures/` as PDF (vector) or high-res PNG

3. **`code/mri_demo/`** — Python code for multi-coil MRI reconstruction demo
   - Generate or download a phantom image
   - Simulate multiple receiver coils with spatially-varying sensitivity maps
   - Simulate k-space undersampling (Cartesian) independently per coil
   - Solve with CG (explicit implementation, no black-box solver)
   - Compare with basic coil-combined zero-filled reconstruction
   - Plot reconstruction error vs iteration count
   - Dependencies: numpy, matplotlib, scipy

## Teaching Philosophy
- Start with concrete examples before abstract theory
- Prefer visual intuition over dense algebra
- Show code alongside math so students can experiment
- Every formula should be accompanied by an explanation of "what it means"

## Quality Standards
- Code must run without errors on a fresh Python 3.10+ environment
- All plots must have labels, titles, and legends
- LaTeX must compile without errors (run pdflatex at least once)
- Slides must be self-contained (no "read the textbook" references)
- Comments in code should explain the math, not just the mechanics
