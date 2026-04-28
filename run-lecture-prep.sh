#!/bin/bash
set -eo pipefail

# ============================================================================
#  Lecture Preparation Workflow — Student/Teacher Agent Loop
#  Topic: Conjugate Gradient for Linear Inverse Problems & MRI Reconstruction
#  Audience: Undergraduate students
#
#  Usage:
#    ./run-lecture-prep.sh [--resume] [output_dir]
#
#    --resume    Skip completed rounds and continue from where it left off
#    output_dir  Directory for all materials (default: ./cg-lecture-output)
# ============================================================================

# --- Argument parsing ---
RESUME=false
WORK_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=true
            shift
            ;;
        -*)
            echo "Unknown flag: $1"
            echo "Usage: $0 [--resume] [output_dir]"
            exit 1
            ;;
        *)
            WORK_DIR="$1"
            shift
            ;;
    esac
done

WORK_DIR="${WORK_DIR:-$(pwd)/cg-lecture-output}"
# Resolve to absolute path (does not require directory to exist yet)
if [[ "$WORK_DIR" != /* ]]; then WORK_DIR="$(pwd)/$WORK_DIR"; fi
WORK_DIR="$(cd "$(dirname "$WORK_DIR")" && pwd)/$(basename "$WORK_DIR")"
ROUNDS=3
FEEDBACK_DIR="$WORK_DIR/feedback"
SLIDES_DIR="$WORK_DIR/slides"
CODE_DIR="$WORK_DIR/code"
PROGRESS_FILE="$WORK_DIR/.progress"

# --- Resume logic ---
START_ROUND=1
START_PHASE="student"   # "student" or "teacher"

if $RESUME && [[ -f "$PROGRESS_FILE" ]]; then
    source "$PROGRESS_FILE" 2>/dev/null || true
    if [[ -n "$LAST_ROUND" && -n "$LAST_PHASE" ]]; then
        if [[ "$LAST_PHASE" == "teacher" ]]; then
            START_ROUND=$((LAST_ROUND + 1))
            START_PHASE="student"
        elif [[ "$LAST_PHASE" == "student" ]]; then
            START_ROUND="$LAST_ROUND"
            START_PHASE="teacher"
        fi
    fi
    echo "Resuming from round $START_ROUND, phase: $START_PHASE"
elif $RESUME; then
    echo "No progress file found at $PROGRESS_FILE — starting from round 1"
fi

if [[ "$START_ROUND" -gt "$ROUNDS" ]]; then
    echo "All $ROUNDS rounds already complete. Nothing to resume."
    exit 0
fi

mkdir -p "$WORK_DIR" "$FEEDBACK_DIR" "$SLIDES_DIR" "$CODE_DIR"

# --- Git helpers ---
git_init() {
    if [[ ! -d "$WORK_DIR/.git" ]]; then
        git -C "$WORK_DIR" init
        git -C "$WORK_DIR" config user.name "Workflow"
        git -C "$WORK_DIR" config user.email "workflow@lecture.local"
        git -C "$WORK_DIR" add -A
        git -C "$WORK_DIR" commit --author="Workflow <workflow@lecture.local>" \
            -m "Initial: project scaffold and CLAUDE.md" || true
    fi
}

git_commit_student() {
    local round=$1
    git -C "$WORK_DIR" add -A
    git -C "$WORK_DIR" commit \
        --author="Student <student@lecture.local>" \
        -m "Round $round: Student — create/improve lecture materials" || true
}

git_commit_teacher() {
    local round=$1
    git -C "$WORK_DIR" add -A
    git -C "$WORK_DIR" commit \
        --author="Teacher <teacher@lecture.local>" \
        -m "Round $round: Teacher — review and feedback" || true
}

# Helper: write progress after each phase completes
write_progress() {
    echo "LAST_ROUND=$1"  > "$PROGRESS_FILE"
    echo "LAST_PHASE=$2" >> "$PROGRESS_FILE"
}

# ============================================================================
#  CLAUDE.md — project context loaded by both agents
# ============================================================================
cat > "$WORK_DIR/CLAUDE.md" << 'CLAUDE_MD_EOF'
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
CLAUDE_MD_EOF

# ============================================================================
#  STUDENT PROMPTS
# ============================================================================

read -r -d '' STUDENT_TASK_ROUND_1 << 'STUDENT_EOF' || true
## Your Role
You are a **diligent student** preparing lecture materials for an undergraduate class. Your work will be reviewed by a professor who will give detailed feedback. Take pride in producing clear, accurate, and engaging materials.

## Your Task
Create a COMPLETE first draft of lecture slides and course materials on the **Conjugate Gradient algorithm for linear inverse problems, with application to multi-coil MRI reconstruction**.

### Specific Deliverables

1. **Lecture Slides** — write `slides/lecture.tex` (LaTeX Beamer)
   - Use `\documentclass{beamer}` with a clean theme (metropolis or Madrid)
   - Title slide with topic and "Lecture Notes"
   - Section 1: Motivation — why do we need iterative solvers? (1-2 frames)
   - Section 2: Linear systems as optimization — least squares, normal equations (2-3 frames)
   - Section 3: Steepest descent — algorithm, geometric intuition, why it's slow (2-3 frames)
   - Section 4: Conjugate Gradient — A-orthogonal directions, derivation, algorithm pseudocode (4-5 frames)
   - Section 5: Convergence — relation to condition number, eigenvalue distribution (2-3 frames)
   - Section 6: Multi-Coil MRI Reconstruction — how MRI uses multiple receiver coils, the SENSE model, CG-based reconstruction (4-5 frames)
   - Section 7: Summary and further reading (1 frame)
   - TOTAL: ~18-22 frames
   - Use `\begin{align*} ... \end{align*}` for multi-line equations
   - Use `\includegraphics{figures/filename.pdf}` for figures
   - After writing, compile with `cd slides && pdflatex -interaction=nonstopmode lecture.tex` to verify it builds

2. **Python Illustration Code** — write to `code/illustrations/`
   - `steepest_descent_vs_cg.py` — Generate contour plots of a 2D quadratic form, trace the iterations of steepest descent and CG side-by-side. Save figures to `slides/figures/` as PDF.
   - `eigenvalue_illustration.py` — Illustrate how eigenvalue distribution affects CG convergence. Save figure to `slides/figures/` as PDF.
   - `mri_acquisition_model.py` — Diagram showing the multi-coil MRI forward model: (image × coil sensitivities → per-coil FFT → k-space → undersampling → noisy measurements per coil). Save figure to `slides/figures/` as PDF.
   - Make sure `mkdir -p slides/figures` is called and all figures are saved there.

3. **Multi-Coil MRI Reconstruction Demo** — write to `code/mri_demo/`
   - `reconstruct.py` — Complete pipeline:
     a. Create/generate a phantom image (e.g., Shepp-Logan via skimage or a simple geometric phantom)
     b. Generate multiple coil sensitivity maps (e.g., 4-8 coils) using Gaussian or Biot-Savart-inspired spatial profiles across the FOV
     c. Simulate per-coil k-space data acquisition with Cartesian undersampling (e.g., R=2-4 acceleration)
     d. Formulate the SENSE reconstruction as a linear system:
        Forward model: for each coil i, y_i = M * FFT(S_i ⊙ x) + noise
        Cost: min_x sum_i ||M F S_i x - y_i||^2 + λ * R(x)
        Normal equations: (sum_i S_i^H F^H M F S_i + λ L^T L) x = sum_i S_i^H F^H y_i
     e. Implement CG from scratch (do NOT use scipy.sparse.linalg.cg or similar)
     f. Reconstruct with CG and compare against simple coil-combined zero-filled IFFT
     g. Plot: original phantom, coil sensitivity maps, zero-filled reconstruction, CG reconstruction, and error-vs-iteration curve
   - `README.md` — How to install dependencies and run the demo, with explanation of the SENSE model

4. **`requirements.txt`** at the project root listing all Python dependencies.

### Important
- Write ALL files now. Do not leave anything as a sketch or outline.
- Code must be runnable. Include `if __name__ == "__main__":` blocks.
- Use `# %%` cell markers in Python files so students can run them interactively in VS Code.
- Target language: American English.

Start by reading the CLAUDE.md at the project root for full context.
Then create every file listed above.
STUDENT_EOF

read -r -d '' STUDENT_TASK_IMPROVE << 'STUDENT_IMPROVE_EOF' || true
## Your Role
You are a **diligent student** who has received detailed feedback from your professor. You must now improve your lecture materials based on that feedback.

## CRITICAL: You MUST use the Write tool
After improving files, you MUST call the Write tool to save your changes. Do not just describe what you would change — actually edit the files. The professor needs to see updated files on disk to review.

## Your Task
Read the professor's latest feedback from `FEEDBACK_FILE`, then revise ALL your materials to address every point raised.

### What to do
1. **Read the feedback** — it is at `FEEDBACK_FILE` (this is the exact path)
2. **Go through each piece of feedback** point by point:
   - If the professor found an error → fix it
   - If an explanation is unclear → rewrite it more clearly
   - If slides are too dense → split them or add more explanation
   - If code needs changes → update the code
   - If LaTeX doesn't compile → fix the errors
   - If figures need improvement → improve the plotting scripts
   - If the multi-coil MRI explanation needs work → elaborate on coil sensitivities, SENSE model, or CG applicability
3. **Update all files** using the Write tool: `slides/`, `code/illustrations/`, and `code/mri_demo/`
4. **Regenerate figures** by running the illustration scripts after updating them
5. **Recompile LaTeX** — run `cd slides && pdflatex -interaction=nonstopmode lecture.tex` to verify

### Deliverables
All the same files as before, but improved:
- `slides/lecture.tex` — updated LaTeX Beamer slides
- `code/illustrations/*.py` — updated illustration scripts
- `slides/figures/*.pdf` — regenerated figures
- `code/mri_demo/reconstruct.py` — updated multi-coil reconstruction demo
- `code/mri_demo/README.md` — updated if needed
- `requirements.txt` — updated if needed

### Key Principle
Do NOT ignore any feedback. If you disagree with a point, add a brief `% NOTE:` comment in the LaTeX or a `# NOTE:` comment in Python explaining your reasoning. But address everything.

Start by reading the CLAUDE.md at the project root for context, then read the feedback file, then make all improvements.
STUDENT_IMPROVE_EOF

# ============================================================================
#  TEACHER PROMPT
# ============================================================================

read -r -d '' TEACHER_REVIEW << 'TEACHER_EOF' || true
## Your Role
You are an **experienced professor** reviewing a student's draft lecture materials. You want the student to succeed and produce excellent materials. Be thorough, specific, and constructive.

## CRITICAL: You MUST use the Write tool
Your review is useless if the student cannot read it. After reviewing the materials, you MUST call the Write tool to create `FEEDBACK_FILE` (use this exact path). Do NOT just summarize your review in the chat — the student agent needs the physical file. This is the only deliverable that matters for your role.

## Your Task
Review ALL materials the student has produced so far, then USE THE WRITE TOOL to save your detailed feedback.

### What to review
1. **Lecture slides** — read `slides/lecture.tex`
2. **Illustration code** — read all `.py` files in `code/illustrations/`
3. **MRI demo code** — read `code/mri_demo/reconstruct.py` and `code/mri_demo/README.md`
4. **Generated figures** — check `slides/figures/` for quality
5. **LaTeX compilation** — if `slides/lecture.pdf` exists, check it; if not, try compiling

### Review criteria
- **Technical accuracy** — Is the math correct? Is the CG algorithm correctly described and implemented? Are convergence properties stated accurately? Is the multi-coil SENSE formulation correct?
- **Pedagogical clarity** — Can an undergraduate with basic linear algebra follow this? Are concepts introduced in the right order? Is jargon explained? Are coil sensitivities and the multi-coil model explained clearly?
- **Visual quality** — Are diagrams clear and properly labeled? Do they actually illustrate the concept? Are coil sensitivity maps shown?
- **Code quality** — Does the code run? Is it readable? Are comments helpful (explain why, not what)?
- **LaTeX quality** — Does it compile? Are Beamer features used well (uncover, pause, blocks, etc.)?
- **Completeness** — Are all required sections present? Are there gaps?
- **MRI application** — Is the connection between CG and multi-coil MRI reconstruction clearly explained? Is the SENSE reconstruction model correctly presented?

### Output format
Use the Write tool to create `FEEDBACK_FILE` with this structure:

# Professor's Review — Round <N>

## Summary
(2-3 sentences overall assessment)

## Slides (LaTeX Beamer)
### Strengths
- ...
### Issues to Fix
- [ ] **Critical** (title): detailed explanation
- [ ] **Important** (title): detailed explanation
- [ ] **Minor** (title): detailed explanation

## Illustration Code
### Strengths
- ...
### Issues to Fix
- [ ] ... (same format)

## MRI Demo Code
### Strengths
- ...
### Issues to Fix
- [ ] ... (same format)

## Overall Guidance for Next Round
(2-3 highest-priority improvements)

Be specific. Instead of "improve slide 5", write "Frame 5: the definition of A-orthogonal is missing a crucial condition — add that p_i^T A p_j = 0 for i ≠ j".

### Order of operations
1. Read all student files
2. Analyze them thoroughly
3. Call the Write tool to create FEEDBACK_FILE (the exact path provided above) with your complete review
4. Then you may optionally summarize in the chat
TEACHER_EOF

# ============================================================================
#  Initialize git and make initial commit
# ============================================================================
git_init

# ============================================================================
#  MAIN LOOP
# ============================================================================

echo "============================================================"
echo "  Lecture Preparation — Student/Teacher Agent Workflow"
echo "  Topic: Conjugate Gradient & Multi-Coil MRI Reconstruction"
echo "  Rounds: $ROUNDS"
echo "  Output: $WORK_DIR"
if $RESUME; then
    echo "  Mode:   Resume (starting round $START_ROUND, phase: $START_PHASE)"
fi
echo "============================================================"
echo ""

for round in $(seq "$START_ROUND" $ROUNDS); do

    # Determine starting phase for this round
    if [[ "$round" -eq "$START_ROUND" ]] && [[ "$START_PHASE" == "teacher" ]]; then
        SKIP_STUDENT=true
    else
        SKIP_STUDENT=false
    fi

    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  ROUND $round / $ROUND                                                    │"
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""

    # =======================================================================
    #  PHASE 1: STUDENT (creates or improves materials)
    # =======================================================================
    if ! $SKIP_STUDENT; then
        echo "▶ [STUDENT] Phase — Round $round"
        echo ""

        if [ "$round" -eq 1 ]; then
            echo "  Task: Create initial draft of all materials"
            set +e
            (cd "$WORK_DIR" && claude \
                --allowedTools Read,Bash,Python,WebSearch,WebFetch,Write \
                -p "$STUDENT_TASK_ROUND_1")
            STUDENT_EXIT=$?
            set -e
        else
            FEEDBACK_FILE="$FEEDBACK_DIR/round-$((round - 1)).md"
            if [[ ! -f "$FEEDBACK_FILE" ]]; then
                echo ""
                echo "✖ ERROR: Feedback file $FEEDBACK_FILE does not exist!"
                echo "  The teacher did not write feedback for the previous round."
                echo "  Run with --resume after fixing, or delete .progress to restart from scratch."
                exit 1
            fi

            # Inject the feedback file path into the prompt
            CURRENT_STUDENT_PROMPT="${STUDENT_TASK_IMPROVE//FEEDBACK_FILE/$FEEDBACK_FILE}"
            echo "  Task: Improve materials based on feedback: $FEEDBACK_FILE"
            echo "  Feedback size: $(wc -l < "$FEEDBACK_FILE") lines"
            set +e
            (cd "$WORK_DIR" && claude \
                --allowedTools Read,Bash,Python,WebSearch,WebFetch,Write \
                -p "$CURRENT_STUDENT_PROMPT")
            STUDENT_EXIT=$?
            set -e
        fi

        echo ""

        # Verify student actually wrote the key deliverable
        if [[ ! -f "$SLIDES_DIR/lecture.tex" ]]; then
            echo "✖ ERROR: Student did not produce slides/lecture.tex"
            echo "  The student phase failed or did not write the expected files."
            exit 1
        fi

        if [[ $STUDENT_EXIT -ne 0 ]]; then
            echo "⚠ WARNING: Student exited with code $STUDENT_EXIT (files may be incomplete)"
        fi

        echo "✓ [STUDENT] Round $round complete"
        write_progress "$round" "student"

        # Commit student changes to git
        git_commit_student "$round"
        echo "  [git] Committed as Student <student@lecture.local>"
        echo ""
    else
        echo "⏭  [STUDENT] Skipped (already completed) — resuming at teacher phase"
        echo ""
    fi

    # =======================================================================
    #  PHASE 2: TEACHER (reviews and gives feedback)
    # =======================================================================
    echo "▶ [TEACHER] Phase — Round $round"
    echo ""

    FEEDBACK_FILE="$FEEDBACK_DIR/round-$round.md"

    # Substitute round number and feedback file path into prompt
    CURRENT_TEACHER_PROMPT="${TEACHER_REVIEW//<N>/$round}"
    CURRENT_TEACHER_PROMPT="${CURRENT_TEACHER_PROMPT//FEEDBACK_FILE/$FEEDBACK_FILE}"

    set +e
    TEACHER_STDOUT=$(mktemp)
    (cd "$WORK_DIR" && claude \
        --allowedTools Read,Bash,Python,WebSearch,WebFetch,Write \
        -p "$CURRENT_TEACHER_PROMPT" 2>&1) | tee "$TEACHER_STDOUT"
    TEACHER_EXIT=${PIPESTATUS[0]}
    set -e

    echo ""

    # Verify teacher actually wrote the feedback file
    if [[ ! -f "$FEEDBACK_FILE" ]]; then
        echo "⚠ Teacher did not use Write tool — saving stdout as fallback"
        sed 's/\x1b\[[0-9;]*m//g' "$TEACHER_STDOUT" > "$FEEDBACK_FILE"
        if [[ ! -s "$FEEDBACK_FILE" ]] || [[ $(wc -l < "$FEEDBACK_FILE") -lt 3 ]]; then
            echo "✖ ERROR: Teacher produced no meaningful output"
            rm -f "$TEACHER_STDOUT"
            exit 1
        fi
    fi
    rm -f "$TEACHER_STDOUT"

    echo "✓ [TEACHER] Review written to $FEEDBACK_FILE ($(wc -l < "$FEEDBACK_FILE") lines)"
    write_progress "$round" "teacher"

    # Commit teacher feedback to git
    git_commit_teacher "$round"
    echo "  [git] Committed as Teacher <teacher@lecture.local>"
    echo ""
done

# ============================================================================
#  Final git log
# ============================================================================
echo "============================================================"
echo "  All $ROUNDS rounds complete!"
echo "  Final materials: $WORK_DIR"
echo "  Slides:    $SLIDES_DIR"
echo "  Code:      $CODE_DIR"
echo "  Feedback:  $FEEDBACK_DIR"
echo ""
echo "  Git history:"
git -C "$WORK_DIR" log --oneline --format="%h %an: %s"
echo "============================================================"
