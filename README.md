# run-lecture-prep.sh — Student/Teacher Agent Workflow for Lecture Generation

A shell script that orchestrates two Claude agents in an iterative feedback loop to produce lecture slides and course materials. The **Student** agent creates and improves materials, the **Teacher** agent reviews them and provides feedback. This repeats for 3 rounds.

## Quick Start

```bash
# Start from scratch
./run-lecture-prep.sh

# Resume after a crash
./run-lecture-prep.sh --resume

# Custom output directory
./run-lecture-prep.sh /path/to/output
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  for round in 1..3:                              │
│                                                  │
│    ┌────────────┐          ┌────────────┐        │
│    │  STUDENT   │ ─produces─▶ │  TEACHER   │        │
│    │  (agent)   │          │  (agent)   │        │
│    └────────────┘          └────────────┘        │
│         ▲                       │                │
│         │     feedback file     │                │
│         └───────────────────────┘                │
│         (rounds 2 & 3 only)                     │
│                                                  │
│    Each phase → git commit with author           │
└──────────────────────────────────┘
```

### How each phase works

**Student phase:**
- Round 1: Creates all materials from scratch (LaTeX slides, Python illustration code, MRI reconstruction demo)
- Rounds 2–3: Reads the Teacher's feedback file from the previous round, addresses every point (fixes errors, improves clarity, regenerates figures, recompiles LaTeX)
- The Student is instructed to use the Write tool for all file changes — it must produce actual files on disk, not just describe what it would do

**Teacher phase:**
- Reads ALL student-produced files (slides, code, figures, README)
- Evaluates against 7 criteria: technical accuracy, pedagogical clarity, visual quality, code quality, LaTeX quality, completeness, MRI application
- Writes a structured review to `feedback/round-N.md` with severity levels (Critical / Important / Minor)
- Feedback is actionable: "Frame 5: the definition of A-orthogonal is missing the condition p_i^T A p_j = 0 for i ≠ j" — not "improve slide 5"

## Key Design Decisions

### Feedback via files (not pipes)
The Teacher writes feedback to `materials/feedback/round-N.md` and the Student reads it in the next round. This persists across invocations, survives crashes, and creates an audit trail of how materials evolved.

### CWD isolation
Each `claude` invocation runs with `cd $WORK_DIR` so relative paths (`slides/lecture.tex`, `code/`, `feedback/`) resolve correctly regardless of where the script is launched from.

### Progress tracking
A `.progress` file records the last completed `(round, phase)` tuple after each phase. On `--resume`, the script reads this file and skips completed work:

```
LAST_ROUND=2
LAST_PHASE=student    # → resume at teacher phase of round 2
```

### Git attribution
Every phase is a separate commit with the agent's identity:

```
f500ed7 Student: Cleanup: remove LaTeX auxiliary files
213ca61 Teacher: Round 3: Teacher — review and feedback
013a2f4 Student: Round 3: Student — create/improve lecture materials
c854ef7 Teacher: Round 2: Teacher — review and feedback
002ad4e Student: Round 2: Student — create/improve lecture materials
b59cef1 Teacher: Round 1: Teacher — review and feedback
c3f6fc1 Student: Round 1: Student — create/improve lecture materials
37d1490 Workflow: Initial: project scaffold and CLAUDE.md
```

### Fallback stdout capture
If the Teacher outputs its review as chat text instead of using the Write tool, the script captures stdout and saves it as the feedback file. If both are empty, the script exits with an error rather than silently advancing.

### Verification gates
After each phase, the script verifies that expected files exist before writing progress. If verification fails, it exits with a clear error message — the `.progress` file is left at the previous phase so `--resume` can retry.

## Running Your Own Topic

Edit the prompts in the script to change the topic:

| Variable | Purpose |
|----------|---------|
| `CLAUDE_MD_EOF` | Project context loaded by both agents |
| `STUDENT_TASK_ROUND_1` | Student's initial task description |
| `STUDENT_TASK_IMPROVE` | Student's improvement task (rounds 2–3) |
| `TEACHER_REVIEW` | Teacher's review task and output format |
| `ROUNDS` | Number of iterations (default: 3) |

## Output Structure

```
materials/
├── slides/
│   ├── lecture.tex          # LaTeX Beamer source
│   ├── lecture.pdf          # Compiled PDF
│   └── figures/             # Generated vector figures
├── code/
│   ├── illustrations/       # Figure-generation scripts
│   └── mri_demo/            # Multi-coil SENSE reconstruction
│       ├── reconstruct.py
│       └── README.md
├── feedback/
│   ├── round-1.md           # Teacher review #1
│   ├── round-2.md           # Teacher review #2
│   └── round-3.md           # Teacher review #3
├── CLAUDE.md                # Project context
└── requirements.txt
```

## Requirements

- [Claude Code](https://claude.com/claude-code) CLI (must be logged in: `claude login` or use third party api, this work is done by deepseek v4 api)
- `pdflatex` with Beamer, TikZ, amsmath, bm packages
- Python 3.10+ with numpy, scipy, matplotlib (for running the generated code)
