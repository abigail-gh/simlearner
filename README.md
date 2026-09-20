# SimLearner

A neuro-symbolic learner simulator for the mathematics domain (simultaneous equations), built to evaluate LLM-based Intelligent Tutoring Systems (LLM-ITS) without needing real students. Adapted from **BEAGLE** (Wang et al., 2026), which demonstrated this approach in the programming domain.

The simulator produces a synthetic student that plans, attempts, and discusses maths problems with a tutor, exhibiting **authentic, imperfect reasoning**: documented misconceptions injected at controlled moments, driven by a learner's simulated knowledge state rather than arbitrary randomness.

## Why

LLMs asked to role-play a struggling student tend to solve problems too well: this is **competency bias**. This project addresses it architecturally, pairing neural generation (GPT-4o) with symbolic components that constrain *what* the model is allowed to produce and *when*.

## Architecture

Six sequential LLM calls per turn, three symbolic components:

1. **Planning** — produces a correct plan and working for the next small step
2. **KC Identification** — classifies which of 13 knowledge components (KCs) that step exercises
3. **Explicit Flaw Injection (EFI)** — generates a specific incorrect working, grounded in a documented misconception for that KC
4. **Student Output** — voices the (possibly flawed) working in character, as the learner's own turn
5. **Tutor** — responds with a guiding question or hint, verifying the maths independently first
6. **Tutor Evaluation** — scores the tutor's response across 8 pedagogical dimensions (Maurya et al., 2025)

Symbolic components, gating the above:

- **Semi-Markov Controller** — governs the learner's metacognitive phase (Planning / Enacting / Monitoring / Reflecting), with Gamma-distributed phase durations and author-supplied transition matrices adapted from BEAGLE
- **Bayesian Knowledge Tracing (BKT)** — tracks per-KC mastery (Corbett & Anderson, 1994), updated only during Monitoring/Reflecting phases
- **Mastery Gate** — decides whether EFI's error actually reaches the learner: Unknown KCs always inject, Partial KCs inject with 50% probability, Mastered KCs never inject

A single `behav_profile` variable (`"low"` or `"high"`) parametrises the Controller's transition matrix, phase durations and initial-state distribution, and BKT's mastery priors. Linguistic style is held constant across both profiles by design (see `decisions-and-learnings` notes for the rationale).

## Repository structure

```
bkt.py                  Bayesian Knowledge Tracing: mastery tracking, KC list, misconceptions
controller.py            Semi-Markov metacognitive controller: phases, transition matrices, durations
tutor_evaluation.py       8-dimension tutor evaluation judge (LLM-as-judge)
evaluator.py              Competency-bias judge: scores whether a turn contains a mistake and
                           whether it matches the injected misconception
simulator_v3.ipynb        Main notebook: run a single simulation interactively, with full
                           two-column HTML logging (conversation | process detail)
run_experiment.py         Batch runner: sweeps configurations across all 7 evaluation problems,
                           logs results to CSV, used for the ablation study
requirements.txt          Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-your-actual-key-here
```

No quotes, no spaces around the `=`, nothing else on that line.

## Running a simulation

Open `simulator_v3.ipynb` and run all cells. Configuration lives near the top:

```python
problem = "Solve the simultaneous equations: 3x + y = 11, 2x − y = 4"
n_turns = 10
BEHAV_PROFILE = "low"    # "low" or "high"
```

### Ablation flags

Each flag isolates one architectural component, switching it off while leaving the rest intact:

| Flag | Effect when `False` |
|---|---|
| `use_planning` | Skip planning; invent a wrong step directly, with no correct plan to corrupt |
| `use_efi` | Skip KC identification and error generation; student receives the correct plan unmodified |
| `use_bkt_gate` | Always inject a misconception, ignoring BKT mastery |
| `use_controller` | No metacognitive phase; BKT never updates (updates are gated on being in Monitoring/Reflecting) |
| `tutor_eval` | Skip the tutor evaluation judge call |

A run produces a saved HTML log (`logs/run_{timestamp}{suffix}.html`) showing the conversation alongside every internal decision: mastery band, KC, injected misconception, and tutor evaluation scores.

## Running the ablation study

```bash
python run_experiment.py
```

Sweeps the configured flag combinations across all 7 evaluation problems (see `Appendix B` in the dissertation for the full problem list), logging results to CSV for analysis.

## Key references

- Wang et al. (2026) — BEAGLE: Behaviour-Enforced Agent for Grounded Learner Emulation (the architecture this project adapts)
- Corbett & Anderson (1994) — Bayesian Knowledge Tracing
- Booth et al. (2017), Al-Rababaha et al. (2020), Johari & Shahrill (2020), Mudavanhu et al. (2023) — the empirical misconception literature grounding EFI's 13 KCs
- Maurya et al. (2025) — the 8-dimension pedagogical taxonomy used by the tutor evaluation judge

## Status

Research prototype built for an MSc dissertation. Not intended for deployment: evaluation to date is self-conducted and small-scale (see Limitations in the dissertation). The Tutor component is a fixed placeholder; in deployment it is intended to be replaced with the actual LLM-ITS under evaluation.
