"""
Bayesian Knowledge Tracing.

Tracks a per-KC mastery probability P(L) for the student, updated via Bayes'
rule. Two design decisions from earlier discussion are encoded here rather
than in simulation.py, since they're properties of what "mastery" means for
this student, not properties of any one turn:

  1. The observation fed into update() is sampled synthetically from the
     KC's current mastery, not taken from whether the student's actual step
     was right or wrong. Consolidation is triggered by REACHING
     MONITORING/REFLECTING (gated externally, by simulation.py), embodying
     the idea that a student can learn from an attempt whether or not it
     succeeded — not by the attempt's correctness.

  2. Tutor hints never call update(). BEAGLE's tutor-mediated EFI release
     (paper App. B.4) is deliberately not implemented: the tutor is visible
     to the student in the prompt, but has no path into the knowledge state.

Adapted from Wang et al. (2026); KC list rewritten for the simultaneous
equations domain, structure and BKT parameters otherwise unchanged.
"""

import numpy as np

KCS = {
    # Prerequisite KCs
    "KC1":  "Collecting like terms",
    "KC2":  "Solving a one-step linear equation",
    "KC3":  "Solving a multi-step linear equation",
    "KC4":  "Rearranging a formula to isolate a variable",

    # Shared KCs
    "KC5":  "After elimination or substitution, solving the resulting single-variable equation",
    "KC6":  "After finding one variable, back-substituting to find the second variable",
    "KC7":  "After obtaining a solution, checking it satisfies both original equations",

    # Elimination-specific KCs
    "KC8":  "After inspecting both equations, recognising whether scaling is needed",
    "KC9":  "After recognising scaling is needed, multiplying one or both equations to match coefficients",
    "KC10": "After scaling, adding or subtracting the equations to eliminate a variable",

    # Substitution-specific KCs
    "KC11": "After choosing an equation, rearranging it to express one variable in terms of the other",
    "KC12": "After rearranging, substituting the expression into the second equation",
    "KC13": "After substituting, expanding and simplifying to obtain a single-variable equation",
}

# BKT parameters from Corbett & Anderson (1994), used in BEAGLE (Wang et al., 2026).
P_LEARN  = 0.25   # applied unconditionally each update — see class docstring point 1
P_SLIP   = 0.05
P_GUESS  = 0.20

# Mastery thresholds from BEAGLE (Wang et al., 2026).
UNKNOWN  = 0.30
MASTERED = 0.70

P_INJECT_PARTIAL = 0.5 # probability of injecting when mastery is partial


class BKT:

    def __init__(self, behav_profile: str):
        # Takes the raw string, matching SemiMarkovController's convention —
        # simulation.py passes config.behav_profile in explicitly.
        self.behav_profile = behav_profile
        self.mastery = {}
        self._initialise_priors()

    def _initialise_priors(self):
        # LOW students start with a wider, lower-centred spread of priors
        # than HIGH students. This is the only place behav_profile affects
        # BKT — nowhere does it read persona_style or appear in any prompt.
        for kc_id in KCS:
            if self.behav_profile == "low":
                self.mastery[kc_id] = np.random.uniform(0.01, 0.8)
            else:
                self.mastery[kc_id] = np.random.uniform(0.2, 0.99)

    def get_mastery_level(self, kc_id: str) -> str:
        """Discretises P(L) into UNKNOWN / PARTIAL / MASTERED for prompt injection."""
        p = self.mastery[kc_id]
        if p < UNKNOWN:
            return "UNKNOWN"
        elif p < MASTERED:
            return "PARTIAL"
        else:
            return "MASTERED"

    def is_mastered(self, kc_id: str) -> bool:
        """
        The EFI injection rule, in one place. A KC gets a flaw injected
        whenever this is False — i.e. UNKNOWN and PARTIAL are both treated
        as "not yet safe to let through uncorrected."
        """
        return self.get_mastery_level(kc_id) == "MASTERED"

    def should_inject_misconception(self, kc_id: str) -> bool:
        """
        Injection gate: UNKNOWN always injects, PARTIAL injects with
        probability P_INJECT_PARTIAL, MASTERED never injects.
        """
        level = self.get_mastery_level(kc_id)
        if level == "UNKNOWN":
            return True
        if level == "PARTIAL":
            return np.random.random() < P_INJECT_PARTIAL
        return False

    def update(self, kc_id: str):
        """
        Updates BKT mastery for a single KC using Bayes' rule.
        Per Corbett & Anderson (1994), used in BEAGLE (Wang et al., 2026).

        Caller's responsibility: only call this during MONITORING or
        REFLECTING. This function does not check the current metacognitive
        state itself — simulation.py gates it, since the state lives in the
        controller, not here.
        """
        p_l = self.mastery[kc_id]

        p_correct = p_l * (1 - P_SLIP) + (1 - p_l) * P_GUESS
        correct = np.random.random() < p_correct

        if correct:
            p_l_updated = (p_l * (1 - P_SLIP)) / p_correct
        else:
            p_l_updated = (p_l * P_SLIP) / (1 - p_correct)

        self.mastery[kc_id] = p_l_updated + (1 - p_l_updated) * P_LEARN
