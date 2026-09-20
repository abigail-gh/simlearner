"""
Semi-Markov behavioural controller.

Governs which metacognitive state (PLANNING, ENACTING, MONITORING,
REFLECTING) the student is in each turn, and for how long. This is what
produces the "wheel-spinning" difference between behav_profile "low" and
"high": the two profiles have different transition matrices and different
expected dwell times, so a low-profile student ends up spending more turns
stuck in ENACTING purely as a consequence of these numbers, not because
anything else in the pipeline treats it as weaker.

Adapted from Wang et al. (2026), BEAGLE — domain changed from Python/physics
to simultaneous equations; the profiles below are rewritten for that domain,
the underlying transition-matrix and Gamma-duration structure is unchanged.
"""

import numpy as np

profiles = {
    "PLANNING": """PLANNING: The Strategist (Thinking Before Acting)
Verbalize Goals: State objectives before acting. "I need to figure out which method to use" not "let me try something."
Deliberate: Consider which approach or equation to use for this small step. Ask questions: "Should I use substitution or elimination?"
Uncertainty: Express uncertainty freely: "I'm not sure how to do that." """,

    "ENACTING": """ENACTING: The Actor (Working Through the Problem)
Verbalize Intent: State micro-goals before acting on a small step. "I am going to substitute x into the first equation."
Reactive Execution: Just do it, don't overthink. "Let's try this" then DO IT.
Trial-and-Error: Quick iterations, not deep analysis. """,

    "MONITORING": """MONITORING: The Spotter (Watching the Working)
Commanding Tone: Use imperatives to check values. "Check the x value," "Look at this step."
Data Obsessed: Quote specific numbers. "I got x=4 but I expected x=3" not "it looks wrong."
Track Progress: Note what changed. "That step is wrong" or "It's still not matching." """,

    "REFLECTING": """REFLECTING: The Critic (Making Sense of What Happened)
Seek Understanding: Think about WHY. "I think the reason is..." or "This makes sense because..."
Emotional Range: Express satisfaction ("I get it now!") or frustration ("This is so confusing.")
Look Backward: Reference prior attempts. "Last time I did this wrong because..." """
}


def get_profile(meta: str) -> str:
    """
    Returns the language-pattern block for a given metacognitive state.
    Used both in the system prompt (via prompts.py) and, when use_controller
    is False, keyed on config.fixed_meta_state instead of a sampled one.
    """
    return profiles[meta]


# Author-supplied transition probabilities, kept verbatim here for provenance.
# Rows are rounded to 4dp and a couple sum to 1.0001/0.9999 instead of exactly
# 1.0, which np.random.choice rejects — normalized per-row below so the class
# attribute actually used at runtime is always a valid distribution.
_raw_transition_matrix = {
    "low": {
        "PLANNING":   {"PLANNING": 0.5313, "ENACTING": 0.3125, "MONITORING": 0.1250, "REFLECTING": 0.0313},
        "ENACTING":   {"PLANNING": 0.2286, "ENACTING": 0.4000, "MONITORING": 0.3143, "REFLECTING": 0.0571},
        "MONITORING": {"PLANNING": 0.1875, "ENACTING": 0.4375, "MONITORING": 0.0625, "REFLECTING": 0.3125},
        "REFLECTING": {"PLANNING": 0.1250, "ENACTING": 0.6250, "MONITORING": 0.0000, "REFLECTING": 0.2500},
    },
    "high": {
        "PLANNING":   {"PLANNING": 0.3714, "ENACTING": 0.2571, "MONITORING": 0.2286, "REFLECTING": 0.1429},
        "ENACTING":   {"PLANNING": 0.6000, "ENACTING": 0.0500, "MONITORING": 0.2000, "REFLECTING": 0.1500},
        "MONITORING": {"PLANNING": 0.2500, "ENACTING": 0.2500, "MONITORING": 0.2500, "REFLECTING": 0.2500},
        "REFLECTING": {"PLANNING": 0.2727, "ENACTING": 0.4545, "MONITORING": 0.1818, "REFLECTING": 0.0909},
    }
}


def _normalize_rows(matrix):
    """Ensures every row sums to exactly 1.0, required by np.random.choice."""
    return {
        persona: {
            state: {k: v / sum(row.values()) for k, v in row.items()}
            for state, row in rows.items()
        }
        for persona, rows in matrix.items()
    }


class SemiMarkovController:
    """
    Two-level state model: a first-order Markov chain over metacognitive
    states, plus an independently-fit Gamma distribution for how long each
    state lasts. The Gamma (rather than the geometric duration a plain
    Markov chain implies) is what lets "getting stuck" be modelled as
    genuinely overdispersed rather than forced into a fixed mean/variance
    relationship — see BEAGLE App. A.2 for the empirical justification.
    """

    # shape = 1/CV², scale = mean/shape — chosen to preserve the CV ratios
    # from Wang et al. (2026) while reducing the means to the shorter
    # timescale of a single simultaneous-equations problem.
    gamma_params = {
        "low": {
            "PLANNING":   {"shape": 1.5625, "scale": 0.6400},
            "ENACTING":   {"shape": 0.5487, "scale": 5.4678},
            "MONITORING": {"shape": 2.9726, "scale": 0.6728},
            "REFLECTING": {"shape": 2.5194, "scale": 0.3969},
        },
        "high": {
            "PLANNING":   {"shape": 0.6830, "scale": 2.9283},
            "ENACTING":   {"shape": 2.1001, "scale": 0.9523},
            "MONITORING": {"shape": 1.6025, "scale": 1.8722},
            "REFLECTING": {"shape": 1.6867, "scale": 1.1857},
        }
    }

    transition_matrix = _normalize_rows(_raw_transition_matrix)

    initial_state_probs = {
        "low":  {"PLANNING": 0.75, "ENACTING": 0.25},
        "high": {"PLANNING": 0.80, "ENACTING": 0.20},
    }

    def __init__(self, behav_profile):
        # Takes the raw string, not a RunConfig, so this class stays
        # reusable outside the simulation pipeline (e.g. for standalone
        # analysis of the transition dynamics). simulation.py is
        # responsible for passing config.behav_profile in.
        self.behav_profile = behav_profile
        self.current_meta = self._sample_initial_meta()
        self.turns_remaining = self._sample_duration(self.current_meta)

    def _sample_initial_meta(self):
        probs = self.initial_state_probs[self.behav_profile]
        states = list(probs.keys())
        weights = list(probs.values())
        return np.random.choice(states, p=weights)

    def _sample_duration(self, meta):
        params = self.gamma_params[self.behav_profile][meta]
        duration = np.random.gamma(params["shape"], params["scale"])
        return max(1, round(duration))

    def get_state(self):
        """The state to use for THIS turn — read before calling step()."""
        return self.current_meta

    def _sample_next_meta(self):
        probs = self.transition_matrix[self.behav_profile][self.current_meta]
        states = list(probs.keys())
        weights = list(probs.values())
        return np.random.choice(states, p=weights)

    def step(self):
        """
        Advances the controller by one turn. Call this AFTER using
        get_state() for the current turn, not before — it may change
        current_meta if the sampled duration has run out.
        """
        self.turns_remaining -= 1
        if self.turns_remaining <= 0:
            self.current_meta = self._sample_next_meta()
            self.turns_remaining = self._sample_duration(self.current_meta)
