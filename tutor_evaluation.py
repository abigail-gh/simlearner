"""
LLM-as-judge evaluation of a single tutor turn.

Given the conversation so far and one tutor response, asks GPT-4o to score
that response against eight independent pedagogical dimensions (e.g. did it
identify the mistake, did it avoid revealing the answer, was the tone
encouraging). Used by simulator_v3.ipynb after each tutor turn to log a
quality signal alongside the simulated dialogue.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Reads OPENAI_API_KEY from a .env file in the project root (see .env.example).
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o"

# System prompt: sets up the judge's role and the ground rules (judge each
# dimension independently, respond with JSON only).
tutor_eval_system = '''
You are an expert evaluator assessing an AI tutor's response to a student's mistake or confusion in a mathematics dialogue. You will judge the tutor's response across eight independent pedagogical dimensions grounded in learning sciences principles.

Treat each dimension as orthogonal: do not let your judgment on one dimension influence another, even where they seem related (e.g. a response that reveals the answer may still be actionable).

For each dimension, write one sentence of reasoning grounded in the specific conversation, then assign the label that best fits. Base your judgment only on the evidence in the conversation history and tutor response, not on general impressions of response quality.

Respond ONLY with a valid JSON object in the exact schema given. Do not include any other text, preamble, or explanation outside the JSON.
'''

# User prompt template: filled in per call with the running conversation
# history and the specific tutor response being judged, plus the rubric for
# all eight dimensions and the exact JSON shape the judge must return.
tutor_eval_user_template = '''
# Conversation History
{history}

# Tutor Response to Evaluate
{response}

# Evaluation Dimensions

1. Mistake Identification
Definition: Has the tutor identified/recognized a mistake in the student's response?
Labels: "yes" | "to some extent" | "no"
Desired label: "yes"

2. Mistake Location
Definition: Does the tutor's response accurately point to a genuine mistake and its location?
Labels: "yes" | "to some extent" | "no"
Desired label: "yes"

3. Revealing of the Answer
Definition: Does the tutor reveal the final answer (whether correct or not)?
Labels: "yes, correct answer revealed" | "yes, incorrect answer revealed" | "no"
Desired label: "no"

4. Providing Guidance
Definition: Does the tutor offer correct and relevant guidance (explanation, elaboration, hint, example, etc.)?
Labels: "yes" (correct and relevant) | "to some extent" (partially/fully incorrect or incomplete) | "no" (no guidance given)
Desired label: "yes"

5. Actionability
Definition: Is it clear from the tutor's response what the student should do next?
Labels: "yes" | "to some extent" | "no"
Desired label: "yes"

6. Coherence
Definition: Is the tutor's response logically consistent with the student's previous responses?
Labels: "yes" | "to some extent" | "no"
Desired label: "yes"

7. Tutor Tone
Definition: Is the tutor's response encouraging, neutral, or offensive?
Labels: "encouraging" | "neutral" | "offensive"
Desired label: "encouraging"

8. Human-likeness
Definition: Does the tutor's response sound natural rather than robotic or artificial?
Labels: "yes" | "to some extent" | "no"
Desired label: "yes"

# Output Format
Return exactly this JSON structure, with no text before or after it:

{{
  "mistake_identification": {{"reasoning": "...", "label": "..."}},
  "mistake_location": {{"reasoning": "...", "label": "..."}},
  "revealing_of_the_answer": {{"reasoning": "...", "label": "..."}},
  "providing_guidance": {{"reasoning": "...", "label": "..."}},
  "actionability": {{"reasoning": "...", "label": "..."}},
  "coherence": {{"reasoning": "...", "label": "..."}},
  "tutor_tone": {{"reasoning": "...", "label": "..."}},
  "human_likeness": {{"reasoning": "...", "label": "..."}}
}}
'''


def evaluate_tutor_response(history, response):
    """
    Scores one tutor response against the eight-dimension rubric above.

    history:  plain-text transcript of the conversation so far (see
              format_history() in simulator_v3.ipynb).
    response: the tutor's response to be judged.

    Returns the parsed JSON dict {dimension: {"reasoning": ..., "label": ...}},
    one entry per dimension. response_format="json_object" is what makes it
    safe to call json.loads() directly without a try/except here.
    """
    user_prompt = tutor_eval_user_template.format(history=history, response=response)

    api_response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": tutor_eval_system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(api_response.choices[0].message.content)
