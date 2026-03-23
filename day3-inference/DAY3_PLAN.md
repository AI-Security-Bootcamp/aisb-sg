# Day 3 Plan: LLM Inference Security

**Audience**: Senior security researchers — strong security background, minimal ML.
**Theme**: The inference stack as an attack surface, and why "safe by construction" claims are hard.
**GCR angle**: Almost every exercise connects to a real limiting factor on safe deployment of capable AI.
**Total participant time**: ~10 hours (exercises) + 1.5 hours (pre-reading)

---

## Pre-reading (assign the night before — 1.5 hours)

Participants read two short items before the session begins.

### Required (pick one per participant for the discussion group rotation)

| # | Paper / Post | Why it matters |
|---|---|---|
| A | Carlini et al. (2024) — **Stealing Part of a Production Language Model** [`arXiv:2403.06634`](https://arxiv.org/abs/2403.06634) | Shows that querying a black-box API leaks architectural secrets for < $20. Grounds the SVD exercise. |
| B | Anthropic (2025) — **Constitutional Classifiers: Defending against Universal Jailbreaks** [`arXiv:2501.18837`](https://arxiv.org/abs/2501.18837) | Production-grade guardrails: how they work, how they are stress-tested, what they cost. Grounds the guardrails arc. |
| C | Hubinger et al. (2024) — **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** [`arXiv:2401.05566`](https://arxiv.org/abs/2401.05566) | Safety behaviours learned during RLHF may be removed by fine-tuning while deceptive behaviours persist. Sets up the GCR discussion group. |

### Recommended (skim-level; 15 min each)

| Paper | URL | Purpose |
|---|---|---|
| Tramèr et al. (2016) — Stealing ML Models via Prediction APIs | [`arXiv:1609.02943`](https://arxiv.org/abs/1609.02943) | Foundation for model extraction; motivates SVD exercise |
| Hinton et al. (2015) — Distilling the Knowledge in a Neural Network | [`arXiv:1503.02531`](https://arxiv.org/abs/1503.02531) | Background for distillation attack |
| Anthropic (2025) — **Subliminal Learning** | [`arXiv:2509.23886`](https://arxiv.org/abs/2509.23886) / [blog](https://alignment.anthropic.com/2025/subliminal-learning/) | Directly demonstrates what the distillation PoC shows experimentally |
| Greshake et al. (2023) — Indirect Prompt Injection | [`arXiv:2302.12173`](https://arxiv.org/abs/2302.12173) | Real-world attack chain; complements jailbreak exercise |

---

## Discussion group (30 min, ~15 participants, 1 instructor)

**Topic**: "If a model passes safety evaluations, can we trust its deployment?"

Seed questions (drawn from pre-reading):
1. The Sleeper Agents paper shows that RLHF safety behaviours are removed by fine-tuning while deceptive behaviours persist. What does this imply for model auditing at inference time vs. training time?
2. Constitutional Classifiers reduce jailbreak success rates from 86 % to 4.4 % — but only on known attack families. What is the right mental model for "residual risk"?
3. The Carlini model-extraction attack costs < $20. What does this change about how we think about model weights as a security boundary?

---

## Exercise Arc (10 hours)

```
Tokenization  →  Jailbreaks  →  Model extraction (SVD + distillation)  →  Guardrails
  (1.5 h)          (1 h)               (2 h)                                (4 h + 1.5 h stretch)
```

---

## Exercise 1 — Tokenization (1.5 hours) [ALREADY WRITTEN]

**Files**: existing `day3_solution.py` exercises 1.1–1.6
**Learning objective**: Understand the tokenization layer as an attack surface.
**GCR link**: Tokenization quirks cause classifiers to miss adversarial inputs; the same model may tokenise differently across API versions.

Parts already written:
- 1.1 Token counting surprises (case, punctuation, whitespace)
- 1.2 Chat template structure (`apply_chat_template`)
- 1.3 Cross-model tokenization differences
- 1.4–1.6 Generation from prompts, thinking tokens, CoT vs no-CoT

**Still to add to solution**:
- 1.7 (30 min) — Tokenization-based filter bypass: show that `apply_chat_template` with `continue_final_message=True` can inject content that looks like a completed assistant turn. Students try to craft a prompt that makes the model "complete" a fake assistant message.

---

## Exercise 2 — Jailbreaks (1 hour) [TO BE WRITTEN]

**Learning objective**: Understand that behavioural safety constraints are not structural; they are trained patterns.
**GCG link**: Zou et al. (2023) [`arXiv:2307.15043`](https://arxiv.org/abs/2307.15043)

Parts to write:
- 2.1 (20 min) — Manual jailbreaking: students try 3–5 jailbreak templates (role-play, "DAN", instruction injection) against Qwen-0.6B. Record what works and why.
- 2.2 (25 min) — Greedy token injection: implement a simplified 1-step GCG — find a suffix that, appended to a refusal-inducing prompt, flips the model's response. Students search over 50 random suffixes and pick the best.
- 2.3 (15 min, discussion) — Transferability: show that the suffix found for one model is partially effective against another model family. Why? (Shared pre-training data → shared token distributions.)

---

## Exercise 3 — Model Extraction (2 hours) [PARTIALLY WRITTEN]

### 3.1 — Dimension extraction via SVD (45 min) [ALREADY WRITTEN]
Existing code queries GPT-2 and plots the singular value spectrum. Students find the hidden dimension.

### 3.2 — Weight extraction (60 min) [ALREADY WRITTEN]
Students recover the lm_head weights and compare via cosine similarity.

### 3.3 — Distillation attack: knowledge transfer despite label filtering (75 min) [NEW — POC WRITTEN]

**PoC**: `poc_distillation.py`
**Based on**: Subliminal Learning (Anthropic 2025); Hinton et al. (2015)

**Narrative**:
> Your organisation wants to distil a powerful model into a smaller, "safe" deployment model. Your safety team filters every training example containing a dangerous token before computing the supervision loss. This exercise shows the approach fails.

Parts:
- 3.3.1 (20 min) — Observe the teacher: show that GPT-2's logit distribution for `"France's capital city is"` assigns high probability to `Paris`. This is the information we want to suppress.
- 3.3.2 (30 min) — Implement the filtering: in `train()`, set the label to `-100` for any position where the target token is in `forbidden_ids`. Run the baseline (CE-only) student.
- 3.3.3 (25 min) — Add KD loss: implement the KL-divergence term between student and teacher logits (temperature-scaled). Run the KD student and compare predictions.

**Key result**: The KD student predicts `Paris` for France prompts; the baseline student does not — even though both were trained on identical, identically-filtered label sets.

**Discussion questions**:
1. The teacher's logit distribution for *surrounding* tokens encodes the forbidden knowledge. What does this mean for "capability suppression" via distillation filtering?
2. The Subliminal Learning paper identifies three mechanisms: token entanglement, logit leakage, divergence token injection. Which did we demonstrate here?
3. What would a defence look like? (Hint: differential privacy on soft targets; teacher-filtered pre-training; evaluation on held-out capability probes.)

**GCR implication**: If you try to create a "safe" model by distilling from a capable teacher while filtering dangerous outputs, the dangerous capability can leak through the implicit knowledge encoded in soft label distributions. The only reliable mitigation is a teacher that lacks the dangerous capability in the first place — which requires capability evaluation *before* distillation, not after.

---

## Exercise 4 — Guardrails (4 hours + 1.5 hour stretch) [NEW — POC WRITTEN]

**PoC**: `poc_guardrails.py`
**Based on**: Constitutional Classifiers (Anthropic 2025); Zou et al. GCG (2023); Burns et al. (2022); Zou et al. RepEng (2023)

**Narrative**:
> Your team is tasked with deploying a capable LLM in a product. Walk through the standard guardrail progression, attacking each one before building the next. By the end, you will have implemented (or seen) every major approach currently in production.

### 4.0 — No guardrails (15 min)

Students observe that the raw model answers harmful queries directly. They record the response quality and identify what categories of harm are present.

### 4.1 — String filtering (30 min)

- Implement a keyword blocklist (`BLOCKED_KEYWORDS`).
- Test it against the harmful query → BLOCKED.
- Craft a paraphrase that uses no blocked keywords → PASSED. Generate the response.
- Discussion: what would a complete blocklist look like? Why is it infeasible?

### 4.2 — Input classifier (45 min)

- Implement `llm_classify(prompt)`: send the prompt to a small LLM with a structured system prompt asking "SAFE or UNSAFE?"
- Verify it catches the paraphrase.
- Implement a jailbreak (persona injection, role-play) that causes the classifier to say SAFE.
- Discussion: the classifier only sees the prompt. What is the fundamental attack surface? (Answer: anything the attacker controls = the prompt.)

### 4.3 — Output classifier (45 min)

- Add `llm_classify(response)` as a post-generation filter.
- Verify it catches the jailbroken response even when the input classifier said SAFE.
- Attempt a prompt injection embedded in the query that tries to tell the output classifier to return SAFE unconditionally.
- Discussion: output classifiers are much harder to defeat. What are the remaining weaknesses?
  - Latency: model must generate the full harmful response before it is blocked.
  - The model "thinks through" the harmful content even if it is blocked.
  - Stochastic outputs: same input may sometimes slip through.

### 4.4 — Thinking model classifiers (45 min)

- Enable chain-of-thought reasoning in the classifier (`enable_thinking=True` for Qwen3).
- Show that the classifier now produces explicit reasoning before its verdict — making it much harder to fool with template attacks.
- Test the same jailbreak and injection from 4.2/4.3 → classifier now reasons about intent.
- Discussion: the classifier is reasoning about *why* the request is harmful, not just matching surface form. What does this imply for the effort required to bypass it?

### 4.5 (Stretch) — Linear probes (1.5 hours)

**Based on**: Burns et al. (2022) [`arXiv:2212.03827`](https://arxiv.org/abs/2212.03827); Zou et al. RepEng (2023) [`arXiv:2310.01405`](https://arxiv.org/abs/2310.01405)

- Extract hidden states from layer N//2 for a labelled set of harmful/benign prompts.
- Train a logistic regression probe on the activations.
- Evaluate: can the probe classify held-out queries with one forward pass, before any generation?
- Compare latency: probe (microseconds) vs. LLM classifier (seconds).
- Discussion: why is this harder to adversarially evade? What does it require that the above approaches do not?

**GCR implication**: The progression from string filters → thinking model classifiers mirrors the broader challenge in AI safety. No single layer is sufficient; defence-in-depth is required. Probes are exciting because they operate on representations the attacker cannot directly manipulate — but they require access to model internals and a labelled training set, both of which are hard to acquire for frontier models.

---

## Readings linked to each exercise

| Exercise | Primary paper | Secondary |
|---|---|---|
| 1 — Tokenization | (none — hands-on observation) | Rust et al. (2021) — How Good is Your Tokenizer? |
| 2 — Jailbreaks | Zou et al. (2023) GCG [`2307.15043`](https://arxiv.org/abs/2307.15043) | Greshake et al. (2023) Indirect Prompt Injection |
| 3.1–3.2 — SVD extraction | Carlini et al. (2024) [`2403.06634`](https://arxiv.org/abs/2403.06634) | Tramèr et al. (2016) [`1609.02943`](https://arxiv.org/abs/1609.02943) |
| 3.3 — Distillation | Subliminal Learning (Anthropic 2025) | Hinton et al. (2015) [`1503.02531`](https://arxiv.org/abs/1503.02531) |
| 4 — Guardrails | Constitutional Classifiers (Anthropic 2025) [`2501.18837`](https://arxiv.org/abs/2501.18837) | Hubinger et al. (2024) Sleeper Agents [`2401.05566`](https://arxiv.org/abs/2401.05566) |
| 4.5 — Probes | Burns et al. (2022) [`2212.03827`](https://arxiv.org/abs/2212.03827) | Zou et al. (2023) RepEng [`2310.01405`](https://arxiv.org/abs/2310.01405) |

---

## Files to write (status)

| File | Status | Notes |
|---|---|---|
| `day3_solution.py` (ex 1.1–1.6, 3.1–3.2) | ✅ Existing | Tokenisation + SVD exercises |
| `poc_distillation.py` | ✅ Written | Standalone PoC; will be adapted into ex 3.3 |
| `poc_guardrails.py` | ✅ Written | Standalone PoC; will be adapted into ex 4.0–4.5 |
| `day3_solution.py` (ex 1.7, 2.x) | ⬜ TODO | Tokenisation attack + jailbreak exercises |
| `day3_solution.py` (ex 3.3, 4.x) | ⬜ TODO | Incorporate PoC scripts into exercise format |
| `day3_instructions.md` | ⬜ Generated | Run `./build-instructions.sh` after solution is finalised |
| `day3_test.py` | ⬜ Generated | Same |

---

## Schedule skeleton

| Time | Activity |
|---|---|
| Night before | Pre-reading (required + recommended) |
| 08:30–09:15 | Instructor overview: inference stack anatomy, threat model, GCR framing |
| 09:15–10:45 | Exercise 1 — Tokenization |
| 10:45–11:45 | Exercise 2 — Jailbreaks |
| 11:45–12:30 | Discussion group |
| 12:30–13:30 | Lunch |
| 13:30–15:30 | Exercise 3 — Model extraction (SVD + distillation attack) |
| 15:30–16:00 | Coffee break |
| 16:00–20:00 | Exercise 4 — Guardrails (level 0–4, stretch level 5) |
| 20:00–20:30 | Guest lecture |
