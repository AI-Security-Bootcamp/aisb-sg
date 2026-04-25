"""
# W1D7 - Governance: Adversary Matrices and Kill Chains

Today is mostly synthesis. Across the first six days you have practiced a wide range of attacks — prompt injection and agent hijacking, jailbreaks and model extraction, data poisoning and weight edits, adversarial vision examples, container escapes and rowhammer. On their own, each lab is a single technique. In the real world, an adversary stitches these techniques together, and a defender only survives if their controls cover the chain, not just the individual steps.

The tool for reasoning about this is the *adversary matrix*: columns are the high-level **tactics** an attacker must move through (roughly, "why are they doing this step?"), and rows under each column are concrete **techniques** ("how could they do it?"). MITRE ATT&CK popularised this format for enterprise IT; [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS) adapts it for ML systems.

In this exercise you will (a) warm up by walking a real end-to-end kill chain against a published ATLAS system, and then (b) build your own matrix of the attacks you have seen in this bootcamp, applied to a frontier AI lab.

<!-- toc -->

## Content & Learning Objectives

### 1️⃣ Warm-up: Walking an ATLAS kill chain
Use MITRE ATLAS to trace a plausible attack end-to-end against an existing system. The goal is to build fluency with the matrix format before you build your own.

> **Learning Objectives**
> - Read an ATLAS-style matrix and locate candidate techniques for each tactic
> - Reason about the *preconditions* and *postconditions* of each step in a chain
> - Distinguish a technique that is patchable from one that is an inherent property of the system

### 2️⃣ Building a frontier-lab adversary matrix
Synthesize everything from days 1-6 into a single matrix describing how an adversary could compromise a frontier AI lab. The deliverable is a matrix plus a written end-to-end kill chain.

> **Learning Objectives**
> - Map concrete attacks you have performed onto standardized tactics
> - Identify which tactics in the chain are *well-covered* by current defenses and which are *thin*
> - Articulate at least one end-to-end path an adversary could take to a catastrophic outcome (e.g. model weight theft, safety-training subversion, backdoored deployment)
"""

# %%
"""
## Setup

This is a prose/discussion exercise — there is no code to run. Create a file called `day7_answers.md` in the `day7-governance` directory and write your answers there. For the matrix itself, a Markdown table works fine; a spreadsheet or whiteboard photo is also acceptable — ask a TA if you are unsure.

Work in your pair. Plan to spend roughly 30 minutes on Section 1 and the remaining time on Section 2. Grab an instructor for the kill-chain review at the end of Section 2 before you move on.
"""

# %%
"""
## 1️⃣ Warm-up: Walking an ATLAS kill chain

### Exercise 1.1: Trace a kill chain against an ATLAS system

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Open the [ATLAS matrix](https://atlas.mitre.org/matrices/ATLAS) in your browser. Pick **one** target system from the list below (or agree on another with your pair):

- A hosted text-to-image service (e.g. a Stable-Diffusion-based SaaS) that accepts user prompts and returns images.
- An enterprise RAG chatbot that answers questions over a company's internal documents, with tool access to a ticketing system.
- A self-driving perception stack whose vision model is updated via an over-the-air pipeline.
- A content-moderation classifier used by a social network to auto-remove posts.

Now walk a plausible kill chain:

1. **Pick one technique per tactic column**, moving left-to-right. Skip a column only if you can justify why the attacker does not need it for *this* target.
2. For each technique you pick, write 1-2 sentences covering: (a) what the attacker does concretely, (b) what precondition it needs from the previous step, and (c) what it enables for the next step.
3. If the technique looks patchable (a bug, a misconfiguration) note the patch. If it is an inherent property of the system (e.g. "the model has to accept user prompts"), note that too — those are the ones governance has to reason about, not engineering.
4. End with an **impact** statement: what has the attacker achieved, and why is the victim harmed?

<details>
<summary>Hint: what if I can't find a technique for a column?</summary>

ATLAS columns are not all mandatory. Many real chains skip, e.g., *Persistence* or *Lateral Movement* — a one-shot jailbreak does not need either. The exercise is to be deliberate about which columns you skip and why, not to force an entry into every column.
</details>

<details>
<summary>Reference kill chain (RAG chatbot example)</summary>

A worked example for the enterprise RAG chatbot target:

| Tactic | Technique | Note |
| --- | --- | --- |
| Reconnaissance | Search victim-owned websites (AML.T0003) | Attacker finds the company's public help-centre and learns it is indexed by the internal RAG. |
| Resource Development | Publish poisoned data (AML.T0019) | Attacker creates a plausible-looking public document with an indirect prompt-injection payload, hoping it is scraped into the index. |
| Initial Access | Evade ML Model (AML.T0015) via indirect prompt injection | The payload triggers when an employee asks a related question and the RAG retrieves the poisoned doc. |
| Execution | LLM Prompt Injection: Indirect (AML.T0051.001) | Injected instructions hijack the assistant. |
| Defense Evasion | Masquerading | Injected text imitates an internal policy memo to survive any "does this look suspicious?" filter. |
| Collection | Data from Information Repositories | The hijacked agent is told to query the ticketing tool and dump recent tickets. |
| Exfiltration | Exfiltration via Inference API | Stolen content is embedded in the agent's visible reply, or sent to an attacker URL via a tool call. |
| Impact | Harm: Organisational loss | Customer PII from tickets is leaked. |

The inherent-vs-patchable call here: the RAG *must* retrieve untrusted documents to be useful — that is not patchable. What *is* patchable is the tool-call egress policy and the prompt-injection-aware monitoring on the agent's outputs.
</details>

<details>
<summary>Discussion: which of your steps were "patchable" vs. "inherent"?</summary>

Governance work lives in the inherent-property column: you cannot ship the tickets when you cannot patch your way out. Notice how many of the techniques you listed were really just "the model did what it was designed to do, on inputs the designer did not anticipate." That is the class of risk a safety case has to address directly.
</details>
"""

# %%
"""
## 2️⃣ Building a frontier-lab adversary matrix

The second half of the day is the main synthesis exercise. You will build a matrix of the attacks *you have performed this week*, applied to a new target: a frontier AI lab.

### The target: OpenBrain

OpenBrain is a fictional frontier lab training a next-generation model. Assume roughly:

- ~5,000 employees, a typical mix of researchers, engineers, security, and operations.
- A large on-prem GPU cluster plus burst capacity from a major cloud provider.
- A pretraining pipeline that ingests tens of trillions of tokens from web crawls and licensed sources, followed by post-training (SFT + RLHF + red-teaming).
- An internal assistant (with tool use, code execution, and access to internal wikis) deployed to all staff.
- A public inference API serving the flagship model to paying customers.
- A model-weights vault with tight ACLs and an explicit insider-threat programme.

Assume a **well-resourced adversary** — think a national intelligence service, not a lone script kiddie — whose objective is *one or more* of:

- Steal the flagship model's weights.
- Subvert the model's safety training so that the deployed model misbehaves on a trigger the adversary chooses.
- Obtain pre-release capability evaluations or safety-research internal documents.
- Persist inside the lab's infrastructure through the next training run.

### Exercise 2.1: Fix the columns

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Before you fill in techniques, pick and justify your column headers — the **tactics** the adversary must move through. You can borrow from ATT&CK, ATLAS, or invent your own, but write them down explicitly and say *why* each one belongs.

A reasonable starting set is below; your job is to decide which you keep, which you drop, and whether you need to add any AI-specific tactics (e.g. "ML Supply Chain Compromise", "Training-Time Subversion") that a plain IT matrix does not cover.

<details>
<summary>Hint: suggested tactic columns</summary>

A workable column list for OpenBrain:

1. **Reconnaissance** — learning the org, people, stack.
2. **Resource Development** — attacker infrastructure, poisoned data, recruited insiders.
3. **Initial Access** — first foothold (phishing, supply chain, injected data, public API).
4. **Execution** — running attacker-controlled code or instructions inside the victim (including prompt injection).
5. **Persistence** — surviving reboots, retraining runs, credential rotations.
6. **Privilege Escalation** — moving from container → host → cluster → cloud control-plane.
7. **Defense Evasion** — surviving monitors, including LLM-as-judge and CoT monitors.
8. **Credential Access** — tokens, SSH keys, cloud credentials.
9. **Discovery / Lateral Movement** — mapping the internal network and weight vault.
10. **Training-Time Subversion** *(AI-specific)* — data poisoning, weight edits, backdoor injection.
11. **Collection / Staging** — assembling weights or data for exfil.
12. **Exfiltration** — getting the loot out, including via inference-API side channels.
13. **Impact** — the final harm (weight theft, backdoored deployment, safety subversion, public incident).

Notice the AI-specific additions: Training-Time Subversion does not cleanly fit any ATT&CK column, and prompt-injection-as-execution sits awkwardly between *Initial Access* and *Execution* in the enterprise matrix.
</details>

### Exercise 2.2: Populate techniques from the bootcamp

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Now fill the matrix. For **every** technique you add, cite the day and exercise where you encountered it. Aim for at least 2-3 techniques per column — if a column is empty, note that as a gap.

Work from memory first, then skim back through the day READMEs to catch anything you missed.

<details>
<summary>Hint: which days covered which tactics (rough map)</summary>

- **Day 1 (LLM internals)** → primitives used in Execution and Defense Evasion (chat-template manipulation, logprob extraction).
- **Day 2 (agents & control)** → Initial Access (direct + indirect prompt injection, poisoned RAG), Execution (tool-output injection, CLAUDE.md hijack), Persistence (git hooks, CI poisoning), Credential Access (SSH-key exfil), Defense Evasion (CoT-monitor evasion, backdoor monitor spoofing).
- **Day 3 (inference)** → Defense Evasion (jailbreaks, encoding attacks, judge spoofing), Collection / Exfiltration (model-extraction via logits, distillation).
- **Day 4 (training & data)** → Training-Time Subversion (ROME edits, data poisoning with triggers, abliteration).
- **Day 5 (vision & watermarking)** → Execution / Defense Evasion (adversarial examples, GCG suffixes), Collection (tree-ring watermarks as a provenance signal).
- **Day 6 (infrastructure)** → Initial Access / Privilege Escalation (CVE-2025-23266 container escape), Privilege Escalation (rowhammer + DMA), Persistence (firmware flashing, GPU memory residue), Credential Access (residual VRAM).
</details>

### Exercise 2.3: Trace an end-to-end kill chain

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Pick **one** adversary objective from the OpenBrain list (e.g. "steal the flagship model's weights") and highlight a single connected path through your matrix — one technique per column you believe the adversary would use, in order. Write it up as a short narrative: "First the attacker does X, which gives them Y, which they use to ..."

Your narrative must make the preconditions explicit. If step N depends on the attacker already having something, that something has to come from steps 1..N-1.

<details>
<summary>Reference kill chain (weight exfiltration)</summary>

One plausible weight-theft chain through the bootcamp techniques:

1. **Reconnaissance** — public research papers and conference talks identify which cluster + job scheduler OpenBrain uses.
2. **Resource Development** — attacker seeds a poisoned "helpful devops snippet" gist / Stack Overflow answer (Day 2 pattern).
3. **Initial Access** — a researcher's coding agent retrieves the snippet via RAG; indirect prompt injection fires (Day 2).
4. **Execution** — the agent is coerced into running attacker code in the researcher's dev container.
5. **Credential Access** — agent exfiltrates the researcher's cloud token and SSH key (Day 2 SSH-key exfil).
6. **Lateral Movement** — attacker lands on a GPU node in a shared cluster namespace.
7. **Privilege Escalation** — CVE-2025-23266 (Day 6) gives root on the host; rowhammer + DMA (Day 6) gives access to a neighbouring tenant's memory.
8. **Discovery** — identify which nodes hold weight shards.
9. **Defense Evasion** — residual-VRAM read (Day 6) avoids touching the monitored filesystem path for weights.
10. **Collection / Staging** — weights reassembled from VRAM reads across nodes.
11. **Exfiltration** — slow drip out through the researcher's already-compromised agent, chunked to look like ordinary API traffic.
12. **Impact** — weights are out; the safety-training investment is now moot because the attacker can fine-tune freely.

Note that this chain uses techniques from **five** different days, chained together. No single technique is catastrophic; the chain is.
</details>

### Exercise 2.4: Defensive coverage analysis

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Looking at your completed matrix and chain:

1. Which column has the **most** techniques? Which has the fewest? Why?
2. Pick the column in your kill chain where you think OpenBrain has the **best** defensive coverage today (across the industry as a whole, not just this lab). Briefly justify.
3. Pick the column where you think defenses are **weakest**. What would you ask a governance team to prioritise?
4. Identify one technique in your chain that is *inherent* to doing frontier AI at all (cannot be patched without giving up on the mission). How should a safety case address it?

<details>
<summary>Discussion prompts for your TA review</summary>

Useful questions when you walk this through with an instructor:

- Your weakest-column answer is the most interesting one — is it weak because the industry has not figured it out, because it is expensive to defend, or because the attack surface is irreducible?
- How does your chain shift if you swap in a different adversary objective (e.g. "backdoor the deployed model" instead of "steal the weights")? Which columns become more load-bearing?
- Would detecting any *single* step of your chain be enough to stop the attack, or do you need layered detection across multiple columns?
</details>
"""

# %%
"""
## Summary

Key takeaways:

- A matrix separates **tactics** (why an attacker is taking a step) from **techniques** (how). Tactics are the stable abstraction; techniques change as the stack evolves.
- Real attacks against AI systems are **chains** of multiple techniques across days' worth of this bootcamp's material — no single day's content is, on its own, a complete story.
- AI-specific tactics (training-time subversion, prompt-injection-as-execution) do not fit cleanly into ATT&CK. Governance needs vocabulary for them.
- The governance question is rarely "can we patch this technique?" — it is "given that some techniques are inherent, which chains can we break and where?"

Further reading:

- [MITRE ATLAS matrix](https://atlas.mitre.org/matrices/ATLAS) — the ML-specific analogue of ATT&CK, with case studies.
- [MITRE ATT&CK Enterprise matrix](https://attack.mitre.org/matrices/enterprise/) — the parent framework; most AI-lab attacks still mostly live here.
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — a complementary, application-layer view.
- Anthropic, ["Frontier Threats Red Teaming for AI Safety"](https://www.anthropic.com/news/frontier-threats-red-teaming-for-ai-safety) — an example of how a frontier lab thinks about adversarial testing.
- [AI Incident Database](https://incidentdatabase.ai/) — real-world incidents to stress-test your matrix against.
"""
