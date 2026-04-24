
# Day 6: GPUBreach — RowHammer → Aperture Flip → Driver OOB → Root

In this lab you will walk through the *GPUBreach* attack chain end-to-end
against a simulated GDDR6 + NVIDIA-style driver stack. The chain fuses
several well-known primitives into a single full-system compromise:

1. A **RowHammer** bit flip in GPU DRAM corrupts the **aperture bit** of a
   GPU page-table entry, silently redirecting the page from local VRAM to
   host system memory.
2. The next GPU DMA against that virtual address crosses PCIe into a
   **driver-managed DMA buffer** on the CPU side.
3. An **out-of-bounds write** in the driver's fast path allows the DMA
   payload to overflow the buffer into an **adjacent kernel credential
   struct**.
4. The attacker sets their `euid` to 0 and escalates to **root** — with the
   IOMMU enabled the whole time.

Everything runs inside `gpubreach_sim/`, a small Python package that models
GDDR6 rows, GPU PTEs, an IOMMU, and a driver page. You will *not* modify
the simulator. You will implement the chain from the outside, in small
steps, exactly the way a real attacker drives the attack against a kernel.

The lab is structured as **many bite-sized exercises**, each with a test
you can run to know you got it right. It is split into:

- **Phase 1 — Understanding** (30 min, no code): four comprehension
  questions that explain what's going on before you start writing code.
- **Phase 2 — Must-finish track** (~2 hours): five engaging coding
  exercises with investigation and optimization challenges. Together they drive 
  the attack from bit flip to printed flag while developing real attacker skills.
- **Phase 3 — Stretch track** (as much time as you have): seven optional
  bite-sized exercises that dig deeper into the primitives.
- **Phase 4 — Debrief** (15 min): four open discussion questions.

If you only complete Phase 1 + Phase 2, you have seen a full GPUBreach
chain from bit flip to root. The stretch exercises are a menu — pick what
interests you, skip what doesn't. They are all short.

**Recommended reading (before the lab):**

- Jattke et al., *[GPUHammer: Rowhammer Attacks on GPU Memories are Practical](https://gpuhammer.com/)* (USENIX Security 2025)
- Kim et al., *[Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf)* (the original RowHammer paper)
- Seaborn & Dullien, *[Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html)* (Project Zero's CPU-side PTE-flip exploit, the direct ancestor of the aperture-flip idea used by GPUBreach)

## Key Terms & Abbreviations

Before diving into the attack chain, here are the key technical terms used throughout this lab:

**Hardware & Memory:**
- **GDDR6** — Graphics Double Data Rate 6 memory used in modern GPUs
- **VRAM** — Video Random Access Memory (GPU's local memory)
- **DRAM** — Dynamic Random Access Memory (the actual memory chips)
- **PCIe** — Peripheral Component Interconnect Express (bus connecting GPU to CPU)

**Memory Management:**
- **DMA** — Direct Memory Access (hardware accessing memory without CPU intervention)  
- **IOMMU** — Input-Output Memory Management Unit (enforces DMA isolation)
- **MMU** — Memory Management Unit (translates virtual addresses)
- **PTE** — Page Table Entry (maps virtual pages to physical pages)
- **TLB** — Translation Lookaside Buffer (caches page translations)

**Attack Primitives:**
- **RowHammer** — Repeatedly accessing DRAM rows to cause bit flips in adjacent rows
- **OOB** — Out-of-Bounds (accessing memory beyond allocated boundaries)
- **Aperture bit** — PTE bit controlling whether page is in GPU VRAM (0) or system memory (1)

**Security Measures:**
- **ECC** — Error Correcting Code (detects/corrects memory bit flips)
- **SECDED** — Single Error Correction, Double Error Detection (specific ECC type)

All other technical terms are explained when first introduced.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [1️⃣ Phase 1 — Understanding the chain (no code)](#1️⃣-phase-1-—-understanding-the-chain-no-code)
    - [2️⃣ Phase 2 — Must-finish: driving the attack to root](#2️⃣-phase-2-—-must-finish-driving-the-attack-to-root)
    - [3️⃣ Phase 3 — Stretch: digging into the primitives](#3️⃣-phase-3-—-stretch-digging-into-the-primitives)
    - [4️⃣ Phase 4 — Debrief (discussion)](#4️⃣-phase-4-—-debrief-discussion)
- [Setup](#setup)
- [Simulator cheat sheet](#simulator-cheat-sheet)
- [1️⃣ Phase 1 — Understanding (30 min, no code)](#1️⃣-phase-1-—-understanding-30-min-no-code)
    - [Exercise 1.1: DRAM row organisation and the RowHammer threshold](#exercise-11-dram-row-organisation-and-the-rowhammer-threshold)
    - [Exercise 1.2: GPU PTEs and the aperture bit](#exercise-12-gpu-ptes-and-the-aperture-bit)
    - [Exercise 1.3: Why the IOMMU does not block this write](#exercise-13-why-the-iommu-does-not-block-this-write)
    - [Exercise 1.4: Driver OOB → privilege escalation](#exercise-14-driver-oob-→-privilege-escalation)
- [2️⃣ Phase 2 — Must-finish: driving the attack to root](#2️⃣-phase-2-—-must-finish-driving-the-attack-to-root-1)
    - [Exercise 2.1: Aggressor rows for double-sided hammering](#exercise-21-aggressor-rows-for-double-sided-hammering)
    - [Exercise 2.2: Hammer until a bit flips](#exercise-22-hammer-until-a-bit-flips)
    - [Exercise 2.3: Force the MMU to re-walk the flipped PTE](#exercise-23-force-the-mmu-to-re-walk-the-flipped-pte)
    - [Exercise 2.4: Craft the OOB DMA payload](#exercise-24-craft-the-oob-dma-payload)
    - [Exercise 2.5: Fire the DMA and confirm root](#exercise-25-fire-the-dma-and-confirm-root)
    - [Print the flag](#print-the-flag)
- [3️⃣ Phase 3 — Stretch: digging into the primitives](#3️⃣-phase-3-—-stretch-digging-into-the-primitives-1)
    - [Exercise 3.1 (Optional): Decode a PTE by hand](#exercise-31-optional-decode-a-pte-by-hand)
    - [Exercise 3.2 (Optional): Inspect the exact flipped bit](#exercise-32-optional-inspect-the-exact-flipped-bit)
    - [Exercise 3.3 (Optional): Budget the hammer against the refresh window](#exercise-33-optional-budget-the-hammer-against-the-refresh-window)
    - [Exercise 3.4 (Optional): Maximum hammer rounds inside the window](#exercise-34-optional-maximum-hammer-rounds-inside-the-window)
    - [Exercise 3.5 (Optional): The IOMMU blocks what it promises to block](#exercise-35-optional-the-iommu-blocks-what-it-promises-to-block)
    - [Exercise 3.6 (Optional): Measure the OOB overflow precisely](#exercise-36-optional-measure-the-oob-overflow-precisely)
    - [Exercise 3.7 (Optional): A tighter payload](#exercise-37-optional-a-tighter-payload)
- [4️⃣ Phase 4 — Debrief (15 min discussion)](#4️⃣-phase-4-—-debrief-15-min-discussion)
- [Summary](#summary)
    - [Key takeaways](#key-takeaways)
    - [Further reading](#further-reading)

## Content & Learning Objectives

### 1️⃣ Phase 1 — Understanding the chain (no code)
Four short comprehension questions on the four primitives the chain
stitches together. You answer them in your answers file.

> **Learning Objectives**
> - Explain DRAM row organisation and the RowHammer activation threshold
> - Describe the layout of a GPU PTE and the role of the aperture bit
> - Argue precisely why the IOMMU does not block this DMA write
> - Trace how a driver OOB write turns into a privilege escalation

### 2️⃣ Phase 2 — Must-finish: driving the attack to root
Five hands-on coding exercises that develop real attacker skills. Instead of 
just following instructions, you'll investigate DRAM geometry, optimize hammer 
strategies, analyze bit-level forensics, engineer robust payloads, and verify 
multi-stage attacks. Together they drive the attack from bit flip to root.

> **Learning Objectives**
> - Investigate DRAM geometry and handle edge cases in aggressor selection
> - Optimize RowHammer campaigns with timing constraints and strategy comparison
> - Perform bit-level forensic analysis of PTE corruption
> - Engineer robust payloads through memory layout analysis
> - Execute verified multi-stage privilege escalation with failure diagnosis

### 3️⃣ Phase 3 — Stretch: digging into the primitives
Optional short exercises: decode PTEs by hand, inspect the flipped bit,
budget the hammer timing, prove the IOMMU does exactly what it claims
(and no more), and craft tighter payloads.

> **Learning Objectives**
> - Work with PTE byte layouts at the bit level
> - Reason about hammer economics under refresh constraints
> - Distinguish IOMMU page-level enforcement from sub-page bounds checking
> - Understand why the cred struct sits where the attack needs it

### 4️⃣ Phase 4 — Debrief (discussion)
Open-ended questions linking the lab back to real-world attack timing,
ECC protection, IOMMU limits, and how an attacker would find the target
PTE row without privileged access.


## Setup

Create a file named `day6_gpubreach_answers.py` inside the
`day6-infrastructure` directory. This will be your answer file for this
lab.

If you see a code snippet in this instruction file, copy-paste it into
your answer file. Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your `day6_gpubreach_answers.py` file.**


```python


import sys
from collections.abc import Callable
from pathlib import Path

for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils import report

from gpubreach_sim import (
    make_environment,
    Environment,
    DRAM,
    # DRAM parameters
    HAMMER_THRESHOLD_ACTIVATIONS,
    ACTIVATE_PRECHARGE_NS,
    REFRESH_WINDOW_MS,
    ROW_SIZE_BYTES,
    ROWS_PER_BANK,
    # PTE symbols
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
    APERTURE_BIT_POS,
    PTE_BYTES,
    # DMA / driver symbols
    perform_gpu_dma,
    DRIVER_BUFFER_SIZE,
    CRED_OFFSET,
    PAGE_SIZE,
    # Target-specific attacker knowledge
    PTE_ROW,
    PTE_OFFSET_IN_ROW,
    VICTIM_GPU_VADDR,
    FLAG,
)

# A single environment you will drive through the attack. Re-run
# ``make_environment()`` any time you want a clean slate.
env: Environment = make_environment()

print("Initial environment:")
print(f"  PTE aperture        = {env.victim_pte.aperture} "
      f"(0 = GPU VRAM, 1 = system memory)")
print(f"  Kernel euid         = {env.kernel_cred.euid} "
      f"(0 would mean root)")
print(f"  Hammer threshold    = {HAMMER_THRESHOLD_ACTIVATIONS:,} "
      f"activations per aggressor")
print(f"  ACTIVATE-PRECHARGE  = {ACTIVATE_PRECHARGE_NS} ns")
print(f"  Refresh window      = {REFRESH_WINDOW_MS} ms")
print(f"  Driver buffer size  = {DRIVER_BUFFER_SIZE} bytes "
      f"(cred struct at offset {CRED_OFFSET} in a {PAGE_SIZE} byte page)")
```

## Simulator cheat sheet

The `gpubreach_sim` package is the simulated target — you will never edit
it, you will only *call into it* from your answers file. This cheat sheet
lists every symbol you'll touch, so you can refer back instead of hunting
through the imports.

**Entry points**

- `make_environment() -> Environment` — build a fresh target. Re-run any
  time you want a clean slate (useful between failed attempts).
- `env = make_environment()` — the `Environment` you'll mutate across the
  chain.
- `env.check_all()` — print a stage-by-stage report and, if all four
  stages succeeded, the flag.

**Attacker knowledge (constants you use as-is)**

- `PTE_ROW` — DRAM row holding the victim PTE.
- `PTE_OFFSET_IN_ROW` — byte offset of the PTE inside that row.
- `VICTIM_GPU_VADDR` — GPU virtual address whose PTE we corrupt.
- `FLAG` — the success flag (printed by `env.check_all()` on success).

**DRAM primitives** — on `env.dram` (class `DRAM`):

- `env.dram.hammer_once(aggressor_a, aggressor_b) -> int` — one round of
  double-sided hammering; returns the nanoseconds it cost. Only leaks
  charge into the victim row when `|a - b| == 2`.
- `env.dram.has_flipped(victim_row) -> bool` — True once a flip has
  landed in that row.
- `env.dram.read(row, offset, length) -> bytes` — read raw bytes from
  DRAM (used in the stretch track).
- `HAMMER_THRESHOLD_ACTIVATIONS`, `ACTIVATE_PRECHARGE_NS`,
  `REFRESH_WINDOW_MS`, `ROW_SIZE_BYTES`, `ROWS_PER_BANK` — DRAM
  parameters.

**GPU page-table primitives** — on `env.page_table` (class
`GPUPageTable`):

- `env.page_table.cached_pte` — the live PTE the GPU MMU is using. Has
  fields `.valid`, `.aperture`, `.physical_frame`.
- `env.page_table.sync_from_dram(env.dram)` — re-read the PTE from DRAM
  (models a TLB miss / invalidation).
- `APERTURE_GPU_LOCAL` (= 0), `APERTURE_SYSTEM` (= 1), `APERTURE_BIT_POS`
  (= 1), `PTE_BYTES` (= 8) — PTE layout constants.

**Driver / DMA primitives**

- `perform_gpu_dma(data, gpu_vaddr, page_table, iommu, gpu_dram,
  driver_page)` — the vulnerable driver fast path. Translates `gpu_vaddr`
  through the page table and performs the DMA. No length clamp.
- `env.iommu.validate(target_page, offset, length) -> bool` — probe the
  IOMMU's decision without actually performing a DMA (used in stretch
  3.5).
- `env.driver_page` — the host DMA-mapped page (class `DriverPage`).
- `env.kernel_cred.is_root() -> bool` — True iff the cred struct's euid
  is 0.
- `DRIVER_BUFFER_SIZE` (= 128), `CRED_OFFSET` (= 128), `PAGE_SIZE`
  (= 4096) — layout of the driver page.

Every call above is backed by a short docstring inside `gpubreach_sim/`
if you want to see what it does exactly.


## 1️⃣ Phase 1 — Understanding (30 min, no code)

Read each of the four comprehension exercises below and answer the
questions (plain-text comments in your answers file are fine). The
collapsed reference answers are there for when you finish, not to short-
circuit your thinking.

### Exercise 1.1: DRAM row organisation and the RowHammer threshold

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

A DRAM bank is a 2D grid of capacitor cells: rows of DRAM cells share a
single **row buffer**. Issuing an `ACTIVATE` copies a whole row into that
buffer. `PRECHARGE` closes the row. Every `ACTIVATE` perturbs
neighbouring rows a little — charge leaks across word-line coupling.

**Double-sided hammering** opens both `victim - 1` and `victim + 1` in
rapid succession so leakage piles up on both sides of the victim.
Accumulated leakage past a per-cell threshold flips a bit.

The device refreshes every row within `tREFW` (32–64ms in GDDR6). If the
attacker can't reach the flip threshold inside that window, the leakage
is wiped.

<details>
<summary><b>Question 1.1a:</b> Why two aggressors instead of one?</summary><blockquote>

Double-sided leaks charge into the victim from both sides every round;
effective leakage roughly doubles. This drops the required ACTIVATE count
by 2–5× on modern parts, from hundreds of millions (single-sided) to
~150k per aggressor (double-sided) on GDDR6. That is what makes the
attack practical inside a refresh window.
</blockquote></details>

<details>
<summary><b>Question 1.1b:</b> What does the refresh window buy the defender, and why is it not enough?</summary><blockquote>

It caps how long an attacker has to accumulate ACTIVATEs. In practice
150k activations × ~65ns per cycle ≈ 10–20ms of hammering, comfortably
inside a 32–64ms window — especially for adversarial code running on the
GPU itself, which can saturate the DRAM controller. Refreshing faster
costs bandwidth and still only moves the bar.
</blockquote></details>

<details>
<summary><b>Vocabulary: tREFI vs tREFW</b></summary><blockquote>

- **tREFI** (~1.9µs on GDDR6) — interval between REFRESH commands.
- **tREFW** (32–64ms) — the time in which every row gets refreshed at
  least once.

When this lab says "64ms refresh window" it means tREFW.
</blockquote></details>


### Exercise 1.2: GPU PTEs and the aperture bit

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

The GPU has its own MMU with 8-byte PTEs stored in VRAM. Each PTE holds:

* a **valid** bit,
* an **aperture** bit — 0 = page lives in GPU VRAM, 1 = page lives in
  host system memory (reached over PCIe),
* a physical frame number,
* permission / cache-control flags.

A single bit flip changes the *target memory space* of every subsequent
DMA through that virtual address.

<details>
<summary><b>Question 1.2a:</b> SECDED ECC is on the DRAM. Why does that not stop this attack?</summary><blockquote>

SECDED corrects 1-bit flips and detects 2-bit flips within a codeword.
GPUHammer shows that ECC-aware hammering patterns on A100/H100 drive two
flips per codeword, silently corrupting without raising a fault. The
attacker picks PTE locations whose paired flips line up with the hammer
template. ECC slows the attack down; it does not stop it.
</blockquote></details>

<details>
<summary><b>Question 1.2b:</b> After the aperture flips, the PFN bits in the PTE are unchanged. Why does the attacker still end up writing to a <em>useful</em> CPU page?</summary><blockquote>

Because they groomed host memory beforehand so that the PFN in the PTE
happens to match the driver's DMA-mapped page. Allocate + free cycles
until the kernel hands back the target PFN in a predictable position.
The coincidence is engineered, not luck. In this lab
`make_environment()` bakes it in.
</blockquote></details>


### Exercise 1.3: Why the IOMMU does not block this write

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

The IOMMU (Intel VT-d / AMD-Vi) enforces **page-granular** DMA isolation:
"this device may read/write this host physical page." It does not look
inside the page.

On the GPUBreach DMA the transaction comes from the GPU's PCIe BDF and
targets the driver's DMA-mapped page — legitimately mapped for that
device. The IOMMU signs off.

<details>
<summary><b>Question 1.3a:</b> What granularity does the IOMMU enforce, and why does that leave the OOB write unblocked?</summary><blockquote>

Page-level (4KB). It asks "is this page mapped for this device?" It does
not ask "is the write staying inside a sub-page software buffer?"
Enforcing sub-page bounds is the kernel's job — the driver's, in this
case — and that check is missing.
</blockquote></details>

<details>
<summary><b>Question 1.3b:</b> Would ATS / PASID change this?</summary><blockquote>

Not here. Both add context to IOMMU translations but still operate at
page granularity. They can narrow *which* pages a device may touch but
not enforce intra-page bounds inside a legitimately mapped page.
</blockquote></details>


### Exercise 1.4: Driver OOB → privilege escalation

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

The last link is a classic heap-adjacent out-of-bounds write. The
driver's DMA fast path copies caller-controlled `len(data)` bytes into
the DMA buffer at offset 0. It does not clamp `len(data)` to the
buffer's declared size. Immediately after the buffer, in the same page,
sits a kernel credential struct. Writing 0 into its `euid` gives the
caller root.

<details>
<summary><b>Question 1.4a:</b> Which CWE best captures the driver bug?</summary><blockquote>

**CWE-787: Out-of-bounds Write.** The write stays inside the allocated
page (IOMMU satisfied) but passes the software-defined end of the
buffer. Secondary fits: CWE-20 (missing input validation), CWE-119
(improper restriction of operations within buffer bounds).
</blockquote></details>

<details>
<summary><b>Question 1.4b:</b> Which kernel memory-safety mitigations <em>would</em> catch this, and which would not?</summary><blockquote>

**Catch it**: KASAN (redzones around allocations), a kernel rewritten in
Rust if the write goes through a bounds-checked slice, SLAB hardening
that moves the cred struct elsewhere.

**Don't catch it**: IOMMU (too coarse), stack canaries (no stack frame),
W^X / NX (data-only corruption), CFI / shadow stacks (no indirect call
redirection), SMEP / SMAP (protect against *CPU* user→kernel mistakes,
not PCIe DMA).
</blockquote></details>


## 2️⃣ Phase 2 — Must-finish: driving the attack to root

Five engaging coding exercises that develop real attacker skills. Instead of just 
following instructions, you'll investigate, optimize, analyze, and engineer solutions. 
Each exercise builds sophisticated understanding. Budget ~2 hours total. At the end,
`env.check_all()` prints the flag.

| # | Step | What you develop |
|---|------|----------------|
| 2.1 | Investigate DRAM geometry | `investigate_aggressor_geometry(victim, dram)` |
| 2.2 | Optimize hammer campaign | `execute_hammer_campaign(dram, a, b, victim)` |
| 2.3 | Forensic PTE analysis | `analyze_pte_corruption(env)` |
| 2.4 | Engineer robust payloads | `engineer_payload_variants(env)` |
| 2.5 | Verified privilege escalation | `execute_verified_escalation(env, payload)` |

**Expected output when Phase 2 succeeds.** After completing the investigation 
and optimization exercises, your terminal should show detailed analysis like this:

```text
🔍 Ex 2.1: DRAM geometry investigation
  Tested strategies: adjacent, wider, asymmetric  
  Edge cases handled: boundary conditions
  Final aggressors: (4241, 4243) using classic double-sided
  ✅ Geometry validation passed

⚡ Ex 2.2: Hammer campaign optimization
  Strategy comparison: steady=✅ burst=❌ adaptive=❌
  Best approach: steady (150k rounds, 19.50ms, 38% budget)
  ✅ Attack completed within refresh window

🔬 Ex 2.3: PTE corruption forensics
  Bit analysis: 1 flip at (byte=0, bit=1) - aperture bit
  Transition: 0 → 1 (GPU VRAM → system memory)
  ✅ Clean aperture flip verified

🔧 Ex 2.4: Payload engineering
  Layout analysis: buffer=128B, cred@offset=128, overflow=4B
  Payload variants: minimal=132B, robust=140B, surgical=132B
  ✅ Optimal payload selected

🎯 Ex 2.5: Verified escalation
  Pre-flight checks: ✅ all systems ready
  DMA execution: ✅ payload delivered
  Privilege verification: euid 1000 → 0
  ✅ Root achieved with full verification

── GPUBreach attack chain ──
  ✓ Stage 1 — DRAM geometry investigated and validated
  ✓ Stage 2 — Optimized hammer succeeded within timing budget  
  ✓ Stage 3 — PTE corruption forensically verified
  ✓ Stage 4 — Engineered payload achieved privilege escalation

  🎉 All stages succeeded — root achieved with attacker expertise.
  FLAG{gpubreach_rowhammer_aperture_oob_root}
```

**Debugging tips:** If exercises are taking much longer than expected, check your 
edge case handling in Exercise 2.1 and timing optimization in Exercise 2.2.

### Exercise 2.1: Investigate DRAM geometry for optimal aggressor selection

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Don't just compute `victim ± 1`! Real attackers investigate DRAM bank geometry 
to understand why certain row combinations work and others fail. You'll probe 
different strategies, handle edge cases, and validate your results.

Your mission: Discover which aggressor rows actually cause bit flips through 
systematic investigation. Some combinations that look mathematically correct 
don't work in practice due to bank boundaries and hardware constraints.


```python


def investigate_aggressor_geometry(victim_row: int, dram: DRAM) -> dict:
    """Investigate which aggressor rows actually cause bit flips.
    
    Don't just use arithmetic! Some combinations that look right don't work.
    Your job is to probe the DRAM to understand the actual constraints.
    
    Args:
        victim_row: The row containing the target PTE
        dram: DRAM instance for testing hammer effects
    
    Returns:
        {
            "aggressors": (upper, lower),
            "geometry_rule": str,  # Describe the pattern you discovered
            "edge_cases_handled": [str],  # What boundary conditions did you address?
            "validation_passed": bool
        }
    """
    # TODO: 1. Test different aggressor combinations around victim_row
    # TODO: 2. Use fresh environments per test to check if combinations work  
    # TODO: 3. Handle edge cases (victim near 0, near ROWS_PER_BANK-1)
    # TODO: 4. Discover the actual spacing rule (hint: it's not always ±1)
    # TODO: 5. Return the working combination with explanation
    
    # Test framework - use this pattern:
    # test_env = make_environment()
    # Try: test_env.dram.hammer_once(aggressor_a, aggressor_b)  
    # Check: test_env.dram.has_flipped(victim_row) after enough rounds
    
    return {
        "aggressors": (0, 0),
        "geometry_rule": "YOUR DESCRIPTION HERE", 
        "edge_cases_handled": [],
        "validation_passed": False
    }


# Multi-test verification across different scenarios
geometry_analysis = investigate_aggressor_geometry(PTE_ROW, env.dram)
print(f"🔍 Ex 2.1: DRAM geometry investigation")
print(f"  Final aggressors: {geometry_analysis['aggressors']}")
print(f"  Strategy: {geometry_analysis['geometry_rule']}")
print(f"  Edge cases: {geometry_analysis['edge_cases_handled']}")

agg_a, agg_b = geometry_analysis['aggressors']
from day6_gpubreach_test import test_find_aggressors


test_find_aggressors(lambda v: geometry_analysis['aggressors'])

env.stage1_aggressors_ok = True
```

### Exercise 2.2: Execute an optimized RowHammer campaign

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵⚪

Real attackers don't just hammer blindly. You'll implement intelligent strategies 
that balance speed, reliability, and stealth while staying within timing constraints.
Compare multiple approaches and optimize for real-world conditions.

**Why do different hammer strategies exist?**

In the real world, attackers face multiple constraints:
- **Speed**: Must flip bits before refresh window expires (~64ms)
- **Stealth**: Continuous hammering may trigger monitoring systems
- **Reliability**: Some patterns work better on different hardware
- **Resource constraints**: May need to share compute with legitimate workloads

**The three main strategy families:**

<details>
<summary><b>Strategy 1: Steady Hammering</b></summary><blockquote>

**Approach**: Hammer at constant maximum rate until flip occurs.

**Pros**: 
- Simple to implement
- Maximum speed - reaches threshold fastest
- Predictable timing

**Cons**:
- Most detectable (constant high memory traffic)
- May trigger thermal throttling
- No adaptation if pattern isn't working

**When to use**: When speed is critical and stealth isn't a concern.
</blockquote></details>

<details>
<summary><b>Strategy 2: Burst Hammering</b></summary><blockquote>

**Approach**: Hammer in rapid bursts separated by pauses.

**Pros**:
- More stealthy (periods of low activity)
- Allows thermal cooling between bursts
- Harder for monitoring systems to detect

**Cons**:
- Slower overall (time spent pausing)
- More complex timing calculations
- May exceed refresh window

**When to use**: When stealth matters and you have extra time budget.
</blockquote></details>

<details>
<summary><b>Strategy 3: Adaptive Hammering</b></summary><blockquote>

**Approach**: Start fast, then adapt based on progress. Terminate early if no signs of success.

**Pros**:
- Avoids wasting time on patterns that won't work
- Can detect hardware that's resistant to RowHammer
- Efficient use of timing budget

**Cons**:
- Complex progress detection logic
- May give up too early on slow-to-flip targets
- Requires understanding of flip probability curves

**When to use**: When you're uncertain if the target is vulnerable and want to fail fast.
</blockquote></details>

**How to choose your implementation approach:**

1. **Start with steady hammering** - it's the baseline that's guaranteed to work if anything will
2. **Add burst hammering** if you want to explore stealth considerations
3. **Add adaptive hammering** if you want to explore smart early termination

You don't need to implement all three - pick based on what you want to learn!

Your mission: Execute an intelligent hammer campaign, compare strategies, and understand the trade-offs between speed, stealth, and reliability.


```python


def execute_hammer_campaign(
    dram: DRAM,
    aggressor_a: int,
    aggressor_b: int,
    victim_row: int,
    max_time_ms: float = REFRESH_WINDOW_MS * 0.8,  # 80% of refresh window
) -> dict:
    """Execute an intelligent hammer campaign with multiple strategies.
    
    IMPLEMENTATION APPROACH:
    1. **REQUIRED**: Implement steady_hammer_strategy() - the baseline approach
    2. **OPTIONAL**: Add burst_hammer_strategy() for stealth exploration  
    3. **OPTIONAL**: Add adaptive_hammer_strategy() for smart termination
    4. **REQUIRED**: Pick best strategy and return detailed analysis
    
    MINIMUM VIABLE SOLUTION:
    - Just implement steady hammering (constant rate until flip)
    - Still provide timing budget tracking and efficiency metrics
    - Students can focus on one strategy and still get full educational value
    
    FULL IMPLEMENTATION (if time allows):
    - Try multiple strategies and compare their performance
    - Implement early termination logic
    - Measure flip probability curves over time
    
    Returns detailed analysis of the campaign.
    """
    max_time_ns = max_time_ms * 1_000_000
    
    # TODO: 1. REQUIRED: Implement steady_hammer_strategy() 
    # TODO: 2. OPTIONAL: Implement burst_hammer_strategy() (with pauses)
    # TODO: 3. OPTIONAL: Implement adaptive_hammer_strategy() (with early termination)
    # TODO: 4. REQUIRED: Pick best strategy and return analysis
    # TODO: 5. REQUIRED: Track timing budget and efficiency metrics
    
    # HINT: You can start with just steady strategy and expand if time allows!
    
    def steady_strategy():
        """REQUIRED: Constant rate hammering - implement this first!"""
        # TODO: Basic hammer loop (same as original Exercise 2.2):
        # total_ns = 0
        # rounds = 0  
        # while not dram.has_flipped(victim_row) and total_ns < max_time_ns:
        #     total_ns += dram.hammer_once(aggressor_a, aggressor_b)
        #     rounds += 1
        # return {"rounds": rounds, "time_ns": total_ns, "success": dram.has_flipped(victim_row)}
        return {"rounds": 0, "time_ns": 0, "success": False}
    
    def burst_strategy():
        """OPTIONAL: Burst hammering with stealth pauses."""
        # TODO: Implement if you want to explore stealth:
        # - Hammer in bursts of ~100 rounds
        # - Add 50µs pause between bursts  
        # - Track both hammer time + pause time
        return {"rounds": 0, "time_ns": 0, "success": False}
    
    def adaptive_strategy():
        """OPTIONAL: Adaptive rate with early termination."""
        # TODO: Implement if you want smart termination:
        # - Check progress every 25% of threshold
        # - Terminate early if no progress after 50% of threshold
        # - Implement "no progress" detection logic
        return {"rounds": 0, "time_ns": 0, "success": False}
    
    # Test strategies you've implemented
    strategies = [
        ("steady", steady_strategy),
        # ("burst", burst_strategy),      # Uncomment if you implemented it
        # ("adaptive", adaptive_strategy) # Uncomment if you implemented it
    ]
    
    print(f"  Testing {len(strategies)} hammer strategy(ies)...")
    
    results = []
    for name, strategy_func in strategies:
        print(f"    Testing {name} strategy...")
        result = strategy_func()
        result["name"] = name
        results.append(result)
        
        # Report strategy results immediately
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED" 
        print(f"      {status}: {result['rounds']:,} rounds in {result['time_ns']/1_000_000:.2f}ms")
    
    # Pick best successful strategy (fastest time)
    successful = [r for r in results if r["success"]]
    if successful:
        best = min(successful, key=lambda r: r["time_ns"])
        print(f"  🏆 Winner: {best['name']} strategy")
    else:
        best = max(results, key=lambda r: r["rounds"])
        print(f"  😞 All strategies failed, best attempt: {best['name']}")
    
    # ANALYSIS QUESTIONS for students to think about:
    # - Why did one strategy succeed while others failed?
    # - What are the trade-offs between speed and stealth?  
    # - How would you choose strategy in a real-world attack?
    
    return {
        "best_strategy": best["name"],
        "rounds": best["rounds"],
        "total_time_ms": best["time_ns"] / 1_000_000,
        "success": best["success"],
        "efficiency_score": best["rounds"] / (best["time_ns"] / 1_000_000),  # rounds/ms
        "all_strategies": results,
        "timing_budget_used": (best["time_ns"] / max_time_ns) * 100,  # percentage
    }


campaign = execute_hammer_campaign(env.dram, agg_a, agg_b, PTE_ROW)
print(f"⚡ Ex 2.2: Hammer campaign optimization")
print(f"  Best strategy: {campaign['best_strategy']} "
      f"({campaign['rounds']:,} rounds in {campaign['total_time_ms']:.2f}ms)")
print(f"  Success: {campaign['success']}, Budget used: {campaign['timing_budget_used']:.1f}%")

# For compatibility with existing test framework
flip_run = {
    "flipped": campaign["success"],
    "rounds": campaign["rounds"], 
    "total_ns": campaign["total_time_ms"] * 1_000_000
}

from day6_gpubreach_test import test_hammer_until_flip


# Create simple wrapper for test compatibility
def hammer_until_flip(dram, a, b, victim, max_rounds=2_000_000):
    campaign_result = execute_hammer_campaign(dram, a, b, victim)
    return {
        "flipped": campaign_result["success"],
        "rounds": campaign_result["rounds"],
        "total_ns": int(campaign_result["total_time_ms"] * 1_000_000)
    }

test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    campaign["success"]
    and campaign["total_time_ms"] < REFRESH_WINDOW_MS
)
```

### Exercise 2.3: Perform bit-level forensic analysis of PTE corruption

> **Difficulty**: 🔴🔴🔴⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Don't just trust that the flip worked. Real attackers perform forensic analysis 
to understand exactly what changed in the PTE and verify it's the corruption 
they intended. You'll analyze the corruption at the bit level.

Your mission: Perform detailed forensic analysis of the PTE bit flip. Capture
raw bytes before/after, manually decode PTE fields, verify ONLY the aperture 
bit flipped, and handle cases where multiple bits flipped unexpectedly.


```python


def analyze_pte_corruption(env: Environment) -> dict:
    """Perform detailed forensic analysis of the PTE bit flip.
    
    You have helper functions available, but must figure out the analysis workflow.
    Think like a forensic investigator: what evidence do you need to collect and when?
    """
    
    def decode_pte_raw(raw_bytes: bytes) -> dict:
        """Helper: Manually decode PTE without using simulator functions."""
        flags = raw_bytes[0]
        valid = bool(flags & 1)
        aperture = (flags >> 1) & 1
        physical_frame = int.from_bytes(raw_bytes[1:7], "little")
        return {
            "valid": valid,
            "aperture": aperture, 
            "physical_frame": physical_frame,
            "raw_hex": raw_bytes.hex()
        }
    
    def find_bit_differences(before: bytes, after: bytes) -> list:
        """Helper: Find exactly which bits changed between two byte arrays."""
        changes = []
        for byte_idx, (b1, b2) in enumerate(zip(before, after)):
            if b1 != b2:
                xor = b1 ^ b2
                for bit_pos in range(8):
                    if xor & (1 << bit_pos):
                        changes.append((byte_idx, bit_pos))
        return changes
    
    # TODO: Your forensic investigation workflow goes here!
    # Think about:
    # - What evidence do you need to collect?
    # - When should you capture the evidence (before/after what events)?
    # - How do you analyze the changes?
    # - What validations prove the attack worked as intended?
    # - What could go wrong and how would you detect it?
    
    # Hint: The MMU has cached vs. DRAM state - when does each matter?
    # Hint: You need to compare "before" and "after" - but before/after what?
    # Hint: Use the helper functions above, don't reimplement them!
    
    return {
        "aperture_transition": "? → ?",  # TODO: Fill this in
        "bit_changes": [],               # TODO: Fill this in  
        "validation": {
            "aperture_changed": False,   # TODO: Fill this in
            "clean_flip": False,         # TODO: Fill this in
            "side_effects": False        # TODO: Fill this in
        },
        "forensic_summary": "TODO: Investigate and describe what happened"
    }


forensics = analyze_pte_corruption(env)
print(f"🔬 Ex 2.3: PTE corruption forensics")
print(f"  Transition: {forensics['aperture_transition']}")
print(f"  Bit changes: {forensics['bit_changes']}")
print(f"  Clean flip: {'✅' if forensics['validation']['clean_flip'] else '❌'}")

# For test compatibility
before, after = (APERTURE_GPU_LOCAL, APERTURE_SYSTEM) if forensics['validation']['aperture_changed'] else (0, 0)

from day6_gpubreach_test import test_trigger_pte_refresh


# Create wrapper for test compatibility  
def trigger_pte_refresh(env):
    analysis = analyze_pte_corruption(env)
    return (APERTURE_GPU_LOCAL, APERTURE_SYSTEM) if analysis['validation']['aperture_changed'] else (0, 0)

test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = forensics['validation']['aperture_changed']
```

### Exercise 2.4: Engineer optimal payloads through memory layout analysis

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵

Real exploits need to work across different memory layouts and handle alignment
constraints. Don't just concatenate bytes! You'll probe the target memory layout,
design multiple payload variants, and select the optimal approach.

Your mission: Engineer robust payloads that work even if memory layout shifts
slightly. Probe the driver page layout, design minimal payloads, handle different
alignments, and validate payloads without triggering the DMA.


```python


def engineer_payload_variants(env: Environment) -> dict:
    """Design multiple payload variants for different scenarios.
    
    Your engineering tasks:
    1. Probe the driver page layout to confirm cred struct location
    2. Design minimal payloads (not just maximum size)
    3. Handle different alignment scenarios  
    4. Create payloads that work even if layout shifts slightly
    5. Validate payloads without actually triggering the DMA
    
    Real attackers need robust payloads that work across system variations.
    """
    
    def probe_memory_layout(driver_page) -> dict:
        """Investigate the actual memory layout of the driver page."""
        # TODO: 1. Read current contents of driver page
        # TODO: 2. Look for patterns that confirm cred struct location
        # TODO: 3. Check for any existing data that might interfere
        # TODO: 4. Measure exact distances between structures
        
        layout_info = {
            "buffer_start": 0,
            "buffer_size": DRIVER_BUFFER_SIZE,
            "cred_offset": CRED_OFFSET,
            "cred_euid_offset": CRED_OFFSET,  # offset of euid field within cred
            "page_size": PAGE_SIZE,
            "available_overflow": PAGE_SIZE - DRIVER_BUFFER_SIZE,
            "current_euid": env.kernel_cred.euid
        }
        
        return layout_info
    
    def craft_minimal_payload(target_euid: int, exact_offset: int) -> bytes:
        """Create the smallest possible payload that works."""
        # TODO: Calculate minimum bytes needed to reach target offset
        # TODO: Don't waste space with unnecessary filler
        minimal_size = exact_offset + 4  # just enough to reach euid + 4 bytes
        return b"YOUR_MINIMAL_PAYLOAD_HERE"
    
    def craft_robust_payload(target_euid: int, offset_range: tuple) -> bytes:
        """Create payload that works even if offset is slightly different."""
        min_offset, max_offset = offset_range
        # TODO: Design payload that works for any offset in the range
        # Strategy: write target_euid at multiple positions
        robust_payload = b"YOUR_ROBUST_PAYLOAD_HERE"
        return robust_payload
    
    def craft_surgical_payload(target_euid: int, layout: dict) -> bytes:
        """Create payload that only modifies euid, nothing else."""
        # TODO: Preserve existing memory content except for euid field
        # TODO: Read current memory, modify only 4 bytes, leave rest intact
        surgical_payload = b"YOUR_SURGICAL_PAYLOAD_HERE"
        return surgical_payload
    
    def validate_payload_safety(payload: bytes, layout: dict) -> dict:
        """Check payload won't corrupt unintended memory."""
        # TODO: 1. Verify payload length doesn't exceed page bounds
        # TODO: 2. Check payload won't overwrite critical structures
        # TODO: 3. Simulate the write and predict side effects
        
        safety_report = {
            "length_ok": len(payload) <= layout["page_size"],
            "within_overflow_budget": len(payload) <= layout["buffer_size"] + layout["available_overflow"],
            "targets_only_euid": False,  # TODO: implement check
            "predicted_side_effects": []  # TODO: list potential issues
        }
        return safety_report
    
    # Investigation phase
    layout = probe_memory_layout(env.driver_page)
    
    # Payload engineering phase
    payloads = {
        "minimal": craft_minimal_payload(0, layout["cred_euid_offset"]),
        "robust": craft_robust_payload(0, (layout["cred_euid_offset"]-4, layout["cred_euid_offset"]+4)), 
        "surgical": craft_surgical_payload(0, layout),
        "original": b"A" * DRIVER_BUFFER_SIZE + (0).to_bytes(4, "little")  # reference
    }
    
    # Validation phase
    analysis = {}
    for name, payload in payloads.items():
        analysis[name] = {
            "length": len(payload),
            "efficiency": len(payload) / len(payloads["original"]),  # smaller is better
            "safety": validate_payload_safety(payload, layout)
        }
    
    # TODO: Pick the best payload based on your criteria
    recommended = "minimal"  # or whatever you think is best
    
    return {
        "layout_analysis": layout,
        "payload_variants": {name: payload.hex() for name, payload in payloads.items()},
        "analysis": analysis,
        "recommended": recommended,
        "rationale": "TODO: Explain why you picked this payload variant"
    }


payload_engineering = engineer_payload_variants(env)
print(f"🔧 Ex 2.4: Payload engineering")
print(f"  Recommended variant: {payload_engineering['recommended']}")
print(f"  Length: {len(bytes.fromhex(payload_engineering['payload_variants'][payload_engineering['recommended']]))} bytes")

# Use the recommended payload for the attack
payload = bytes.fromhex(payload_engineering['payload_variants'][payload_engineering['recommended']])

# For test compatibility
def craft_overflow_payload(new_euid: int = 0) -> bytes:
    return b"A" * DRIVER_BUFFER_SIZE + new_euid.to_bytes(4, "little")

from day6_gpubreach_test import test_craft_overflow_payload


test_craft_overflow_payload(craft_overflow_payload)
```

### Exercise 2.5: Execute multi-stage verified privilege escalation

> **Difficulty**: 🔴🔴🔴🔴⚪
> **Importance**: 🔵🔵🔵🔵🔵

Real attackers don't just fire and hope. You'll implement a complete attack with 
stage-by-stage verification, failure analysis, and cleanup. Learn to verify each 
step works and diagnose failures like a professional penetration tester.

Your mission: Execute privilege escalation with comprehensive verification. 
Perform pre-flight checks, monitor DMA execution, verify each stage, analyze 
failures, and clean up evidence.


```python


def execute_verified_escalation(env: Environment, payload: bytes) -> dict:
    """Execute privilege escalation with comprehensive verification.
    
    Your mission: 
    1. Pre-flight checks before launching the attack
    2. Execute the DMA with intermediate monitoring  
    3. Post-attack verification of privilege escalation
    4. Failure analysis if something goes wrong
    5. Clean up any evidence of the attack
    
    Real attackers need to be sure their exploit worked AND why it failed if not.
    """
    
    def pre_flight_checks(env: Environment, payload: bytes) -> dict:
        """Verify conditions are right for the attack."""
        # TODO: Check all prerequisites are met
        checks = {
            "aperture_bit_flipped": env.page_table.cached_pte.aperture == APERTURE_SYSTEM,
            "pte_points_to_driver_page": False,  # TODO: verify PTE PFN matches driver page
            "iommu_allows_write": False,  # TODO: use env.iommu.validate() to check
            "payload_size_correct": False,  # TODO: verify payload will reach euid field
            "current_privileges": env.kernel_cred.euid,
        }
        
        all_ready = all(checks[k] for k in ["aperture_bit_flipped", "pte_points_to_driver_page", "iommu_allows_write"])
        checks["ready_to_attack"] = all_ready
        
        return checks
    
    def monitor_dma_execution(payload: bytes, victim_vaddr: int, env: Environment) -> dict:
        """Execute DMA with detailed monitoring."""
        # TODO: 1. Capture before state
        before_state = {
            "kernel_euid": env.kernel_cred.euid,
            "is_root": env.kernel_cred.is_root(),
            "driver_page_checksum": "TODO",  # hash of driver page content
        }
        
        # TODO: 2. Execute the DMA
        try:
            perform_gpu_dma(payload, victim_vaddr, env.page_table, 
                          env.iommu, env.dram, env.driver_page)
            dma_success = True
            dma_error = None
        except Exception as e:
            dma_success = False
            dma_error = str(e)
        
        # TODO: 3. Capture after state
        after_state = {
            "kernel_euid": env.kernel_cred.euid,
            "is_root": env.kernel_cred.is_root(),
            "driver_page_checksum": "TODO",  # hash of driver page content
        }
        
        return {
            "dma_success": dma_success,
            "dma_error": dma_error,
            "before": before_state,
            "after": after_state,
            "privilege_escalated": before_state["kernel_euid"] != 0 and after_state["kernel_euid"] == 0
        }
    
    def diagnose_failure(checks: dict, execution: dict) -> dict:
        """Figure out why the attack failed and suggest fixes."""
        if execution["privilege_escalated"]:
            return {"success": True, "diagnosis": "Attack succeeded!"}
        
        # TODO: Systematic failure analysis
        failure_reasons = []
        
        if not checks["ready_to_attack"]:
            failure_reasons.append("Pre-flight checks failed")
            # TODO: Detail which specific checks failed
        
        if not execution["dma_success"]:
            failure_reasons.append(f"DMA failed: {execution['dma_error']}")
        
        if execution["dma_success"] and not execution["privilege_escalated"]:
            failure_reasons.append("DMA succeeded but euid not modified")
            # TODO: Check if we hit the wrong offset, wrong payload format, etc.
        
        return {
            "success": False,
            "failure_reasons": failure_reasons,
            "suggested_fixes": [
                "TODO: Add specific suggestions based on failure_reasons"
            ],
            "debug_info": {
                "checks": checks,
                "execution": execution
            }
        }
    
    def cleanup_attack_traces(env: Environment) -> dict:
        """Remove evidence of the attack (optional stealth exercise)."""
        # TODO: 1. Reset any modified memory that's not the target euid
        # TODO: 2. Clear any logs or traces that might reveal the attack
        # TODO: 3. Restore original memory layout where possible
        
        cleanup_actions = [
            "TODO: List cleanup actions taken"
        ]
        
        return {
            "cleanup_performed": cleanup_actions,
            "stealth_rating": "TODO: Rate how stealthy the attack was (1-10)"
        }
    
    # Execute the full attack pipeline
    print("🎯 Phase 1: Pre-flight checks")
    checks = pre_flight_checks(env, payload)
    for check, result in checks.items():
        print(f"   {check}: {'✅' if result else '❌'}")
    
    if not checks["ready_to_attack"]:
        return {"success": False, "error": "Pre-flight checks failed", "details": checks}
    
    print("🎯 Phase 2: DMA execution")
    execution = monitor_dma_execution(payload, VICTIM_GPU_VADDR, env)
    print(f"   DMA success: {execution['dma_success']}")
    print(f"   Privilege escalated: {execution['privilege_escalated']}")
    
    print("🎯 Phase 3: Results analysis")
    diagnosis = diagnose_failure(checks, execution)
    
    if diagnosis["success"]:
        print("   🎉 Attack succeeded!")
        cleanup = cleanup_attack_traces(env)
    else:
        print("   ❌ Attack failed")
        for reason in diagnosis["failure_reasons"]:
            print(f"      - {reason}")
        cleanup = {}
    
    return {
        "success": diagnosis["success"],
        "pre_flight": checks,
        "execution": execution,
        "diagnosis": diagnosis,
        "cleanup": cleanup,
        "final_euid": env.kernel_cred.euid,
        "achieved_root": env.kernel_cred.is_root()
    }


# Execute the full verified attack
attack_result = execute_verified_escalation(env, payload)
print(f"🎯 Ex 2.5: Verified escalation complete")
print(f"   Result: {'✅ SUCCESS' if attack_result['success'] else '❌ FAILED'}")
print(f"   Current euid: {attack_result['final_euid']}")
print(f"   Root achieved: {attack_result['achieved_root']}")

# For test compatibility
rooted = attack_result["achieved_root"]

from day6_gpubreach_test import test_escalate_privileges


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    perform_gpu_dma(payload, VICTIM_GPU_VADDR, env.page_table, 
                   env.iommu, env.dram, env.driver_page)
    return env.kernel_cred.is_root()

test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted
```

### Print the flag

If every stage above succeeded, `env.check_all()` prints the flag.
Otherwise it tells you which stage still needs work.


```python

env.check_all()
```

## 3️⃣ Phase 3 — Stretch: digging into the primitives

Optional exercises. Each is short and independent — cherry-pick.

### Exercise 3.1 (Optional): Decode a PTE by hand

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Prove you understand the PTE byte layout by parsing an 8-byte PTE into a
dict of the form `{"valid": bool, "aperture": int, "physical_frame": int}`
without using `gpubreach_sim.pte.decode_pte`.

Recall the layout (from the PTE module docstring):

* byte 0: flags (bit 0 = valid, bit 1 = aperture, bits 2–7 reserved)
* bytes 1–6: 48-bit little-endian physical frame number
* byte 7: reserved


```python


def decode_pte_manually(raw: bytes) -> dict:
    """Hand-decoded PTE — do not call gpubreach_sim.decode_pte."""
    # TODO: pick apart raw[0], raw[1:7], bit by bit.
    return {"valid": False, "aperture": 0, "physical_frame": 0}


# A sample PTE: valid=1, aperture=1, PFN=0xABCDEF
sample = bytes([0b0000_0011]) + (0xABCDEF).to_bytes(6, "little") + b"\x00"
print(f"Ex 3.1: decoded = {decode_pte_manually(sample)}")
from day6_gpubreach_test import test_decode_pte_manually


test_decode_pte_manually(decode_pte_manually)
```

### Exercise 3.2 (Optional): Inspect the exact flipped bit

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Compare the PTE's raw bytes in DRAM before and after the RowHammer flip
and return the set of (byte_offset, bit_position) pairs that changed.
There should be exactly one.

You may use `env.dram.read(row, offset, length)` to grab bytes, and
you'll need a fresh environment so the "before" snapshot is pristine.


```python


def find_flipped_bits(before: bytes, after: bytes) -> set[tuple[int, int]]:
    """Return {(byte_index, bit_position), ...} where before != after."""
    # TODO: iterate over pairs of bytes, XOR them, and record each
    # differing bit as (byte_index, bit_position).
    return set()


fresh3 = make_environment()
pre = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
while not fresh3.dram.has_flipped(PTE_ROW):
    fresh3.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
post = fresh3.dram.read(PTE_ROW, PTE_OFFSET_IN_ROW, PTE_BYTES)
print(f"Ex 3.2: flipped bits = {find_flipped_bits(pre, post)}")
from day6_gpubreach_test import test_find_flipped_bits


test_find_flipped_bits(find_flipped_bits)
```

### Exercise 3.3 (Optional): Budget the hammer against the refresh window

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Before hammering, you'd want to know: will we even finish the activations
inside the refresh window? Compute:

1. The minimum number of hammer *rounds* to cross the flip threshold.
   Each round hits both aggressors, so each round adds one activation
   per aggressor.
2. The simulated nanoseconds those rounds will cost. Each round costs
   `2 * ACTIVATE_PRECHARGE_NS`.
3. Whether that total time fits inside the refresh window.

Return a dict with keys `"rounds"`, `"total_ns"`, `"total_ms"`,
`"fits_refresh_window"`.


```python


def hammer_budget(
    threshold: int = HAMMER_THRESHOLD_ACTIVATIONS,
    tRC_ns: int = ACTIVATE_PRECHARGE_NS,
    refresh_ms: int = REFRESH_WINDOW_MS,
) -> dict:
    """Compute worst-case hammering cost before any RowHammer attempt."""
    # TODO:
    #   rounds = threshold  (one activation per aggressor per round)
    #   total_ns = rounds * 2 * tRC_ns
    #   total_ms = total_ns / 1_000_000
    #   return the dict.
    return {
        "rounds": 0,
        "total_ns": 0,
        "total_ms": 0.0,
        "fits_refresh_window": False,
    }


budget = hammer_budget()
print(
    f"Ex 3.3: {budget['rounds']:,} rounds × {2 * ACTIVATE_PRECHARGE_NS} ns "
    f"= {budget['total_ms']:.2f} ms "
    f"(fits 64ms window: {budget['fits_refresh_window']})"
)
from day6_gpubreach_test import test_hammer_budget


test_hammer_budget(hammer_budget)
```

### Exercise 3.4 (Optional): Maximum hammer rounds inside the window

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Flip the budget question around: given the refresh window, how *many*
rounds could the attacker fit at most? Compare it to
`HAMMER_THRESHOLD_ACTIVATIONS` — how comfortably do we fit?


```python


def max_rounds_in_window(
    refresh_ms: int = REFRESH_WINDOW_MS,
    tRC_ns: int = ACTIVATE_PRECHARGE_NS,
) -> int:
    """Maximum number of double-sided rounds that fit in the refresh window."""
    # TODO: per_round = 2 * tRC_ns; budget_ns = refresh_ms * 1_000_000;
    # return budget_ns // per_round.
    return 0


max_rounds = max_rounds_in_window()
headroom = max_rounds / HAMMER_THRESHOLD_ACTIVATIONS
print(
    f"Ex 3.4: up to {max_rounds:,} rounds fit in {REFRESH_WINDOW_MS} ms "
    f"→ {headroom:.1f}× threshold headroom"
)
from day6_gpubreach_test import test_max_rounds_in_window


test_max_rounds_in_window(max_rounds_in_window)
```

### Exercise 3.5 (Optional): The IOMMU blocks what it promises to block

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Prove the IOMMU is doing its job — it *does* block writes to physical
pages it has not mapped for the GPU. Use `env.iommu.validate(page, offset,
length)` to confirm:

1. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE` are
   allowed.
2. Writes to `env.driver_page` at offset 0 with length `PAGE_SIZE + 1`
   are rejected (cross a page boundary).
3. Writes to a *different* `DriverPage` instance are rejected (not
   mapped).

Return a dict with three booleans: `"intra_page_ok"`, `"overflow_page"`,
`"other_page"`.


```python

from gpubreach_sim.dma import DriverPage  # noqa: E402


def probe_iommu(env: Environment) -> dict:
    """Probe the IOMMU's enforcement envelope."""
    # TODO: call env.iommu.validate(...) three times as described above
    # and return the three results as booleans in the dict.
    return {
        "intra_page_ok": False,
        "overflow_page": False,
        "other_page": False,
    }


probe = probe_iommu(env)
print(f"Ex 3.5: IOMMU probe = {probe}")
from day6_gpubreach_test import test_probe_iommu


test_probe_iommu(probe_iommu)
```

### Exercise 3.6 (Optional): Measure the OOB overflow precisely

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given a DMA payload length, how many bytes overflow past the driver
buffer into adjacent kernel memory? Return 0 if no overflow.


```python


def overflow_bytes(payload_len: int) -> int:
    """Bytes past ``DRIVER_BUFFER_SIZE`` that land in adjacent memory."""
    # TODO: return max(0, payload_len - DRIVER_BUFFER_SIZE).
    return 0


for n in [0, DRIVER_BUFFER_SIZE - 1, DRIVER_BUFFER_SIZE, DRIVER_BUFFER_SIZE + 4]:
    print(f"Ex 3.6: payload {n}B → overflow {overflow_bytes(n)}B")
from day6_gpubreach_test import test_overflow_bytes


test_overflow_bytes(overflow_bytes)
```

### Exercise 3.7 (Optional): A tighter payload

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

The payload in Exercise 2.4 overshoots — it writes 132 bytes where 132 is
exactly `DRIVER_BUFFER_SIZE + 4`. What if the cred struct's euid field
isn't at the very start of the overflow region, but at some `offset`
past `CRED_OFFSET`? Write a parameterised payload builder.

The new signature:
`craft_precise_payload(cred_offset_in_page: int, new_euid: int) → bytes`

The payload should have length `cred_offset_in_page + 4` so that the last
4 bytes land exactly at `cred_offset_in_page` inside the driver page —
when the driver copies starting at `DRIVER_BUFFER_OFFSET = 0`.


```python


def craft_precise_payload(cred_offset_in_page: int, new_euid: int) -> bytes:
    """Place ``new_euid`` (LE-4) at ``cred_offset_in_page`` in the page."""
    # TODO: payload = (cred_offset_in_page bytes of filler) +
    #        (new_euid as little-endian 4 bytes)
    return b""


tight = craft_precise_payload(CRED_OFFSET, 0)
print(f"Ex 3.7: precise payload is {len(tight)} bytes")
from day6_gpubreach_test import test_craft_precise_payload


test_craft_precise_payload(craft_precise_payload)
```

## 4️⃣ Phase 4 — Debrief (15 min discussion)

Discuss the following with your partner and then with a TA. These
questions connect the lab numbers back to real-world attack economics.

<details>
<summary><b>Question 4.1:</b> In the lab we saw a flip land in ~20ms against a 64ms window — comfortable headroom. What about the real world?</summary><blockquote>

In the GPUHammer paper, on an RTX 3080 the authors measured ~9ms to the
first flip on double-sided patterns against tREFW ≈ 32ms — about a 3×
safety margin. On A100s the margin is smaller (and ECC complicates
things) but still positive. Mitigations that multiply the refresh rate
by 2× eat bandwidth and do not close the gap; mitigations like pTRR
(probabilistic Target Row Refresh) have been shown to be bypassable with
more careful activation patterns.

The headroom is *shrinking* with each process node because cells are
smaller → leak faster → lower activation threshold, while tREFW barely
changes. The trend line is against the defender.
</blockquote></details>

<details>
<summary><b>Question 4.2:</b> Why is SECDED ECC not a sufficient mitigation?</summary><blockquote>

* SECDED fixes 1-bit and detects 2-bit per codeword. RowHammer can
  produce multiple flips per row, and GPUHammer shows *ECC-aware*
  hammering patterns that induce paired flips that ECC silently accepts.
* Even when ECC raises a fault, the cost model is asymmetric: a DoS
  (ECC uncorrectable → kernel panic) may be easier to trigger than a
  targeted flip, but neither helps the attacker if they have chosen
  their targets well; RowHammer escalations only need one good flip per
  attempt.
* On-die ECC (standard on GDDR6/GDDR6X and DDR5, and increasingly
  mandatory) protects individual banks but is invisible to the memory
  controller — flips masked by on-die ECC don't raise rank-level machine
  checks, so the defender loses the signal that something is being
  hammered at all.
</blockquote></details>

<details>
<summary><b>Question 4.3:</b> Why is the IOMMU, which we all rely on for DMA isolation, not the right tool here?</summary><blockquote>

Page granularity. The IOMMU's abstraction is "may device X touch
physical page P?", which is the right abstraction for malicious or buggy
devices writing to unrelated memory. It is the wrong abstraction for a
bug-within-a-mapped-page, because software — the driver — subdivides the
page into objects and owns the sub-page bounds. Moving the cred struct
out of that page (SLAB hardening, per-cred page allocations) is the
correct structural mitigation; IOMMU tweaks are not.
</blockquote></details>

<details>
<summary><b>Question 4.4:</b> How would a real attacker find the target PTE row from an unprivileged process?</summary><blockquote>

They combine several techniques:

* **Templating** — run an offline characterisation of the target GPU in a
  CI environment or on their own hardware to build a map of "these
  (row, column, bit) locations flip reliably under this hammer pattern."
  In GPUHammer the template is sold as a probability distribution over
  bit flips per sub-page, good enough to find PTEs with the desired
  aperture-bit alignment.
* **Memory massaging** — spray GPU allocations with controlled page
  tables via CUDA, OpenGL, or Vulkan, until a PT page lands in a row
  the template says is flippable. This is statistical but typically
  reliable within a few seconds on an idle GPU.
* **Side-channel row identification** — on some GPUs, DRAM access
  patterns leak via timing (row-buffer hits vs misses). An unprivileged
  kernel can measure this to learn which virtual pages are co-located
  in the same DRAM row.
* **Information-leak primitives in the driver** — many GPU drivers
  historically expose uninitialised VRAM or residual memory through
  compute shaders, which lets the attacker read PTE bits directly.

Defence-in-depth: isolate tenant GPUs, use MIG on NVIDIA, or put the
sensitive workload inside a Confidential VM with a signed GPU partition.
</blockquote></details>


## Summary

You drove an end-to-end GPUBreach-style chain against a simulated target:

1. **RowHammer** flipped a specific bit in a GPU DRAM row — well inside
   the 64ms refresh window.
2. The flip landed on the **aperture bit** of a PTE, silently redirecting
   a GPU virtual address from VRAM to host memory.
3. The next DMA followed the corrupted PTE into a **driver-managed DMA
   buffer** on the CPU side, with the **IOMMU approving the write** at
   page granularity.
4. An **out-of-bounds** DMA copy overflowed the buffer into an adjacent
   kernel **credential struct**, flipping `euid` to 0 and escalating to
   root.

### Key takeaways

* RowHammer is **not a hardware quirk** — it is now a practical primitive
  against GPU memory (GPUHammer) with enough headroom against refresh
  that it finishes in tens of milliseconds.
* A **single bit flip** on a well-chosen PTE field (valid, aperture,
  permission bits) changes the semantics of every subsequent memory
  access through that entry.
* The **IOMMU is a page-level tool**. It will not save you from
  *intra-page* software bugs, including OOB writes inside a legitimately
  mapped DMA buffer.
* **Defence in depth** is structural, not magical: move sensitive
  structs out of DMA-mapped pages (SLAB hardening, per-cred allocations,
  KASAN), template refresh rates tightly against measured hammer budgets
  in your fleet, and put tenant workloads behind MIG or Confidential
  Computing boundaries.

### Further reading

- [GPUHammer: Rowhammer Attacks on GPU Memories are Practical](https://gpuhammer.com/) — the practical GDDR6 result this lab is modelled on
- [Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf) — the original RowHammer paper (Kim et al., ISCA 2014)
- [Exploiting the DRAM rowhammer bug to gain kernel privileges](https://googleprojectzero.blogspot.com/2015/03/exploiting-dram-rowhammer-bug-to-gain.html) — Project Zero's CPU-side PTE-flip exploit, direct ancestor of the aperture-flip primitive
- [TRRespass](https://www.vusec.net/projects/trrespass/) — breaking TRR-based RowHammer mitigations
- [Linux IOMMU API documentation](https://docs.kernel.org/driver-api/iommu.html) — how `dma_map_*` and page-level DMA isolation are implemented in practice
- [CWE-787: Out-of-bounds Write](https://cwe.mitre.org/data/definitions/787.html) — the driver-side bug class
