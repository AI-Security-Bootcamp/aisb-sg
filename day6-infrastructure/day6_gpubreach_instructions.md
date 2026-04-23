
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
- **Phase 2 — Must-finish track** (~30–45 min): five tiny coding
  exercises, one per step of the chain. Together they drive the attack
  from bit flip to printed flag.
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
Exactly five bite-sized coding exercises, each with a test — one per
step of the chain. Together they bit-flip a PTE, re-walk it through the
GPU MMU, and DMA an oversized payload into a kernel cred struct to get
root.

> **Learning Objectives**
> - Compute aggressor rows for double-sided hammering
> - Drive the hammer loop in cycle-accurate terms
> - Force the GPU MMU to pick up the corrupted entry
> - Craft a payload that exploits an intra-page OOB
> - Observe an end-to-end privilege escalation

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

Five tiny coding exercises — one for each step in the chain. Each has a
test you run immediately after. Budget ~30–45 minutes total. At the end,
`env.check_all()` prints the flag.

| # | Step | What you write |
|---|------|----------------|
| 2.1 | Pick the right DRAM rows to hammer | `find_aggressors(victim_row)` |
| 2.2 | Hammer until a bit flips | `hammer_until_flip(dram, a, b, row)` |
| 2.3 | Make the GPU MMU pick up the flip | `trigger_pte_refresh(env)` |
| 2.4 | Build an oversized DMA payload | `craft_overflow_payload(euid=0)` |
| 2.5 | Fire the DMA and confirm root | `escalate_privileges(env, payload)` |

**Expected output when Phase 2 succeeds.** After running every cell
through to `env.check_all()`, your terminal should look like this (exact
numbers for rounds/ns will match on every machine because the flip row
is deterministic):

```text
Ex 2.1: aggressors for PTE_ROW=4242 → 4241, 4243
  Aggressor geometry correct!
Ex 2.2: flipped=True after 150,000 rounds (19.50 ms)
  Hammer loop and cycle accounting correct!
Ex 2.3: aperture 0 → 1 (expected 0 → 1)
  PT resync propagated the flip!
Ex 2.4: payload=132 bytes (128 filler + 4 euid)
  Payload layout correct!
Ex 2.5: root achieved? True
  End-to-end escalation succeeded!
── GPUBreach attack chain ──
  ✓ Stage 1 — aggressor rows identified
  ✓ Stage 2 — flip landed inside the 64ms refresh window
  ✓ Stage 3 — aperture bit flipped in the live PTE
  ✓ Stage 4 — OOB DMA wrote euid=0 into the cred struct

  🎉 All stages succeeded — root achieved.
  FLAG{gpubreach_rowhammer_aperture_oob_root}
```

If Phase 2 is taking **minutes** instead of **milliseconds**, you almost
certainly have a `|a - b| ≠ 2` bug in `find_aggressors` — double-check
Exercise 2.1 before anything else.

### Exercise 2.1: Aggressor rows for double-sided hammering

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Given the row that holds the target PTE, return the two aggressor rows
that sandwich it. `DRAM.hammer_once(agg_a, agg_b)` only leaks into the
victim row when `|agg_a - agg_b| == 2` with the victim between them.


```python


def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    # TODO: Return (upper, lower) such that:
    #   1. The two values differ by exactly 2.
    #   2. Their midpoint is `victim_row`.
    pass


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 2.1: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")
from day6_gpubreach_test import test_find_aggressors


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True
```

### Exercise 2.2: Hammer until a bit flips

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

Now drive the loop against the live DRAM. Call
`env.dram.hammer_once(agg_a, agg_b)` repeatedly, summing the ns it
returns, until `env.dram.has_flipped(PTE_ROW)` becomes True. Return the
number of rounds and the total ns.


```python


def hammer_until_flip(
    dram: DRAM,
    aggressor_a: int,
    aggressor_b: int,
    victim_row: int,
    max_rounds: int = 2_000_000,
) -> dict:
    """Drive the DRAM into a RowHammer flip on ``victim_row``."""
    # TODO:
    # 1. Initialise total_ns and rounds counters.
    # 2. While `dram.has_flipped(victim_row)` is False (and you are
    #    below `max_rounds`):
    #      - call dram.hammer_once(aggressor_a, aggressor_b)
    #      - add its return value (nanoseconds) to total_ns
    #      - increment rounds
    # 3. Return {"rounds": rounds, "total_ns": total_ns,
    #            "flipped": dram.has_flipped(victim_row)}
    return {"rounds": 0, "total_ns": 0, "flipped": False}


flip_run = hammer_until_flip(env.dram, agg_a, agg_b, PTE_ROW)
print(
    f"Ex 2.2: flipped={flip_run['flipped']} after "
    f"{flip_run['rounds']:,} rounds "
    f"({flip_run['total_ns'] / 1_000_000:.2f} ms)"
)
from day6_gpubreach_test import test_hammer_until_flip


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)
```

### Exercise 2.3: Force the MMU to re-walk the flipped PTE

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵⚪

The DRAM bit is flipped, but the GPU MMU's cached copy still says
"aperture = GPU VRAM". Call `page_table.sync_from_dram(env.dram)` to
re-read the PTE bytes from DRAM, then return the `(before, after)` pair
of aperture values.

A real attacker triggers this re-walk with a GPU context switch, an
explicit TLB flush from the driver, or simply by waiting for the cache
line to be evicted.


```python


def trigger_pte_refresh(env: Environment) -> tuple[int, int]:
    """Resync the PT from DRAM. Return (before_aperture, after_aperture)."""
    # TODO:
    # 1. Read env.page_table.cached_pte.aperture into `before`.
    # 2. Call env.page_table.sync_from_dram(env.dram).
    # 3. Read env.page_table.cached_pte.aperture into `after`.
    # 4. Return (before, after).
    return 0, 0


before, after = trigger_pte_refresh(env)
print(f"Ex 2.3: aperture {before} → {after} (expected 0 → 1)")
from day6_gpubreach_test import test_trigger_pte_refresh


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)
```

### Exercise 2.4: Craft the OOB DMA payload

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

You need a byte string for the DMA payload such that:

1. Its length is exactly `DRIVER_BUFFER_SIZE + 4` bytes. The first
   `DRIVER_BUFFER_SIZE` bytes fill the driver buffer; the last 4 bytes
   overflow into the `euid` field of the cred struct.
2. The last 4 bytes encode the integer `0` (root's euid) as a 4-byte
   little-endian number — matching how `KernelCred.euid` is serialised.

You can put any content in the first `DRIVER_BUFFER_SIZE` bytes. The
convention is to use `b"A"` so the hexdump is easy to read.


```python


def craft_overflow_payload(new_euid: int = 0) -> bytes:
    """Return a DMA payload that overflows the driver buffer into euid."""
    # TODO:
    # 1. Fill `DRIVER_BUFFER_SIZE` bytes (e.g. b"A" repeated).
    # 2. Append the little-endian 4-byte encoding of `new_euid`.
    # 3. Return filler + euid_bytes.
    return b""


payload = craft_overflow_payload()
print(
    f"Ex 2.4: payload={len(payload)} bytes "
    f"({DRIVER_BUFFER_SIZE} filler + 4 euid)"
)
from day6_gpubreach_test import test_craft_overflow_payload


test_craft_overflow_payload(craft_overflow_payload)
```

### Exercise 2.5: Fire the DMA and confirm root

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Hand the payload to `perform_gpu_dma`. The simulator resolves the PTE,
validates the transaction with the IOMMU (which approves — the page is
mapped), and writes the payload into the driver page. The overflow lands
on the cred struct and `env.kernel_cred.is_root()` flips to True.

Call `perform_gpu_dma` with the positional arguments
`(payload, VICTIM_GPU_VADDR, env.page_table, env.iommu, env.dram,
env.driver_page)`, then return `env.kernel_cred.is_root()`.


```python


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    # TODO:
    # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
    #    env.page_table, env.iommu, env.dram, env.driver_page).
    # 2. Return env.kernel_cred.is_root().
    return False


rooted = escalate_privileges(env, payload)
print(f"Ex 2.5: root achieved? {rooted}")
from day6_gpubreach_test import test_escalate_privileges


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
