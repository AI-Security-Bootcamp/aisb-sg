# GPUBreach Lab — Instructor / TA Notes

These notes are for the TAs running the room. They cover expected
timing, common pitfalls by exercise, a fast-forward snippet for students
stuck on early steps, and full-prose answers to the Phase-4 debrief
questions (an expanded version of the collapsibles in the solution).

**Do not share this file with students.** The answer key lives in
`day6_gpubreach_reference.py`.

## Before the session

1. `python day6-infrastructure/smoke_test.py` on the reference machine
   should print `OK`. If anyone reports a FAIL, their venv / Python
   version is wrong — fix before they open the lab.
2. The reference file runs the full chain end-to-end in under a second
   on a laptop:
   ```bash
   time .venv/bin/python day6-infrastructure/day6_gpubreach_reference.py
   # real ~0.5s, user ~0.4s
   ```
   If a student says "it's been running for 5 minutes", something is
   wrong — almost always a `|a-b| ≠ 2` bug in `find_aggressors`.

## Expected per-exercise runtimes

| Exercise | Expected wall time | If it's much longer |
|----------|-------------------|---------------------|
| 2.1 `find_aggressors` | instant | — |
| 2.2 `hammer_until_flip` | ~0.3s (150k rounds) | bad aggressors → `max_rounds=2M` burned → ~5s then "not flipped" |
| 2.3 `trigger_pte_refresh` | instant | student forgot `sync_from_dram` |
| 2.4 `craft_overflow_payload` | instant | — |
| 2.5 `escalate_privileges` | instant | student passed wrong env (one that isn't flipped+synced) |
| 3.1–3.7 | all instant | — |

## Common pitfalls by exercise

### 2.1 `find_aggressors`
- Returning `(victim, victim+2)` instead of `(victim-1, victim+1)` — the
  midpoint is wrong; the test catches it.
- Returning `(victim-2, victim+2)` — differ by 4, not 2. The test
  message mentions midpoint *and* distance; point them at both.

### 2.2 `hammer_until_flip`
- Forgetting to accumulate the return value of `hammer_once` into
  `total_ns`. The assertion "total_ns must accumulate hammer_once return"
  names the fix explicitly.
- Looping forever because aggressors from 2.1 are wrong: the loop hits
  `max_rounds=2_000_000` and returns `flipped=False`. Push them back to
  2.1.
- Calling `hammer_once(victim_row)` instead of `hammer_once(aggressor_a,
  aggressor_b)` — a signature confusion. The docstring for
  `DRAM.hammer_once` in `gpubreach_sim/dram.py` is the reference.

### 2.3 `trigger_pte_refresh`
- Calling `env.page_table.sync_from_dram()` without the `env.dram`
  argument.
- Reading `aperture` *after* the sync for `before` (so `before == after`
  and the test fails). The point is to snapshot before the sync.

### 2.4 `craft_overflow_payload`
- Using big-endian: `new_euid.to_bytes(4, "big")`. The kernel cred is
  little-endian on x86/ARM64.
- Filling with 0x00 bytes: works, but makes the hexdump unreadable.
  Cosmetic — don't correct.
- Making the filler `DRIVER_BUFFER_SIZE - 4` long (thinking the total
  has to equal `DRIVER_BUFFER_SIZE`) — the test asserts total length is
  `DRIVER_BUFFER_SIZE + 4`.

### 2.5 `escalate_privileges`
- Reusing the *same* `env` across the test and the chain-driving prints
  — the test constructs a fresh env internally, so this is fine. But
  students sometimes re-run 2.5 after 2.5 has already rooted `env`, then
  wonder why "nothing changed" — `env.kernel_cred.is_root()` is already
  True.
- Passing arguments to `perform_gpu_dma` in the wrong order. The
  positional order is `(data, gpu_vaddr, page_table, iommu, gpu_dram,
  driver_page)` — matches what the cheat sheet lists.

### Stretch — quick flags
- **3.1** students may count bits from the MSB. PTE layout is LSB-first:
  bit 0 = valid, bit 1 = aperture.
- **3.5** some students try to "fix" the IOMMU by mapping the other page
  into the IOMMU so the write is allowed. Not the point — explain that
  the test's job is to *characterise* the IOMMU's real behaviour.
- **3.7** students sometimes pass `cred_offset_in_page < DRIVER_BUFFER_SIZE`
  thinking it generalises 2.4. The solution asserts otherwise — the
  point is a precise OOB overwrite at an offset *past* the buffer.

## Fast-forward helper for stuck students

If a pair is still on 2.2 with 10 minutes left in Phase 2, paste the
following into their answers file so they can see the chain finish and
move on to Phase 3 or 4. Explain what it's doing as you paste it.

```python
# Fast-forward: land the flip and resync the PT from DRAM.
# This is what 2.2 + 2.3 do — we're skipping your implementations to
# unblock the rest of the chain.
while not env.dram.has_flipped(PTE_ROW):
    env.dram.hammer_once(PTE_ROW - 1, PTE_ROW + 1)
env.page_table.sync_from_dram(env.dram)

env.stage1_aggressors_ok = True
env.stage2_flipped_in_refresh_window = True
env.stage3_aperture_changed = True
```

Then they can continue with 2.4 and 2.5 normally.

## Phase-4 debrief — expanded answers

The collapsibles in the solution are deliberately terse. Use these
expanded versions if a discussion stalls.

**4.1 — Real-world headroom.** GPUHammer (USENIX '25) reports 9ms to
first flip on an RTX 3080 against a 32ms tREFW — roughly 3× margin.
Mitigations that double the refresh rate roughly halve effective memory
bandwidth (tREFI fires twice as often; rows spend more time being
refreshed and less serving reads). pTRR / TRR-style mitigations have
been shown bypassable by TRRespass (VU Amsterdam) with more careful
activation patterns that evade the counter-based targeting heuristics.
The trend is against the defender because smaller cells leak faster,
the activation threshold keeps dropping, and tREFW barely changes with
process node.

**4.2 — Why SECDED ECC is insufficient.** Three independent reasons:
(a) GPUHammer demonstrates *ECC-aware* hammering patterns that produce
two flips per codeword in a direction SECDED silently accepts — the
codeword is still a valid codeword, it just encodes different data;
(b) even when ECC raises machine checks, the attacker only needs a
single successful escalation, and the per-attempt success rate can be
tuned up by choosing codewords with more favourable flip templates;
(c) on-die ECC (standard on GDDR6/DDR5) corrects flips invisibly to the
memory controller, so the defender loses the telemetry that would
otherwise let them detect a hammering attempt in progress. End-to-end
ECC (between controller and DIMM) helps here; on-die alone does not.

**4.3 — Why the IOMMU doesn't help.** The IOMMU's contract is "may
device X touch physical page P?" — the right question for DMA from
buggy or malicious devices into unrelated kernel memory. It is the
*wrong* question for bug-within-a-mapped-page because the driver owns
the sub-page structure. Any intra-page bounds check has to live in the
driver or in a slab allocator that knows not to co-locate sensitive
objects next to DMA buffers. The correct structural fix is SLAB
hardening (per-cred page allocations, `SLAB_ACCOUNT` segregation, or
fine-grained allocator randomisation) — not IOMMU tweaks.

**4.4 — How to find the target PTE row from userspace.**
- Offline **templating**: characterise the target GPU on lab hardware
  (or a CI runner with identical silicon). Build a probability map of
  "these (row, column, bit) sites flip reliably under this hammer
  pattern." GPUHammer publishes this kind of template as a distribution
  over sub-page bits.
- Online **memory massaging**: spray GPU allocations with pages whose
  contents you control (via CUDA/Vulkan/OpenGL), freeing and allocating
  in patterns that coax the kernel's page-table allocator into handing
  back pages in rows the template says will flip. This is statistical
  but typically closes within a few seconds on an idle GPU.
- **Side channels**: DRAM row-buffer conflicts leak timing — an
  unprivileged kernel can measure this to infer which virtual pages
  are co-located in the same DRAM row. (CMU's DRAMA side channel is
  the canonical reference on CPU DRAM; the GPU analogue is newer but
  works.)
- **Driver information leaks**: historically NVIDIA, AMD and Mesa
  drivers have all leaked uninitialised VRAM via compute shaders, which
  can disclose PTE bits directly. Patch levels matter here.

Defence-in-depth for operators: use MIG / partitioned GPUs for
multi-tenant inference, pin tenant GPUs to VMs with dedicated silicon
where possible, and wrap sensitive workloads in Confidential Computing
(AMD SEV-SNP + signed GPU partitions, Intel TDX + Hopper confidential
containers, etc.).
