# %%
"""
Starter file for the GPUBreach lab.

This file is pre-scaffolded with the five Phase 2 (must-finish) exercises.
Read `day6_gpubreach_instructions.md` alongside this file — the
instructions have the full prose, the cheat sheet, and Phase 1 / 3 / 4.

Work in order, top to bottom. Each `# %%` block is a cell you can run in
VS Code / Jupyter / ipython. Replace the `TODO` bodies with your solution,
then run the test call immediately below the exercise. When all five
Phase-2 tests pass, `env.check_all()` prints the flag.

**Run this file cell-by-cell, not top-to-bottom.** Each exercise cell
will crash until you implement its function — that is expected. Run
Exercise 2.1 first (setup + `find_aggressors`), then each subsequent
exercise in order.
"""

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


# %%
# Exercise 2.1 — find_aggressors(victim_row) -> (upper, lower)

def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    # TODO: Return (upper, lower) such that:
    #   1. The two values differ by exactly 2.
    #   2. Their midpoint is `victim_row`.
    pass


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 2.1: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")


@report
def test_find_aggressors(solution: Callable[[int], tuple[int, int]]):
    for victim in [PTE_ROW, 100, 7777, 42]:
        a, b = solution(victim)
        assert abs(a - b) == 2, (
            f"aggressors for victim={victim} must differ by 2, got ({a}, {b})"
        )
        assert (a + b) // 2 == victim, (
            f"victim={victim} must lie between aggressors, midpoint was "
            f"{(a + b) // 2}"
        )
    print("  Aggressor geometry correct!")


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True


# %%
# Exercise 2.2 — hammer_until_flip: drive the hammer loop

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


@report
def test_hammer_until_flip(solution: Callable[..., dict]):
    fresh = make_environment()
    victim = fresh.dram.flip_location[0]
    a, b = victim - 1, victim + 1
    res = solution(fresh.dram, a, b, victim)
    assert res["flipped"], f"expected a flip to land, got {res}"
    assert res["rounds"] >= HAMMER_THRESHOLD_ACTIVATIONS, (
        f"expected at least {HAMMER_THRESHOLD_ACTIVATIONS:,} rounds, "
        f"got {res['rounds']:,}"
    )
    expected_ns = res["rounds"] * 2 * ACTIVATE_PRECHARGE_NS
    assert res["total_ns"] == expected_ns, (
        f"total_ns must accumulate hammer_once return; "
        f"expected {expected_ns}, got {res['total_ns']}"
    )
    other = make_environment()
    no_flip = solution(other.dram, 0, 5, other.dram.flip_location[0], max_rounds=1000)
    assert not no_flip["flipped"], (
        "hammer_until_flip must stop at max_rounds if aggressors are wrong"
    )
    print("  Hammer loop and cycle accounting correct!")


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)


# %%
# Exercise 2.3 — trigger_pte_refresh: make the GPU MMU re-walk the PTE

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


@report
def test_trigger_pte_refresh(solution: Callable[[Environment], tuple[int, int]]):
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    assert fresh.page_table.cached_pte.aperture == APERTURE_GPU_LOCAL, (
        "cached PTE must still be stale before sync_from_dram runs"
    )
    pair = solution(fresh)
    assert pair == (APERTURE_GPU_LOCAL, APERTURE_SYSTEM), (
        f"expected (0, 1), got {pair}"
    )
    assert fresh.page_table.cached_pte.aperture == APERTURE_SYSTEM, (
        "after refresh, cached_pte must reflect APERTURE_SYSTEM"
    )
    print("  PT resync propagated the flip!")


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)


# %%
# Exercise 2.4 — craft_overflow_payload: build the OOB DMA payload

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


@report
def test_craft_overflow_payload(solution: Callable[..., bytes]):
    p = solution()
    assert isinstance(p, (bytes, bytearray)), (
        f"payload must be bytes, got {type(p).__name__}"
    )
    assert len(p) == DRIVER_BUFFER_SIZE + 4, (
        f"expected {DRIVER_BUFFER_SIZE + 4} bytes, got {len(p)}"
    )
    assert int.from_bytes(p[-4:], "little") == 0, (
        f"last 4 bytes should encode euid=0 little-endian, "
        f"got {p[-4:].hex()}"
    )
    p2 = solution(42)
    assert int.from_bytes(p2[-4:], "little") == 42, (
        f"parameterised new_euid should be respected, got {p2[-4:].hex()}"
    )
    print("  Payload layout correct!")


test_craft_overflow_payload(craft_overflow_payload)


# %%
# Exercise 2.5 — escalate_privileges: fire the DMA, confirm root

def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    # TODO:
    # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
    #    env.page_table, env.iommu, env.dram, env.driver_page).
    # 2. Return env.kernel_cred.is_root().
    return False


rooted = escalate_privileges(env, payload)
print(f"Ex 2.5: root achieved? {rooted}")


@report
def test_escalate_privileges(solution: Callable[..., bool]):
    fresh = make_environment()
    v = fresh.dram.flip_location[0]
    while not fresh.dram.has_flipped(v):
        fresh.dram.hammer_once(v - 1, v + 1)
    fresh.page_table.sync_from_dram(fresh.dram)
    overflow_payload = b"A" * DRIVER_BUFFER_SIZE + (0).to_bytes(4, "little")
    assert not fresh.kernel_cred.is_root(), (
        "sanity: cred must start non-root"
    )
    ok = solution(fresh, overflow_payload)
    assert ok is True, "escalate_privileges should return True on success"
    assert fresh.kernel_cred.is_root(), (
        "kernel_cred.euid should be 0 after the OOB write"
    )
    fresh2 = make_environment()
    while not fresh2.dram.has_flipped(v):
        fresh2.dram.hammer_once(v - 1, v + 1)
    fresh2.page_table.sync_from_dram(fresh2.dram)
    tiny = b"A" * (DRIVER_BUFFER_SIZE - 1)
    ok2 = solution(fresh2, tiny)
    assert ok2 is False, (
        "a payload shorter than DRIVER_BUFFER_SIZE must NOT escalate"
    )
    print("  End-to-end escalation succeeded!")


test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted


# %%
# Check everything — this prints the flag on success.
env.check_all()


# %%
# Stretch exercises (3.1–3.7) live in day6_gpubreach_instructions.md.
# Copy-paste the ones you want to try from there as additional `# %%`
# cells below this one.
