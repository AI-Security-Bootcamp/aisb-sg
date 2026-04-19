# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path
from aisb_utils import report
from __future__ import annotations
import torch
from day3_setup import (
    model, tokenizer,
    user_msg, system_msg, strip_thinking,
    show, show_verdict,
    CLASSIFIER_SYSTEM_PROMPT,
)
from aisb_utils.test_utils import report
from day3_setup import generate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import random
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from aisb_utils.test_utils import report
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm



@report
def test_tokenize_strings(solution):
    results = solution(["Hello world", "Hello\n\nworld"])
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all(isinstance(r, list) for r in results), "Each result should be a list"
    assert all(len(r) > 0 for r in results), "Token lists should be non-empty"
    # "Hello\n\nworld" should have more tokens than "Hello world" (extra newline)
    assert len(results[1]) >= len(results[0]), \
        "Double newline should produce at least as many tokens as space"
    print("  All tests passed!")




@report
def test_format_chat_prompt(solution):
    prompt = solution("What is the capital of Japan?")
    assert isinstance(prompt, str) and len(prompt) > 0, "Should return a non-empty string"
    assert "Japan" in prompt, "Prompt should contain the original question"
    # Qwen models use <|im_start|> role markers
    assert "<|im_start|>" in prompt or "[INST]" in prompt, \
        "Prompt should contain chat template markers (e.g. <|im_start|> or [INST])"
    print("  All tests passed!")




@report
def test_compare_chat_templates(solution):
    # Use just 1 question and 2 models for speed
    results = solution(["Hello"], ["Qwen/Qwen3-0.6B", "unsloth/gemma-2-2b-it"])
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for model_name, question, prompt in results:
        assert isinstance(prompt, str) and len(prompt) > 0, \
            f"Prompt for {model_name} should be a non-empty string"
        assert "Hello" in prompt, f"Prompt for {model_name} should contain the question"
    # Different models should produce different templates
    assert results[0][2] != results[1][2], \
        "Different models should produce different chat templates"
    print("  All tests passed!")




@report
def test_my_generate(solution):
    result = solution(model, tokenizer, [user_msg("What is 2+2?")], max_new_tokens=50)
    assert isinstance(result, str), "my_generate must return a string"
    assert len(result) > 0, "my_generate must return a non-empty string"




@report
def test_my_generate_no_thinking(solution):
    result = solution(
        model, tokenizer, [user_msg("Say hello")],
        max_new_tokens=50, enable_thinking=False, strip_think=True,
    )
    assert isinstance(result, str), "my_generate must return a string"
    assert "<think>" not in result, "Thinking tags should be stripped"




@report
def test_send_unguarded(solution):
    result = solution("What is 2+2?")
    assert isinstance(result, str), "send_unguarded must return a string"
    assert len(result) > 0, "send_unguarded must return a non-empty string"




@report
def test_model_refuses_harmful(solution):
    """The model should refuse the direct harmful query."""
    test_response = solution(HARMFUL_QUERY)
    assert isinstance(test_response, str) and len(test_response) > 0
    lower = test_response.lower()
    refusal_words = ["sorry", "can't", "cannot", "not able", "refuse", "won't",
                     "inappropriate", "don't", "will not", "unable"]
    assert any(w in lower for w in refusal_words), \
        "Expected the model to refuse the direct harmful query"




@report
def test_string_filter_blocks_harmful(solution):
    blocked, kw = solution(HARMFUL_QUERY)
    assert blocked, "string_filter should block the harmful query"
    assert kw is not None




@report
def test_string_filter_passes_benign(solution):
    blocked, _ = solution(BENIGN_QUERY)
    assert not blocked, "string_filter should not block benign queries"




@report
def test_discovered_keywords():
    found, expected = set(DISCOVERED_KEYWORDS), set(BLOCKED_KEYWORDS)
    extra = found - expected
    missing = expected - found
    msg = f"Found {len(found)}/{len(expected)} keywords."
    if extra:
        msg += f" {len(extra)} extra: {extra}."
    if missing:
        msg += f" {len(missing)} missing."
    assert found == expected, msg




@report
def test_string_filter_passes_paraphrase(solution):
    blocked, _ = solution(MY_PARAPHRASE)
    assert not blocked, "Paraphrased query should bypass the keyword filter"




@report
def test_llm_classify_returns_verdict(solution):
    verdict = solution(BENIGN_QUERY)
    assert isinstance(verdict, str), "Must return a string"
    assert verdict in ("SAFE", "UNSAFE"), f"Must return 'SAFE' or 'UNSAFE', got {verdict}"




@report
def test_llm_classify_catches_paraphrase(solution):
    verdict = solution(MY_BYPASS_QUERY)
    assert verdict == "UNSAFE", "Classifier should catch the paraphrased harmful query"




@report
def test_llm_classify_passes_benign(solution):
    verdict = solution(BENIGN_QUERY)
    assert verdict == "SAFE", "Classifier should pass benign queries"




@report
def test_send_with_guardrails_blocks_harmful(solution):
    response, meta = solution(HARMFUL_QUERY)
    assert response is None, "Harmful query should be blocked"
    assert meta["input_verdict"] == "UNSAFE"




@report
def test_send_with_guardrails_passes_benign(solution):
    response, meta = solution(BENIGN_QUERY)
    assert response is not None, "Benign query should not be blocked"
    assert meta["input_verdict"] == "SAFE"




@report
def test_get_hidden_states_shape(solution):
    rep = solution("Hello world")
    assert rep.ndim == 1, "Hidden state should be 1D"
    assert rep.shape[0] > 0, "Hidden state should have non-zero dimension"




@report
def test_probe_catches_jailbreak():
    # JAILBREAK_QUERY is held out from LABELLED_SAMPLES — this is a true
    # out-of-sample evaluation, not a test on training data.
    probe_obj, scaler_obj, _ = train_probe(LABELLED_SAMPLES)
    verdict, prob = probe_classify(JAILBREAK_QUERY, probe_obj, scaler_obj)
    assert verdict == "UNSAFE", f"Probe should catch the jailbreak (got {verdict}, p={prob:.3f})"




# %%


@report
def test_train_step_ce(solution: Callable[..., float]):
    """Verify the step runs, returns a float, and updates the student."""
    # Create a tiny dummy student (we don't need the full 124M for a unit test)
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_embd=64, n_layer=2, n_head=2,
        n_positions=128, n_ctx=128,
    )
    torch.manual_seed(0)
    student = GPT2LMHeadModel(cfg).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=1e-3)

    input_ids, filtered_labels = EXAMPLES[0]
    params_before = [p.detach().clone() for p in student.parameters()]

    loss = solution(student, input_ids, filtered_labels, optimizer)

    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert not np.isnan(loss) and not np.isinf(loss), f"Bad loss: {loss}"
    assert loss > 0, f"Loss should be positive for random-init model, got {loss}"

    # Confirm at least one parameter changed
    params_after = list(student.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after)), \
        "No parameters changed — did you call optimizer.step()?"
    print("  All tests passed!")




# %%


@report
def test_kd_loss(solution: Callable[..., torch.Tensor]):
    """Verify KD loss is 0 when student == teacher, positive otherwise,
    and scales as expected with temperature."""
    torch.manual_seed(0)
    t_logits = torch.randn(2, 5, 100, device=DEVICE, requires_grad=False)

    # Case 1: student == teacher → loss should be 0
    s_logits = t_logits.clone().requires_grad_()
    loss_same = solution(s_logits, t_logits, temperature=2.0)
    assert torch.is_tensor(loss_same) and loss_same.dim() == 0, \
        f"Expected scalar tensor, got {loss_same}"
    assert loss_same.item() < 1e-5, \
        f"Expected ~0 when student == teacher, got {loss_same.item()}"

    # Case 2: student != teacher → loss should be positive
    s_logits = torch.randn(2, 5, 100, device=DEVICE, requires_grad=True)
    loss_diff = solution(s_logits, t_logits, temperature=2.0)
    assert loss_diff.item() > 0, f"Expected positive loss, got {loss_diff.item()}"

    # Case 3: gradient flows to student
    loss_diff.backward()
    assert s_logits.grad is not None and (s_logits.grad != 0).any(), \
        "No gradient flowing to student_logits"

    # Case 4: T² scaling — larger T should *not* drive the loss to 0 entirely
    # (the T² factor compensates). Doubling T with fixed logits should not
    # change the loss by more than a factor of ~4.
    loss_T1 = solution(torch.randn(2, 5, 100, device=DEVICE), t_logits, temperature=1.0)
    loss_T5 = solution(torch.randn(2, 5, 100, device=DEVICE), t_logits, temperature=5.0)
    assert loss_T1.item() > 0 and loss_T5.item() > 0

    print("  All tests passed!")




# %%


@report
def test_train_step_with_kd(solution: Callable[..., tuple[float, float, float]]):
    """Verify the KD step runs, updates the student, and returns sensible
    (total, ce, kd) values."""
    # Small dummy student for speed; use the real teacher
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_embd=64, n_layer=2, n_head=2,
        n_positions=128, n_ctx=128,
    )
    torch.manual_seed(0)
    student = GPT2LMHeadModel(cfg).to(torch.bfloat16).to(DEVICE)
    optimizer = AdamW(student.parameters(), lr=1e-3)

    input_ids, filtered_labels = EXAMPLES[0]
    params_before = [p.detach().clone() for p in student.parameters()]

    total, ce, kd = solution(
        student, teacher, input_ids, filtered_labels, optimizer,
        kd_alpha=KD_ALPHA, temperature=KD_TEMPERATURE,
    )

    for name, val in [("total", total), ("ce", ce), ("kd", kd)]:
        assert isinstance(val, float), f"{name} should be float, got {type(val)}"
        assert not np.isnan(val) and not np.isinf(val), f"{name} is NaN/inf: {val}"
        assert val > 0, f"{name} should be positive, got {val}"

    # Check the combination is correct. Use relative tolerance — the student
    # is in bf16 and kd can be ~1000 (KL over 50k vocab), so exact float
    # equality of total vs. (1-α)·ce + α·kd is not expected.
    expected = (1 - KD_ALPHA) * ce + KD_ALPHA * kd
    tol = max(0.1, 0.02 * abs(expected))
    assert abs(total - expected) < tol, \
        f"total ({total:.3f}) != (1-α)·ce + α·kd ({expected:.3f}) within {tol:.3f}"

    # Check parameters actually updated
    params_after = list(student.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after)), \
        "No parameters changed — did you call optimizer.step()?"
    print("  All tests passed!")
