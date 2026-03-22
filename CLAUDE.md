# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo

`aisb-sg` — AI Security Bootcamp, Singapore iteration. Private repo: `git@github.com:pranavgade20/aisb-sg.git`.

## Exercise Format

Solution files (`dayN_solution.py`) are the source of truth. ```dayN_instructions.md``` and ```dayN_test.py``` are generated from these — never edit the output files directly.
After the end of your editing session, run the build script on the solution files you've updated to ensure the output files are updated.

### File structure per day
```
dayN-name/
├── dayN.py               ← (optional) baseline file that is broken up into the exercise chunks
├── dayN_solution.py      ← author edits this file to make the exercises
├── dayN_instructions.md  ← generated (read-only)
└── dayN_test.py          ← generated (read-only)
```

### Solution file anatomy
```python
# %%                          ← VS Code cell marker
"""
# Title - Markdown heading

<!-- toc -->                   ← auto-generates Table of Contents here

## Exercise N: Name
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Instruction text here in markdown...
"""

# Code that goes into the skeleton file
def my_function():
    if "SOLUTION":             ← stripped out in instructions/test
        return actual_answer
    else:
        # TODO: implement
        pass

if "SKIP":                     ← block omitted entirely from output
    ...

if "REFERENCE_ONLY":           ← only in reference build, not instructions
    ...

@report                        ← decorator that prints ✅/❌ pass/fail
def test_my_function(solution):
    assert solution("input") == "expected"

test_my_function(my_function)

"""
<details><summary>Hint 1</summary>
Hint content...
</details>
"""
```

### Build system
- **`build-instructions.sh`** — entrypoint, sets up venv, delegates to Python
- **`aisb_utils/build_instructions.py`** — orchestrates build; supports `--watch`, `--force`, `--reference` flags
- **`aisb_utils/solution_parsing.py`** (libcst) — AST transformer that strips solutions, extracts markdown, generates TOC, wraps hints in `<blockquote>`
- **`aisb_utils/test_utils.py`** — `@report` decorator for colored pass/fail output

**Build command:** `./build-instructions.sh` generates read-only `*_instructions.md` and `*_test.py`. Use `--watch` for live regeneration, `--force` to rebuild all.

### Key conventions
- Students create a separate `dayN_answers.py` and copy snippets into it
- Hints use progressive `<details><summary>Hint N</summary>` blocks
- Exercises use emoji difficulty/importance scales
- Generated files are chmod 444 (read-only) to prevent accidental edits

## Day README Format

Each day folder has a `README.md` with:
- `- [ ]` / `- [x]` checkboxes for topic tracking
- `*Italics*` for ML prerequisites
- `**Bold**` for day themes
- Nested bullets for sub-topics and exercise steps

## Key Dependencies

- Python 3.11+
- Ruff (line length 120, ignores E402/F401/F811)
- libcst for AST-based solution parsing
- PyTorch for ML exercises
