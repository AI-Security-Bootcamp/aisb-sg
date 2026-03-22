"""
Convert solution Python scripts containing triple-quoted Markdown strings to a Markdown document.
"""

import argparse
import asyncio
import glob
import logging
import os

from solution_parsing import build, build_reference_py

logging.basicConfig(level=logging.INFO)


def needs_build(inpath: str, outpaths: list[str]):
    mod_in = os.stat(inpath).st_mtime
    for outpath in outpaths:
        if not os.path.exists(outpath):
            return True
        mod_out = os.stat(outpath).st_mtime
        if mod_in > mod_out:
            return True
    return False


def build_instructions(inpath, instruction_path, test_path):
    ro_files = [instruction_path, test_path]
    for ro_file in ro_files:
        if os.path.exists(ro_file):
            # Temporarily make writable for editing
            os.chmod(ro_file, 0o644)
    with (
        open(inpath, "r") as infile,
        open(instruction_path, "w") as instruction_file,
        open(test_path, "w") as test_file,
    ):
        build(infile, instruction_file, test_file)
    for ro_file in ro_files:
        # Make these files read-only
        if os.path.exists(ro_file):
            os.chmod(ro_file, 0o444)


def build_reference(inpath, reference_path, test_path):
    if os.path.exists(reference_path):
        # Temporarily make writable for editing
        os.chmod(reference_path, 0o644)
    with (
        open(inpath, "r") as infile,
        open(reference_path, "w") as reference_file,
    ):
        build_reference_py(infile, reference_file, test_path)
    # Set reference file to read and execute (544)
    os.chmod(reference_path, 0o544)


def build_all(force=False, files: list[str] | None = None, reference=False):
    if files is None:
        files = glob.glob("**/*_solution.py", recursive=True)
    for file in files:
        assert file.endswith("_solution.py")

    for inpath in files:
        instruction_file = inpath.replace("_solution.py", "_instructions.md")
        test_file = inpath.replace("_solution.py", "_test.py")
        reference_file = inpath.replace("_solution.py", "_reference.py")
        try:
            if needs_build(inpath, [instruction_file, test_file]) or force:
                build_instructions(inpath, instruction_file, test_file)
            if reference and (needs_build(inpath, [reference_file]) or force):
                build_reference(inpath, reference_file, test_file)
        except SyntaxError as e:
            print("Syntax error - will not rebuild until next save.")
            print(e)
        except UnicodeDecodeError as e:
            print("Unicode error - will not rebuild until next save.")
            print(e)


async def watch_for_changes(interval_secs=2, files=None):
    """Watch and rebuild on input changes."""
    while True:
        build_all(files=files)
        await asyncio.sleep(interval_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Specific solution files to process (optional)")
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Watch solution files for changes and rebuild automatically",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild even if files are up to date",
    )
    parser.add_argument(
        "--reference",
        "-r",
        action="store_true",
        help="Generate reference files containing only SOLUTION blocks",
    )
    args = parser.parse_args()
    if args.watch:
        asyncio.run(watch_for_changes(files=args.files if args.files else None))
    else:
        build_all(force=args.force, files=args.files if args.files else None, reference=args.reference)
