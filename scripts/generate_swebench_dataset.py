"""
Script to generate predictions for SWE-bench Lite (princeton-nlp/SWE-bench_Lite).
This version uses 'uv venv' and 'uv pip' / 'uv run ra-aid' commands to manage everything in the environment.

It:
- Loads the SWE-bench Lite dataset
- For each instance, clones (or reuses) the repo at the specified commit
- Creates or reuses a dedicated Python virtual environment via `uv venv`
- Installs `ra-aid` in editable mode + any project dependencies via `uv pip`
- Also installs the cloned project itself in editable mode if it appears to be a Python package
- Calls `uv run ra-aid` to generate a patch
- Writes out predictions in JSON format

No progress bar or spinner is used, allowing `ra-aid` output to stream directly.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from git import Repo
from rich.logging import RichHandler

# If you'd like to override Python versions for specific repos:
PYTHON_VERSION_OVERRIDES = {
    # "someorg/somerepo": "3.9",
}


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "generate_dataset.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = RichHandler(
        rich_tracebacks=True, show_time=False, show_path=False
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(console_handler)


def load_dataset_safely() -> Optional[Any]:
    """Load SWE-bench Lite dataset with error handling."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None


def create_output_dirs() -> Tuple[Path, Path]:
    """Create base/log directory structure."""
    date_str = datetime.now().strftime("%Y%m%d")
    base_dir = Path("evaluation") / "default" / f"{date_str}_raaid"
    log_dir = base_dir / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, log_dir


def uv_venv(repo_dir: Path, repo_name: str, force_venv: bool) -> None:
    """
    Create (or reuse) a .venv in 'repo_dir' using 'uv venv'.
    If force_venv is True, we remove .venv first.

    Example command:
      uv venv .venv --python=3.9
    """
    venv_dir = repo_dir / ".venv"
    if venv_dir.exists() and force_venv:
        logging.info(f"Removing existing .venv at {venv_dir}")
        shutil.rmtree(venv_dir)

    python_version = PYTHON_VERSION_OVERRIDES.get(repo_name, None) or "3.12"
    cmd = ["uv", "venv"]
    if python_version:
        cmd.append(f"--python={python_version}")
    cmd.append(".venv")

    try:
        subprocess.run(cmd, cwd=repo_dir, check=True)
    except Exception as e:
        logging.error(f"Failed to create venv in {repo_dir}: {e}")


def uv_pip_install(repo_dir: Path, args: List[str]) -> None:
    """
    Run 'uv pip install ...' in the specified repo_dir.
    Example: uv_pip_install(repo_dir, ["--upgrade", "pip"])
    """
    cmd = ["uv", "pip", "install"] + args
    try:
        subprocess.run(cmd, cwd=repo_dir, check=True)
    except Exception as e:
        logging.error(f"Failed to run uv pip install {args}: {e}")


def uv_run_raaid(repo_dir: Path, prompt: str) -> Optional[str]:
    """
    Call 'uv run ra-aid' with the given prompt in the environment,
    streaming output directly to the console (capture_output=False).
    Returns the patch if successful, else None.
    """
    cmd = ["uv", "run", "ra-aid", "--cowboy-mode", "-m", prompt]
    # We are NOT capturing output, so it streams live:
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            text=True,
            check=False,  # We manually handle exit code
        )
        if result.returncode != 0:
            logging.error("ra-aid returned non-zero exit code.")
            return None
    except subprocess.TimeoutExpired:
        logging.error("ra-aid timed out")
        return None
    except Exception as e:
        logging.error(f"ra-aid error: {e}")
        return None

    # Collect patch
    patch = get_git_patch(repo_dir)
    return patch


def get_git_patch(repo_dir: Path) -> Optional[str]:
    """Generate a git patch from the current changes in `repo_dir`."""
    try:
        repo = Repo(repo_dir)
        if not repo.is_dirty():
            logging.info("No changes detected in repository.")
            return None
        patch_text = repo.git.diff(unified=3)
        if not patch_text.strip():
            return None
        if not any(line.startswith("+") for line in patch_text.splitlines()):
            return None
        return patch_text
    except Exception as e:
        logging.error(f"Failed to generate patch: {e}")
        return None


def setup_venv_and_deps(repo_dir: Path, repo_name: str, force_venv: bool) -> None:
    """
    - uv venv .venv --python=xxx (optional)
    - uv pip install --upgrade pip
    - uv pip install --upgrade setuptools wheel  (so pkg_resources etc. are available)
    - uv pip install -e <ra-aid local path>
    - If pyproject.toml -> uv pip install .
    - If requirements.txt -> uv pip install -r requirements.txt
    - If requirements-dev.txt -> uv pip install -r requirements-dev.txt
    - If there's a setup.py or pyproject => uv pip install -e .
    """
    uv_venv(repo_dir, repo_name, force_venv)

    # 1) upgrade pip
    uv_pip_install(repo_dir, ["--upgrade", "pip"])

    # 2) ensure setuptools & wheel are installed/up to date
    uv_pip_install(repo_dir, ["--upgrade", "setuptools", "wheel"])

    # 3) install ra-aid from local path
    script_dir = Path(__file__).resolve().parent
    ra_aid_root = script_dir.parent  # one level up from scripts
    uv_pip_install(repo_dir, ["-e", str(ra_aid_root)])

    # 4) optional pyproject
    pyproject_path = repo_dir / "pyproject.toml"
    if pyproject_path.is_file():
        uv_pip_install(repo_dir, ["."])

    # 5) optional requirements.txt
    req_file = repo_dir / "requirements.txt"
    if req_file.is_file():
        uv_pip_install(repo_dir, ["-r", "requirements.txt"])

    # 6) optional requirements-dev.txt
    req_dev_file = repo_dir / "requirements-dev.txt"
    if req_dev_file.is_file():
        uv_pip_install(repo_dir, ["-r", "requirements-dev.txt"])

    # 7) install the cloned project in editable mode if it's a Python package
    setup_path = repo_dir / "setup.py"
    if pyproject_path.is_file() or setup_path.is_file():
        logging.info("Installing cloned project in editable mode.")
        uv_pip_install(repo_dir, ["-e", "."])


def build_prompt(
    problem_statement: str, fail_tests: List[str], pass_tests: List[str]
) -> str:
    """
    Construct the prompt text from problem_statement, FAIL_TO_PASS, PASS_TO_PASS.
    """
    prompt = f"{problem_statement}\n\nTests that need to be fixed:\n```\n"
    for t in fail_tests:
        prompt += f"- {t}\n"
    prompt += "```\n\n"
    if pass_tests:
        prompt += "Tests that must remain passing:\n```\n"
        for t in pass_tests:
            prompt += f"- {t}\n"
        prompt += "```\n\n"
    prompt += "\n\nYou must run all above tests both **before and after** making changes, and ensure they pass as you do your work. Do not write any new test cases."
    return prompt


def process_instance(
    instance: Dict[str, Any], projects_dir: Path, reuse_repo: bool, force_venv: bool
) -> Dict[str, Any]:
    """
    Process a single dataset instance without a progress bar/spinner.
    - Clone or reuse the repo at projects_dir/<instance_id>
    - Checkout commit
    - Create or reuse a .venv in that repo
    - Install ra-aid + any project dependencies
    - Build prompt, run ra-aid (output streamed to console)
    - Return prediction dict
    """
    inst_id = instance.get("instance_id", "<unknown>")
    repo_name = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    fail_tests = instance.get("FAIL_TO_PASS", [])
    pass_tests = instance.get("PASS_TO_PASS", [])

    if isinstance(fail_tests, str):
        fail_tests = [fail_tests]
    if isinstance(pass_tests, str):
        pass_tests = [pass_tests]

    if "github.com" not in repo_name:
        repo_url = f"https://github.com/{repo_name}.git"
    else:
        repo_url = repo_name

    checkout_dir = projects_dir / f"{inst_id}"

    try:
        if not checkout_dir.exists():
            logging.info(f"Cloning {repo_url} -> {checkout_dir}")
            repo = Repo.clone_from(repo_url, checkout_dir)
        else:
            if reuse_repo:
                logging.info(f"Reusing existing directory: {checkout_dir}")
                repo = Repo(checkout_dir)
            else:
                logging.info(f"Deleting existing directory: {checkout_dir}")
                shutil.rmtree(checkout_dir)
                repo = Repo.clone_from(repo_url, checkout_dir)

        # checkout correct commit
        repo.git.checkout(commit)

        # set up venv + deps
        setup_venv_and_deps(checkout_dir, repo_name, force_venv)

        # build prompt, run ra-aid
        prompt_text = build_prompt(problem_statement, fail_tests, pass_tests)
        patch = uv_run_raaid(checkout_dir, prompt_text)

        return {
            "instance_id": inst_id,
            "model_patch": patch if patch else "",
            "model_name_or_path": "ra-aid",
        }

    except Exception as e:
        logging.error(f"Failed to process {repo_url}:{commit} - {e}")
        return {
            "instance_id": inst_id,
            "model_patch": "",
            "model_name_or_path": "ra-aid",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions for SWE-bench Lite using uv + ra-aid (no progress bar)."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to store prediction file"
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        required=True,
        help="Directory where projects will be cloned.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Number of instances to process (default: all)",
    )
    parser.add_argument(
        "--reuse-repo",
        action="store_true",
        help="If set, do not delete an existing repo directory. We'll reuse it.",
    )
    parser.add_argument(
        "--force-venv",
        action="store_true",
        help="If set, recreate the .venv even if it exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Create base/log dirs and set up logging
    base_dir, log_dir = create_output_dirs()
    setup_logging(log_dir, args.verbose)
    logging.info("Starting script")

    # Ensure projects dir
    args.projects_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset_safely()
    if dataset is None:
        sys.exit(1)

    # Combine dev + test
    all_data = list(dataset["dev"]) + list(dataset["test"])

    # Ensure output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = args.output_dir / "predictions.json"
    predictions: List[Dict[str, str]] = []

    limit = args.num_instances if args.num_instances else len(all_data)

    # Just a simple for loop - no progress bar
    logging.info(f"Processing up to {limit} instances.")
    for i, inst in enumerate(all_data):
        if i >= limit:
            break

        logging.info(f"=== Instance {i+1}/{limit}, ID={inst.get('instance_id')} ===")
        pred = process_instance(
            inst, args.projects_dir, args.reuse_repo, args.force_venv
        )
        predictions.append(pred)

    # Save predictions
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    logging.info("Done generating predictions.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception:
        logging.exception("Unhandled error occurred.")
        sys.exit(1)
