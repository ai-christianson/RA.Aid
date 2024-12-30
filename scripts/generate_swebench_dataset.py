#!/usr/bin/env python3
"""
Script to generate predictions for SWE-bench Lite (princeton-nlp/SWE-bench_Lite).
This script:
- Loads the SWE-bench Lite dataset
- For each instance, clones the repo at the specified commit into a user-defined projects directory
- Creates a dedicated Python virtual environment in the cloned repo using 'uv venv'
  (the default system Python is used unless overridden in the `PYTHON_VERSION_OVERRIDES` dictionary)
- Installs `ra-aid` (in editable mode) plus any project dependencies from:
    - pyproject.toml (pip install .)
    - requirements.txt
    - requirements-dev.txt
- Forms a prompt from the instance fields (problem_statement, FAIL_TO_PASS, PASS_TO_PASS)
- Calls ra-aid (from the venv) to create a patch
- Writes out predictions in the required JSON format

Additionally, we provide an internal dictionary for per-project Python version overrides:
  e.g.:

    PYTHON_VERSION_OVERRIDES = {
        "org/repo": "3.9",
        "some-other-org/another-repo": "3.10",
    }

If a repo name is not found in that dictionary, this script will just use the default system Python.

Required parameters:
  --projects-dir : Directory where all repos are cloned.

Optional parameters:
  --cleanup      : If set, remove the cloned repos after processing.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from datasets import load_dataset
from git import Repo
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


# If you'd like to override Python versions for specific repos:
# For example: "pandas-dev/pandas": "3.9"
PYTHON_VERSION_OVERRIDES = {
    # "org/repo": "3.9",
    # "another-org/another-repo": "3.10",
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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(console_handler)


def load_dataset_safely() -> Optional[Any]:
    """Load SWE-bench Lite dataset with error handling."""
    try:
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


def install_local_raaid(pip_path: Path) -> None:
    """
    Install ra-aid (in editable mode) into the local environment.
    We assume that this script lives in <repo_root>/scripts, so the
    root directory is one level up from __file__.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # one level up
    try:
        subprocess.run(
            [str(pip_path), "install", "-e", str(repo_root)],
            cwd=str(repo_root),
            check=True
        )
    except Exception as e:
        logging.error(f"Failed to install ra-aid in editable mode from {repo_root}: {e}")


def setup_repo_venv(repo_dir: Path, repo_name: str) -> Path:
    """
    Create a Python virtual environment in `repo_dir/.venv` using `uv venv`.
    Installs:
      - local ra-aid (editable mode)
      - pyproject.toml => pip install .
      - requirements.txt => pip install -r ...
      - requirements-dev.txt => pip install -r ...

    Steps to determine Python version:
      1) Check the PYTHON_VERSION_OVERRIDES dict for the given repo_name.
         If found, use that as the --python=<version> argument.
      2) Otherwise, let uv pick the default system Python.

    Returns:
        Path to the .venv directory
    """
    venv_dir = repo_dir / ".venv"

    # Check for Python version override
    python_version = PYTHON_VERSION_OVERRIDES.get(repo_name, None)

    # Construct the uv command
    uv_cmd = ["uv", "venv"]
    if python_version:
        uv_cmd.append(f"--python={python_version}")
    uv_cmd.append(str(venv_dir))

    try:
        subprocess.run(uv_cmd, cwd=repo_dir, check=True)
    except Exception as e:
        logging.error(f"Failed to create venv in {repo_dir} using uv: {e}")
        return venv_dir  # Return anyway for partial info

    pip_path = venv_dir / "bin" / "pip"

    # Upgrade pip
    try:
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            cwd=repo_dir,
            check=False
        )
    except Exception as e:
        logging.error(f"Failed to upgrade pip in {venv_dir}: {e}")

    # 1) Install ra-aid in editable mode from our local repo
    install_local_raaid(pip_path)

    # 2) If pyproject.toml is present, install local project
    pyproject_path = repo_dir / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            subprocess.run(
                [str(pip_path), "install", "."],
                cwd=repo_dir,
                check=True
            )
        except Exception as e:
            logging.error(f"Failed to install project from pyproject.toml in {repo_dir}: {e}")

    # 3) If requirements.txt is present
    req_path = repo_dir / "requirements.txt"
    if req_path.is_file():
        try:
            subprocess.run(
                [str(pip_path), "install", "-r", str(req_path)],
                cwd=repo_dir,
                check=True
            )
        except Exception as e:
            logging.error(f"Failed to install from requirements.txt: {e}")

    # 4) If requirements-dev.txt is present
    req_dev_path = repo_dir / "requirements-dev.txt"
    if req_dev_path.is_file():
        try:
            subprocess.run(
                [str(pip_path), "install", "-r", str(req_dev_path)],
                cwd=repo_dir,
                check=True
            )
        except Exception as e:
            logging.error(f"Failed to install from requirements-dev.txt: {e}")

    return venv_dir


def run_raaid(
    repo_dir: Path,
    venv_dir: Path,
    problem_statement: str,
    fail_tests: List[str],
    pass_tests: List[str]
) -> Optional[str]:
    """
    Run ra-aid on the problem statement (using the local venv), returning a generated patch if possible.
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

    # Use ra-aid from the local venv
    raaid_exe = venv_dir / "bin" / "ra-aid"
    impl_cmd = [
        str(raaid_exe),
        '--cowboy-mode',
        '-m', prompt,
    ]

    try:
        impl_result = subprocess.run(
            impl_cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        if impl_result.returncode != 0:
            logging.error("ra-aid returned non-zero exit code.")
            logging.debug(f"stdout: {impl_result.stdout}")
            logging.debug(f"stderr: {impl_result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logging.error("ra-aid implementation phase timed out.")
        return None
    except Exception as e:
        logging.error(f"ra-aid error: {e}")
        return None

    repo = Repo(repo_dir)
    patch = get_git_patch(repo)
    return patch


def get_git_patch(repo: Repo) -> Optional[str]:
    """Generate a git patch for current changes."""
    if not repo.is_dirty():
        logging.info("No repo changes detected.")
        return None
    try:
        patch = repo.git.diff(unified=3)
        if not patch or not patch.strip():
            return None
        if not any(line.startswith('+') for line in patch.splitlines()):
            return None
        return patch
    except Exception as e:
        logging.error(f"Failed to generate patch: {e}")
        return None


def process_instance(
    instance: Dict[str, Any],
    projects_dir: Path,
    cleanup: bool
) -> Dict[str, Any]:
    """
    Process a single dataset instance:
    - Clone the repo into projects_dir/<instance_id>
    - Checkout commit
    - Build a local Python venv in that repo (checking override dict)
    - Install ra-aid + any project dependencies
    - Build prompt from problem_statement, FAIL_TO_PASS, PASS_TO_PASS
    - Return dict in required format:
        {
            "instance_id": ...,
            "model_patch": ...,
            "model_name_or_path": ...
        }
    - If cleanup is True, remove the cloned repo after generating a patch
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

    # Build GitHub URL
    if "github.com" not in repo_name:
        repo_url = f"https://github.com/{repo_name}.git"
    else:
        repo_url = repo_name

    checkout_dir = projects_dir / f"{inst_id}"
    patch_str = None

    try:
        if checkout_dir.exists():
            logging.info(f"Removing pre-existing directory: {checkout_dir}")
            shutil.rmtree(checkout_dir)

        # Clone and checkout
        repo = Repo.clone_from(repo_url, checkout_dir)
        repo.git.checkout(commit)

        # Set up local Python venv & install dependencies
        venv_dir = setup_repo_venv(checkout_dir, repo_name=repo_name)

        # Run ra-aid
        patch_str = run_raaid(
            checkout_dir,
            venv_dir,
            problem_statement,
            fail_tests,
            pass_tests
        )

    except Exception as e:
        logging.error(f"Failed to process {repo_url}:{commit} - {e}")
        return {
            "instance_id": inst_id,
            "model_patch": "",
            "model_name_or_path": "ra-aid"
        }
    finally:
        if cleanup:
            logging.info(f"Cleaning up directory: {checkout_dir}")
            shutil.rmtree(checkout_dir, ignore_errors=True)

    return {
        "instance_id": inst_id,
        "model_patch": patch_str if patch_str else "",
        "model_name_or_path": "ra-aid"
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions for SWE-bench Lite using ra-aid."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to store prediction file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Number of instances to process (default: all)"
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        required=True,
        help="Directory where projects will be cloned. Must exist or can be created."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="If set, remove the cloned repos after generating the patch."
    )
    args = parser.parse_args()

    base_dir, log_dir = create_output_dirs()
    setup_logging(log_dir, args.verbose)
    logging.info("Starting script")

    args.projects_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_safely()
    if dataset is None:
        sys.exit(1)

    # Combine 'dev' and 'test' splits for this dataset (there is no 'train')
    all_data = list(dataset["dev"]) + list(dataset["test"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = args.output_dir / "predictions.json"
    predictions = []

    limit = args.num_instances if args.num_instances else len(all_data)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False
    ) as progress:
        task = progress.add_task("Processing instances...", total=limit)
        for i, inst in enumerate(all_data):
            if i >= limit:
                break
            try:
                pred = process_instance(
                    inst,
                    projects_dir=args.projects_dir,
                    cleanup=args.cleanup
                )
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error processing instance: {inst.get('instance_id', '')} - {e}")
            finally:
                progress.advance(task)

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    logging.info("Done generating predictions.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Unhandled error occurred.")
        sys.exit(1)
