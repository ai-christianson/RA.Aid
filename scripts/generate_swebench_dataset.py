#!/usr/bin/env python3
"""
Script to generate predictions for SWE-bench Lite (princeton-nlp/SWE-bench_Lite).
This script:
- Loads the SWE-bench Lite dataset
- Clones each repo at the specified commit
- Forms a prompt from the instance fields (problem_statement, FAIL_TO_PASS, PASS_TO_PASS)
- Calls ra-aid to create a patch
- Writes out predictions in the required JSON format
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from datasets import load_dataset
from git import Repo
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


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


def run_raaid(
    repo_dir: Path,
    problem_statement: str,
    fail_tests: List[str],
    pass_tests: List[str]
) -> Optional[str]:
    """Run ra-aid on the problem statement, returning a generated patch if possible."""
    # Create prompt
    prompt = f"{problem_statement}\n\nTests that need to be fixed:\n```\n"
    for t in fail_tests:
        prompt += f"- {t}\n"
    prompt += "```\n\n"
    if pass_tests:
        prompt += "Tests that must remain passing:\n```\n"
        for t in pass_tests:
            prompt += f"- {t}\n"
        prompt += "```\n\n"

    # Implementation phase
    impl_cmd = [
        'ra-aid',
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

    # Collect patch
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


def process_instance(instance: Dict[str, Any], output_repo_dir: Path) -> Dict[str, Any]:
    """
    Process a single dataset instance:
    - Clone the repo
    - Checkout commit
    - Build prompt from problem_statement, FAIL_TO_PASS, PASS_TO_PASS
    - Return dict in required format:
        {
            "instance_id": ...,
            "model_patch": ...,
            "model_name_or_path": ...
        }
    """
    inst_id = instance.get("instance_id", "<unknown>")
    repo_name = instance["repo"]
    commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    fail_tests = instance.get("FAIL_TO_PASS", [])
    pass_tests = instance.get("PASS_TO_PASS", [])

    # Convert to lists if they're strings
    if isinstance(fail_tests, str):
        fail_tests = [fail_tests]
    if isinstance(pass_tests, str):
        pass_tests = [pass_tests]

    # Attempt to build a github url if not provided
    # If 'repo' is "org/repo", create https://github.com/org/repo.git
    if "github.com" not in repo_name:
        repo_url = f"https://github.com/{repo_name}.git"
    else:
        repo_url = repo_name

    patch_str = None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            # Clone & checkout
            repo = Repo.clone_from(repo_url, tmp_path)
            repo.git.checkout(commit)
        except Exception as e:
            logging.error(f"Failed to clone/check out {repo_url}:{commit} - {e}")
            return {
                "instance_id": inst_id,
                "model_patch": "",
                "model_name_or_path": "ra-aid"
            }
        # Run ra-aid
        patch_str = run_raaid(tmp_path, problem_statement, fail_tests, pass_tests)

    # Return required prediction structure
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
    args = parser.parse_args()

    base_dir, log_dir = create_output_dirs()
    setup_logging(log_dir, args.verbose)
    logging.info("Starting script")

    dataset = load_dataset_safely()
    if dataset is None:
        sys.exit(1)

    # Combine "dev" and "test" splits (no "train" in this dataset)
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
                pred = process_instance(inst, args.output_dir)
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