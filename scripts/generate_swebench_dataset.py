#!/usr/bin/env python3
"""
Script to generate SWE-bench dataset for RA.Aid evaluation.
This is a work in progress and is not yet functional.

This script handles:
- Loading the SWE-bench Lite dataset
- Creating dated output directories
- Setting up logging infrastructure
- Processing dataset instances (placeholder)
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
from typing import Optional, Tuple, Dict, Any

from datasets import load_dataset
from git import Repo
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure logging with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        verbose: Whether to enable debug logging
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "generate_dataset.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(console_handler)

def load_dataset_safely() -> Optional[dict]:
    """Load SWE-bench dataset with error handling.
    
    Returns:
        Dataset object if successful, None otherwise
    """
    try:
        dataset = load_dataset("princeton-nlp/SWE-bench", "default")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None

def create_output_dirs() -> Tuple[Path, Path]:
    """Create dated output directory structure.
    
    Returns:
        Tuple of (output_dir, log_dir) paths
    """
    date_str = datetime.now().strftime("%Y%m%d")
    base_dir = Path("evaluation") / "default" / f"{date_str}_raaid"
    log_dir = base_dir / "logs"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return base_dir, log_dir

def process_dataset_instance(instance: Dict[str, Any], output_dir: Path) -> bool:
    """Process a single dataset instance.
    
    Args:
        instance: Dataset instance containing problem information
        output_dir: Directory to store output files
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Required fields
        logging.debug(f"Instance data: {instance}")
        logging.debug(f"Instance keys: {instance.keys()}")
        
        instance_id = str(instance['id'])  # Use id as unique identifier
        repo_url = instance['repo_url']
        commit_id = instance['code_before']['revision']
        
        # Issue description
        issue_title = instance['issue_title']
        issue_body = instance.get('issue_body', '')  # Optional with default
        issue_desc = f"{issue_title}\n\n{issue_body}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Clone repository
            repo = Repo.clone_from(repo_url, temp_path)
            repo.git.checkout(commit_id)
            
            # Format input for ra-aid
            issue_desc = instance['problem_statement']
            test_info = instance.get('test_information', '')
            
            # Run ra-aid
            patch = run_raaid(temp_path, issue_desc, test_info)
            if not patch:
                return False
                
            # Write prediction
            write_prediction(output_dir, instance_id, patch)
            return True
            
    except Exception as e:
        # Use instance.get() to avoid KeyError in error logging
        instance_id = instance.get('id', '<unknown>')
        logging.error(f"Failed to process instance {instance_id}: {e}")
        return False

def parse_test_information(test_info: str) -> Tuple[list, list]:
    """Parse test information into failing and passing test lists.
    
    Args:
        test_info: Raw test information string
        
    Returns:
        Tuple[list, list]: Lists of (fail_to_pass, pass_to_pass) tests
        
    Raises:
        ValueError: If required test sections are missing or malformed
    """
    fail_to_pass = []
    pass_to_pass = []
    
    # Split into sections
    sections = test_info.split('\n\n')
    current_section = None
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if section.startswith('FAIL_TO_PASS:'):
            current_section = 'fail'
            tests = section.replace('FAIL_TO_PASS:', '').strip().split('\n')
            fail_to_pass.extend(test.strip() for test in tests if test.strip())
            
        elif section.startswith('PASS_TO_PASS:'):
            current_section = 'pass'
            tests = section.replace('PASS_TO_PASS:', '').strip().split('\n')
            pass_to_pass.extend(test.strip() for test in tests if test.strip())
    
    if not fail_to_pass:
        raise ValueError("No FAIL_TO_PASS tests found in test information")
        
    return fail_to_pass, pass_to_pass

def run_raaid(repo_dir: Path, issue_desc: str, test_info: str) -> Optional[str]:
    """Run ra-aid on the problem and capture output.
    
    Args:
        repo_dir: Path to repository directory
        issue_desc: Problem description
        test_info: Additional test information
        
    Returns:
        Optional[str]: Generated patch if successful, None otherwise
    """
    try:
        # Parse test information
        fail_to_pass, pass_to_pass = parse_test_information(test_info)
        
        # Format prompt with clear sections
        prompt = (
            f"{issue_desc}\n\n"
            "Tests that need to be fixed:\n"
            "```\n"
            + "\n".join(f"- {test}" for test in fail_to_pass)
            + "\n```\n\n"
        )
        
        if pass_to_pass:
            prompt += (
                "Tests that must remain passing:\n"
                "```\n"
                + "\n".join(f"- {test}" for test in pass_to_pass)
                + "\n```\n\n"
            )
            
    except ValueError as e:
        logging.error(f"Invalid test information format: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing test information: {e}")
        return None
    
    try:
        # Configure ra-aid with appropriate flags
        cmd = [
            'ra-aid',
            '-m', prompt,
            '--research-only',  # First analyze without implementation
            '--expert-provider', 'openai',  # Use OpenAI for expert knowledge
            '--verbose'  # Enable detailed logging
        ]
        
        # First run - research phase
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for research
        )
        
        if result.returncode != 0:
            logging.error("Research phase failed")
            return None
            
        # Second run - implementation phase
        cmd = [
            'ra-aid',
            '-m', prompt,
            '--expert-provider', 'openai',
            '--verbose'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for implementation
        )
        
        if result.returncode == 0:
            repo = Repo(repo_dir)
            return get_git_patch(repo)
            
        logging.error(f"ra-aid failed with exit code {result.returncode}")
        logging.debug(f"stdout: {result.stdout}")
        logging.debug(f"stderr: {result.stderr}")
        return None
        
    except subprocess.TimeoutExpired:
        logging.error("ra-aid timed out")
        return None
    except Exception as e:
        logging.error(f"Error running ra-aid: {e}")
        return None

def get_git_patch(repo: Repo) -> Optional[str]:
    """Generate a git patch from the current changes.
    
    Args:
        repo: GitPython Repo object
        
    Returns:
        Optional[str]: Formatted patch if valid changes exist
    """
    if not repo.is_dirty():
        logging.error("No changes detected in repository")
        return None
        
    try:
        # Get diff in patch format
        patch = repo.git.diff(unified=3)
        
        # Basic validation
        if not patch or not patch.strip():
            return None
            
        if not any(line.startswith('+') for line in patch.splitlines()):
            return None
            
        return patch
        
    except Exception as e:
        logging.error(f"Failed to generate patch: {e}")
        return None

def write_prediction(output_dir: Path, instance_id: str, patch: str) -> None:
    """Write prediction entry to JSONL file.
    
    Args:
        output_dir: Output directory path
        instance_id: Dataset instance ID
        patch: Generated patch content
    """
    prediction_file = output_dir / "all_preds.jsonl"
    
    entry = {
        "id": instance_id,
        "patch": patch,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "ra_aid_version": subprocess.check_output(
                ['ra-aid', '--version'],
                text=True
            ).strip(),
            "git_hash": subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                text=True
            ).strip()
        }
    }
    
    with open(prediction_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
        
    # Also save individual prediction files for easier inspection
    instance_dir = output_dir / "predictions" / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    with open(instance_dir / "prediction.json", "w") as f:
        json.dump(entry, f, indent=2)

def cleanup_temp_files(temp_dir: Path) -> None:
    """Remove temporary processing files.
    
    Args:
        temp_dir: Directory containing temporary files
    """
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logging.debug(f"Cleaned up temporary directory: {temp_dir}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench dataset for RA.Aid evaluation"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to store processed dataset"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if individual instances fail"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Number of instances to process (default: all)"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for dataset generation script."""
    args = parse_args()
    
    # Create directory structure
    base_dir, log_dir = create_output_dirs()
    
    # Initialize logging
    setup_logging(log_dir, args.verbose)
    logging.info("Starting dataset generation")
    
    # Load dataset
    dataset = load_dataset_safely()
    if dataset is None:
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        total_instances = len(dataset['train'])
        task = progress.add_task("Processing dataset...", total=total_instances)
        
        success_count = 0
        for idx, instance in enumerate(dataset['train']):
            try:
                if process_dataset_instance(instance, args.output_dir):
                    success_count += 1
            except Exception as e:
                # Use instance.get() to avoid KeyError in error logging
                instance_id = instance.get('id', '<unknown>')
                logging.error(f"Failed to process instance {instance_id}: {e}")
            finally:
                progress.advance(task)
                
            if args.num_instances is not None and idx + 1 >= args.num_instances:
                break
                
        progress.stop()
    
    logging.info(f"Dataset generation complete. Processed {success_count}/{total_instances} instances successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.exception("Unhandled error occurred")
        sys.exit(1)
