import shutil
from collections.abc import Generator
from typing import Any

from datasets import load_dataset

from utils import NO_ENV_SETUP, SWEBenchEntry, SWEBenchEnvSetup, SWEBenchSplit, construct_codebase
from codegen.sdk.core.codebase import Codebase


class SWEBenchWrapper:
    """
    A wrapper for the princeton-nlp/SWE-bench dataset that manages loading, organizing,
    and iterating through coding task entries.
    """
    def __init__(self, remove_after_run: bool = False):
        """
        Initialize the wrapper by loading the SWE-bench dataset.
        
        Args:
            remove_after_run: If True, removes downloaded repositories after processing
        """
        print("Loading SWE-bench dataset...")
        self.ds = load_dataset("princeton-nlp/SWE-bench")
        print("SWE-bench dataset loaded.")
        self.remove_after_run = remove_after_run
        self.repo_groups = self.create_repo_groups()

    def create_repo_groups(self) -> dict:
        """
        Organize dataset entries by split (train/dev/test), repository, and environment setup commit.
        This grouping allows efficient iteration and environment setup reuse.
        
        Returns:
            A nested dictionary structure:
            {split: {repo: {environment_setup_commit: [entries]}}}
        """
        # Create a list of all possible splits
        SPLITS: list[SWEBenchSplit] = ["train", "dev", "test"]

        # Create a nested dictionary with explicit type hints
        repo_groups: dict[SWEBenchSplit, dict[str, dict[str, list[Any]]]] = {}

        # Group entries from all splits
        for split in SPLITS:
            repo_groups[split] = {}
            for entry in self.ds[split]:
                repo = entry["repo"]
                environment_setup_commit = entry["environment_setup_commit"]

                # Initialize nested dictionaries if they don't exist
                if repo not in repo_groups[split]:
                    repo_groups[split][repo] = {}
                if environment_setup_commit not in repo_groups[split][repo]:
                    repo_groups[split][repo][environment_setup_commit] = []

                repo_groups[split][repo][environment_setup_commit].append(entry)

        return repo_groups

    def get_entries_for_split(self, split: SWEBenchSplit) -> Generator[tuple[SWEBenchEnvSetup | SWEBenchEntry, Codebase], None, None]:
        """
        Generate entries for a specific dataset split (train/dev/test) along with their parsed codebases.
        
        The function follows this process for each repository:
        1. Constructs the codebase object
        2. For each environment setup commit:
           - Yields the environment setup info
           - For each entry using that environment:
             - Checks out the base commit
             - Yields the entry with parsed codebase
        
        Args:
            split: Which dataset split to process ("train", "dev", or "test")
            
        Yields:
            Tuples of (entry, codebase) where entry is either:
            - SWEBenchEnvSetup for environment setup commits
            - SWEBenchEntry for actual test entries
        """
        # ===== [ For each repo in the split ] =====
        for repo in self.repo_groups[split]:
            print(f"Processing repo: {repo}")
            # construct the codebase for the repo
            codebase = construct_codebase(repo_full_name=repo)
            # ===== [ For each environment setup commit ] =====
            for environment_setup_commit in self.repo_groups[split][repo]:
                print(f"Processing environment setup commit: {environment_setup_commit}")
                # yield the environment setup commit
                if environment_setup_commit:
                    print(f"Checking out environment setup commit: {environment_setup_commit}")
                    codebase.checkout(commit=environment_setup_commit, remote=True)
                    yield SWEBenchEnvSetup(split=split, environment_setup_commit=environment_setup_commit), codebase
                else:
                    print("No environment setup commit found")
                    yield SWEBenchEnvSetup(split=split, environment_setup_commit=NO_ENV_SETUP), codebase
                # ===== [ For each test setup commit ] =====
                for entry in self.repo_groups[split][repo][environment_setup_commit]:
                    print(f"Processing test entry: {entry['instance_id']}")
                    codebase.checkout(commit=entry["base_commit"], remote=True)
                    # yield the test entry with a parsed codebase object
                    yield SWEBenchEntry(entry=entry, split=split), codebase

        if codebase and self.remove_after_run:
            print(f"Cleaning up repo: {repo}")
            # Clean up: remove the repo from the tmp_dir if requested
            shutil.rmtree(f"/tmp/codegen/{repo}")


if __name__ == "__main__":
    # Example usage of the wrapper
    swe_bench_wrapper = SWEBenchWrapper()
    for entry, codebase in swe_bench_wrapper.get_entries_for_split("train"):
        if isinstance(entry, SWEBenchEnvSetup):
            print(f"Environment setup commit: {entry.environment_setup_commit}")
            # install dependencies...
        elif isinstance(entry, SWEBenchEntry):
            print(f"Entry: {entry.entry['instance_id']}")
            problem_statement = entry.entry["problem_statement"]
            print(f"Task: {problem_statement[:20]}")
            # send agent to solve tasks....

        print(f"Number of files: {len(codebase.files)}")
