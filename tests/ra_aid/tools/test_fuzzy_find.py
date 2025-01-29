import pytest
from git import Repo
from git.exc import InvalidGitRepositoryError

from ra_aid.tools import fuzzy_find_project_files


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with some test files"""
    repo = Repo.init(tmp_path)

    # Create some files
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "test_main.py").write_text("def test_main(): pass")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/utils.py").write_text("def util(): pass")
    (tmp_path / "lib/__pycache__").mkdir()
    (tmp_path / "lib/__pycache__/utils.cpython-39.pyc").write_text("cache")

    # Create some untracked files
    (tmp_path / "untracked.txt").write_text("untracked content")
    (tmp_path / "draft.py").write_text("# draft code")

    # Add and commit only some files
    repo.index.add(["main.py", "lib/utils.py"])
    repo.index.commit("Initial commit")

    return tmp_path


def test_basic_fuzzy_search(git_repo):
    """Test basic fuzzy matching functionality"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "utils", "repo_path": str(git_repo)}
    )

    assert len(results) >= 1
    assert any("lib/utils.py" in match[0] for match in results)
    assert all(isinstance(match[1], int) for match in results)


def test_threshold_filtering(git_repo):
    """Test threshold parameter behavior"""
    # Should match with high threshold
    results_high = fuzzy_find_project_files.invoke(
        {"search_term": "main", "threshold": 80, "repo_path": str(git_repo)}
    )
    assert len(results_high) >= 1
    assert any("main.py" in match[0] for match in results_high)

    # Should not match with very high threshold
    results_very_high = fuzzy_find_project_files.invoke(
        {"search_term": "mian", "threshold": 99, "repo_path": str(git_repo)}
    )
    assert len(results_very_high) == 0


def test_max_results_limit(git_repo):
    """Test max_results parameter"""
    max_results = 1
    results = fuzzy_find_project_files.invoke(
        {"search_term": "py", "max_results": max_results, "repo_path": str(git_repo)}
    )
    assert len(results) <= max_results


def test_include_paths_filter(git_repo):
    """Test include_paths filtering"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "py", "include_paths": ["lib/*"], "repo_path": str(git_repo)}
    )
    assert all("lib/" in match[0] for match in results)


def test_exclude_patterns_filter(git_repo):
    """Test exclude_patterns filtering"""
    results = fuzzy_find_project_files.invoke(
        {
            "search_term": "py",
            "exclude_patterns": ["*test*"],
            "repo_path": str(git_repo),
        }
    )
    assert not any("test" in match[0] for match in results)


def test_invalid_threshold():
    """Test error handling for invalid threshold"""
    with pytest.raises(ValueError):
        fuzzy_find_project_files.invoke({"search_term": "test", "threshold": 101})


def test_non_git_repo(tmp_path):
    """Test error handling outside git repo"""
    with pytest.raises(InvalidGitRepositoryError):
        fuzzy_find_project_files.invoke(
            {"search_term": "test", "repo_path": str(tmp_path)}
        )


def test_exact_match(git_repo):
    """Test exact matching returns 100% score"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "main.py", "repo_path": str(git_repo)}
    )
    assert len(results) >= 1
    assert any(match[1] == 100 for match in results)


def test_empty_search_term(git_repo):
    """Test behavior with empty search term"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "", "repo_path": str(git_repo)}
    )
    assert len(results) == 0


def test_untracked_files(git_repo):
    """Test that untracked files are included in search results"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "untracked", "repo_path": str(git_repo)}
    )
    assert len(results) >= 1
    assert any("untracked.txt" in match[0] for match in results)


def test_no_matches(git_repo):
    """Test behavior when no files match the search term"""
    results = fuzzy_find_project_files.invoke(
        {"search_term": "nonexistentfile", "threshold": 80, "repo_path": str(git_repo)}
    )
    assert len(results) == 0
