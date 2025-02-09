from unittest.mock import patch, MagicMock

import pytest

from ra_aid.models_params import ReasoningTier, Capability
from ra_aid.tools.memory import _global_memory

@pytest.fixture(autouse=True)
def mock_global_memory():
    """Mock the global memory configuration."""
    with patch.dict(_global_memory, {"config": {
        "expert_provider": "openai",
        "expert_model": "gpt-4",
        "expert_auto_select_model": False
    }}, clear=True):
        yield
from ra_aid.tools.expert import (
    emit_expert_context,
    expert_context,
    get_best_expert_model_by_reasoning_tier,
    get_best_expert_model_by_reasoning_tier_capabilities_and_specialties,
    read_files_with_limit,
)

@pytest.mark.parametrize(
    "test_id,reasoning_tiers,tier,provider,expected_model,expected_provider",
    [
        (
            "expert_model_exists",
            {
                "model1": {"tier": ReasoningTier.BASIC, "provider": "openai"},
                "model2": {"tier": ReasoningTier.EXPERT, "provider": "anthropic"},
                "model3": {"tier": ReasoningTier.ADVANCED, "provider": "openai"},
            },
            ReasoningTier.EXPERT,
            None,
            "model2",
            "anthropic",
        ),
        (
            "multiple_expert_models",
            {
                "model1": {"tier": ReasoningTier.EXPERT, "provider": "openai"},
                "model2": {"tier": ReasoningTier.EXPERT, "provider": "anthropic"},
                "model3": {"tier": ReasoningTier.ADVANCED, "provider": "openai"},
            },
            ReasoningTier.EXPERT,
            None,
            "model1",
            "openai",
        ),
        (
            "advanced_model_exists",
            {
                "model1": {"tier": ReasoningTier.BASIC, "provider": "openai"},
                "model2": {"tier": ReasoningTier.ADVANCED, "provider": "anthropic"},
                "model3": {"tier": ReasoningTier.NONE, "provider": "openai"},
            },
            ReasoningTier.ADVANCED,
            None,
            "model2",
            "anthropic",
        ),
        (
            "basic_model_exists",
            {
                "model1": {"tier": ReasoningTier.BASIC, "provider": "openai"},
                "model2": {"tier": ReasoningTier.NONE, "provider": "anthropic"},
            },
            ReasoningTier.BASIC,
            None,
            "model1",
            "openai",
        ),
        (
            "specific_provider_match",
            {
                "model1": {"tier": ReasoningTier.EXPERT, "provider": "openai"},
                "model2": {"tier": ReasoningTier.EXPERT, "provider": "anthropic"},
            },
            ReasoningTier.EXPERT,
            "anthropic",
            "model2",
            "anthropic",
        ),
        (
            "provider_not_found",
            {
                "model1": {"tier": ReasoningTier.EXPERT, "provider": "openai"},
                "model2": {"tier": ReasoningTier.EXPERT, "provider": "anthropic"},
            },
            ReasoningTier.EXPERT,
            "gemini",
            "",
            "",
        ),
        (
            "no_matching_tier",
            {
                "model1": {"tier": ReasoningTier.BASIC, "provider": "openai"},
                "model2": {"tier": ReasoningTier.ADVANCED, "provider": "anthropic"},
            },
            ReasoningTier.EXPERT,
            None,
            "",
            "",
        ),
        (
            "empty_reasoning_tiers",
            {},
            ReasoningTier.EXPERT,
            None,
            "",
            "",
        ),
    ],
)
def test_get_best_expert_model_by_reasoning_tier(
    test_id, reasoning_tiers, tier, provider, expected_model, expected_provider, monkeypatch
):
    """Test getting model by reasoning tier with different scenarios."""
    monkeypatch.setattr("ra_aid.tools.expert.reasoning_tiers", reasoning_tiers)
    result_model, result_provider = get_best_expert_model_by_reasoning_tier(tier, provider)
    assert result_model == expected_model
    assert result_provider == expected_provider


@pytest.mark.parametrize(
    "test_id,reasoning_tiers,tier,capabilities,specialties,provider,expected_model,expected_provider",
    [
        (
            "exact_match",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL],
            [],
            None,
            "model1",
            "openai",
        ),
        (
            "multiple_capabilities",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL, Capability.MATHEMATICAL, Capability.CODE_ANALYSIS],
                    "provider": "anthropic",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL, Capability.MATHEMATICAL],
            [],
            None,
            "model1",
            "anthropic",
        ),
        (
            "all_capabilities",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": Capability.list(),
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL, Capability.CODE_GENERATION],
            [],
            None,
            "model1",
            "openai",
        ),
        (
            "specific_provider",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
                "model2": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "anthropic",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL],
            [],
            "anthropic",
            "model2",
            "anthropic",
        ),
        (
            "provider_not_found",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL],
            [],
            "gemini",
            "",
            "",
        ),
        (
            "no_capabilities_requested",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [],
            [],
            None,
            "model1",
            "openai",
        ),
        (
            "missing_capability",
            {
                "model1": {
                    "tier": ReasoningTier.EXPERT,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.MATHEMATICAL],
            [],
            None,
            "",
            "",
        ),
        (
            "wrong_tier",
            {
                "model1": {
                    "tier": ReasoningTier.BASIC,
                    "capabilities": [Capability.LOGICAL],
                    "provider": "openai",
                },
            },
            ReasoningTier.EXPERT,
            [Capability.LOGICAL],
            [],
            None,
            "",
            "",
        ),
        (
            "empty_reasoning_tiers",
            {},
            ReasoningTier.EXPERT,
            [Capability.LOGICAL],
            [],
            None,
            "",
            "",
        ),
    ],
)
def test_get_best_expert_model_by_reasoning_tier_capabilities_and_specialties(
    test_id,
    reasoning_tiers,
    tier,
    capabilities,
    specialties,
    provider,
    expected_model,
    expected_provider,
    monkeypatch,
):
    """Test getting model by reasoning tier, capabilities, and specialties with different scenarios."""
    monkeypatch.setattr("ra_aid.tools.expert.reasoning_tiers", reasoning_tiers)
    result_model, result_provider = get_best_expert_model_by_reasoning_tier_capabilities_and_specialties(
        tier, capabilities, specialties, provider
    )
    assert result_model == expected_model
    assert result_provider == expected_provider

@pytest.fixture
def temp_test_files(tmp_path):
    """Create temporary test files with known content."""
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"

    file1.write_text("Line 1\nLine 2\nLine 3\n")
    file2.write_text("File 2 Line 1\nFile 2 Line 2\n")
    file3.write_text("")  # Empty file

    return tmp_path, [file1, file2, file3]


def test_read_files_with_limit_basic(temp_test_files):
    """Test basic successful reading of multiple files."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(f) for f in files])

    assert "## File:" in result
    assert "Line 1" in result
    assert "File 2 Line 1" in result
    assert str(files[0]) in result
    assert str(files[1]) in result


def test_read_files_with_limit_empty_file(temp_test_files):
    """Test handling of empty files."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(files[2])])  # Empty file
    assert result == ""  # Empty files should be excluded from output


def test_read_files_with_limit_nonexistent_file(temp_test_files):
    """Test handling of nonexistent files."""
    tmp_path, files = temp_test_files
    nonexistent = str(tmp_path / "nonexistent.txt")
    result = read_files_with_limit([str(files[0]), nonexistent])

    assert "Line 1" in result  # Should contain content from existing file
    assert "nonexistent.txt" not in result  # Shouldn't include non-existent file


def test_read_files_with_limit_line_limit(temp_test_files):
    """Test enforcement of line limit."""
    tmp_path, files = temp_test_files
    result = read_files_with_limit([str(files[0]), str(files[1])], max_lines=2)

    assert "truncated" in result
    assert "Line 1" in result
    assert "Line 2" in result
    assert "File 2 Line 1" not in result  # Should be truncated before reaching file 2


@patch("builtins.open")
def test_read_files_with_limit_permission_error(mock_open_func, temp_test_files):
    """Test handling of permission errors."""
    mock_open_func.side_effect = PermissionError("Permission denied")
    tmp_path, files = temp_test_files

    result = read_files_with_limit([str(files[0])])
    assert result == ""  # Should return empty string on permission error


@patch("builtins.open")
def test_read_files_with_limit_io_error(mock_open_func, temp_test_files):
    """Test handling of IO errors."""
    mock_open_func.side_effect = IOError("IO Error")
    tmp_path, files = temp_test_files

    result = read_files_with_limit([str(files[0])])
    assert result == ""  # Should return empty string on IO error


def test_read_files_with_limit_encoding_error(temp_test_files):
    """Test handling of encoding errors."""
    tmp_path, files = temp_test_files

    # Create a file with invalid UTF-8
    invalid_file = tmp_path / "invalid.txt"
    with open(invalid_file, "wb") as f:
        f.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8

    result = read_files_with_limit([str(invalid_file)])
    assert result == ""  # Should return empty string on encoding error


def test_expert_context_management():
    """Test expert context global state management."""
    # Clear any existing context
    expert_context["text"].clear()
    expert_context["files"].clear()

    # Test adding context
    result1 = emit_expert_context.invoke("Test context 1")
    assert "Context added" in result1
    assert len(expert_context["text"]) == 1
    assert expert_context["text"][0] == "Test context 1"

    # Test adding multiple contexts
    result2 = emit_expert_context.invoke("Test context 2")
    assert "Context added" in result2
    assert len(expert_context["text"]) == 2
    assert expert_context["text"][1] == "Test context 2"

    # Test context accumulation
    assert all(
        ctx in expert_context["text"] for ctx in ["Test context 1", "Test context 2"]
    )
