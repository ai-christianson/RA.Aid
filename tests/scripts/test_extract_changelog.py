import pytest

from scripts.extract_changelog import extract_version_content


@pytest.fixture
def basic_changelog():
    return """## [1.2.0]
### Added
- Feature A
- Feature B

## [1.1.0]
### Changed
- Change X
- Change Y
"""


@pytest.fixture
def complex_changelog():
    return """## [2.0.0]
### Breaking
- Major change

## [1.9.0]
### Added
- Feature C
### Fixed
- Bug fix

## [1.8.0]
Some content
"""


def test_basic_version_extraction(basic_changelog):
    """Test extracting a simple version entry"""
    result = extract_version_content(basic_changelog, "1.2.0")
    expected = """## [1.2.0]
### Added
- Feature A
- Feature B"""
    assert result == expected


def test_middle_version_extraction(complex_changelog):
    """Test extracting a version from middle of changelog"""
    result = extract_version_content(complex_changelog, "1.9.0")
    expected = """## [1.9.0]
### Added
- Feature C
### Fixed
- Bug fix"""
    assert result == expected


def test_version_not_found():
    """Test error handling when version doesn't exist"""
    with pytest.raises(ValueError, match="Version 9.9.9 not found in changelog"):
        extract_version_content("## [1.0.0]\nSome content", "9.9.9")


def test_empty_changelog():
    """Test handling empty changelog"""
    with pytest.raises(ValueError, match="Version 1.0.0 not found in changelog"):
        extract_version_content("", "1.0.0")


def test_malformed_changelog():
    """Test handling malformed changelog without proper version headers"""
    content = "Some content\nNo version headers here\n"
    with pytest.raises(ValueError, match="Version 1.0.0 not found in changelog"):
        extract_version_content(content, "1.0.0")


def test_version_with_special_chars():
    """Test handling versions with special regex characters"""
    content = """## [1.0.0-beta.1]
Special version
## [1.0.0]
Regular version"""
    result = extract_version_content(content, "1.0.0-beta.1")
    assert result == "## [1.0.0-beta.1]\nSpecial version"
