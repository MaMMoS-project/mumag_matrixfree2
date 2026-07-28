"""Tests for installed ``tommos`` distribution metadata."""

from importlib.metadata import metadata, version


def test_distribution_metadata_matches_import_package() -> None:
    """Verify installed metadata agrees with the import package."""
    project_metadata = metadata("tommos")
    assert project_metadata["Name"] == "tommos"
    assert version("tommos") == "0.1.0"
    assert project_metadata["Requires-Python"] == ">=3.11"
