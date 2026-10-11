"""Shared pytest configuration.

Hypothesis profiles (BLUEPRINT §5.1): `ci` runs 200 examples per property,
`nightly` runs 2,000. Select with HYPOTHESIS_PROFILE=<name>.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile(
    "nightly",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
