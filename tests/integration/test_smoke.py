"""End-to-end smoke (BLUEPRINT §5.1).

Placeholder until WS0 step 3 lands the canonical layer. Marked integration so
the PR job runs it; it must stay under five minutes.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_smoke_placeholder() -> None:
    import gpd

    assert gpd.__version__
