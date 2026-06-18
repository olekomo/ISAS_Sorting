from pathlib import Path

import pytest

from isas_benchmarl import ISASConfig


@pytest.fixture
def config() -> ISASConfig:
    return ISASConfig.from_yaml(
        Path(__file__).parents[1] / "configs" / "isas_3x3.yaml"
    )
