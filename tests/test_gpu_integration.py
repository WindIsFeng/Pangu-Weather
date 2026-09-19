"""Opt in with PANGU_RUN_INTEGRATION=1; no network request is made by this test."""
import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pangu_weather.cases import parse_time
from pangu_weather.config import load_config
from pangu_weather.era5 import ERA5Source
from pangu_weather.runtime import Runtime
from pangu_weather.schedule import predict


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PANGU_RUN_INTEGRATION") != "1", reason="real-model GPU check is opt-in")
def test_against_official_iterative_chain():
    config = load_config(Path(os.environ.get("PANGU_TEST_CONFIG", "configs/default.yaml")))
    config = replace(config, download_missing=False)
    init = parse_time(os.environ.get("PANGU_TEST_INIT", "2025-09-22T00:00:00Z"))
    source = ERA5Source(config, logging.getLogger("integration"))
    initial, _ = source.get(init)
    runtime = Runtime(config, logging.getLogger("integration"))
    try:
        # Independent reproduction of the official loop, with a separate daily anchor.
        current = anchor = initial
        scheduled = predict(initial, [6, 12, 18, 24, 30], runtime)
        for index in range(5):
            if (index + 1) % 4 == 0:
                current = runtime.run(24, anchor)
                anchor = current
            else:
                current = runtime.run(6, current)
            lead, actual = next(scheduled)
            assert lead == (index + 1) * 6
            for expected, result in zip(current, actual):
                np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        with pytest.raises(StopIteration):
            next(scheduled)
        # Exercise the 3h and 1h weights as well as the 24h and 6h weights.
        outputs = list(predict(initial, [3, 4, 5], runtime))
        assert [lead for lead, _ in outputs] == [3, 4, 5]
    finally:
        runtime.close()
