import csv
from datetime import timedelta

import pytest

from pangu_weather.cases import Case, parse_time, read_cases
from pangu_weather.schedule import build_schedule, greedy_steps, predict


@pytest.mark.parametrize("hours,expected", [(1, (1,)), (3, (3,)), (6, (6,)),
    (23, (6, 6, 6, 3, 1, 1)), (24, (24,)), (30, (24, 6)), (48, (24, 24)),
    (56, (24, 24, 6, 1, 1)), (168, (24,) * 7)])
def test_greedy(hours, expected):
    assert greedy_steps(hours) == expected


@pytest.mark.parametrize("value", [-1, 2.5, True, "24"])
def test_bad_leads(value):
    with pytest.raises(ValueError):
        greedy_steps(value)


def test_minimal_step_counts_and_provenance():
    # Compare with dynamic programming, independently of the greedy implementation.
    best = [0] + [999] * 240
    for h in range(1, len(best)):
        best[h] = 1 + min(best[h-s] for s in (1, 3, 6, 24) if s <= h)
        assert len(greedy_steps(h)) == best[h]
    class Trace:
        def __init__(self):
            self.calls = []
        def run(self, step, state):
            self.calls.append((step, state))
            return state + (step,)
    runtime = Trace()
    leads = list(range(6, 169, 6))
    result = dict(predict((), leads, runtime))
    assert len(runtime.calls) == 28
    assert result[18] == (6, 6, 6)
    assert result[24] == (24,)
    assert result[30] == (24, 6)
    assert result[48] == (24, 24)
    assert result[168] == (24,) * 7
    result = dict(predict((), list(range(1, 58)), Trace()))
    assert all(result[h] == greedy_steps(h) for h in result)
    assert len(build_schedule([56])) == 5


def test_terminal_time_and_short_horizon(init_time):
    case = Case("a", "storm", init_time, 56, 6)
    assert case.leads == list(range(6, 55, 6)) + [56]
    assert (case.init_time + timedelta(hours=case.leads[-1])).isoformat() == "2026-01-03T02:00:00+00:00"
    assert Case("b", "storm", init_time, 3, 6).leads == [3]


def test_csv(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("case_id,storm_id,init_time,forecast_hours,output_interval_hours\na,s,2025-01-01T08:00:00+08:00,56,\nb,s,2025-01-02T00:00:00Z,5,1\n")
    cases = read_cases(path)
    assert cases[0].init_time.hour == 0
    assert cases[0].output_interval_hours == 6
    assert cases[1].leads == [1, 2, 3, 4, 5]
    with path.open("a") as stream:
        stream.write("a,s,2025-01-01T00:00:00Z,24,6\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_cases(path)


@pytest.mark.parametrize("text", ["2025-01-01T00:00:00", "2025-01-01T00:30:00Z"])
def test_reject_ambiguous_time(text):
    with pytest.raises(ValueError):
        parse_time(text)
