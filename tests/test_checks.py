"""Type repair and the computed calendar."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifemate.agent.prompt import calendar_lines  # noqa: E402
from lifemate.controller.checks import clean_args, coerce  # noqa: E402

I, N, B = {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
A, O, S = {"type": "array", "items": {"type": "string"}}, {"type": "object"}, {"type": "string"}


def test_coerce_repairs():
    assert coerce("845000", I) == 845000 and coerce("2.800.000", I) == 2800000 and coerce("2,800,000", I) == 2800000
    assert coerce(2.0, I) == 2 and type(coerce(2.0, I)) is int
    assert coerce("2.5", N) == 2.5 and coerce("3", N) == 3
    assert coerce("true", B) is True and coerce("False", B) is False
    assert coerce("['contact_id']", A) == ["contact_id"] and coerce('["a", "b"]', A) == ["a", "b"]
    assert coerce("contact_id", A) == ["contact_id"]
    assert coerce("{'item_id': 'm1', 'qty': 1}", O) == {"item_id": "m1", "qty": 1}


def test_coerce_leaves_ambiguous_values():
    assert coerce("845000 - 713606", I) == "845000 - 713606"    # arithmetic is not ours to do
    assert coerce("năm mươi", I) == "năm mươi"
    assert coerce("09994384354", S) == "09994384354"            # strings stay strings
    assert coerce("[broken", A) == "[broken"
    assert coerce(True, I) is True                               # bool is never turned into an int
    assert coerce(2.5, I) == 2.5


def test_clean_args_flag_off_is_v1_behaviour():
    spec = {"parameters": {"properties": {"amount": I}, "required": ["amount"]}}
    assert clean_args({"amount": "845000"}, spec)[0] == {"amount": "845000"}
    assert clean_args({"amount": "845000"}, spec, coerce_types=True)[0] == {"amount": 845000}


def test_calendar_lines():
    lines = calendar_lines("2026-11-28T07:10:00+07:00")            # a Saturday
    assert "today Thứ Bảy 2026-11-28" in lines[0] and "tomorrow (mai) Chủ Nhật 2026-11-29" in lines[0]
    assert lines[1].startswith("This week: Thứ Hai 2026-11-23") and lines[1].endswith("Chủ Nhật 2026-11-29")
    assert lines[2].startswith("Next week: Thứ Hai 2026-11-30") and lines[2].endswith("Chủ Nhật 2026-12-06")


if __name__ == "__main__":
    from tests.runner import run
    run(globals())
