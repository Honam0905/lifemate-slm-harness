"""Run test_* functions without pytest (pytest is not installed locally); pytest also collects these files."""
from __future__ import annotations

import sys
import time
import traceback


class Skip(Exception):
    pass


def run(namespace: dict) -> None:
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
