"""Check a LifeMate submission.jsonl against the public_test input before upload.

    python3 -I scripts/check_submission.py SUBMISSION.jsonl DATA_ROOT     (DATA_ROOT holds public_test/)
Checks: one line per sample, same ids in the same order, exactly the keys id / tool_calls / state_hash, a 64-hex
state hash, well-formed calls, size < 5 MB, and the split's own zaic_tools validator. Prints READY TO UPLOAD.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sub, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
ids = [json.loads(l)["id"] for l in open(root / "public_test/public_test.jsonl", encoding="utf-8") if l.strip()]
lines = [l for l in open(sub, encoding="utf-8") if l.strip()]
problems, seen = [], []
for n, l in enumerate(lines, 1):
    try:
        d = json.loads(l)
    except json.JSONDecodeError as e:
        problems.append(f"line {n}: not JSON ({e})")
        continue
    if set(d) != {"id", "tool_calls", "state_hash"}:
        problems.append(f"line {n}: keys {sorted(d)}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(d.get("state_hash", ""))):
        problems.append(f"line {n}: bad state_hash")
    tc = d.get("tool_calls")
    if not isinstance(tc, list) or any(not isinstance(c, dict) or set(c) != {"name", "arguments"}
                                       or not isinstance(c["arguments"], dict) for c in tc):
        problems.append(f"line {n}: bad tool_calls")
    seen.append(d.get("id"))
size = sub.stat().st_size
empty = sum(json.loads(l)["tool_calls"] == [] for l in lines)
print(f"lines {len(lines)} / expected {len(ids)} | same ids in same order: {seen == ids} | "
      f"duplicates: {len(seen) - len(set(seen))} | size {size / 1e6:.2f} MB (limit 5 MB) | empty tool_calls: {empty}")
for p in problems[:20]:
    print("PROBLEM", p)
p = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", str(sub)],
                   cwd=root / "public_test", capture_output=True, text=True)
print("zaic_tools validator exit", p.returncode, "|", (p.stdout + p.stderr).strip()[-600:])
ok = not problems and seen == ids and size < 5_000_000 and p.returncode == 0
print("READY TO UPLOAD" if ok else "NOT READY")
sys.exit(0 if ok else 1)
