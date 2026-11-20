# molab: the 7 Nov recipe run in 4 cells

Recipes, each trained, merged, scored on dev C (A2-s600 transcripts), and compared paired against v2r scored in the
same session on the same GPU:
- **v2rb**: v2r with do-nothing rebalanced (~1.6 h)
- **v2rb_full**: the same data with full fine-tuning (~3 h)
- **v2rb_r128**: LoRA r128 for 3 epochs (~2.5 h)

The job then writes the public submission of the best recipe if it beats v2r on dev.

1. molab → **New notebook** (or the one from 6 Nov). Notebook specs button (app header) → **attach the GPU**.
2. Sidebar **file icon → upload** both parts `lifemate_molab_bundle_1107.zip.part1of2` + `part2of2` (99 MB each). The notebook never prints data (all
   example printing is off), but don't share its link.
3. Paste each block below into its own cell and run them in order. Cell 3 waits until you press **Start**.
4. First run **smoke** (~25 min). It checks every recipe end to end: 3 steps each, 12-sample evals, 5 public samples.
   If it ends with `SMOKE OK`, select **full** and press **Start** again. The full run takes about 8.5 h.
5. **Keep this browser tab open** until cell 3 prints `JOB OK`. Cell 4 then shows the download buttons.

### Cell 1: find the uploaded bundle (or its parts), join the parts, check the zip, unpack it into its own folder
molab uploads are capped at 100 MB, so the bundle comes as `lifemate_molab_bundle_1107.zip.part1of2` and
`...part2of2`. Upload both. This cell joins them and checks every file's CRC.
```python
import marimo as mo
import os, re, sys, subprocess, zipfile, pathlib, time, hashlib

def _find_files():
    _hits = []
    for _base in [pathlib.Path.cwd(), pathlib.Path.home(), pathlib.Path("/tmp"), pathlib.Path("/mnt"), pathlib.Path("/data")]:
        if not _base.exists():
            continue
        for _root, _dirs, _files in os.walk(_base):
            if _root.count(os.sep) - str(_base).count(os.sep) >= 4 or "lifemate" in _root:
                _dirs[:] = []
            _dirs[:] = [d for d in _dirs if not d.startswith(".")]
            _hits += [pathlib.Path(_root, f) for f in _files if f.startswith("lifemate_molab_bundle")]
    return sorted(set(_hits))

def _bundles():
    _files = _find_files()
    _zips = [f for f in _files if f.name.endswith(".zip")]
    _groups = {}
    for _f in _files:
        _m = re.fullmatch(r"(.+\.zip)\.part(\d+)of(\d+)", _f.name)
        if _m:
            _groups.setdefault((_m[1], int(_m[3])), {})[int(_m[2])] = _f
    for (_name, _n), _parts in _groups.items():
        if len(_parts) != _n:
            print(f"{_name}: only parts {sorted(_parts)} of {_n} uploaded - upload the rest, then re-run this cell")
            continue
        _out = pathlib.Path("/tmp") / _name
        if not _out.exists() or _out.stat().st_size != sum(p.stat().st_size for p in _parts.values()):
            with open(_out, "wb") as _w:
                for _i in range(1, _n + 1):
                    _w.write(_parts[_i].read_bytes())
        _zips.append(_out)
    return sorted(set(_zips), key=lambda p: p.stat().st_mtime)

BUNDLES = _bundles()
print("notebook folder:", pathlib.Path.cwd())
print("bundles (oldest first):", BUNDLES)
WORK = pathlib.Path("/tmp") / (BUNDLES[-1].stem if BUNDLES else "lifemate_none")
if BUNDLES and not (WORK / "cloud" / "molab_job.sh").exists():
    print("sha256", hashlib.sha256(BUNDLES[-1].read_bytes()).hexdigest()[:16], "(expected f8bcba314a7dfdc1 for the 1107 bundle)")
    with zipfile.ZipFile(BUNDLES[-1]) as _z:
        _bad = _z.testzip()
        assert _bad is None, f"corrupt file in the bundle: {_bad} - upload the parts again"
        WORK.mkdir(parents=True, exist_ok=True)
        _z.extractall(WORK)
print("work folder:", WORK, "|", sorted(p.name for p in WORK.iterdir()) if WORK.exists() else "NOT YET - upload the bundle, then re-run this cell")
```

### Cell 2: mode + start button
```python
mode = mo.ui.radio(options=["smoke", "full"], value="smoke", label="run")
start = mo.ui.run_button(label="Start (smoke ~25 min / full ~8.5 h; keep this tab open)")
mo.hstack([mode, start], justify="start")
```

### Cell 3: run the job, with timestamped live log
```python
mo.stop(not start.value, mo.md("Choose **smoke** or **full**, then press **Start**."))
_env = dict(os.environ, PYTHON=sys.executable, HF_HOME=str(WORK / "hf"))
_env["SMOKE"] = "1" if mode.value == "smoke" else "0"
# _env["RECIPES"] = "v2rb v2rb_full v2rb_r128"     # default; reorder or shorten here
_proc = subprocess.Popen(["bash", str(WORK / "cloud" / "molab_job.sh")], cwd=WORK, env=_env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open(WORK / "job_console.log", "a") as _log:
    for _line in _proc.stdout:
        _msg = f"{time.strftime('%H:%M:%S')} {_line.rstrip()}"
        print(_msg, flush=True)
        _log.write(_msg + "\n"); _log.flush()
JOB_RC = _proc.wait()
print("job exit code:", JOB_RC, "(0 = OK)")
```

### Cell 4: download buttons (results zip = summary, dev rows, logs, submission; one zip per LoRA adapter)
```python
_ = JOB_RC   # runs after the job
_zips = sorted(WORK.glob("molab_results*.zip")) + sorted(WORK.glob("molab_adapter_*.zip"))
mo.vstack([mo.download(data=_z.read_bytes(), filename=_z.name,
                       label=f"Download {_z.name} ({_z.stat().st_size / 2**20:.0f} MB)") for _z in _zips]) \
    if _zips else mo.md("No zips yet (the job did not finish): copy the log from cell 3 instead.")
```

### What to send back
- The **summary** at the end of cell 3 (one line per recipe: `dev C`, `vs v2r-molab: +x CI [..] won/lost`).
- If something failed: the `=== ... RECIPE FAILED` line and the 5 lines above it.
- The results zip, plus the adapter zip of the winning recipe if it is a LoRA. A full fine-tune stays on molab: say so
  and we push it to a private Hugging Face repo with your token.

### Final fit (Sunday, after the dev run chooses the recipe)
In cell 3, set `_env["FINAL"] = "1"` and `_env["RECIPES"] = "v2rb_all"` (or `v2rb_all_full` / `v2rb_all_r128`,
matching the winner), then run full. The job trains on all 1,545 conversations, runs no dev eval (the model has seen
dev), and writes the public submission.
