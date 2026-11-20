#!/usr/bin/env python3
"""Drive a Colab VM from the local machine through the official Colab CLI.

The Colab CLI runs Python in the VM's Jupyter kernel.  Everything here is built on
`colab exec -s SESSION` with code piped on stdin, so it works unattended.

    remote.py [-s SESSION] sh 'nvidia-smi'                 # run a shell command on the VM
    remote.py [-s SESSION] exec FILE                       # run a local .py file in the kernel (foreground)
    remote.py [-s SESSION] env HF_TOKEN                    # copy local $HF_TOKEN to the VM (never echoed)
    remote.py [-s SESSION] env FOO=bar                     # set a literal
    remote.py [-s SESSION] put LOCAL REMOTE                # upload a file
    remote.py [-s SESSION] get REMOTE LOCAL                # download a file
    remote.py [-s SESSION] bg-start NAME -- python x.py a  # detached job, log in /content/logs/NAME.log
    remote.py [-s SESSION] bg-log NAME [-n 60]             # tail the job log
    remote.py [-s SESSION] bg-status NAME                  # running? exit code?
    remote.py [-s SESSION] bg-stop NAME
    remote.py [-s SESSION] fetch NAME [-n 40] [--grep REGEX]  # download log via file API (robust), print tail
    remote.py [-s SESSION] runner [--hours 20]             # KEEP RUNNING: busy kernel cell = VM stays alive; runs queued jobs
    remote.py [-s SESSION] submit NAME -- cmd args         # queue a job for the runner (file upload, no kernel call)

Background jobs are started with `setsid nohup`, so they survive later `colab exec` calls and
kernel restarts; they die only when the VM is stopped.  Env vars set with `env` are written to
/content/env.sh, which bg-start sources, so HF_TOKEN reaches detached jobs.

Set COLAB_BIN if `colab` is not on PATH.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

LOG_DIR = "/content/logs"
ENV_FILE = "/content/env.sh"


def colab_bin() -> str:
    b = os.environ.get("COLAB_BIN") or shutil.which("colab")
    if not b:
        sys.exit("colab CLI not found: put it on PATH or set COLAB_BIN")
    return b


EXEC_TIMEOUT = float(os.environ.get("COLAB_EXEC_TIMEOUT", "300"))


def run_kernel(code: str, session: str | None, check: bool = True, timeout: float | None = None) -> int:
    """Pipe python `code` into `colab exec` and stream its output (bounded: a lost reply must not hang us)."""
    cmd = [colab_bin(), "exec", "--timeout", str(timeout or EXEC_TIMEOUT)]
    if session:
        cmd += ["-s", session]
    p = subprocess.run(cmd, input=code, text=True)
    if check and p.returncode != 0:
        sys.exit(f"colab exec failed with code {p.returncode}")
    return p.returncode


def kernel_shell(command: str) -> str:
    """Python snippet that runs a shell command inside the kernel and prints its output."""
    return (
        "import subprocess, sys\n"
        f"p = subprocess.run({command!r}, shell=True, capture_output=True, text=True)\n"
        "sys.stdout.write(p.stdout); sys.stderr.write(p.stderr)\n"
        "print(f'[exit {p.returncode}]')\n"
    )


def cmd_sh(a):
    return run_kernel(kernel_shell(a.command), a.session)


def cmd_exec(a):
    cmd = [colab_bin(), "exec", "--timeout", str(a.timeout), "-f", a.file]
    if a.session:
        cmd += ["-s", a.session]
    return subprocess.run(cmd).returncode


def cmd_fetch(a):
    """Download a job's log/rc through the file API (no kernel round-trip) and print the tail."""
    d = os.environ.get("VMLOG_DIR") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "vmlogs")
    os.makedirs(d, exist_ok=True)
    sess = ["-s", a.session] if a.session else []
    status = "running (no exit code yet)"
    for ext in ("rc", "log"):
        local = os.path.join(d, f"{a.name}.{ext}")
        if os.path.exists(local):
            os.remove(local)
        try:
            subprocess.run([colab_bin(), "download", *sess, f"{LOG_DIR}/{a.name}.{ext}", local],
                           capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            print(f"[fetch] download of {a.name}.{ext} timed out")
        if ext == "rc" and os.path.exists(local):
            status = f"finished, exit code {open(local).read().strip()}"
    print(f"[fetch] {a.name}: {status}")
    log = os.path.join(d, f"{a.name}.log")
    if not os.path.exists(log):
        print("[fetch] no log on the VM yet")
        return 1
    with open(log, "rb") as f:
        text = f.read().decode("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in text.splitlines() if l.strip()]
    if a.grep:
        import re
        lines = [l for l in lines if re.search(a.grep, l)]
    for l in lines[-a.n:]:
        print(l[:a.width])
    print(f"[fetch] full log: {log}")
    return 0


def cmd_env(a):
    lines = []
    for item in a.items:
        if "=" in item:
            k, v = item.split("=", 1)
        else:  # copy from local environment, value never printed
            k = item
            v = os.environ.get(k)
            if v is None:
                sys.exit(f"{k} is not set in the local environment")
        lines.append((k, v))
    # written via python so the values never appear in a shell command line
    code = (
        "import os\n"
        f"os.makedirs({LOG_DIR!r}, exist_ok=True)\n"
        f"pairs = {json.dumps(lines)}\n"
        f"with open({ENV_FILE!r}, 'a') as f:\n"
        "    for k, v in pairs:\n"
        "        os.environ[k] = v\n"
        "        f.write(f'export {k}={v!r}\\n')\n"
        "print('set:', ', '.join(k for k, _ in pairs))\n"
    )
    return run_kernel(code, a.session)


def cmd_put(a):
    cmd = [colab_bin(), "upload"] + (["-s", a.session] if a.session else []) + [a.local, a.remote]
    return subprocess.run(cmd).returncode


def cmd_get(a):
    cmd = [colab_bin(), "download"] + (["-s", a.session] if a.session else []) + [a.remote, a.local]
    return subprocess.run(cmd).returncode


def cmd_bg_start(a):
    job = " ".join(shlex.quote(x) for x in a.argv)
    log = f"{LOG_DIR}/{a.name}.log"
    pid = f"{LOG_DIR}/{a.name}.pid"
    rc = f"{LOG_DIR}/{a.name}.rc"
    # exit code is recorded by the wrapper so bg-status can report it after the job ends
    inner = f"[ -f {ENV_FILE} ] && source {ENV_FILE}; {job}; echo $? > {rc}"
    code = (
        "import os, subprocess, time\n"
        f"os.makedirs({LOG_DIR!r}, exist_ok=True)\n"
        f"if os.path.exists({rc!r}): os.remove({rc!r})\n"
        f"p = subprocess.Popen(['bash', '-lc', {inner!r}], stdout=open({log!r}, 'ab'), stderr=subprocess.STDOUT,\n"
        "                     stdin=subprocess.DEVNULL, start_new_session=True)\n"
        f"open({pid!r}, 'w').write(str(p.pid))\n"
        "time.sleep(2)\n"
        f"print('started pid', p.pid, '| alive:', p.poll() is None)\n"
        f"print(open({log!r}, 'rb').read()[:2000].decode('utf-8', 'replace'))\n"
    )
    return run_kernel(code, a.session)


def cmd_bg_log(a):
    return run_kernel(kernel_shell(f"tail -n {a.n} {LOG_DIR}/{a.name}.log"), a.session)


def cmd_bg_status(a):
    shell = (
        f"pid=$(cat {LOG_DIR}/{a.name}.pid 2>/dev/null); "
        f"if [ -z \"$pid\" ]; then echo 'no such job'; "
        f"elif kill -0 $pid 2>/dev/null; then echo \"running (pid $pid)\"; "
        f"elif [ -f {LOG_DIR}/{a.name}.rc ]; then echo \"finished, exit code $(cat {LOG_DIR}/{a.name}.rc)\"; "
        f"else echo 'not running, no exit code recorded (killed?)'; fi; "
        f"ls -la {LOG_DIR}/{a.name}.log 2>/dev/null"
    )
    return run_kernel(kernel_shell(shell), a.session)


def cmd_watch(a):
    """Stream a background job's log from inside the kernel until it finishes (or max_seconds)."""
    code = f'''
import os, sys, time
log, rc = "{LOG_DIR}/{a.name}.log", "{LOG_DIR}/{a.name}.rc"
t0, pos = time.time(), 0
def dump():
    global pos
    if os.path.exists(log):
        with open(log, "rb") as f:
            f.seek(pos); chunk = f.read(); pos += len(chunk)
        if chunk:
            sys.stdout.write(chunk.decode("utf-8", "replace")); sys.stdout.flush()
while True:
    dump()
    if os.path.exists(rc):
        print(f"[watch] job finished, exit code {{open(rc).read().strip()}}, {{(time.time() - t0) / 60:.1f}} min watched"); break
    if time.time() - t0 > {a.max_seconds}:
        print("[watch] job still running; watch window over"); break
    time.sleep({a.interval})
'''
    cmd = [colab_bin(), "exec", "--timeout", str(a.max_seconds + 120)] + (["-s", a.session] if a.session else [])
    return subprocess.run(cmd, input=code, text=True).returncode


# The Colab backend keeps a VM alive only while its kernel is busy.  Detached jobs leave the kernel idle and the
# VM is reclaimed after ~20 min (seen 2026-10-31 and 2026-11-01).  The runner is one long kernel cell that keeps
# the kernel busy and starts jobs dropped into QUEUE_DIR as files (uploaded via the file API by `submit`).
QUEUE_DIR = "/content/queue"
RUNNER_CODE = r'''
import glob, os, subprocess, time
Q, L, ENV, SECONDS = "@Q@", "@L@", "@ENV@", @SECONDS@
os.makedirs(Q, exist_ok=True)
os.makedirs(L, exist_ok=True)
t0 = beat = time.time()
print("[runner] up; jobs are *.job files in", Q, flush=True)
while time.time() - t0 < SECONDS and not os.path.exists(Q + "/STOP"):
    for f in sorted(glob.glob(Q + "/*.job")):
        name = os.path.basename(f)[:-4]
        cmd = open(f).read().strip()
        os.rename(f, f + ".taken")
        rc, log = f"{L}/{name}.rc", f"{L}/{name}.log"
        if os.path.exists(rc):
            os.remove(rc)
        inner = f"[ -f {ENV} ] && source {ENV}; {cmd}; echo $? > {rc}"
        p = subprocess.Popen(["bash", "-lc", inner], stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        open(f"{L}/{name}.pid", "w").write(str(p.pid))
        print(f"[runner] {time.strftime('%H:%M:%S')} started {name} (pid {p.pid})", flush=True)
    if time.time() - beat > 900:
        beat = time.time()
        print(f"[runner] alive {(beat - t0) / 3600:.2f} h", flush=True)
    time.sleep(15)
print("[runner] exiting", flush=True)
'''


def cmd_runner(a):
    code = (RUNNER_CODE.replace("@Q@", QUEUE_DIR).replace("@L@", LOG_DIR).replace("@ENV@", ENV_FILE)
            .replace("@SECONDS@", str(int(a.hours * 3600))))
    cmd = [colab_bin(), "exec", "--timeout", str(int(a.hours * 3600) + 600)] + (["-s", a.session] if a.session else [])
    # if this client disconnects the cell keeps running on the VM (the kernel stays busy)
    return subprocess.run(cmd, input=code, text=True).returncode


def cmd_submit(a):
    import tempfile
    job = shlex.join(a.argv)
    with tempfile.NamedTemporaryFile("w", suffix=".job", delete=False) as f:
        f.write(job + "\n")
        tmp = f.name
    cmd = [colab_bin(), "upload"] + (["-s", a.session] if a.session else []) + [tmp, f"{QUEUE_DIR}/{a.name}.job"]
    rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
    os.remove(tmp)
    print(f"[submit] {a.name}: {job[:160]}" if rc == 0 else f"[submit] upload failed ({rc})")
    return rc


def cmd_bg_stop(a):
    shell = f"pid=$(cat {LOG_DIR}/{a.name}.pid 2>/dev/null) && kill -- -$pid 2>/dev/null; kill $pid 2>/dev/null; echo stopped"
    return run_kernel(kernel_shell(shell), a.session)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", default=os.environ.get("COLAB_SESSION"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sh"); p.add_argument("command"); p.set_defaults(fn=cmd_sh)
    p = sub.add_parser("exec"); p.add_argument("file"); p.add_argument("--timeout", type=int, default=1800); p.set_defaults(fn=cmd_exec)
    p = sub.add_parser("runner"); p.add_argument("--hours", type=float, default=20); p.set_defaults(fn=cmd_runner)
    p = sub.add_parser("submit"); p.add_argument("name"); p.add_argument("argv", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("fetch"); p.add_argument("name"); p.add_argument("-n", type=int, default=40); p.add_argument("--grep", default=None); p.add_argument("--width", type=int, default=400); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("env"); p.add_argument("items", nargs="+"); p.set_defaults(fn=cmd_env)
    p = sub.add_parser("put"); p.add_argument("local"); p.add_argument("remote"); p.set_defaults(fn=cmd_put)
    p = sub.add_parser("get"); p.add_argument("remote"); p.add_argument("local"); p.set_defaults(fn=cmd_get)
    p = sub.add_parser("bg-start"); p.add_argument("name"); p.add_argument("argv", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_bg_start)
    p = sub.add_parser("bg-log"); p.add_argument("name"); p.add_argument("-n", type=int, default=60); p.set_defaults(fn=cmd_bg_log)
    p = sub.add_parser("bg-status"); p.add_argument("name"); p.set_defaults(fn=cmd_bg_status)
    p = sub.add_parser("bg-stop"); p.add_argument("name"); p.set_defaults(fn=cmd_bg_stop)
    p = sub.add_parser("watch"); p.add_argument("name"); p.add_argument("--max_seconds", type=int, default=1500); p.add_argument("--interval", type=int, default=20); p.set_defaults(fn=cmd_watch)
    a = ap.parse_args()
    if a.cmd == "submit":
        a.argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
        if not a.argv:
            sys.exit("submit needs a command after --")
    if a.cmd == "bg-start":
        if a.argv and a.argv[0] == "--":
            a.argv = a.argv[1:]
        if not a.argv:
            sys.exit("bg-start needs a command after --")
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
