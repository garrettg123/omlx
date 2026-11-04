#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated, bounded runner for one Elephant trial (plan S2; stdlib only).

Modes: --dry-run (default; prechecks + plan, no child), --fake-child SCRIPT (planned launch, SCRIPT
replaces omlx.cli), --real-launch (needs --a1-approved + --auth-mode). Trial root R holds models/
logs/ pid receipts.jsonl home/ omlx-home/. The child gets HOME=R/home, OMLX_BASE_PATH=R/omlx-home,
HF_HUB_OFFLINE=1 and an allowlisted env, in its own process group under sandbox-exec (denies
guarded-root writes, non-loopback egress). A Python audit hook also blocks and records guarded
writes, but native writes (subprocesses, C code) bypass it and are stopped by the sandbox alone,
uncounted. The before/after snapshot (settings.json sha256+mtime, top-level mtime of each guarded
root) is the backstop for writes the hook can't see; it misses edits deeper inside a root. Also
sandbox-exec is deprecated, so each run first self-tests the real profile and stops
(isolation_unenforceable) unless a native probe write to a guarded dir is denied. Ceilings may only
be tightened. A stop means "process stopped", not proof GPU work was cancelled. The shared server
PID is only probed via os.kill(pid, 0). Exit 0 = clean run, 2 = refused before launch, 3 =
stopped/failed; any runner fault after spawn is runner_error (exit 3; group still killed, trial
cleaned, stop/cleanup/final emitted). Scope (mock contract only - not live inference): --fake-child
runs prove only this runner's args, env, isolation, ceilings and cleanup. Real oMLX honouring
--base-path/HOME and no-key loopback auth acceptability remain T1 pre-load / A1 gates for Garrett.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, shutil, signal, socket, subprocess, sys, time  # noqa: E401
from pathlib import Path

ROOT, GIB = Path(__file__).resolve().parents[1], 1 << 30
SHARED_PID, SHARED_PORT = 98445, 8000
SANDBOX, SETTINGS = "/usr/bin/sandbox-exec", Path.home() / ".omlx" / "settings.json"
DEFAULT_GUARDED = (Path.home() / ".omlx", Path.home() / "Library/Application Support/oMLX",
                   Path.home() / ".config/omlx", Path.home() / ".cache/huggingface")
DEFAULT_PARENT = Path.home() / ".cache" / "laya-elephant-trial"
# name: (ceiling, direction) - "max": override must be <= ceiling; "min": >=.
CEILINGS = {"walltime_s": (1200.0, "max"), "rss_bytes": (16 * GIB, "max"),
            "trial_bytes": (6 * GIB, "max"), "window_s": (300.0, "max"),
            "load_s": (300.0, "max"), "grace_s": (10.0, "max"),
            "sample_s": (1.0, "max"), "min_free_bytes": (150 * GIB, "min")}
RECEIPT_KEYS = frozenset(
    "event ts run_id mode reason pid pgid port limit value signal detail spawned ok argv "
    "env_keys limits trial_root cleanup group_pids_left port_closed trial_root_bytes "
    "models_deleted shared_pid_alive_before shared_pid_alive_after settings_present "
    "settings_unchanged guard_write_events guard_read_events guarded_roots_unchanged "
    "elapsed_s exit_code sandbox trial_root_bytes_after errors receipts_lost".split()) | set(CEILINGS)
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{6,}|bearer\s+\S+|api[_-]?key|token|secret|"
                       r"password|[A-Za-z0-9+_-]{40,})", re.I)
ENV_PASS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "LOGNAME", "SHELL", "TERM")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Runs inside the child before the target: block + record Python-level guarded writes.
BOOTSTRAP = r"""
import json, os, runpy, sys
_ev = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
_roots = tuple(r.rstrip('/') + '/' for r in json.loads(sys.argv[2]))
_W = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_WOPS = set('os.mkdir os.rename os.remove os.rmdir os.symlink os.link os.truncate os.chmod '
            'os.chown os.utime os.chflags os.setxattr os.removexattr shutil.rmtree '
            'shutil.copyfile shutil.copytree shutil.move shutil.copymode shutil.copystat'.split())
_ALL, _busy = _WOPS | {'open', 'os.listdir', 'os.scandir'}, []
def _guarded(p):
    if isinstance(p, (str, bytes, os.PathLike)):
        rp = os.path.realpath(os.path.abspath(os.fsdecode(p))) + '/'
        return rp if rp.startswith(_roots) else None
def _hook(event, args):
    if _busy or not args or event not in _ALL:
        return
    _busy.append(1)
    try:
        if event == 'open':
            mode, flags = args[1], args[2] or 0
            write = bool(flags & _W) or (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            hits = [_guarded(args[0])]
        elif event in _WOPS:
            write, hits = True, [_guarded(a) for a in args[:2]]
        else:
            write, hits = False, [_guarded(args[0])]
        hit = next((h for h in hits if h), None)
        if hit:
            rec = {'kind': 'write' if write else 'read', 'op': event, 'path': hit if write else None}
            os.write(_ev, (json.dumps(rec) + '\n').encode())
        if hit and write:
            raise PermissionError(1, 'guarded root write blocked', hit)
    finally:
        _busy.clear()
sys.addaudithook(_hook)
_t, _rest = sys.argv[3], sys.argv[4:]
sys.argv = _rest if _t == '-m' else [_t] + _rest
if _t == '-m':
    runpy.run_module(_rest[0], run_name='__main__', alter_sys=True)
else:
    runpy.run_path(_t, run_name='__main__')
"""

class Stop(Exception):
    """A stop rule fired before launch."""

def scrub(obj):
    """Keep allowlisted keys only; drop any secret-looking string value."""
    clean = lambda v: not (isinstance(v, str) and SECRET_RE.search(v))  # noqa: E731
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k in RECEIPT_KEYS and clean(v)}
    return [scrub(v) for v in obj if clean(v)] if isinstance(obj, list) else obj

class Receipts:
    def __init__(self, run_id: str, path: Path | None = None):
        self.run_id, self.path, self.lost = run_id, path, 0

    def emit(self, event: str, **fields) -> dict:
        rec = scrub({"event": event, "ts": round(time.time(), 3), "run_id": self.run_id, **fields})
        try:  # a lost receipt is counted (run fails closed) but never aborts stop/cleanup
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
        except OSError:
            self.lost += 1
        return rec

def resolve_limits(overrides: dict) -> dict:
    limits = {k: c for k, (c, _) in CEILINGS.items()}
    for key, val in ((k, v) for k, v in overrides.items() if v is not None):
        ceiling, direction = CEILINGS[key]
        if val <= 0 or (val > ceiling if direction == "max" else val < ceiling):
            raise Stop(f"ceiling_relaxation_refused:{key}")
        limits[key] = val
    return limits

def settings_snapshot(path: Path = SETTINGS) -> dict:
    """sha256 + mtime only; contents are never kept, printed or logged."""
    try:
        with open(path, "rb") as fh:  # read-only; only the digest leaves this block
            st, digest = os.fstat(fh.fileno()), hashlib.file_digest(fh, "sha256").hexdigest()
    except FileNotFoundError:
        return {"present": False}
    return {"present": True, "sha256": digest, "mtime_ns": st.st_mtime_ns}

def shared_pid_alive(pid: int = SHARED_PID) -> bool:
    try:
        os.kill(pid, 0)  # existence probe only: signal 0 is never delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True

def real(p: Path) -> Path:
    return Path(os.path.realpath(os.path.abspath(os.path.expanduser(str(p)))))

def is_under(child: Path, roots) -> bool:
    return any(real(child) == real(r) or real(r) in real(child).parents for r in roots)

def _size(path: str) -> int:
    try:
        return os.lstat(path).st_size
    except OSError:  # vanished mid-walk while the child is still writing
        return 0

def tree_bytes(path: Path) -> int:
    return sum(_size(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)

def roots_fingerprint(roots) -> list:
    """Top-level mtime only (changes on any entry create/delete/rename)."""
    return [os.stat(r).st_mtime_ns if os.path.isdir(r) else None for r in roots]

def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0

def pick_port(requested: int | None) -> int:
    if requested == SHARED_PORT:
        raise Stop("port_is_shared")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", requested or 0))
        except OSError:
            raise Stop("port_in_use") from None
        return s.getsockname()[1]

def pending_download_tasks(base: Path) -> bool:
    try:
        return bool(json.loads((base / "hf_download_tasks.json").read_text("utf-8") or "[]"))
    except FileNotFoundError:
        return False
    except ValueError:
        return True

def group_rss_and_pids(pgid: int) -> tuple[int, list]:
    """Live members of the group per ps (zombies excluded); a failing ps raises, never 'empty'."""
    out = subprocess.run(["ps", "-axo", "pgid=,pid=,rss=,stat="], capture_output=True, text=True,
                         timeout=5, check=True).stdout
    rows = [r for r in map(str.split, out.splitlines())
            if len(r) == 4 and r[0] == str(pgid) and not r[3].startswith("Z")]
    return sum(int(r[2]) for r in rows) * 1024, [int(r[1]) for r in rows]

def sandbox_profile(roots) -> str:
    rules = ["(version 1)", "(allow default)", '(deny network-outbound (remote ip "*:*"))',
             '(allow network-outbound (remote ip "localhost:*"))']
    paths = {str(Path(os.path.abspath(os.path.expanduser(str(x))))) for x in roots}
    paths |= {str(real(x)) for x in roots}  # literal and resolved spellings
    return "".join(rules + [f"(deny file-write* (subpath {json.dumps(r)}))" for r in sorted(paths)])

def build_launch(trial: Path, port: int, roots, python: str, fake: str | None,
                 parent_env: dict | None = None) -> tuple[list, dict]:
    parent_env = os.environ if parent_env is None else parent_env
    env = {k: parent_env[k] for k in ENV_PASS if k in parent_env}
    home = trial / "home"
    env.update({"HOME": str(home), "OMLX_BASE_PATH": str(trial / "omlx-home"),
                "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)})
    serve = ["serve", "--model-dir", str(trial / "models"), "--base-path",
             str(trial / "omlx-home"), "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "warning", "--no-cache", "--no-hf-cache", "--headless"]
    target = [fake] + serve if fake else ["-m", "omlx.cli"] + serve
    events = str(trial / "logs" / "guard-events.jsonl")
    roots_json = json.dumps(sorted({str(real(r)) for r in roots}))
    return [SANDBOX, "-p", sandbox_profile(roots), python, "-B", "-c", BOOTSTRAP, events,
            roots_json] + target, env

def guard_counts(trial: Path) -> tuple[int, int]:
    path = trial / "logs" / "guard-events.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    kinds = [json.loads(line).get("kind") for line in lines if line]
    return kinds.count("write"), kinds.count("read")

def sandbox_enforces(trial: Path, roots) -> bool:
    """Pre-launch self-test: native write to a guarded probe dir denied, to R allowed."""
    probe, ok = trial / "logs" / "probe-guarded", trial / "logs" / "probe-ok"
    probe.mkdir()
    cmd = [SANDBOX, "-p", sandbox_profile([*roots, probe]), "/bin/sh", "-c",
           '/usr/bin/touch "$1" && ! /usr/bin/touch "$2/x" 2>/dev/null', "sh", ok, probe]
    try:
        res = subprocess.run(cmd, env={"PATH": "/usr/bin:/bin"}, capture_output=True, timeout=10)
        good = res.returncode == 0 and ok.exists() and not (probe / "x").exists()
    except (OSError, subprocess.TimeoutExpired):
        good = False
    shutil.rmtree(probe, ignore_errors=True)
    ok.unlink(missing_ok=True)
    return good

def clean_trial(trial: Path, reason: str, retain_models: bool) -> None:
    """Byte cap: keep only receipts.jsonl + pid ("delete partial"); else drop home/, models/."""
    keep = ("receipts.jsonl", "pid") if reason == "trial_root_bytes" else None
    doomed = [p for p in trial.iterdir() if p.name not in keep] if keep else \
        [trial / "home"] + ([] if retain_models else [trial / "models"])
    for p in doomed:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

def kill_group(proc: subprocess.Popen, pgid: int, grace: float, rc: Receipts) -> list:
    """SIGTERM then SIGKILL this pgid only (never another PID); returns live members left."""
    if pgid <= 1 or pgid == SHARED_PID or pgid == os.getpgrp():
        raise RuntimeError("refusing to signal a foreign process group")
    gone = lambda: proc.poll() is not None and not group_rss_and_pids(pgid)[1]  # noqa: E731
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            detail, wait = "sent", grace if sig == signal.SIGTERM else 5.0
        except (ProcessLookupError, PermissionError) as exc:  # EPERM: exited, unreaped leader
            detail, wait = "ESRCH" if isinstance(exc, ProcessLookupError) else "EPERM", 0.0
        rc.emit("signal", pgid=pgid, signal=sig.name, detail=detail)
        deadline = time.monotonic() + wait
        while not gone() and time.monotonic() < deadline:
            time.sleep(0.05)
        if gone():  # poll() reaped the leader and ps shows no live member
            return []
    return group_rss_and_pids(pgid)[1] or [proc.pid]  # leader alive outside its group: escaped

def watch(proc, pgid, trial, port, limits, rc, start) -> str:
    ready_at = None
    while True:
        t = time.monotonic()
        if proc.poll() is not None:  # a blocked write is the reason even if the child then exits
            return "guarded_write" if guard_counts(trial)[0] else "child_exited"
        if ready_at is None and port_open(port):
            ready_at = t
            rc.emit("ready", port=port, elapsed_s=round(t - start, 3))
        free_short = limits["min_free_bytes"] - shutil.disk_usage("/").free  # bytes below floor
        checks = [("walltime", t - start, limits["walltime_s"]),
                  ("rss", group_rss_and_pids(pgid)[0], limits["rss_bytes"]),
                  ("trial_root_bytes", tree_bytes(trial), limits["trial_bytes"]),
                  ("guarded_write", guard_counts(trial)[0], 0), ("free_space", free_short, 0),
                  ("load_deadline", t - start, limits["load_s"]) if ready_at is None
                  else ("inference_window", t - ready_at, limits["window_s"])]
        for reason, value, limit in checks:
            if value > limit:
                rc.emit("limit", reason=reason, value=value, limit=limit)
                return reason
        if (trial / "done").exists():
            return "done"
        time.sleep(limits["sample_s"])

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--trial-parent", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--guarded-root", type=Path, action="append", default=[], help="extra root")
    ap.add_argument("--port", type=int)
    ap.add_argument("--python", default=sys.executable)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--fake-child", metavar="SCRIPT")
    mode.add_argument("--real-launch", action="store_true")
    ap.add_argument("--a1-approved", action="store_true")
    ap.add_argument("--auth-mode", choices=["loopback-no-key"],
                    help="fork's existing loopback bind without key (omlx/utils/network.py)")
    ap.add_argument("--retain-models", action="store_true")
    for key, (ceiling, _) in CEILINGS.items():  # tighten-only overrides
        ap.add_argument("--" + key.replace("_", "-"), type=type(ceiling))
    return ap.parse_args(argv)

def main(argv=None) -> int:
    args, start = parse(argv), time.monotonic()
    mode = "real" if args.real_launch else "fake" if args.fake_child else "dry"
    rc, roots = Receipts(args.run_id), list(DEFAULT_GUARDED) + list(args.guarded_root)
    trial = real(args.trial_parent) / args.run_id
    try:
        limits, is_real = resolve_limits({k: getattr(args, k) for k in CEILINGS}), mode == "real"
        for bad, why in (  # first failing precheck wins; all are read-only
            (not RUN_ID_RE.match(args.run_id), "bad_run_id"),
            (is_under(trial, roots) or is_under(args.trial_parent, roots), "trial_root_guarded"),
            (trial.exists(), "trial_root_exists"),
            (is_real and not args.a1_approved, "real_launch_not_authorized"),
            (is_real and not args.auth_mode, "auth_unresolved"),
            (is_real and not (ROOT / "omlx" / "cli.py").is_file(), "fork_source_missing"),
            (not os.access(SANDBOX, os.X_OK), "isolation_unenforceable"),
            (shutil.disk_usage("/").free < limits["min_free_bytes"], "free_space")):
            if bad:
                raise Stop(why)
        shared_before, snap_before, fp_before = shared_pid_alive(), settings_snapshot(), roots_fingerprint(roots)
        for sub in ("models", "logs", "home", "omlx-home"):
            (trial / sub).mkdir(parents=True, exist_ok=False)
        rc.path = trial / "receipts.jsonl"
        if pending_download_tasks(trial / "omlx-home"):
            raise Stop("pending_download_tasks")
        if not sandbox_enforces(trial, roots):
            raise Stop("isolation_unenforceable")
        port = pick_port(args.port)
        fake = str(real(Path(args.fake_child))) if args.fake_child else None
        cmd, env = build_launch(trial, port, roots, args.python, fake)
        rc.emit("plan", mode=mode, port=port, trial_root=str(trial), limits=limits,
                argv=cmd[3:4] + cmd[9:], env_keys=sorted(env), sandbox=True,
                shared_pid_alive_before=shared_before, settings_present=snap_before["present"])
    except Stop as exc:
        return finish(rc, mode, str(exc), False, start, exit_code=2)
    if mode == "dry":
        return finish(rc, mode, "dry_run_ok", False, start, exit_code=0)
    with open(trial / "logs" / "child.log", "wb") as log:
        proc = subprocess.Popen(cmd, env=env, cwd=str(trial), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pgid, errors, reason = proc.pid, [], "runner_error"  # from here on faults are recorded
    try:
        (trial / "pid").write_text(f"{proc.pid}\n")
        rc.emit("spawned", pid=proc.pid, pgid=pgid, port=port)
        reason = watch(proc, pgid, trial, port, limits, rc, start)
    except BaseException as exc:  # incl. Ctrl-C: the group must never outlive the runner
        errors.append(f"watch:{type(exc).__name__}")
    left = attempt(errors, "kill_group", [pgid], kill_group, proc, pgid, limits["grace_s"], rc)
    reason = ("runner_error" if errors else "child_failed"  # fail closed: crashed child/sandbox
              if reason == "child_exited" and proc.returncode != 0 else reason)
    rc.emit("stop", reason=reason, detail="group_not_cleared" if left else "process stopped",
            exit_code=proc.returncode, errors=list(errors))
    size = tree_bytes(trial)  # measured before clean_trial deletes anything
    writes, reads = attempt(errors, "guard_counts", (None, None), guard_counts, trial)
    attempt(errors, "clean_trial", None, clean_trial, trial, reason, args.retain_models)
    after = attempt(errors, "settings", {"present": None}, settings_snapshot)
    ps_left = left or attempt(errors, "ps", (0, [pgid]), group_rss_and_pids, pgid)[1]
    port_closed = not attempt(errors, "port", True, port_open, port)
    roots_same = attempt(errors, "roots", None, roots_fingerprint, roots) == fp_before
    cleanup = {"group_pids_left": ps_left, "port_closed": port_closed, "errors": errors,
               "trial_root_bytes": size, "trial_root_bytes_after": tree_bytes(trial),
               "models_deleted": not (trial / "models").exists(), "guarded_roots_unchanged": roots_same,
               "shared_pid_alive_before": shared_before, "shared_pid_alive_after": shared_pid_alive(),
               "settings_present": after["present"], "settings_unchanged": after == snap_before,
               "guard_write_events": writes, "guard_read_events": reads}
    reason = "runner_error" if errors else reason
    ok = (not errors and not ps_left and writes == 0 and port_closed and roots_same
          and cleanup["shared_pid_alive_after"] == shared_before and after == snap_before)
    rc.emit("cleanup", cleanup=cleanup, ok=ok)
    good = reason in ("child_exited", "done") and ok
    return finish(rc, mode, reason, True, start, exit_code=0 if good else 3, cleanup=cleanup)

def attempt(errors: list, step: str, default, fn, *args):
    """Post-spawn step: a fault is recorded (run ends runner_error, exit 3), never raised."""
    try:
        return fn(*args)
    except Exception as exc:
        errors.append(f"{step}:{type(exc).__name__}")
        return default

def finish(rc, mode, reason, spawned, start, exit_code, cleanup=None) -> int:
    exit_code = 3 if rc.lost and exit_code == 0 else exit_code  # unrecorded run is no success
    rec = rc.emit("final", mode=mode, reason=reason, spawned=spawned, exit_code=exit_code,
                  elapsed_s=round(time.monotonic() - start, 3), cleanup=cleanup, receipts_lost=rc.lost)
    print(json.dumps(rec, sort_keys=True))
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
