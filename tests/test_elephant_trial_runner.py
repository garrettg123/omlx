# SPDX-License-Identifier: Apache-2.0
"""Elephant isolated trial runner (plan S2): stop rules, isolation and cleanup.

mock contract only - not live inference. No oMLX process is launched, no model
is loaded and nothing is downloaded: every child is a fake Python script run in
place of ``omlx.cli``. All writes go to temp dirs; the real default guarded
roots (``~/.omlx`` etc.) are only read (sha256 + mtime of settings.json, and a
directory listing in the read probe). The shared server PID is only probed
with ``os.kill(pid, 0)`` by the runner itself. Stdlib unittest only.
"""

import ast
import contextlib
import importlib.util
import io
import json
import os
import socket
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "elephant_trial_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("elephant_trial_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_runner()
FAST = ["--sample-s", "0.1", "--grace-s", "0.5"]


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            if k != "limits":  # limits keys are ceiling names (also allowlisted)
                yield from _walk_keys(v)
            else:
                yield from v.keys()
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="elephant-s2-")
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.parent = self.tmp / "trials"
        self.parent.mkdir()
        self.scripts = self.tmp / "scripts"
        self.scripts.mkdir()
        # host free space must not decide results; the runner's 150 GiB floor is unchanged
        free = mock.patch.object(R.shutil, "disk_usage", return_value=mock.Mock(free=1 << 50))
        free.start()
        self.addCleanup(free.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def script(self, name, body):
        path = self.scripts / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return str(path)

    def run_main(self, *args, run_id="t1"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = R.main(["--trial-parent", str(self.parent), "--run-id", run_id, *args])
        final = json.loads(buf.getvalue().strip().splitlines()[-1])
        return code, final

    def receipts(self, run_id="t1"):
        path = self.parent / run_id / "receipts.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def event(self, name, run_id="t1"):
        return [r for r in self.receipts(run_id) if r["event"] == name]


class DryRunAndCeilings(RunnerCase):
    def test_01_dry_run_plans_without_spawning(self):
        code, final = self.run_main()
        self.assertEqual((code, final["reason"], final["spawned"]), (0, "dry_run_ok", False))
        trial = self.parent / "t1"
        for sub in ("models", "logs", "home", "omlx-home"):
            self.assertTrue((trial / sub).is_dir(), sub)
        self.assertFalse((trial / "pid").exists())
        plan = self.event("plan")[0]
        self.assertTrue(plan["sandbox"])
        self.assertEqual(plan["trial_root"], str(trial))
        self.assertEqual(plan["limits"]["walltime_s"], 1200.0)
        self.assertNotEqual(plan["port"], R.SHARED_PORT)

    def test_02_ceilings_match_plan_section_5(self):
        gib = 1 << 30
        self.assertEqual(R.resolve_limits({}), {
            "walltime_s": 1200.0, "rss_bytes": 16 * gib, "trial_bytes": 6 * gib,
            "window_s": 300.0, "load_s": 300.0, "grace_s": 10.0, "sample_s": 1.0,
            "min_free_bytes": 150 * gib})
        self.assertEqual(R.SHARED_PID, 98445)  # literal: the shared server is never touched

    def test_03_tightening_allowed_relaxation_refused(self):
        self.assertEqual(R.resolve_limits({"walltime_s": 5.0})["walltime_s"], 5.0)
        self.assertEqual(R.resolve_limits({"min_free_bytes": 200 << 30})["min_free_bytes"],
                         200 << 30)
        for key, val in (("walltime_s", 1201.0), ("rss_bytes", (16 << 30) + 1),
                         ("trial_bytes", (6 << 30) + 1), ("window_s", 301.0),
                         ("load_s", 301.0), ("min_free_bytes", (150 << 30) - 1),
                         ("walltime_s", 0.0), ("sample_s", -1.0)):
            with self.assertRaises(R.Stop) as cm:
                R.resolve_limits({key: val})
            self.assertEqual(str(cm.exception), f"ceiling_relaxation_refused:{key}")
        code, final = self.run_main("--walltime-s", "5000")
        self.assertEqual((code, final["reason"], final["spawned"]),
                         (2, "ceiling_relaxation_refused:walltime_s", False))
        self.assertFalse((self.parent / "t1").exists())

    def test_04_bad_run_id_and_existing_trial_root(self):
        code, final = self.run_main(run_id="../escape")
        self.assertEqual((code, final["reason"]), (2, "bad_run_id"))
        self.assertEqual(self.run_main()[0], 0)
        code, final = self.run_main()
        self.assertEqual((code, final["reason"], final["spawned"]), (2, "trial_root_exists", False))

    def test_05_free_space_floor_stops_before_launch(self):
        code, final = self.run_main("--min-free-bytes", str(1 << 60))
        self.assertEqual((code, final["reason"], final["spawned"]), (2, "free_space", False))
        self.assertFalse((self.parent / "t1").exists())

    def test_06_pending_download_tasks_detected(self):
        base = self.tmp / "base"
        base.mkdir()
        self.assertFalse(R.pending_download_tasks(base))
        (base / "hf_download_tasks.json").write_text("[]")
        self.assertFalse(R.pending_download_tasks(base))
        (base / "hf_download_tasks.json").write_text('[{"repo": "x"}]')
        self.assertTrue(R.pending_download_tasks(base))
        (base / "hf_download_tasks.json").write_text("{corrupt")
        self.assertTrue(R.pending_download_tasks(base))  # unreadable counts as pending

    def test_07_settings_snapshot_is_hash_and_mtime_only(self):
        path = self.tmp / "settings.json"
        self.assertEqual(R.settings_snapshot(path), {"present": False})
        path.write_text('{"auth": {"api_key": "do-not-leak-me"}}')
        snap = R.settings_snapshot(path)
        self.assertEqual(set(snap), {"present", "sha256", "mtime_ns"})
        self.assertNotIn("do-not-leak-me", json.dumps(snap))
        self.assertEqual(len(snap["sha256"]), 64)

    def test_08_shared_port_refused(self):
        with self.assertRaises(R.Stop) as cm:
            R.pick_port(R.SHARED_PORT)
        self.assertEqual(str(cm.exception), "port_is_shared")

    def test_09_port_conflict_stops_run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            busy = s.getsockname()[1]
            code, final = self.run_main("--port", str(busy))
        self.assertEqual((code, final["reason"], final["spawned"]), (2, "port_in_use", False))


class LaunchContract(RunnerCase):
    def test_10_launch_argv_and_env(self):
        trial = self.tmp / "R"
        parent_env = {"PATH": "/usr/bin", "OMLX_API_KEY": "sk-shouldneverleak123",
                      "OMLX_FOO": "1", "HF_HOME": "/x", "HF_HUB_CACHE": "/y", "HOME": "/Users/x"}
        argv, env = R.build_launch(trial, 40123, [self.tmp / "g"], "/usr/bin/python3", None,
                                   parent_env=parent_env)
        self.assertEqual(argv[:2], [R.SANDBOX, "-p"])
        serve = argv[argv.index("omlx.cli") + 1:]
        self.assertEqual(argv[argv.index("omlx.cli") - 1], "-m")
        self.assertEqual(serve[0], "serve")
        pairs = dict(zip(serve[1::2], serve[2::2]))
        self.assertEqual(pairs["--base-path"], str(trial / "omlx-home"))
        self.assertEqual(pairs["--model-dir"], str(trial / "models"))
        self.assertEqual(pairs["--host"], "127.0.0.1")
        self.assertEqual(pairs["--port"], "40123")
        for flag in ("--no-cache", "--no-hf-cache", "--headless"):
            self.assertIn(flag, serve)
        self.assertNotIn("--api-key", argv)
        self.assertFalse(any("shouldneverleak" in a for a in argv))
        self.assertEqual(env["HOME"], str(trial / "home"))
        self.assertEqual(env["OMLX_BASE_PATH"], str(trial / "omlx-home"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertTrue(env["XDG_CONFIG_HOME"].startswith(str(trial)))
        self.assertTrue(env["XDG_CACHE_HOME"].startswith(str(trial)))
        self.assertEqual([k for k in env if k.startswith("OMLX_")], ["OMLX_BASE_PATH"])
        for k in ("OMLX_API_KEY", "HF_HOME", "HF_HUB_CACHE"):
            self.assertNotIn(k, env)
        self.assertFalse(any("shouldneverleak" in v for v in env.values()))

    def test_10b_parent_api_key_never_reaches_receipts(self):
        with mock.patch.dict(os.environ, {"OMLX_API_KEY": "sk-parentsecretvalue99"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = R.main(["--trial-parent", str(self.parent), "--run-id", "t1"])
        self.assertEqual(code, 0)
        text = (self.parent / "t1" / "receipts.jsonl").read_text() + buf.getvalue()
        self.assertNotIn("sk-parentsecretvalue99", text)
        self.assertNotIn("OMLX_API_KEY", self.event("plan")[0]["env_keys"])

    def test_10c_sandbox_profile_denies_guarded_writes_and_egress(self):
        prof = R.sandbox_profile([self.tmp / "g"])
        self.assertIn('(deny network-outbound (remote ip "*:*"))', prof)
        self.assertIn('(allow network-outbound (remote ip "localhost:*"))', prof)
        self.assertIn(f'(deny file-write* (subpath {json.dumps(str(self.tmp / "g"))}))', prof)

    def test_15_real_launch_gates(self):
        code, final = self.run_main("--real-launch")
        self.assertEqual((code, final["reason"], final["spawned"]),
                         (2, "real_launch_not_authorized", False))
        code, final = self.run_main("--real-launch", "--a1-approved")
        self.assertEqual((code, final["reason"], final["spawned"]), (2, "auth_unresolved", False))
        self.assertFalse((self.parent / "t1").exists())

    def test_16_runner_imports_stdlib_only_and_fits_budget(self):
        src = RUNNER.read_text(encoding="utf-8")
        self.assertLessEqual(len(src.splitlines()), 400)
        banned = {"omlx", "mlx", "numpy", "pydantic"}
        for node in ast.walk(ast.parse(src)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                self.assertNotIn(name.split(".")[0], banned, name)

    def test_17_trial_root_refused_under_omlx_home(self):
        parent = Path.home() / ".omlx" / f"elephant-s2-refuse-{os.getpid()}"
        self.assertFalse(parent.exists())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = R.main(["--trial-parent", str(parent), "--run-id", "t1"])
        final = json.loads(buf.getvalue().strip().splitlines()[-1])
        self.assertEqual((code, final["reason"], final["spawned"]),
                         (2, "trial_root_guarded", False))
        self.assertFalse(parent.exists())


class Receipts(RunnerCase):
    def test_14_receipt_allowlist_and_secret_drop(self):
        rec = R.Receipts("x").emit("stop", reason="walltime", api_key="abc", password="p",
                                   detail="Bearer sk-abcdefghijk", extra="kept?")
        self.assertEqual(set(rec), {"event", "ts", "run_id", "reason"})
        rec = R.Receipts("x").emit("plan", argv=["--host", "sk-abcdefghijkl", "127.0.0.1"])
        self.assertEqual(rec["argv"], ["--host", "127.0.0.1"])
        self.run_main()
        for line in self.receipts():
            self.assertLessEqual(set(_walk_keys(line)), R.RECEIPT_KEYS)


class FakeChildRuns(RunnerCase):
    def test_11_guarded_write_blocked_and_stops(self):
        guard = self.tmp / "guard"
        guard.mkdir()
        (guard / "keep.txt").write_text("x")
        before = sorted(os.listdir(guard)), os.stat(guard).st_mtime_ns
        fake = self.script("w.py", f"""
            import time
            for attempt in range(3):
                try:
                    open({str(guard / "probe.txt")!r}, "w").write("nope")
                except OSError:
                    pass
            time.sleep(999)
        """)
        code, final = self.run_main("--fake-child", fake, "--guarded-root", str(guard),
                                    "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (3, "guarded_write"))
        self.assertEqual((sorted(os.listdir(guard)), os.stat(guard).st_mtime_ns), before)
        self.assertEqual((guard / "keep.txt").read_text(), "x")
        cleanup = final["cleanup"]
        self.assertGreaterEqual(cleanup["guard_write_events"], 1)
        self.assertTrue(cleanup["guarded_roots_unchanged"])
        self.assertEqual(cleanup["group_pids_left"], [])

    def test_12_read_probe_with_real_default_roots(self):
        omlx = Path.home() / ".omlx"
        snap_before = R.settings_snapshot()
        fake = self.script("r.py", f"""
            import os
            p = {str(omlx)!r}
            if os.path.isdir(p):
                os.listdir(p)
        """)
        code, final = self.run_main("--fake-child", fake, "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (0, "child_exited"))
        cleanup = final["cleanup"]
        self.assertEqual(cleanup["guard_write_events"], 0)
        self.assertTrue(cleanup["settings_unchanged"])
        self.assertTrue(cleanup["guarded_roots_unchanged"])
        self.assertEqual(cleanup["settings_present"], snap_before["present"])
        self.assertEqual(R.settings_snapshot(), snap_before)
        if not snap_before["present"]:
            self.assertEqual(snap_before, {"present": False})
        if omlx.is_dir():
            self.assertGreaterEqual(cleanup["guard_read_events"], 1)
        code, final = self.run_main(run_id="t2")  # dry run, same real roots
        self.assertEqual((code, final["reason"]), (0, "dry_run_ok"))
        self.assertEqual(R.settings_snapshot(), snap_before)

    def test_13_walltime_kills_whole_group_and_cleans_up(self):
        fake = self.script("s.py", """
            import os, subprocess, sys, time
            g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(999)"])
            open("grandchild.pid", "w").write(str(g.pid))
            os.makedirs("models", exist_ok=True)
            open("models/weights.bin", "wb").write(b"0" * 4096)
            time.sleep(999)
        """)
        t0 = time.monotonic()
        code, final = self.run_main("--fake-child", fake, "--walltime-s", "1.5", *FAST)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual((code, final["reason"], final["spawned"]), (3, "walltime", True))
        trial = self.parent / "t1"
        gpid = int((trial / "grandchild.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(gpid, 0)
        c = final["cleanup"]
        self.assertEqual(c["group_pids_left"], [])
        self.assertTrue(c["port_closed"])
        self.assertTrue(c["models_deleted"])
        self.assertFalse((trial / "models").exists())
        self.assertIsInstance(c["trial_root_bytes"], int)
        self.assertEqual(c["shared_pid_alive_before"], c["shared_pid_alive_after"])
        self.assertTrue(c["settings_unchanged"])
        events = [r["event"] for r in self.receipts()]
        for ev in ("plan", "spawned", "limit", "signal", "stop", "cleanup", "final"):
            self.assertIn(ev, events)
        stop = self.event("stop")[0]
        self.assertEqual(stop["detail"], "process stopped")
        self.assertEqual(self.event("signal")[0]["signal"], "SIGTERM")
        pid = int((trial / "pid").read_text())
        self.assertEqual(self.event("spawned")[0]["pid"], pid)
        self.assertNotEqual(pid, R.SHARED_PID)

    def test_13b_sigterm_ignored_escalates_to_sigkill(self):
        fake = self.script("i.py", """
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(999)
        """)
        # 2.5 s: walltime counts from main() start, incl. the sandbox self-test, so leave the
        # child time to install SIG_IGN before SIGTERM lands.
        code, final = self.run_main("--fake-child", fake, "--walltime-s", "2.5", *FAST)
        self.assertEqual(final["reason"], "walltime")
        sigs = [r["signal"] for r in self.event("signal")]
        self.assertEqual(sigs, ["SIGTERM", "SIGKILL"])
        self.assertEqual(final["cleanup"]["group_pids_left"], [])

    def test_13c_rss_and_trial_bytes_stops(self):
        sleeper = self.script("z.py", "import time\ntime.sleep(999)\n")
        code, final = self.run_main("--fake-child", sleeper, "--rss-bytes", "1",
                                    "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (3, "rss"))
        t1 = self.parent / "t1"  # non-byte stop: home/ and models/ go, logs/ stay
        self.assertEqual([(t1 / d).exists() for d in ("home", "models", "logs")],
                         [False, False, True])
        stage = self.tmp / "stage"  # staged outside R, renamed in: files land whole
        stage.mkdir()
        writer = self.script("b.py", f"""
            import os, time
            s = {str(stage)!r}
            open(s + "/p.bin", "wb").write(b"0" * 3_000_000)
            open(s + "/b.bin", "wb").write(b"0" * 1_000_000)
            os.makedirs("home/.cache", exist_ok=True)
            os.rename(s + "/b.bin", "blob.partial")  # ~1 MB: under the 2 MB cap
            os.rename(s + "/p.bin", "home/.cache/partial.bin")  # now ~4 MB
            time.sleep(999)
        """)
        code, final = self.run_main("--fake-child", writer, "--trial-bytes", "2000000",
                                    "--walltime-s", "10", *FAST, run_id="t2")
        self.assertEqual((code, final["reason"]), (3, "trial_root_bytes"))
        limit = self.event("limit", "t2")[0]
        self.assertGreater(limit["value"], limit["limit"])
        t2, c = self.parent / "t2", final["cleanup"]  # byte cap -> kill, delete partial
        self.assertFalse((t2 / "home" / ".cache" / "partial.bin").exists())
        self.assertFalse((t2 / "blob.partial").exists())
        self.assertEqual(sorted(os.listdir(t2)), ["pid", "receipts.jsonl"])
        self.assertGreaterEqual(c["trial_root_bytes"], 4_000_000)
        self.assertLess(c["trial_root_bytes_after"], 100_000)

    def test_13g_nonzero_child_exit_fails_closed(self):
        crash = self.script("x.py", "raise SystemExit(7)\n")
        code, final = self.run_main("--fake-child", crash, "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"], final["spawned"]), (3, "child_failed", True))

    def test_13h_broken_or_noop_sandbox_stops_before_launch(self):
        fake = self.script("n.py", "")
        for i, prof in enumerate(("(version 1)(syntax-error", "(version 1)(allow default)")):
            with mock.patch.object(R, "sandbox_profile", return_value=prof):
                code, final = self.run_main("--fake-child", fake, run_id=f"s{i}")
            self.assertEqual((code, final["reason"], final["spawned"]),
                             (2, "isolation_unenforceable", False), prof)

    def test_13i_native_write_denied_by_sandbox_but_uncounted(self):
        guard = self.tmp / "guard"
        guard.mkdir()
        fake = self.script("t.py", f"""
            import subprocess
            subprocess.run(["/usr/bin/touch", {str(guard / 'native.txt')!r}])
        """)
        code, final = self.run_main("--fake-child", fake, "--guarded-root", str(guard),
                                    "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (0, "child_exited"))
        self.assertEqual(os.listdir(guard), [])  # the sandbox, not the hook, stopped it
        self.assertEqual(final["cleanup"]["guard_write_events"], 0)

    def test_13d_load_deadline_and_inference_window(self):
        sleeper = self.script("z.py", "import time\ntime.sleep(999)\n")
        code, final = self.run_main("--fake-child", sleeper, "--load-s", "0.5",
                                    "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (3, "load_deadline"))
        server = self.script("p.py", """
            import socket, sys, time
            port = int(sys.argv[sys.argv.index("--port") + 1])
            s = socket.socket(); s.bind(("127.0.0.1", port)); s.listen()
            time.sleep(999)
        """)
        code, final = self.run_main("--fake-child", server, "--window-s", "0.5",
                                    "--walltime-s", "10", *FAST, run_id="t2")
        self.assertEqual((code, final["reason"]), (3, "inference_window"))
        self.assertEqual(len(self.event("ready", "t2")), 1)
        self.assertTrue(final["cleanup"]["port_closed"])

    def test_13e_done_marker_and_retain_models(self):
        fake = self.script("d.py", """
            import os, time
            open("models/w.bin", "wb").write(b"1" * 100)
            open("done", "w").close()
            time.sleep(999)
        """)
        code, final = self.run_main("--fake-child", fake, "--retain-models", "--walltime-s", "10",
                                    *FAST)
        self.assertEqual((code, final["reason"]), (0, "done"))
        self.assertFalse(final["cleanup"]["models_deleted"])
        self.assertTrue((self.parent / "t1" / "models" / "w.bin").exists())

    def test_13f_kill_group_refuses_foreign_groups(self):
        proc = mock.Mock()
        with mock.patch.object(R.os, "killpg") as killpg:  # nothing real is ever signalled
            for pgid in (0, 1, R.SHARED_PID, os.getpgrp()):
                with self.assertRaises(RuntimeError):
                    R.kill_group(proc, pgid, 0.1, R.Receipts("x"))
        killpg.assert_not_called()


class KillAndCleanupRobustness(RunnerCase):
    def tail(self, run_id):
        return [r["event"] for r in self.receipts(run_id)][-3:]

    def test_18_killpg_eperm_reaps_leader_and_finishes(self):
        quick = self.script("q.py", "pass\n")  # exits by itself; killpg is mocked, nothing signalled
        with mock.patch.object(R.os, "killpg", side_effect=PermissionError(1, "EPERM")):
            code, final = self.run_main("--fake-child", quick, "--walltime-s", "10", *FAST)
        self.assertEqual((code, final["reason"]), (0, "child_exited"))
        self.assertEqual([s["detail"] for s in self.event("signal")], ["EPERM"])
        self.assertEqual((final["cleanup"]["group_pids_left"], final["cleanup"]["errors"]), ([], []))
        self.assertEqual(self.tail("t1"), ["stop", "cleanup", "final"])

    def test_18b_eperm_with_live_members_reports_them(self):
        proc = mock.Mock(pid=424242, poll=lambda: None)  # never reaped: members must be reported
        with mock.patch.object(R.os, "killpg", side_effect=PermissionError(1, "EPERM")) as killpg, \
                mock.patch.object(R, "group_rss_and_pids", return_value=(0, [424243])):
            self.assertEqual(R.kill_group(proc, 424242, 0.1, R.Receipts("x")), [424243])
        sent = [c.args for c in killpg.call_args_list]
        self.assertEqual(sent, [(424242, R.signal.SIGTERM), (424242, R.signal.SIGKILL)])

    def test_18c_post_spawn_faults_still_emit_stop_cleanup_final(self):
        quick, args = self.script("q.py", "pass\n"), ("--walltime-s", "10", *FAST)
        with mock.patch.object(R, "kill_group", return_value=[999999]):  # members survived
            code, final = self.run_main("--fake-child", quick, *args)
        self.assertEqual((code, self.event("stop")[0]["detail"]), (3, "group_not_cleared"))
        self.assertEqual(final["cleanup"]["group_pids_left"], [999999])
        with mock.patch.object(R, "kill_group", side_effect=RuntimeError("boom")):
            code, final = self.run_main("--fake-child", quick, *args, run_id="t2")
        self.assertEqual((code, final["reason"]), (3, "runner_error"))
        self.assertEqual(final["cleanup"]["errors"], ["kill_group:RuntimeError"])
        self.assertEqual(self.tail("t2"), ["stop", "cleanup", "final"])
        sleeper = self.script("z.py", "import time\ntime.sleep(999)\n")
        with mock.patch.object(R, "watch", side_effect=KeyboardInterrupt):  # Ctrl-C mid-run
            code, final = self.run_main("--fake-child", sleeper, *args, run_id="t3")
        self.assertEqual((code, final["reason"]), (3, "runner_error"))
        self.assertEqual(final["cleanup"]["group_pids_left"], [])  # real child group still killed
        self.assertEqual(self.tail("t3"), ["stop", "cleanup", "final"])
        with self.assertRaises(ProcessLookupError):
            os.kill(int((self.parent / "t3" / "pid").read_text()), 0)

    def test_18d_guarded_write_then_exit_never_crashes(self):
        (guard := self.tmp / "guard").mkdir()
        fake = self.script("x.py", f"""
            try:
                open({str(guard / "probe.txt")!r}, "w")
            except OSError:
                pass
        """)  # one blocked write, then exit 0: the round-2 crash case (EPERM race)
        for i in range(5):
            code, final = self.run_main("--fake-child", fake, "--guarded-root", str(guard),
                                        "--walltime-s", "10", *FAST, run_id=f"r{i}")
            self.assertEqual((code, final["reason"], final["spawned"]), (3, "guarded_write", True))
            self.assertEqual(final["cleanup"]["group_pids_left"], [])
            self.assertEqual(self.tail(f"r{i}"), ["stop", "cleanup", "final"])
        self.assertEqual(os.listdir(guard), [])


if __name__ == "__main__":
    unittest.main()
