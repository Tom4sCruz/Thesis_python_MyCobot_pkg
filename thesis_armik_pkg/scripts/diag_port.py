#!/usr/bin/env python3
"""
SERIAL PORT DIAGNOSTIC -- is something else using the arm's port?
=================================================================

Symptoms this is for: pyserial's "device reports readiness to read but
returned no data (device disconnected or multiple access on port?)", armik's
"could not read joint angles after N attempts", and streamed moves that
stutter. All three happen when a SECOND process has the arm's serial port
open: it eats the robot's replies and interleaves its own traffic with
armik's setpoint stream. If it survives a reboot, something starts it at boot.

    sudo python3 scripts/diag_port.py                     # full report (needs sudo to see
                                                          # other users' processes)
    sudo python3 scripts/diag_port.py --active            # + 50 get_angles reads, if the port is free
    python3 scripts/diag_port.py --port /dev/ttyUSB0      # another port

Nothing here moves the arm. Without --active nothing is even sent to it.

What it checks
--------------
1. WHO HAS THE PORT OPEN  every process with a file descriptor on the port
   (PID, user, start time, parent, command line) -- plus fuser / lsof.
2. WHAT STARTS AT BOOT    running services that look related (getty, serial,
   mycobot, ros, server, python, ...), the serial consoles on this UART,
   @reboot cron entries, desktop autostart files, rc.local, shell rc files --
   and which of those changed in the last RECENT_DAYS days.
3. KERNEL CONSOLE          whether /proc/cmdline puts a console on the port.
4. PASSIVE LISTEN          open the port, send NOTHING, listen for a few
   seconds. The arm only talks when asked, so any bytes = someone else is
   talking to it.
5. EXCLUSIVE OPEN          whether another pyserial user holds an exclusive lock.
6. ACTIVE READS (--active) 50 x get_angles through armik, no motion.
7. ENVIRONMENT             pymycobot version, groups, port permissions.

Paste the whole output back.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import grp
import os
import pwd
import re
import shutil
import statistics
import stat
import subprocess
import time

RECENT_DAYS = 14
LISTEN_S = 3.0
N_ACTIVE_READS = 50
SERVICE_PATTERN = re.compile(r"getty|serial|mycobot|elephant|ros|server|python|node|socket|uart",
                             re.IGNORECASE)
RC_PATTERN = re.compile(r"python|ttyTHS|ttyUSB|ttyACM|mycobot|ros2? (run|launch)", re.IGNORECASE)

_results = []          # (status, title) per check, for the summary


# ---------------------------------------------------------------- output
def banner(n, title):
    print(f"\n{'=' * 72}\n[{n}] {title}\n{'=' * 72}")


def verdict(status, title, detail=""):
    _results.append((status, title))
    print(f"--> {status}: {title}" + (f" -- {detail}" if detail else ""))


def run(cmd, timeout=10):
    """Output of a shell command (stdout+stderr), or None if it can't run."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- 1. holders
def _proc_info(pid):
    info = {"pid": pid}
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip() or "?"
        cmd = " ".join(cmd.split())                      # one line, even for `sh -c '...'`
        info["cmd"] = cmd if len(cmd) <= 160 else cmd[:157] + "..."
    except OSError:
        info["cmd"] = "?"
    try:
        st = os.stat(f"/proc/{pid}")
        info["user"] = pwd.getpwuid(st.st_uid).pw_name
    except (OSError, KeyError):
        info["user"] = "?"
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        info["ppid"] = int(fields[1])
        ticks = int(fields[19])
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        age = up - ticks / os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        info["started"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - age))
        info["age_s"] = age
    except Exception:
        info["ppid"], info["started"], info["age_s"] = None, "?", None
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            units = re.findall(r"/([^/\n]+\.service)", f.read())
        # a system service it runs under -- NOT user@<uid>.service (that is a
        # whole login session; stopping it would log the user out)
        units = [u for u in units if not re.match(r"user(-runtime-dir)?@\d+\.service$", u)]
        info["unit"] = units[-1] if units else None
    except OSError:
        info["unit"] = None
    return info


def port_users(port):
    """Every process (other than this one) with an fd on `port`."""
    try:
        target = os.path.realpath(port)
    except OSError:
        target = port
    me, found, denied = os.getpid(), [], 0
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(d))
        if pid == me:
            continue
        try:
            fds = os.listdir(f"{d}/fd")
        except PermissionError:
            denied += 1
            continue
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.readlink(f"{d}/fd/{fd}")) == target:
                    found.append(_proc_info(pid))
                    break
            except OSError:
                continue
    return found, denied


def check_holders(port):
    banner(1, f"WHO HAS {port} OPEN")
    users, denied = port_users(port)
    if denied:
        print(f"(could not inspect {denied} processes -- run with sudo to see everyone's)")
    for u in users:
        parent = _proc_info(u["ppid"]) if u.get("ppid") else None
        print(f"  PID {u['pid']}  user {u['user']}  started {u['started']}"
              + (f"  (systemd unit: {u['unit']})" if u.get("unit") else ""))
        print(f"      cmd:    {u['cmd']}")
        if parent:
            print(f"      parent: PID {parent['pid']}  {parent['cmd']}")
    for tool in ("fuser -v", "lsof"):
        exe = tool.split()[0]
        if shutil.which(exe):
            out = run(f"{tool} {port}")
            if out:
                print(f"  $ {tool} {port}\n" + "\n".join("    " + l for l in out.splitlines()))
    if users:
        verdict("FAIL", f"{len(users)} other process(es) have {port} open", "see above")
    elif denied:
        verdict("WARN", "no other user found among the processes visible", "re-run with sudo")
    else:
        verdict("PASS", f"nobody else has {port} open")
    return users


# ---------------------------------------------------------------- 2. boot
def _recent(path):
    try:
        return time.time() - os.stat(path).st_mtime < RECENT_DAYS * 86400
    except OSError:
        return False


def check_boot(port):
    banner(2, "WHAT STARTS AT BOOT")
    suspicious = []
    dev = os.path.basename(port)

    out = run("systemctl list-units --type=service --state=running --no-legend --no-pager")
    if out is None:
        print("  (systemctl not available)")
    else:
        hits = [l.split()[0] for l in out.splitlines() if l.strip() and SERVICE_PATTERN.search(l)]
        print("  running services that look related:")
        for h in hits:
            print(f"    {h}")
        print("    (none)" if not hits else "", end="" if hits else "\n")

    for unit in ("nvgetty.service", f"serial-getty@{dev}.service", "ModemManager.service"):
        en = run(f"systemctl is-enabled {unit}") or "?"
        ac = run(f"systemctl is-active {unit}") or "?"
        flag = "  <-- can hold the port" if ac.startswith("active") and "ModemManager" not in unit else ""
        print(f"  {unit:32s} enabled={en.splitlines()[0]:10s} active={ac.splitlines()[0]}{flag}")
        if flag:
            suspicious.append(unit)

    print("  recently changed unit files (last %d days):" % RECENT_DAYS)
    any_recent = False
    for d in ("/etc/systemd/system", "/lib/systemd/system", "/usr/lib/systemd/system",
              *glob.glob("/home/*/.config/systemd/user"), "/root/.config/systemd/user"):
        for f in glob.glob(f"{d}/**/*.service", recursive=True):
            if _recent(f) and not os.path.islink(f):
                any_recent = True
                print(f"    {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.stat(f).st_mtime))}  {f}")
                suspicious.append(f)
    if not any_recent:
        print("    (none)")

    print("  @reboot / cron entries:")
    cron_lines = []
    for f in ["/etc/crontab", *glob.glob("/etc/cron.d/*")]:
        try:
            for l in open(f):
                if "@reboot" in l or RC_PATTERN.search(l):
                    cron_lines.append(f"{f}: {l.strip()}")
        except OSError:
            pass
    for user in sorted({p.pw_name for p in pwd.getpwall() if p.pw_uid >= 1000 or p.pw_uid == 0}):
        out = run(f"crontab -l -u {user} 2>/dev/null")
        for l in (out or "").splitlines():
            if l.strip() and not l.lstrip().startswith("#"):
                cron_lines.append(f"crontab({user}): {l.strip()}")
    for l in cron_lines:
        print(f"    {l}")
        if "@reboot" in l:
            suspicious.append(l)
    if not cron_lines:
        print("    (none)")

    print("  desktop autostart / rc.local / shell rc lines that start python or touch serial:")
    hits = []
    for f in glob.glob("/home/*/.config/autostart/*.desktop") + glob.glob("/etc/xdg/autostart/*.desktop"):
        try:
            txt = open(f).read()
        except OSError:
            continue
        if RC_PATTERN.search(txt) or _recent(f):
            m = re.search(r"^Exec=(.*)$", txt, re.MULTILINE)
            hits.append(f"{f}: Exec={m.group(1) if m else '?'}" + ("   [changed recently]" if _recent(f) else ""))
    for f in ["/etc/rc.local", "/etc/profile", *glob.glob("/home/*/.bashrc"), *glob.glob("/home/*/.profile"),
              *glob.glob("/home/*/.bash_profile"), "/root/.bashrc"]:
        try:
            for i, l in enumerate(open(f), 1):
                if RC_PATTERN.search(l) and not l.lstrip().startswith("#"):
                    hits.append(f"{f}:{i}: {l.strip()}")
        except OSError:
            pass
    for h in hits:
        print(f"    {h}")
    suspicious += [h for h in hits if "autostart" in h]
    if not hits:
        print("    (none)")

    if suspicious:
        verdict("WARN", "things that start at boot and could open the port", "see above")
    else:
        verdict("PASS", "nothing suspicious starts at boot")
    return suspicious


# ---------------------------------------------------------------- 3. console
def check_console(port):
    banner(3, "KERNEL CONSOLE ON THE PORT")
    try:
        cmdline = open("/proc/cmdline").read().strip()
    except OSError:
        cmdline = ""
    print(f"  /proc/cmdline: {cmdline}")
    dev = os.path.basename(port)
    if re.search(rf"console={re.escape(dev)}\b", cmdline):
        verdict("FAIL", f"the kernel console is on {dev}", "it writes boot / log text into the arm's line")
        return True
    verdict("PASS", f"no kernel console on {dev}")
    return False


# ---------------------------------------------------------------- 4/5. port
def check_listen(port, baud):
    banner(4, f"PASSIVE LISTEN ({LISTEN_S:.0f} s, nothing sent)")
    try:
        import serial
    except ImportError:
        verdict("WARN", "pyserial not installed -- skipped")
        return None
    try:
        s = serial.Serial(port, baud, timeout=0.1)
    except Exception as exc:
        verdict("FAIL", "could not open the port", str(exc))
        return None
    data, errors = b"", []
    t0 = time.time()
    try:
        s.reset_input_buffer()
        while time.time() - t0 < LISTEN_S:
            try:
                data += s.read(4096)
            except Exception as exc:          # the "multiple access" error lands here
                errors.append(str(exc))
                time.sleep(0.05)
    finally:
        s.close()
    print(f"  {len(data)} bytes received, {len(errors)} read errors")
    if data:
        print(f"  first bytes: {data[:48].hex(' ')}")
        if b"\xfe\xfe" in data:
            print("  (contains 0xFE 0xFE -- myCobot protocol frames: someone is talking to the arm)")
    for e in sorted(set(errors))[:3]:
        print(f"  read error: {e}")
    if data or errors:
        verdict("FAIL", "traffic on a port nobody here is using",
                "another process is driving the arm" if data else "another process is reading the port")
        return False
    verdict("PASS", "the line is quiet")
    return True


def check_exclusive(port, baud):
    banner(5, "EXCLUSIVE OPEN")
    try:
        import serial
    except ImportError:
        verdict("WARN", "pyserial not installed -- skipped")
        return
    try:
        s = serial.Serial(port, baud, timeout=0.1, exclusive=True)
        s.close()
        verdict("PASS", "exclusive lock taken and released",
                "(only proves no OTHER locking user; non-locking ones show up in [1])")
    except Exception as exc:
        verdict("FAIL", "could not take an exclusive lock", str(exc))


# ---------------------------------------------------------------- 6. active
def check_active(port, baud):
    banner(6, f"ACTIVE READS ({N_ACTIVE_READS} x get_angles, no motion)")
    from armik.connection import ArmConnection, ArmError
    try:
        conn = ArmConnection(port, baud)
    except Exception as exc:
        verdict("FAIL", "ArmConnection could not open the port", str(exc))
        return
    ms, fails = [], 0
    try:
        for _ in range(N_ACTIVE_READS):
            t0 = time.perf_counter()
            try:
                conn.get_angles(retries=1)
                ms.append((time.perf_counter() - t0) * 1000.0)
            except (ArmError, Exception):
                fails += 1
    finally:
        conn.close()
    if ms:
        ms.sort()
        print(f"  median {statistics.median(ms):.1f} ms, p95 {ms[int(0.95 * (len(ms) - 1))]:.1f} ms, "
              f"max {ms[-1]:.1f} ms, failures {fails}/{N_ACTIVE_READS}")
    if fails:
        verdict("FAIL", f"{fails}/{N_ACTIVE_READS} reads failed")
    else:
        verdict("PASS", "every read came back")


# ---------------------------------------------------------------- 7. env
def check_env(port):
    banner(7, "ENVIRONMENT")
    try:
        import pymycobot
        print(f"  pymycobot {getattr(pymycobot, '__version__', '?')}  ({pymycobot.__file__})")
    except ImportError:
        print("  pymycobot NOT installed")
    real_user = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    groups = [g.gr_name for g in grp.getgrall() if real_user in g.gr_mem]
    print(f"  user {real_user}, groups: {', '.join(groups) or '(none)'}")
    try:
        st = os.stat(port)
        print(f"  {port}: {stat.filemode(st.st_mode)} owner {pwd.getpwuid(st.st_uid).pw_name} "
              f"group {grp.getgrgid(st.st_gid).gr_name}")
        verdict("PASS", "port exists")
    except OSError as exc:
        verdict("FAIL", f"{port} not accessible", str(exc))


# ---------------------------------------------------------------- main
def main():
    from armik import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--active", action="store_true",
                    help="also do 50 get_angles reads through armik (only if the port looks free)")
    args = ap.parse_args()

    print(f"serial port diagnostic -- {args.port} @ {args.baud}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    if os.geteuid() != 0:
        print("NOTE: not running as root -- other users' processes are invisible. Use sudo for the full picture.")

    holders = check_holders(args.port)
    boot = check_boot(args.port)
    console = check_console(args.port)
    quiet = check_listen(args.port, args.baud) if not holders else None
    if holders:
        banner(4, "PASSIVE LISTEN")
        verdict("WARN", "skipped -- the port is already held by another process (see [1])")
    check_exclusive(args.port, args.baud)
    if args.active:
        if holders or quiet is False:
            banner(6, "ACTIVE READS")
            verdict("WARN", "skipped -- the port is not free, results would be meaningless")
        else:
            check_active(args.port, args.baud)
    check_env(args.port)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for status, title in _results:
        print(f"  {status:4s}  {title}")
    print()
    if holders:
        for h in holders:
            how = (f"sudo systemctl stop {h['unit']}   (and `sudo systemctl disable {h['unit']}` "
                   f"to stop it starting at boot)") if h.get("unit") else f"sudo kill {h['pid']}"
            print(f"VERDICT: {args.port} is held by PID {h['pid']} ({h['cmd']}) -> stop it: {how}")
        print("It is a shared robot -- check with whoever set that up before disabling it.")
    elif console:
        print(f"VERDICT: the kernel console is on {args.port} -- remove console={os.path.basename(args.port)} "
              f"from the boot arguments (extlinux.conf) and reboot.")
    elif quiet is False:
        print("VERDICT: something is talking on the line but no process holds it -- check [2] for "
              "services, and whether another computer is wired to the same UART.")
    else:
        print("VERDICT: no other user of the port found. If the arm still misbehaves, re-run with --active "
              "and paste the output.")


if __name__ == "__main__":
    main()
