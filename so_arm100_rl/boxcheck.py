"""One rule, in one place: do not start a training run on a busy machine.

Wall-clock and env-steps/s numbers taken while something else was running
are fiction, and the other job's numbers are ruined too. Every entry point
that trains calls this.
"""
import subprocess

LOAD_LIMIT = 4.0


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def require_quiet_box(force=False, limit=LOAD_LIMIT, quiet=False):
    l1 = load1()
    if l1 > limit and not force:
        others = ""
        try:
            out = subprocess.run(["ps", "-eo", "pcpu,args", "--sort=-pcpu"],
                                 capture_output=True, text=True, timeout=5).stdout
            rows = [r for r in out.splitlines()[1:4] if r.strip()]
            others = "\n    " + "\n    ".join(r.strip()[:100] for r in rows)
        except Exception:      # noqa: BLE001
            pass
        raise SystemExit(
            f"\n  1-minute load average is {l1:.2f} (limit {limit}).\n"
            f"  Something else is using this machine.{others}\n\n"
            f"  Not starting. --force exists and you should not use it.\n")
    if not quiet:
        print(f"  box check: 1-min load {l1:.2f}, ok")
    return l1
