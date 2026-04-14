"""
run_scenarios.py — Automated 4-phase Locust load test runner

Phases:
  1. SMOKE   — 5 users, 60s   → Sanity check: does everything work?
  2. RAMP    — 50 users, 120s → Gradual stress: how does latency grow?
  3. SPIKE   — 100 users, 60s → Burst: can we survive sudden spikes?
  4. SOAK    — 30 users, 180s → Endurance: memory leaks, connection exhaustion?

Usage:
  python run_scenarios.py
"""

import subprocess
import sys
import os
import json
import time
import csv
from pathlib import Path

LOCUSTFILE = "locustfile.py"
HOST = "http://localhost:8000"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Phase definitions ──────────────────────────────────────────
PHASES = [
    {
        "name": "smoke",
        "users": 5,
        "spawn_rate": 5,
        "run_time": "60s",
        "tags": ["smoke"],
        "stop_on_fail": True,
    },
    {
        "name": "ramp",
        "users": 50,
        "spawn_rate": 5,
        "run_time": "120s",
        "tags": ["core"],
        "stop_on_fail": False,
    },
    {
        "name": "spike",
        "users": 100,
        "spawn_rate": 50,
        "run_time": "60s",
        "tags": ["core"],
        "stop_on_fail": False,
    },
    {
        "name": "soak",
        "users": 30,
        "spawn_rate": 10,
        "run_time": "180s",
        "tags": ["core", "heavy"],
        "stop_on_fail": False,
    },
]


def check_backend_health():
    """Quick check that the backend is reachable."""
    import urllib.request
    try:
        req = urllib.request.urlopen(f"{HOST}/api/v1/health", timeout=5)
        data = json.loads(req.read())
        if data.get("engine_loaded"):
            return True
        print("⚠  Backend is up but engine not loaded yet.")
        return False
    except Exception as e:
        print(f"FAILED: Backend not reachable at {HOST}: {e}")
        return False


def parse_csv_stats(csv_path):
    """Parse the Locust stats CSV and return a summary dict."""
    summary = {
        "total_requests": 0,
        "total_failures": 0,
        "error_rate_pct": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "peak_rps": 0.0,
        "endpoints": [],
    }

    if not csv_path.exists():
        return summary

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        name = row.get("Name", "")
        if name == "Aggregated":
            summary["total_requests"] = int(row.get("Request Count", 0))
            summary["total_failures"] = int(row.get("Failure Count", 0))
            total = summary["total_requests"] or 1
            summary["error_rate_pct"] = round(
                summary["total_failures"] / total * 100, 2
            )
            summary["avg_latency_ms"] = float(row.get("Average Response Time", 0))
            summary["p95_latency_ms"] = float(row.get("95%", 0))
            summary["peak_rps"] = float(row.get("Requests/s", 0))
        else:
            summary["endpoints"].append(
                {
                    "name": name,
                    "method": row.get("Type", ""),
                    "requests": int(row.get("Request Count", 0)),
                    "failures": int(row.get("Failure Count", 0)),
                    "avg_ms": float(row.get("Average Response Time", 0)),
                    "p95_ms": float(row.get("95%", 0)),
                    "rps": float(row.get("Requests/s", 0)),
                }
            )

    return summary


def print_summary(phase_name, summary):
    """Pretty-print a phase summary."""
    err = summary["error_rate_pct"]
    p95 = summary["p95_latency_ms"]

    err_flag = " RED FAIL" if err > 1 else " OK"
    p95_flag = " RED FAIL" if p95 > 200 else " OK"

    print(f"\n{'='*60}")
    print(f"  STATS  {phase_name.upper()} PHASE RESULTS")
    print(f"{'='*60}")
    print(f"  Total Requests : {summary['total_requests']}")
    print(f"  Failures       : {summary['total_failures']}")
    print(f"  Error Rate     : {err:.2f}%{err_flag}")
    print(f"  Avg Latency    : {summary['avg_latency_ms']:.0f} ms")
    print(f"  P95 Latency    : {p95:.0f} ms{p95_flag}")
    print(f"  Peak RPS       : {summary['peak_rps']:.1f}")
    print()

    if summary["endpoints"]:
        print(f"  {'Endpoint':<30} {'Reqs':>6} {'Fail':>5} {'Avg':>7} {'P95':>7} {'RPS':>7}")
        print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
        for ep in summary["endpoints"]:
            print(
                f"  {ep['name']:<30} {ep['requests']:>6} {ep['failures']:>5} "
                f"{ep['avg_ms']:>6.0f}ms {ep['p95_ms']:>6.0f}ms {ep['rps']:>6.1f}"
            )

    print(f"{'='*60}\n")

    issues = []
    if err > 1:
        issues.append(f"Error rate {err:.2f}% exceeds 1% threshold")
    if p95 > 200:
        issues.append(f"P95 latency {p95:.0f}ms exceeds 200ms threshold")

    return issues


def run_phase(phase):
    """Run a single Locust phase headlessly via CLI."""
    name = phase["name"]
    html_report = REPORTS_DIR / f"{name}_report.html"
    csv_prefix = str(REPORTS_DIR / name)

    # Remove stale CSVs
    for suffix in ["_stats.csv", "_failures.csv", "_stats_history.csv", "_exceptions.csv"]:
        p = Path(csv_prefix + suffix)
        if p.exists():
            p.unlink()

    cmd = [
        sys.executable, "-m", "locust",
        "-f", LOCUSTFILE,
        "--host", HOST,
        "--headless",
        "-u", str(phase["users"]),
        "-r", str(phase["spawn_rate"]),
        "-t", phase["run_time"],
        "--html", str(html_report),
        "--csv", csv_prefix,
        "--only-summary",
    ]

    if phase.get("tags"):
        for tag in phase["tags"]:
            cmd.extend(["--tags", tag])

    print(f"\nSTARTING {name.upper()} phase: {phase['users']} users, "
          f"spawn {phase['spawn_rate']}/s, run {phase['run_time']}")
    print(f"    Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    # Parse results
    csv_path = Path(csv_prefix + "_stats.csv")
    summary = parse_csv_stats(csv_path)
    issues = print_summary(name, summary)

    return {
        "name": name,
        "exit_code": result.returncode,
        "summary": summary,
        "issues": issues,
        "html_report": str(html_report),
    }


def main():
    print("=" * 60)
    print("  FraudShield AI - Load Test Suite")
    print("=" * 60)

    # 1. Pre-flight check
    print("\nCHECKING backend health...")
    if not check_backend_health():
        print("\nFAILED: Cannot proceed. Start the backend first:")
        print("  cd backend && venv\\Scripts\\uvicorn api.main:app --reload")
        sys.exit(1)
    print("SUCCESS: Backend is healthy and engine is loaded.\n")

    all_results = []
    all_issues = {}

    for phase in PHASES:
        result = run_phase(phase)
        all_results.append(result)

        if result["issues"]:
            all_issues[phase["name"]] = result["issues"]

        # Stop-on-fail for smoke
        if phase.get("stop_on_fail") and (result["exit_code"] != 0 or result["issues"]):
            print("STOPPED: SMOKE TEST FAILED - stopping pipeline.")
            print(f"   Issues: {result['issues']}")
            print(f"   Check report: {result['html_report']}")
            sys.exit(1)

        # Brief pause between phases
        if phase != PHASES[-1]:
            print("WAITING: Cooling down 5s before next phase...\n")
            time.sleep(5)

    # ─── Final Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY - All Phases")
    print("=" * 60)

    for r in all_results:
        s = r["summary"]
        status = "OK PASS" if not r["issues"] else "RED ISSUES"
        print(
            f"  {r['name'].upper():<8} {status}  |  "
            f"Reqs: {s['total_requests']:>5}  "
            f"Err: {s['error_rate_pct']:.1f}%  "
            f"P95: {s['p95_latency_ms']:.0f}ms  "
            f"RPS: {s['peak_rps']:.1f}"
        )

    if all_issues:
        print(f"\n⚠  Issues detected in {len(all_issues)} phase(s):")
        for phase_name, issues in all_issues.items():
            for issue in issues:
                print(f"   [{phase_name.upper()}] {issue}")
        print("\n💡  Recommendation: review the HTML reports in reports/ folder.")
    else:
        print("\n✅  All phases passed! Safe to proceed to cloud deployment.")

    print(f"\n📁  Reports saved to: {REPORTS_DIR.resolve()}")
    print()


if __name__ == "__main__":
    main()
