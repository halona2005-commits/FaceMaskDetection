"""
Run entry point for Face Mask Detection with Live Alert System

Menu options:
1) Run Real-Time Detection
2) Run Optimized Detection
3) Open Web Dashboard
4) View Session Logs
5) View Violation Snapshots
6) Exit

This script imports and calls into the project's modules where possible
and uses subprocess for components that run standalone (Flask dashboard).
"""

import sys
import subprocess
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_realtime():
    try:
        from day9_realtime import main as realtime_main
        realtime_main()
    except Exception as e:
        print("Error running real-time detection:")
        traceback.print_exception(e, e, e.__traceback__)


def run_optimized():
    try:
        import day14_optimized
        day14_optimized.main()
    except Exception as e:
        print("Error running optimized detection:")
        traceback.print_exception(e, e, e.__traceback__)


def open_dashboard():
    try:
        python = sys.executable
        script = PROJECT_ROOT / 'day13_dashboard.py'
        if not script.exists():
            print(f"Dashboard script not found: {script}")
            return

        print("Starting web dashboard (will block this menu until stopped).")
        proc = subprocess.Popen([python, str(script)], cwd=str(PROJECT_ROOT))
        print("Dashboard started. Access it at http://0.0.0.0:5000/ in your browser.")
        print("Press Ctrl+C to stop the dashboard and return to this menu.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Stopping dashboard...")
            proc.terminate()
            proc.wait(timeout=5)

    except Exception as e:
        print("Error launching dashboard:")
        traceback.print_exception(e, e, e.__traceback__)


def view_session_logs():
    try:
        log_file = PROJECT_ROOT / 'dashboard_log.csv'
        if not log_file.exists():
            print("No session log found.")
            return

        print(f"Showing last 50 lines from {log_file}")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[-50:]:
            print(line.rstrip())

    except Exception as e:
        print("Error reading session logs:")
        traceback.print_exception(e, e, e.__traceback__)


def view_violation_snapshots():
    try:
        snaps_dir = PROJECT_ROOT / 'alerts' / 'snapshots'
        if not snaps_dir.exists() or not snaps_dir.is_dir():
            print("No snapshots directory found.")
            return

        files = sorted(list(snaps_dir.glob('*')), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print("No snapshot files found.")
            return

        print(f"Recent snapshots in {snaps_dir}:")
        for p in files[:50]:
            print(f"- {p.name}  ({p.stat().st_size} bytes)")

    except Exception as e:
        print("Error listing snapshots:")
        traceback.print_exception(e, e, e.__traceback__)


def main():
    while True:
        print('\nFace Mask Detection - Main Menu')
        print('1) Run Real-Time Detection')
        print('2) Run Optimized Detection')
        print('3) Open Web Dashboard')
        print('4) View Session Logs')
        print('5) View Violation Snapshots')
        print('6) Exit')

        choice = input('Select an option [1-6]: ').strip()
        if choice == '1':
            run_realtime()
        elif choice == '2':
            run_optimized()
        elif choice == '3':
            open_dashboard()
        elif choice == '4':
            view_session_logs()
        elif choice == '5':
            view_violation_snapshots()
        elif choice == '6':
            print('Exiting. Goodbye!')
            return
        else:
            print('Invalid choice, please select a number between 1 and 6.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Fatal error:')
        traceback.print_exception(e, e, e.__traceback__)
        sys.exit(1)
