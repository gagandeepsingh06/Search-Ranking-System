"""
run.py
------
Starts both FastAPI backend and React Vite frontend
at the same time with a single command.

Run with:
    python run.py
"""

import subprocess
import sys
import os


def run():

    print("\n" + "="*50)
    print("  STARTING AI SEARCH RANKING SYSTEM")
    print("="*50)

    # Start FastAPI backend
    print("\n Starting FastAPI backend on http://localhost:8000")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.search_api:app", "--reload", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Start React Vite frontend
    print(" Starting React Vite frontend on http://localhost:5173")
    frontend_cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    
    # On Windows, we use shell=True to run npm command scripts reliably
    react_process = subprocess.Popen(
        "npm run dev",
        cwd=frontend_cwd,
        shell=True
      )

    print("\n" + "="*50)
    print("  BOTH SERVERS RUNNING")
    print("  API:      http://localhost:8000")
    print("  Frontend: http://localhost:5173")
    print("  Press Ctrl+C to stop both")
    print("="*50 + "\n")

    try:
        # Keep running until Ctrl+C
        api_process.wait()
        react_process.wait()

    except KeyboardInterrupt:
        print("\n\nStopping servers...")
        api_process.terminate()
        # On Windows Popen with shell=True creates child processes; terminating the parent shell doesn't kill children directly,
        # but using taskkill or standard terminate handles it well or closing shell is sufficient for local development.
        react_process.terminate()
        print("Both servers stopped.")


if __name__ == "__main__":
    run()