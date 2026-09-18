import streamlit as st
_RUNS_KEY = "runs"

def init():
    st.session_state.setdefault(_RUNS_KEY, [])

def all_runs():
    return st.session_state[_RUNS_KEY]

def add(run):
    st.session_state[_RUNS_KEY].append(run)

def current():
    runs = all_runs()
    return runs[-1] if runs else None

def recent(limit=10):
    return list(reversed(all_runs()[-limit:]))

def stats():
    runs = all_runs()
    total_runs = len(runs)
    total_rows = sum(r.num_rows for r in runs)
    subsystems_used = len({r.subsystem_key for r in runs})
    return {"total_runs": total_runs, "total_rows": total_rows, "subsystems_used": subsystems_used}