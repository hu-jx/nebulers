import streamlit as st
from state import runs as runs_state
from views.helpers import render as render_template

def render(limit=10):
    st.markdown("#### Run History")

    recent = runs_state.recent(limit=limit)
    if not recent:
        render_template("empty_card", message="no runs yet")
        return

    for i, run in enumerate(recent):
        _render_row(i, run)

def _render_row(index, run):
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 1.5])
    with c1:
        st.markdown(f"<span class='badge-subsystem'>{run.subsystem_label}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='activity-name'>{run.filename[:28]}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"{run.num_rows} rows")
    with c4:
        st.markdown(f"<div class='activity-time'>{run.timestamp}</div>", unsafe_allow_html=True)
    with c5:
        st.download_button(
            "csv",
            data=run.csv_bytes,
            file_name=f"{run.subsystem_key}_predictions.csv",
            mime="text/csv",
            key=f"dl_{index}",
            use_container_width=True,
        )