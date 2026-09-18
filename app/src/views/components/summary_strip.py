import streamlit as st
from models.subsystem import SUBSYSTEMS
from state import runs as runs_state
from views.helpers import render as render_template

def render():
    st.markdown("#### Summary")

    stats = runs_state.stats()

    with st.container(key="summary_bar"):
        c1, c2, c3 = st.columns(3)
        with c1:
            render_template("bar_stat", label="total runs", value=stats["total_runs"], accent_class="")
        with c2:
            render_template("bar_stat", label="rows processed", value=stats["total_rows"], accent_class="lavender")
        with c3:
            render_template(
                "bar_stat",
                label="subsystems covered",
                value=f"{stats['subsystems_used']}/{len(SUBSYSTEMS)}",
                accent_class="orange",
            )