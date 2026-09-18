import streamlit as st
from views.components import subsystem_panel, results_panel, summary_strip, run_history
from views.helpers import spacer

def render():
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        subsystem, uploaded_file = subsystem_panel.render()
    with right_col:
        results_panel.render(subsystem, uploaded_file)

    spacer("md")
    summary_strip.render()

    spacer("md")
    run_history.render()