import streamlit as st
from controllers import run_controller
from state import runs as runs_state
from views.helpers import render as render_template

def render(subsystem, uploaded_files):
    with st.container(key="results_square"):
        st.markdown("#### Predictions")

        if not uploaded_files:
            render_template("empty_message", message="upload a dataset to run predictions")
            return

        if st.button("run predictions", use_container_width=True):
            run_controller.execute(subsystem, uploaded_files)

        run = runs_state.current()
        if run is None:
            render_template("empty_message", message="click run predictions to see results")
            return

        st.dataframe(run.preview_rows, use_container_width=True, hide_index=True)
        st.download_button(
            "download predictions csv",
            data=run.csv_bytes,
            file_name=f"{run.subsystem_key}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )