import streamlit as st
from models.subsystem import SUBSYSTEMS

def render():
    with st.container(key="subsystem_square"):
        st.markdown("#### Subsystem")

        options = {s.label: s for s in SUBSYSTEMS}
        label = st.selectbox("select subsystem", list(options.keys()), label_visibility="collapsed")
        subsystem = options[label]
        st.caption(subsystem.task)

        uploaded_files = st.file_uploader(
            "drop sensor data here",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("csv or xlsx, one or more files")

        return subsystem, uploaded_files