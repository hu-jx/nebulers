import streamlit as st
from models.subsystem import SUBSYSTEMS
from views.helpers import render as render_template

def render():
    st.markdown("#### Subsystems")
    cols = st.columns(2)
    for i, subsystem in enumerate(SUBSYSTEMS):
        with cols[i % 2]:
            render_template(
                "subsystem_card",
                label=subsystem.label,
                task=subsystem.task,
                description=subsystem.description,
            )