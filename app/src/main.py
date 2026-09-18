#code for app here
import streamlit as st
from state import runs as runs_state, page as page_state
from views.helpers import load_css, spacer
from views.components import subsystem_footer
from views.pages import main_page, subsystems_page, about_page


st.set_page_config(
    page_title="nebulers",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

runs_state.init()
page_state.init()

PAGES = {
    "main":       ("main",       main_page),
    "subsystems": ("subsystems", subsystems_page),
    "about":      ("about",      about_page),
}

with st.sidebar:
    st.markdown('<div class="sidebar-logo">PS3</div>', unsafe_allow_html=True)
    for key, (label, _) in PAGES.items():
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            page_state.switch_to(key)
    spacer("sm")
    st.caption("<caption placeholder>")

title_col, repo_col = st.columns([3, 1])
with title_col:
    st.markdown('<div class="page-title">title placeholder</div>', unsafe_allow_html=True)
with repo_col:
    spacer("sm")
    st.link_button("view repo", "https://github.com/hu-jx/nebulers", use_container_width=True)

spacer("md")

PAGES[page_state.current()][1].render()

subsystem_footer.render()