#code for app here
import streamlit as st

from state import runs as runs_state
from views.helpers import load_css, spacer
from views.pages import main_page


st.set_page_config(
    page_title="nebulers",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
runs_state.init()


with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">NebulaX</div>',
        unsafe_allow_html=True,
    )

    spacer("sm")
    st.caption("problem statement 3!")


title_col, repo_col = st.columns([3, 1])

with title_col:
    st.markdown(
        """
        <div class="page-title">nebulers</div>
        <div class="page-sub">train the trains.. badumtss</div>
        """,
        unsafe_allow_html=True,
    )

with repo_col:
    spacer("sm")
    st.link_button(
        "view repository",
        "https://github.com/hu-jx/nebulers",
        use_container_width=True,
    )


spacer("md")

main_page.render()