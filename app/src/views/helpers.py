from pathlib import Path
from string import Template
from textwrap import dedent
import streamlit as st
_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = _ROOT / "templates"


def load_css(filename="styles.css"):
    css = (_ROOT / filename).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render(template_name, **fields):
    path = _TEMPLATES / f"{template_name}.html"
    raw = dedent(path.read_text()).strip()
    html = Template(raw).safe_substitute(**fields)
    st.markdown(html, unsafe_allow_html=True)

def spacer(size="md"):
    st.markdown(f"<div class='spacer-{size}'></div>", unsafe_allow_html=True)