import io
import pandas as pd

def _read_one(uploaded_file):
    if uploaded_file.name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))

def load(uploaded_files):
    frames = []
    for f in uploaded_files:
        frame = _read_one(f)
        frame.insert(0, "source_file", f.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)