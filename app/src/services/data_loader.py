import io

import pandas as pd


# reads one uploaded file using the format expected by the selected subsystem
def _read_one(uploaded_file, subsystem):
    raw = io.BytesIO(
        uploaded_file.getvalue()
    )

    is_shm = (
        subsystem.key == "shm"
    )

    header = (
        None
        if is_shm
        else 0
    )

    if uploaded_file.name.lower().endswith(
        ".xlsx"
    ):
        return pd.read_excel(
            raw,
            header=header,
        )

    return pd.read_csv(
        raw,
        header=header,
    )


# combines uploaded files while keeping the original filename for per-file prediction
def load(uploaded_files, subsystem):
    frames = []

    for uploaded_file in uploaded_files:
        frame = _read_one(
            uploaded_file,
            subsystem,
        )

        frame.insert(
            0,
            "source_file",
            uploaded_file.name,
        )

        frames.append(
            frame
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )
