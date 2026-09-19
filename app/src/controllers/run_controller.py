from datetime import datetime
from models.run import Run
from services import data_loader, export, model_loader
from state import runs as runs_state


def execute(subsystem, uploaded_files):
    dataframe = data_loader.load(uploaded_files)
    model = model_loader.load(subsystem)
    predictions_df, csv_bytes = export.save_predictions(model, dataframe)
    filename = uploaded_files[0].name if len(uploaded_files) == 1 else f"{len(uploaded_files)} files"

    run = Run(
        subsystem_key=subsystem.key,
        subsystem_label=subsystem.label,
        filename=filename,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_rows=len(dataframe),
        csv_bytes=csv_bytes,
        preview_rows=predictions_df.head(5).to_dict(orient="records"),
    )

    runs_state.add(run)