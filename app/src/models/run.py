from dataclasses import dataclass

@dataclass
class Run:
    subsystem_key: str
    subsystem_label: str
    filename: str
    timestamp: str
    num_rows: int
    csv_bytes: bytes
    preview_rows: list