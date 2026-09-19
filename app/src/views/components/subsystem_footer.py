from models.subsystem import SUBSYSTEMS
from state import runs as runs_state
from views.helpers import render as render_template

def render():
    covered = {r.subsystem_key for r in runs_state.all_runs()}
    statuses = {s.key: ("done" if s.key in covered else "pending") for s in SUBSYSTEMS}
    render_template(
        "subsystem_footer",
        door_status=statuses["door"],
        rail_status=statuses["rail_corrugation"],
        shm_status=statuses["shm"],
    )