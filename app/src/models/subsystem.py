from dataclasses import dataclass

@dataclass
class Subsystem:
    key: str
    label: str
    task: str
    description: str

SUBSYSTEMS = [
    Subsystem(
        key="door",
        label="Door",
        task="binary classification",
        description="flag abnormal-resistance door-open/close cycles from door sensor data",
    ),
    Subsystem(
        key="rail_corrugation",
        label="Rail Corrugation",
        task="multi-class classification",
        description="distinguish Normal vs. Side I vs. Side II rail corrugation from multi-channel axle-box vibration and shock data",
    ),
    Subsystem(
        key="shm",
        label="SHM",
        task="regression",
        description="estimate cumulative fatigue damage from dynamic stress time series",
    ),
]