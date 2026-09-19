NORMAL_LABEL = "Normal"
ABNORMAL_LABEL = "Abnormal resistance"
NUMBER_PHASE_BINS = 5
SIGNAL_COLUMNS = {
    "current": "Motor current(mA)",
    "voltage": "Motor Voltage(10mV)",
    "back_emf": "Motor electrodynamic force",
    "position": "Door leaf position",
}
DOOR_MODEL_PATH = "nebulers/app/models/door_model.joblib"
PREDICTED_TEST_FOR_DOOR = "nebulers/Optional_Items/Door/datasets/door_predictions.csv"