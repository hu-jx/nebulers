import re

import numpy as np
import pandas as pd


FS = 10000.0
SPEED_SENSOR_TEETH = 90
WHEEL_DIAMETER_M = 0.85
TRANSITIONS_PER_REV = SPEED_SENSOR_TEETH * 2

FREQUENCY_BANDS = [
    (0, 25),
    (25, 50),
    (50, 75),
    (75, 100),
    (100, 150),
    (150, 250),
    (250, 500),
    (500, 1000),
    (1000, 2500),
    (2500, 5000),
]

PER_CAR_BANDS = [
    (50, 75),
    (75, 100),
]

WAVELENGTH_BANDS_M = [
    (0.02, 0.04),
    (0.04, 0.06),
    (0.06, 0.10),
    (0.10, 0.15),
    (0.15, 0.25),
    (0.25, 0.40),
    (0.40, 0.60),
]

MIN_CORRUGATION_WAVELENGTH_M = 0.02
MAX_CORRUGATION_WAVELENGTH_M = 0.60
EPS = 1e-12


# reads the sensor type, position, car and side from a column name
def parse_column(column):
    match = re.match(
        r"(Vibration|Shock) of bearing in position (\d+) of car (\d+)",
        column,
    )

    if match is None:
        return None

    sensor_type = match.group(1)
    position = int(match.group(2))
    car = int(match.group(3))
    side = "I" if position % 2 == 1 else "II"

    return sensor_type, position, car, side


# turns the digital rotating-speed signal into wheel speed in m/s
def calculate_train_speed(speed_signal):
    binary = (np.asarray(speed_signal) >= 0.5).astype(int)
    transition_count = int(np.sum(np.diff(binary) != 0))
    duration = len(binary) / FS

    if transition_count == 0:
        return 0.0

    rotation_hz = transition_count / (TRANSITIONS_PER_REV * duration)
    return rotation_hz * np.pi * WHEEL_DIAMETER_M


# returns the mean power spectrum across a group of sensors
def get_spectrum(dataframe, columns):
    values = dataframe[columns].to_numpy(dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)

    fft_values = np.fft.rfft(values, axis=0)
    power = np.abs(fft_values) ** 2
    mean_power = power.mean(axis=1)

    frequencies = np.fft.rfftfreq(values.shape[0], d=1.0 / FS)

    return frequencies, mean_power


# calculates rms across all samples and sensors in a group
def get_rms(dataframe, columns):
    values = dataframe[columns].to_numpy(dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(values ** 2)))


# sums spectral power inside one frequency band
def band_energy(frequencies, power, low_hz, high_hz):
    if high_hz >= FS / 2:
        mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    else:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)

    return float(np.sum(power[mask]))


# gives the common text label used for a frequency band
def band_name(low_hz, high_hz):
    return f"{int(low_hz)}_{int(high_hz)}"


# gives the common text label used for a wavelength band
def wavelength_name(low_m, high_m):
    return f"{int(round(low_m * 100))}_{int(round(high_m * 100))}cm"


# creates the speed-dependent frequency mask used by the diagnostic features
def corrugation_search_mask(frequencies, speed_mps):
    if speed_mps > 0:
        low_hz = speed_mps / MAX_CORRUGATION_WAVELENGTH_M
        high_hz = speed_mps / MIN_CORRUGATION_WAVELENGTH_M
    else:
        low_hz = 20.0
        high_hz = 1000.0

    low_hz = max(low_hz, 5.0)
    high_hz = min(high_hz, FS / 2.0)

    return (frequencies >= low_hz) & (frequencies <= high_hz)


# gets peak concentration and spectral entropy inside the corrugation search range
def spectral_shape_features(power, mask):
    band_power = power[mask]

    if len(band_power) < 5:
        return 0.0, 0.0

    total_power = float(np.sum(band_power))

    if total_power <= EPS:
        return 0.0, 0.0

    peak_concentration = float(np.max(band_power) / (total_power + EPS))

    probabilities = band_power / (total_power + EPS)
    spectral_entropy = float(
        -np.sum(probabilities * np.log(probabilities + EPS))
        / np.log(len(probabilities))
    )

    return peak_concentration, spectral_entropy


# measures how much peak strength varies across the eight cars for one side
def car_peak_logpower_std(
    dataframe,
    grouped_columns,
    side,
    frequencies,
    mask,
    global_peak_frequency,
):
    if not np.isfinite(global_peak_frequency):
        return 0.0

    global_peak_index = int(
        np.argmin(np.abs(frequencies - global_peak_frequency))
    )

    car_peak_powers = []

    for car in range(1, 9):
        columns = grouped_columns[("Vibration", car, side)]
        car_frequencies, car_power = get_spectrum(dataframe, columns)

        if len(car_power[mask]) == 0:
            continue

        peak_index = int(
            np.argmin(np.abs(car_frequencies - frequencies[global_peak_index]))
        )
        car_peak_powers.append(car_power[peak_index])

    if len(car_peak_powers) == 0:
        return 0.0

    return float(np.std(np.log1p(np.asarray(car_peak_powers))))


# calculates per-car side i/ii asymmetry inside the speed-dependent search range
def dynamic_car_asymmetry(dataframe, grouped_columns, mask):
    log_ratios = []

    for car in range(1, 9):
        _, power_i = get_spectrum(
            dataframe,
            grouped_columns[("Vibration", car, "I")],
        )
        _, power_ii = get_spectrum(
            dataframe,
            grouped_columns[("Vibration", car, "II")],
        )

        energy_i = float(np.sum(power_i[mask]))
        energy_ii = float(np.sum(power_ii[mask]))

        log_ratios.append(
            np.log((energy_i + EPS) / (energy_ii + EPS))
        )

    log_ratios = np.asarray(log_ratios)

    return {
        "vib_dynamic_car_logratio_min": float(np.min(log_ratios)),
        "vib_dynamic_car_logratio_max": float(np.max(log_ratios)),
        "vib_dynamic_car_logratio_std": float(np.std(log_ratios)),
    }


# extracts the same 94 features used to train the saved model
def extract_features(dataframe, source_name="recording"):
    dataframe = dataframe.copy()

    if "source_file" in dataframe.columns:
        dataframe = dataframe.drop(columns=["source_file"])

    if "Rotating speed" not in dataframe.columns:
        raise ValueError(f"{source_name}: missing Rotating speed column")

    grouped_columns = {
        (sensor_type, car, side): []
        for sensor_type in ["Vibration", "Shock"]
        for car in range(1, 9)
        for side in ["I", "II"]
    }

    side_columns = {
        (sensor_type, side): []
        for sensor_type in ["Vibration", "Shock"]
        for side in ["I", "II"]
    }

    for column in dataframe.columns:
        info = parse_column(column)

        if info is None:
            continue

        sensor_type, _, car, side = info
        grouped_columns[(sensor_type, car, side)].append(column)
        side_columns[(sensor_type, side)].append(column)

    for key, columns in grouped_columns.items():
        if len(columns) != 4:
            raise ValueError(
                f"{source_name}: expected 4 channels for {key}, found {len(columns)}"
            )

    speed_mps = calculate_train_speed(
        dataframe["Rotating speed"].to_numpy()
    )

    features = {}

    # keeps overall vibration/shock level as context without using a hard amplitude threshold
    for short_name, sensor_type in [
        ("vib", "Vibration"),
        ("shock", "Shock"),
    ]:
        rms_i = get_rms(dataframe, side_columns[(sensor_type, "I")])
        rms_ii = get_rms(dataframe, side_columns[(sensor_type, "II")])

        features[f"{short_name}_rms_I"] = rms_i
        features[f"{short_name}_rms_II"] = rms_ii
        features[f"{short_name}_rms_ratio"] = (
            (rms_i + EPS) / (rms_ii + EPS)
        )

    # whole-side vibration spectra are reused for the fixed frequency and wavelength features
    vib_freq_i, vib_power_i = get_spectrum(
        dataframe,
        side_columns[("Vibration", "I")],
    )
    vib_freq_ii, vib_power_ii = get_spectrum(
        dataframe,
        side_columns[("Vibration", "II")],
    )

    for low_hz, high_hz in FREQUENCY_BANDS:
        name = band_name(low_hz, high_hz)

        energy_i = band_energy(
            vib_freq_i,
            vib_power_i,
            low_hz,
            high_hz,
        )
        energy_ii = band_energy(
            vib_freq_ii,
            vib_power_ii,
            low_hz,
            high_hz,
        )

        log_i = np.log1p(energy_i)
        log_ii = np.log1p(energy_ii)

        features[f"vib_energy_I_{name}"] = float(log_i)
        features[f"vib_energy_II_{name}"] = float(log_ii)
        features[f"vib_ratio_{name}"] = float(
            (energy_i + EPS) / (energy_ii + EPS)
        )
        features[f"vib_diff_{name}"] = float(log_i - log_ii)

    # per-car ratios keep local side differences in the two useful frequency bands
    for low_hz, high_hz in PER_CAR_BANDS:
        name = band_name(low_hz, high_hz)
        car_log_ratios = []

        for car in range(1, 9):
            frequencies_i, power_i = get_spectrum(
                dataframe,
                grouped_columns[("Vibration", car, "I")],
            )
            frequencies_ii, power_ii = get_spectrum(
                dataframe,
                grouped_columns[("Vibration", car, "II")],
            )

            energy_i = band_energy(
                frequencies_i,
                power_i,
                low_hz,
                high_hz,
            )
            energy_ii = band_energy(
                frequencies_ii,
                power_ii,
                low_hz,
                high_hz,
            )

            car_log_ratios.append(
                np.log((energy_i + EPS) / (energy_ii + EPS))
            )

        car_log_ratios = np.asarray(car_log_ratios)

        features[f"car_log_ratio_max_{name}"] = float(
            np.max(car_log_ratios)
        )
        features[f"car_log_ratio_min_{name}"] = float(
            np.min(car_log_ratios)
        )
        features[f"car_log_ratio_median_{name}"] = float(
            np.median(car_log_ratios)
        )
        features[f"car_log_ratio_std_{name}"] = float(
            np.std(car_log_ratios)
        )

    # wavelength features shift the spectrum into a speed-normalised physical scale
    positive_mask = vib_freq_i > 0
    positive_frequencies = vib_freq_i[positive_mask]
    positive_power_i = vib_power_i[positive_mask]
    positive_power_ii = vib_power_ii[positive_mask]

    wavelengths = (
        speed_mps / positive_frequencies
        if speed_mps > 0
        else None
    )

    for low_m, high_m in WAVELENGTH_BANDS_M:
        name = wavelength_name(low_m, high_m)

        if wavelengths is None:
            energy_i = 0.0
            energy_ii = 0.0
        else:
            mask = (
                (wavelengths >= low_m)
                & (wavelengths < high_m)
            )

            energy_i = float(np.sum(positive_power_i[mask]))
            energy_ii = float(np.sum(positive_power_ii[mask]))

        log_i = np.log1p(energy_i)
        log_ii = np.log1p(energy_ii)

        features[f"wl_energy_I_{name}"] = float(log_i)
        features[f"wl_energy_II_{name}"] = float(log_ii)
        features[f"wl_ratio_{name}"] = float(
            (energy_i + EPS) / (energy_ii + EPS)
            if speed_mps > 0
            else 1.0
        )
        features[f"wl_diff_{name}"] = float(
            log_i - log_ii
            if speed_mps > 0
            else 0.0
        )

    # adds the 12 diagnostic features that improved the final cv score
    vib_mask = corrugation_search_mask(
        vib_freq_i,
        speed_mps,
    )

    vib_peak_i, vib_entropy_i = spectral_shape_features(
        vib_power_i,
        vib_mask,
    )
    vib_peak_ii, vib_entropy_ii = spectral_shape_features(
        vib_power_ii,
        vib_mask,
    )

    features["vib_I_peak_concentration"] = vib_peak_i
    features["vib_II_peak_concentration"] = vib_peak_ii
    features["vib_peak_concentration_logratio_I_II"] = float(
        np.log((vib_peak_i + EPS) / (vib_peak_ii + EPS))
    )

    features["vib_I_spectral_entropy"] = vib_entropy_i
    features["vib_II_spectral_entropy"] = vib_entropy_ii
    features["vib_entropy_diff_I_II"] = float(
        vib_entropy_i - vib_entropy_ii
    )

    shock_freq_i, shock_power_i = get_spectrum(
        dataframe,
        side_columns[("Shock", "I")],
    )
    shock_freq_ii, shock_power_ii = get_spectrum(
        dataframe,
        side_columns[("Shock", "II")],
    )

    shock_mask_i = corrugation_search_mask(
        shock_freq_i,
        speed_mps,
    )
    shock_mask_ii = corrugation_search_mask(
        shock_freq_ii,
        speed_mps,
    )

    _, shock_entropy_i = spectral_shape_features(
        shock_power_i,
        shock_mask_i,
    )
    _, shock_entropy_ii = spectral_shape_features(
        shock_power_ii,
        shock_mask_ii,
    )

    features["shock_entropy_diff_I_II"] = float(
        shock_entropy_i - shock_entropy_ii
    )

    band_power_i = vib_power_i[vib_mask]
    band_power_ii = vib_power_ii[vib_mask]
    band_frequencies = vib_freq_i[vib_mask]

    if len(band_power_i) > 0:
        peak_frequency_i = float(
            band_frequencies[int(np.argmax(band_power_i))]
        )
        peak_frequency_ii = float(
            band_frequencies[int(np.argmax(band_power_ii))]
        )
    else:
        peak_frequency_i = np.nan
        peak_frequency_ii = np.nan

    features["vib_I_car_peak_logpower_std"] = (
        car_peak_logpower_std(
            dataframe,
            grouped_columns,
            "I",
            vib_freq_i,
            vib_mask,
            peak_frequency_i,
        )
    )

    features["vib_II_car_peak_logpower_std"] = (
        car_peak_logpower_std(
            dataframe,
            grouped_columns,
            "II",
            vib_freq_ii,
            vib_mask,
            peak_frequency_ii,
        )
    )

    features.update(
        dynamic_car_asymmetry(
            dataframe,
            grouped_columns,
            vib_mask,
        )
    )

    return features


class RailCorrugationModel:
    # wraps the saved model bundle so the rest of the app can call .predict()
    def __init__(self, bundle):
        required = {
            "svm",
            "xgb",
            "meta_model",
            "feature_columns",
            "labels",
        }
        missing = required - set(bundle)

        if missing:
            raise ValueError(
                f"rail model bundle is missing: {sorted(missing)}"
            )

        self.svm = bundle["svm"]
        self.xgb = bundle["xgb"]
        self.meta_model = bundle["meta_model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.labels = bundle["labels"]

    # predicts one label per uploaded recording and returns submission-ready rows
    def predict(self, dataframe):
        if "source_file" not in dataframe.columns:
            raise ValueError("rail input is missing source_file")

        feature_rows = []
        file_ids = []

        for source_file, group in dataframe.groupby(
            "source_file",
            sort=False,
        ):
            raw_data = group.drop(
                columns=["source_file"]
            ).reset_index(drop=True)

            features = extract_features(
                raw_data,
                source_name=source_file,
            )

            feature_rows.append(features)
            file_ids.append(source_file)

        feature_df = pd.DataFrame(feature_rows)
        feature_df = feature_df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)

        missing_features = [
            column
            for column in self.feature_columns
            if column not in feature_df.columns
        ]

        if missing_features:
            raise ValueError(
                f"missing model features: {missing_features}"
            )

        X = feature_df[self.feature_columns]

        meta_features = np.hstack([
            self.svm.predict_proba(X),
            self.xgb.predict_proba(X),
        ])

        prediction_ids = self.meta_model.predict(
            meta_features
        )

        predictions = [
            self._label_for(int(value))
            for value in prediction_ids
        ]

        return pd.DataFrame({
            "file_id": file_ids,
            "prediction": predictions,
        })

    # handles either dict or list label mappings from the saved bundle
    def _label_for(self, value):
        if isinstance(self.labels, dict):
            if value in self.labels:
                return self.labels[value]
            return self.labels[str(value)]

        return self.labels[value]
