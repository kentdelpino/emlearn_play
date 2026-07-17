
# - ----------------------------- - #
# - emlearn play by Kent del Pino - #
# - ----------------------------- - #

# This forces setuptools to boot up and map distutils
import setuptools
setuptools.setup
import os

print('0. Import needed tools')

import numpy as np
import pandas as pd
import emlearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

def pause():
    programPause = input("Press the <ENTER> key to continue...")

# Point to Datasets, the pEMP CSVs
data_path = "./SOON-pEMP/"
# List of sets to be used
file_mapping = {
    1: data_path + "01 - m1_half_shaft_speed_no_mechanical_load.csv",  # Normal operation
    2: data_path + "02 - m1_load_0.5Nm_half_speed.csv",  # Normal with load
    3: data_path + "03 - m1_mechanically_imbalanced_half_speed.csv",  # Mechanical Fault 1
    4: data_path + "04 - m1_mechanically_imbalanced_half_speed.csv",  # Mechanical Fault 2
    5: data_path + "05 - m1_mechanically_imbalanced_load_0.5Nm_half_speed.csv",  # Mechanical Fault 3
}

# We want to work in Int32 (native word size), the HW-ADC or SPI-Acc sensor is ~14bit 
ADC_MAX = 16383  # up-to 14 significant bits (Note: here unsigned)
WINDOW_SIZE = 32 

print("1. Loading datasets and splitting CHRONOLOGICALLY before windowing...")
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for scenario_id, filename in file_mapping.items():
    if not os.path.exists(filename):
        continue

    df = pd.read_csv(filename)
    vibration_channels = ["AccX", "AccY", "AccZ"]
    df_cleaned = df[vibration_channels].dropna()
    X_scenario = df_cleaned.to_numpy(dtype=np.float32)
    
    # Define targets
    target_class = 0 if scenario_id in [1, 2] else 1

    # --- CRITICAL FIX: Split raw data chronologically (70% Train, 30% Test) ---
    split_idx = int(len(X_scenario) * 0.7)
    X_scenario_train = X_scenario[:split_idx]
    X_scenario_test = X_scenario[split_idx:]

    # Function to apply stride tricks cleanly on a single isolated block of data
    def build_windows(data_block):
        num_samples = len(data_block) - WINDOW_SIZE + 1
        if num_samples <= 0:
            return None
        shape = (num_samples, WINDOW_SIZE, data_block.shape[1])
        strides = (data_block.strides[0], data_block.strides[0], data_block.strides[1])
        windows = np.lib.stride_tricks.as_strided(data_block, shape=shape, strides=strides)
        return windows.reshape(num_samples, -1)

    # Process train block
    X_win_train = build_windows(X_scenario_train)
    if X_win_train is not None:
        X_train_list.append(X_win_train)
        y_train_list.append(np.full(shape=(len(X_win_train),), fill_value=target_class))

    # Process test block
    X_win_test = build_windows(X_scenario_test)
    if X_win_test is not None:
        X_test_list.append(X_win_test)
        y_test_list.append(np.full(shape=(len(X_win_test),), fill_value=target_class))

# Combine all scenario pieces into final train/test matrices
X_train_raw = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

X_test_raw = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)


print("2. Normalising and converting to XX-bit ADC FIXED-POINT values...")
# Find scaling baselines globally using ONLY training data to prevent baseline leakage
X_min_base = np.nanmin(X_train_raw[:, :3], axis=0)
X_max_base = np.nanmax(X_train_raw[:, :3], axis=0)

X_min = np.tile(X_min_base, WINDOW_SIZE)
X_max = np.tile(X_max_base, WINDOW_SIZE)
denom = (X_max - X_min) + 1e-8

# Scale training data
X_train_scaled = (X_train_raw - X_min) / denom
X_train = (X_train_scaled * ADC_MAX).astype(np.int32)

# Scale testing data using the training parameters
X_test_scaled = (X_test_raw - X_min) / denom
X_test = (X_test_scaled * ADC_MAX).astype(np.int32)


print("3. Training Fixed-Point Random Forest Classifier...")
# We use the clean, un-shuffled chronological partitions
model = RandomForestClassifier(n_estimators=14, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Present what has been learned
y_pred = model.predict(X_test)
test_f1 = f1_score(y_test, y_pred)
print(f"\n--- NEW EVALUATION METRICS WITH WINDOW_SIZE={WINDOW_SIZE} ---")
print(f"Test F1-Score: {test_f1:.4f}")
print("\nClassification Summary:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fault"]))
print(" ")


# --- STEP 4: EMLEARN EXPORT TO C HEADER ---
print("4. Generating 'model.h' C file with emlearn...")

c_model = emlearn.convert(model, method="inline", dtype='int32_t')
c_model.save(file="model.h", name="motor_vibration")

print("Success! 'model.h' has been generated. Ready for your MSPM0 project.")
print()


# --- PICK C TEST SAMPLES DIRECTLY FROM X_test ---
normal_idx = np.where(y_test == 0)[0][0]
fault_idx = np.where(y_test == 1)[0][0]

sample_normal_c = np.concatenate([X_test[normal_idx], X_test[normal_idx + 1], X_test[normal_idx + 2], X_test[normal_idx + 3]])
sample_fault_c = np.concatenate([X_test[fault_idx], X_test[fault_idx + 1], X_test[fault_idx + 2], X_test[fault_idx + 3]])

pause()
print("------------------------------------------------------------->")
print(" If no sensor, try some of this test-data")
print()
print(f"// Expected Class: 0 (Normal)")  # The [384], can just be []
print(f"const int32_t sample_normal[384] = {str(sample_normal_c.tolist()).replace('[', '{').replace(']', '}')};\n")

print(f"// Expected Class: 1 (Fault)")
print(f"const int32_t sample_fault[384] = {str(sample_fault_c.tolist()).replace('[', '{').replace(']', '}')};\n")

