
from machine import Pin
import dht
import time
import json
import math
import gc

# ============================================
# 1. CONFIGURATION
# ============================================

TEST_MODE = True

SAMPLE_INTERVAL = 1.5 if TEST_MODE else 2.5
WINDOW_SIZE = 10

# ============================================
# 2. HARDWARE INITIALIZATION
# ============================================

sensor = dht.DHT22(Pin(15))
led = Pin(16, Pin.OUT)

led.off()

# ============================================
# 3. LOAD TRAINED AI MODELS
# ============================================


# ============================================
# LOAD MODELS AND MEASURE RAM
# ============================================

# Measure free RAM before loading
gc.collect()
ram_before = gc.mem_free()

# Load model data once
with open("model_data.json", "r") as f:
    model_data = json.load(f)

# Measure free RAM after loading
gc.collect()
ram_after = gc.mem_free()

# Extract model parameters
mean = model_data["mean"]
std = model_data["std"]

float_model = model_data["float32"]
int8_model = model_data["int8"]

print("Models loaded successfully!")

print("")
print("MODEL RAM MEASUREMENT")
print("------------------------------")
print("Free RAM before loading:", ram_before, "bytes")
print("Free RAM after loading:", ram_after, "bytes")
print(
    "RAM used by loaded model data:",
    ram_before - ram_after,
    "bytes"
)


# ============================================
# 4. AUTOMATIC TEST DATA
# ============================================

# Format: temperature, humidity, expected anomaly
#
# 10 normal readings
# 2 moderate temperature changes
# 2 high-temperature anomalies
# 2 high-humidity anomalies
# 10 normal recovery readings

test_data = (
    [(24.0, 50.0, False)] * 10
    + [(28.0, 50.0, True)] * 2
    + [(35.0, 50.0, True)] * 2
    + [(24.0, 85.0, True)] * 2
    + [(24.0, 50.0, False)] * 10
)

# ============================================
# 5. DATA STORAGE
# ============================================

temp_history = []
hum_history = []

reading_count = 0

float_times = []
int8_times = []

# Confusion matrices: TP, TN, FP, FN
float_results = [0, 0, 0, 0]
int8_results = [0, 0, 0, 0]

# ============================================
# 6. FEATURE EXTRACTION
# ============================================

def get_features(temp, hum):

    if len(temp_history) == 0:
        avg_temp = temp
        avg_hum = hum
    else:
        avg_temp = sum(temp_history) / len(temp_history)
        avg_hum = sum(hum_history) / len(hum_history)

    temp_dev = temp - avg_temp
    hum_dev = hum - avg_hum

    features = [
        temp,
        hum,
        temp_dev,
        hum_dev
    ]

    normalized = []

    for i in range(4):
        normalized.append(
            (features[i] - mean[i]) / std[i]
        )

    return normalized


def update_history(temp, hum):

    temp_history.append(temp)
    hum_history.append(hum)

    if len(temp_history) > WINDOW_SIZE:
        temp_history.pop(0)
        hum_history.pop(0)

# ============================================
# 7. NEURAL NETWORK INFERENCE
# ============================================

def predict(features, model, quantized=False):

    W1 = model["W1"]
    b1 = model["b1"]

    W2 = model["W2"]
    b2 = model["b2"][0]

    if quantized:
        scale1 = model["scale1"]
        scale2 = model["scale2"]
    else:
        scale1 = 1.0
        scale2 = 1.0

    # Hidden layer: 4 inputs -> 8 neurons
    hidden = []

    for j in range(8):

        value = b1[j]

        for i in range(4):
            value += (
                features[i]
                * W1[i][j]
                * scale1
            )

        # ReLU activation
        hidden.append(max(0.0, value))

    # Output layer: 8 neurons -> 1 output
    output = b2

    for j in range(8):
        output += (
            hidden[j]
            * W2[j][0]
            * scale2
        )

    # Sigmoid activation
    output = max(-50.0, min(50.0, output))

    probability = 1.0 / (
        1.0 + math.exp(-output)
    )

    return probability

# ============================================
# 8. SENSOR DATA COLLECTION
# ============================================

def get_reading(index):

    if TEST_MODE:
        return test_data[index]

    sensor.measure()

    return (
        sensor.temperature(),
        sensor.humidity(),
        None
    )

# ============================================
# 9. EVALUATION FUNCTIONS
# ============================================

def update_confusion_matrix(
    expected, predicted, results
):

    if expected and predicted:
        results[0] += 1

    elif not expected and not predicted:
        results[1] += 1

    elif not expected and predicted:
        results[2] += 1

    else:
        results[3] += 1


def print_model_results(name, results):

    tp, tn, fp, fn = results

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total * 100
        if total > 0 else 0
    )

    precision = (
        tp / (tp + fp) * 100
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn) * 100
        if (tp + fn) > 0 else 0
    )

    print("")
    print(name)
    print("------------------------------")

    print("True Positives:", tp)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    print("Accuracy:", round(accuracy, 2), "%")
    print("Precision:", round(precision, 2), "%")
    print("Recall:", round(recall, 2), "%")

# ============================================
# 10. MAIN PROGRAM
# ============================================

print("")
print("================================")
print("TinyML Environmental Anomaly Detector")
print("================================")

if TEST_MODE:
    print("MODE: AUTOMATIC TEST")
else:
    print("MODE: LIVE DHT22 SENSOR")

print("Sampling interval:", SAMPLE_INTERVAL, "s")
print("Model architecture: 4-8-1")
print("Optimization: INT8 Weight Quantization")

while True:

    # Stop after all test measurements
    if TEST_MODE and reading_count >= len(test_data):
        break

    try:

        # Read sensor or simulated values
        temperature, humidity, expected = (
            get_reading(reading_count)
        )

        # Validate sensor readings
        if not (-40 <= temperature <= 80):
            raise ValueError("Invalid temperature")

        if not (0 <= humidity <= 100):
            raise ValueError("Invalid humidity")

        reading_count += 1

        # Extract and normalize features
        features = get_features(
            temperature,
            humidity
        )

        # --------------------------------
        # FLOAT32 INFERENCE
        # --------------------------------

        start = time.ticks_us()

        float_score = predict(
            features,
            float_model,
            False
        )

        float_latency = time.ticks_diff(
            time.ticks_us(),
            start
        )

        # --------------------------------
        # INT8 WEIGHT INFERENCE
        # --------------------------------

        start = time.ticks_us()

        int8_score = predict(
            features,
            int8_model,
            True
        )

        int8_latency = time.ticks_diff(
            time.ticks_us(),
            start
        )

        float_times.append(float_latency)
        int8_times.append(int8_latency)

        # --------------------------------
        # ANOMALY CLASSIFICATION
        # --------------------------------

        float_anomaly = float_score >= 0.5
        int8_anomaly = int8_score >= 0.5

        # LED controlled by quantized model
        led.value(1 if int8_anomaly else 0)

        # --------------------------------
        # EVALUATION
        # --------------------------------

        if TEST_MODE:

            update_confusion_matrix(
                expected,
                float_anomaly,
                float_results
            )

            update_confusion_matrix(
                expected,
                int8_anomaly,
                int8_results
            )

        # --------------------------------
        # CONSOLE OUTPUT
        # --------------------------------

        print("")
        print("------------------------------")
        print("Reading:", reading_count)

        print(
            "Temperature:",
            temperature,
            "C"
        )

        print(
            "Humidity:",
            humidity,
            "%"
        )

        print(
            "Float32 anomaly score:",
            round(float_score, 4)
        )

        print(
            "Int8 anomaly score:",
            round(int8_score, 4)
        )

        print(
            "Float32 latency:",
            float_latency,
            "us"
        )

        print(
            "Int8 latency:",
            int8_latency,
            "us"
        )

        if TEST_MODE:
            print("Expected anomaly:", expected)

        print("Float32 prediction:", float_anomaly)
        print("Int8 prediction:", int8_anomaly)

        if int8_anomaly:
            print("STATUS: ANOMALY DETECTED")
            print("LED: ON")
        else:
            print("STATUS: NORMAL")
            print("LED: OFF")

        # Update history AFTER inference
        update_history(
            temperature,
            humidity
        )

    except (OSError, ValueError) as error:

        led.off()
        print("ERROR:", error)

        if TEST_MODE:
            break

    # No unnecessary delay after final test
    if not (
        TEST_MODE
        and reading_count >= len(test_data)
    ):
        time.sleep(SAMPLE_INTERVAL)

# ============================================
# 11. FINAL TEST SUMMARY
# ============================================

if TEST_MODE:

    led.off()

    print("")
    print("================================")
    print("FINAL ON-DEVICE AI BENCHMARK")
    print("================================")

    print("Total measurements:", reading_count)

    if float_times and int8_times:

        avg_float = (
            sum(float_times) / len(float_times)
        )

        avg_int8 = (
            sum(int8_times) / len(int8_times)
        )

        print(
            "Average Float32 latency:",
            round(avg_float, 2),
            "us"
        )

        print(
            "Average Int8 latency:",
            round(avg_int8, 2),
            "us"
        )

        if avg_float > 0:

            latency_change = (
                (avg_int8 - avg_float)
                / avg_float * 100
            )

            print(
                "Latency change:",
                round(latency_change, 2),
                "%"
            )

    # Model parameter sizes from Colab
    float_size = 196
    int8_size = 84

    print("")
    print("MODEL PARAMETER STORAGE")
    print("------------------------------")

    print("Float32:", float_size, "bytes")
    print("Int8:", int8_size, "bytes")

    print(
        "Storage reduction:",
        round(
            (1 - int8_size / float_size) * 100,
            2
        ),
        "%"
    )

    # Accuracy comparison
    print("")
    print("MODEL ACCURACY COMPARISON")
    print("================================")

    print_model_results(
        "FLOAT32 MODEL",
        float_results
    )

    print_model_results(
        "INT8 MODEL",
        int8_results
    )

    # RAM information
    gc.collect()

    print("")
    print("MEMORY INFORMATION")
    print("------------------------------")

    print(
        "Free RAM after test:",
        gc.mem_free(),
        "bytes"
    )

    print("")
    print("================================")
    print("TEST COMPLETED")
    print("LED: OFF")
    print("================================")
