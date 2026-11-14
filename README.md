#AIoT Environmental Anomaly Detection
This project demonstrates environmental anomaly detection using a Raspberry Pi Pico W, DHT22 sensor, and a small neural network running locally in MicroPython.Project Files
- main.py – Sensor acquisition, on-device inference, and LED control.
- diagram.json – Wokwi hardware simulation.
- model_data.json – Trained Float32 and quantized Int8 model parameters.
- train_and_quantize.ipynb – Model training and quantization in Google Colab.
- benchmark_results.md – Accuracy, latency, and memory measurements.
How to Run
1. Open https://wokwi.com/.
2. Create a Raspberry Pi Pico W MicroPython project.
3. Add main.py, diagram.json, and model_data.json.
4. Start the simulation.
5. Open the Serial Monitor to inspect predictions and benchmark results.
Reproduce Model Training
1. Open train_and_quantize.ipynb in Google Colab.
2. Run all notebook cells.
3. Train the neural network and generate the quantized weights.
4. Export the resulting model parameters.
Results
- Synthetic test accuracy: 99.33% for both models.
- Wokwi test accuracy: 100% on 26 scripted readings.
- Float32 inference latency: 1.612 ms.
- Int8 weight inference latency: 1.594 ms.
- Theoretical parameter storage: 196 bytes versus 85 bytes.
- Parameter storage reduction: 56.63%.
The prototype uses synthetic training data and simulated hardware. Results do not establish real-world deployment performance.
