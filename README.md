# 🧠 Neural Network Architectures for Predicting Time-Series Data from CFD Simulations

This project explores the use of deep learning models—**Fully Connected Neural Networks (FCNN)** and **Long Short-Term Memory Networks (LSTM)**—as surrogate models to predict **aerodynamic force coefficients (Cd, Cl)** and **pressure distributions** from CFD simulations of a **transversely oscillating cylinder**.

The goal is to offer a lightweight, fast, and accurate alternative to running repeated, computationally expensive CFD simulations—especially useful for control, optimization, and real-time applications.

---

## 📌 Project Goals

- ✅ Predict time-series aerodynamic behavior from probe and force data
- 🔁 Replace expensive CFD computations with fast neural surrogates
- 📈 Compare performance of FCNN vs LSTM architectures
- 🌍 Generalize across varying amplitudes and frequencies of oscillation

---

## 🌀 CFD Simulation Setup

CFD data is generated using [ml-cfd-lecture by Andre Weiner](https://github.com/AndreWeiner/ml-cfd-lecture) with modifications for an oscillating cylinder setup using OpenFOAM.

### Simulation Parameters

| Parameter               | Value                                  |
|-------------------------|----------------------------------------|
| Flow Type               | 2D Laminar                             |
| Reynolds Number         | 100                                    |
| Solver                  | `pimpleFoam`                           |
| Cylinder Motion         | (y(t) = A*sin(2*pi*f*t))    |
| Amplitude Ratio (A/D)   | 0.5 – 3.5                              |
| Frequency Ratio (f/fₙ)  | 3.0 – 5.0                              |
| Time Step               | 0.01 s                                 |
| Outputs                 | Cd, Cl, 12 pressure probe readings     |

---

## 🧠 Code Overview

| File                  | Description |
|-----------------------|-------------|
| `main.py`             | Controls training and testing |
| `FCNN.py`, `LSTM.py`  | Define FCNN and LSTM architectures |
| `Data_pipeline.py`    | Reads, normalizes, and prepares CFD data |
| `autoregress_train.py`, `autoregress_func.py` | Implements autoregressive sequence modeling |
| `hyperparameter_tuning.py` | Performs parameter tuning using Optuna |
| `plots.py`            | Generates evaluation plots |
| `utils.py`            | Utility functions for normalization, saving models, etc. |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sumanthyeddula/Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations.git
cd Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations
```
### ▶️ Run the Code

- **Training** (`main.py`):
  ```python
  enable_testing_mode = False
  model_type = LSTMModel  # or FCNNModel
  ```
  ```bash
  python main.py
  ```

- **Testing**:
  ```python
  enable_testing_mode = True
  python main.py
  ```

  ---

## 📊 LSTM Model Results

_(Embed actual image links in your repository to replace these placeholders)_

### ✅ Training Loss

![LSTM Loss](./results/LSTM/LSTM__30_loss_plot.png.png)

### ✅ Cd & Cl Time-Series

![LSTM Cd Cl](./results/LSTM/LSTM__30_cd_cl_plot.png.png)

### ✅ Pressure Probes

![LSTM Probes](./results/LSTM/LSTM__30_probe_columns_plot.PNG.png)

### ✅ Cd/Cl Accuracy Contours

![LSTM Contour](./results/LSTM/LSTM__30_cd_cl_contour.PNG.png)

### ✅ Probe Accuracy Contours

![LSTM Probe Contours](./results/LSTM/LSTM__30_probe_contour.PNG.png)

### ✅ Global Accuracy Bar Plot

![LSTM Error Bars](./results/LSTM/LSTM__prediction_accuracy_error_bars_scatter.png.png)

---

## 📊 FCNN Model Results

### ✅ Training Loss

![FCNN Loss](./results/FCNN/FCNN__30_loss_plot.png.png)

### ✅ Cd & Cl Time-Series

![FCNN Cd Cl](./results/FCNN/FCNN__30_cd_cl_plot.png.png)

### ✅ Pressure Probes

![FCNN Probes](./results/FCNN/FCNN__30_probe_columns_plot.PNG.png)

### ✅ Cd/Cl Accuracy Contours

![FCNN Contour](./results/FCNN/FCNN__30_cd_cl_contour.png.png)

### ✅ Probe Accuracy Contours

![FCNN Probe Contours](./results/FCNN/FCNN__30_probe_contour.PNG.png)

### ✅ Global Accuracy Bar Plot

![FCNN Error Bars](./results/FCNN/FCNN__prediction_accuracy_error_bars_scatter.png.png)

---

## 🧠 Model Comparison Summary

| Feature             | LSTM                             | FCNN                             |
|---------------------|----------------------------------|----------------------------------|
| Sequence Modeling   | ✅ Yes                           | ❌ No                            |
| Accuracy (Cd/Cl)    | ⭐ High (~0.9+)                   | ⚠️ Moderate (~0.7–0.8)           |
| Probe Prediction    | ✅ Excellent                     | ⚠️ Decent but variable           |
| Speed               | ⏳ Slower                        | ⚡ Faster                        |
| Ideal Use Case      | Unsteady, high-frequency flows   | Smooth, quasi-steady conditions  |

---

## 📄 Final Report

📘 [Research_Project_final_report_sumanth.pdf](./Research_Project_final_report_sumanth.pdf)

Includes:
- CFD & motion setup
- Neural network architecture
- Results and discussion
- Conclusion and future work

---

## 🙋‍♂️ Author

**Sumanth Reddy Yeddula**  
Master’s Student @ TU Dresden  
📧 sumanthreddyyeddula@gmail.com  

---

> 💬 If you use this work or have questions, feel free to connect or raise an issue.
