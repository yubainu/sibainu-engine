# Sibainu Engine v6.4-Delta (Technical Validation)

![Project Status](https://img.shields.io/badge/Status-Technical%20Validation-success)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)
![Tested on](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB)-green)

**TL;DR**: A hidden-state based pre-emptive auditor achieving **0.9176 ROC-AUC** on an RTX 3050 (4GB). It detects ~60% of hallucinations at a 5% False Signal Rate (FSR).

## 1. Technical Overview
This project demonstrates a lightweight auditing layer that monitors internal **Hidden State Dynamics** to detect hallucinations *before* token generation.

* **Multi-Axis Analysis**: Beyond "Anchor Drift," v6.4 integrates **Layer Dissonance**—the structural inconsistency between transformer layers during anomalous inference.
* **Pre-emptive Detection**: Identifies the "collapse of latent trajectory" prior to the first token being sampled.
* **Theoretical Generalizability**: Validated on **Gemma-2b**. The geometric detection logic is theoretically applicable to any Transformer-based architecture.
* **Ultra-Low Resource**: Adds negligible latency ($O(d)$ per token). Developed and validated on consumer-grade hardware (**RTX 3050 4GB**).

## 2. Validation Resources
* **`evaluation_results_v6.4.csv`**:
    * Final score data from validation.
* **`visualizations/`**:
    * **`ROC_Curve_v6.4.png`**: Primary evidence of **0.9176 AUC**.
* **`sibainu_engine_lite.py`**:
    * Demo script for verifying fundamental detection logic.

### How to Use the Demo Code
This code is designed to run in a Python 3.x environment.

1.  **Execution**: Run the following command to see the engine in action:
    ```bash
    python sibainu_engine_lite.py
    ```
2.  **Performance Evaluation**: To reproduce the benchmark results (ROC-AUC 0.8995), run:
    ```bash
    python evaluate.py
    ```
    * *Note: The scripts will reference the data in `raw_logs.csv`.*

#### Performance Evaluation (Internal Benchmark)

hese metrics are achieved by the full 8-axis engine. The Lite version (1-axis) provided here is for fundamental logic verification.


## 3. Benchmarks (Actual Measurements)

| Metric | Value (v6.4) | Previous (v6.1) | Note |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9176** | 0.8995 | Significant precision improvement. |
| **Recall @ 5% FSR** | **59.70%** | 48.2% (est) | Captures approx. 60% under strict constraints. |
| **Precision** | **91.2%** | 88.5% | Minimizes unnecessary interventions. |
| **Latency** | **< 1ms** | < 1ms | Near-zero overhead on RTX 3050. |

> [!NOTE]
> **Separation Efficiency**: At a 5.0% FSR (False Signal Rate), the engine captures **59.7%** of all hallucinations in the HaluEval-QA dataset.

## 4. Methodology: Layer Dissonance
The v6.4 engine focuses on **"Latent Trajectory Collapse."** When a model begins to hallucinate, the vector transformations between the middle and final layers exhibit a specific type of geometric turbulence (Dissonance) that is statistically absent during factual recall.

## 5. Roadmap
- [ ] **Cross-Model Validation**: Verifying theoretical generalizability across different LLMs.
- [ ] **Training Efficiency**: Applying the theory to filter training data and reduce computational resource costs.

## 6. License / Contact
**License: All Rights Reserved (Proprietary)**
**(C) 2026 sibainu.**
* **Developer**: yubainu
* **YouTube**: [Demonstration Video](https://youtu.be/H1_zDC0SXQ8)
* **Contact**: yubainu98(at)gmail.com
* **NDA**: Available for technical briefs upon request.
