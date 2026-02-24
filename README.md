# Sibainu Engine v6.1-Gamma (Technical Preview)

![Project Status](https://img.shields.io/badge/Status-Technical%20Preview-blue)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)
![Tested on](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB)-green)

This project is a demonstration of a lightweight auditing layer designed to detect and suppress hallucinations (false outputs) in Transformer-based LLMs in real-time by observing geometric fluctuations in Hidden States during inference.

## 1. Geometric Detection Overview
The engine statistically evaluates "trajectory distortion" within the model's internal vector space rather than performing semantic content analysis.

* **Geometric Analysis**: Measures the **"Anchor Drift"**—how the Hidden State of each generated token deviates from the "semantic anchors" defined by the prompt.
* **Real-time Intervention**: Triggers an immediate suppression or control of the generation process the moment the Drift Score exceeds the **preset threshold**.
* **Low Computational Cost**: 
    > Adds only $O(d)$ vector distance calculations per token. This ensures minimal impact on inference throughput, even in local environments such as the RTX 3050 (4GB).



## 2. Public Resources
To ensure verification transparency, the following resources are provided:

* **`sibainu_engine_lite.py`**:
    * A **demo/trial version** with analysis axes limited to "Anchor Drift."
    * Allows for the verification of the fundamental detection logic.
* **`evaluate.py`**:
    * A verification script to automatically measure the performance (Precision/Recall, etc.) of this engine.
* **`raw_logs.csv`**:
    * Contains the raw verification data (IDs, true labels, predicted labels, and drift scores) generated during performance validation on an RTX 3050 (4GB).

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

## 3. Performance Evaluation (Internal Benchmark)

### Validation Process
1.  **Calibration**: Determined the optimal threshold to maximize F1-Score using a validation set of 200 samples.
2.  **Blind Test**: Conducted independent testing on the remaining 800 samples using the fixed threshold.
3.  **Data Source**: Evaluations are based strictly on **actual Hidden State measurements** on RTX 3050, not theoretical predictions.

### Evaluation Metrics
| Metric | Value | Technical Characteristics |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8995** | Confirmed strong correlation between geometric fluctuations and hallucinations. |
| **Precision** | **88.52%** | High precision. Conservative design minimizing false positives. |
| **Recall** | **53.89%** | Captures approx. half of the cases. |
| **FSR** | **7.01%** | False Stop Rate. Minimizes interruption of valid responses. |

<img width="2552" height="1638" alt="score_distribution_fixed" src="https://github.com/user-attachments/assets/359768e3-2107-49dc-817c-642c8510876c" />
<img width="800" height="800" alt="roc_curve" src="https://github.com/user-attachments/assets/b49cde77-d79c-4bfc-b3b8-535d63845dac" />


## 4. Verification Case Studies
Based on the engine protocol, logical deviations were successfully neutralized in the following cases. For details, refer to the [Demonstration Video](https://youtu.be/H1_zDC0SXQ8).

> [!IMPORTANT]
> **Neutralized Examples:**
> 1. **Non-existent Entity Assertion**: e.g., "The Capital of Mars."
> 2. **Fabricated Authority Attribution**: e.g., Using fake names (Dr. George T. Hems).
> 3. **Chronological Anachronism**: e.g., "Silicon Valley" in a 1930s setting.

## 5. Implementation Characteristics
* **External Auditing Layer**: Can be integrated as an external module into existing LLM pipelines.
* **Black-box Agnostic**: Detects anomalies based solely on the geometric behavior of vector data without accessing the LLM's internal weights or hidden logic.
* **Diagnostic Logging**: When a hallucination is detected, the engine outputs the score and the **reason for detection (Reason)** in the logs for post-analysis.
* **Dynamic Resampling**: Lowers the sampling temperature upon threshold breach to trigger deterministic regeneration.

## 6. Roadmap
- [ ] **Multidimensional Integration**: Incorporating Layer Dissonance to aim for 80% Recall.
- [ ] **Geometric Data Cleansing**: Applying detection logic to pre-processing to improve "intellectual purity."
- [ ] **Optimization of Learning Process**: Aiming for a 15% reduction in training resource costs through dynamic gradient control.

## 7. License / Contact
**License: All Rights Reserved (Proprietary)**
**(C) 2026 sibainu.**

This release is for technical demonstration purposes only. Commercial use, reproduction, or redistribution of the code and algorithms without permission is prohibited.

* **Developer**: yubainu
* **YouTube**: [Demonstration Video](https://youtu.be/H1_zDC0SXQ8)
* **Contact**: yubainu98@gmail.com

**Hardware Tested**: NVIDIA GeForce RTX 3050 (4GB)
