# Sibainu Engine v6.4-Delta (Technical Validation)

![Project Status](https://img.shields.io/badge/Status-Technical%20Validation-success)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)
![Tested on](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB)-green)

**TL;DR**: A hidden-state based pre-emptive auditor achieving **0.9176 ROC-AUC** on an RTX 3050 (4GB). It detects ~60% of hallucinations at a 5% False Signal Rate (FSR).




## 1. Technical Overview
This project demonstrates a lightweight auditing layer that monitors internal **Hidden State Dynamics** to detect hallucinations *before* token generation.

* **No Training Required**: Works out-of-the-box with frozen weights. No fine-tuning or prior training on hallucination datasets is necessary.
* **Multi-Axis Analysis**: Beyond "Anchor Drift," v6.4 integrates **Layer Dissonance**—the structural inconsistency between transformer layers during anomalous inference.
* **Pre-emptive Detection**: Identifies the "collapse of latent trajectory" prior to the first token being sampled.
* **Theoretical Generalizability**: Validated on **Gemma-2b**. The geometric detection logic is theoretically applicable to any Transformer-based architecture.
* **Ultra-Low Resource**: Adds negligible latency ($O(d)$ per token). Developed and validated on consumer-grade hardware (**RTX 3050 4GB**).

## 2. Validation Resources
* **`evaluation_results_v6.4.csv`**:
    * Final score data from validation.
* **`visualizations/`**:
    * **`ROC_Curve_v6.4.png`**: Primary evidence of **0.9176 AUC**.

## 🛠 How to Use the Demo Code

### 🔐 Technical Access & Verification

The Sibainu Engine API is currently hosted in a **Private Space** to protect the proprietary **Internal Consistency Metrics (ICM)** logic and prevent unauthorized scraping. 

If you are a technical auditor, AI researcher, or represent an organization interested in evaluating the engine's performance (confirming the **60% Recall at 5% FSR** target), please follow the steps below to request a temporary verification token:

1. **Contact:** Reach out to **yubainu98(at)gmail.com** with your professional affiliation.
2. **Purpose:** Briefly state the scope of your verification (e.g., "HaluEval large-scale benchmarking").
3. **Issuance:** A **read-only access token** will be provided for a limited duration to facilitate your independent audit using the provided scripts.
(Please allow up to 24 hours for a response due to time zone differences.)

### 🖥 Environment Setup
A GPU with at least **4GB VRAM** (e.g., NVIDIA RTX 3050) is required to run the 4-bit (NF4) quantized model.

`pip install torch transformers datasets bitsandbytes accelerate pandas requests`

### 🔑 Obtain Access Token
The Sibainu Engine API is hosted in a **Private Space**. To perform an audit, you must authenticate your requests.

1. Open `demo6.4.py` or `Sibainu_HaluEval_NF4_Scanner.py`.
2. Locate the following configuration line:
   `HF_TOKEN = ""` 
3. **Insert your provided token between the quotes.** (e.g., `HF_TOKEN = "hf_..."`)



### 🚀 Running the Real-time Demo (`demo6.4.py`)
This script executes a 20-token inference with live **Internal Consistency Metrics (ICM)** monitoring.

`python demo6.4.py`

* **Input:** Enter any technical or factual question when prompted.
* **Output:** The engine streams each token along with its real-time risk score. A final verdict (`HIGH_RISK` / `LOW_RISK`) is issued based on the physical threshold of **3.6510**.

### 🔄 Running the Automated Recovery Demo (`recovery_agent_gemma.py`)

This script implements a **Closed-Loop Safety Control** that automatically triggers a re-generation (Recovery Mode) when the engine detects a physical neural anomaly.

* **Input**: Enter a factual or complex question (e.g., "What is the tallest mountain in Japan?").
* **Real-time Monitoring**: The client streams tokens while the remote API issues `CONTINUE` or `RECOVER` commands based on the live FSR5 score.
* **Automatic Recovery**: If the threshold of **3.6510** is breached, the agent immediately aborts the corrupted session and re-runs the inference using **Deterministic Greedy Search** to ensure factual stability.
* **Output**: A final, verified response is delivered after the autonomous correction process.

### 📊 Running the HaluEval Benchmark (`Sibainu_HaluEval_NF4_Scanner.py`)
This script automates the validation process using the official **HaluEval (QA)** dataset from Hugging Face.

`python Sibainu_HaluEval_NF4_Scanner.py`

* **Instructions:**
  1. Enter the **Start ID** (e.g., `0`) to pull from the dataset.
  2. Enter the **Number of Samples** (e.g., `10`) to scan.
* **Result:** A CSV file (e.g., `halueval_results_0_to_9.csv`) will be generated. This file contains raw ICM scores for both "Right" and "Hallucinated" pairs, allowing for immediate ROC-AUC calculation.

## 3. Benchmarks (Actual Measurements)

| Metric | Value (v6.4) | Previous (v6.1) | Note |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9176** | 0.8995 | Significant precision improvement. |
| **Recall @ 5% FSR** | **59.70%** | 48.2% (est) | Captures approx. 60% under strict constraints. (Threshold: 3.6510)|
| **Recall @ 10% FSR** | **74.75%** | 62.5% (est) | Captures approx. 75% under strict constraints. (Threshold: 2.9577)|
| **Precision** | **91.2%** | 88.5% | Minimizes unnecessary interventions. |
| **Latency** | **< 1ms** | < 1ms | Near-zero overhead on RTX 3050. |

> [!NOTE]
> **Separation Efficiency**: At a 5.0% FSR (False Signal Rate), the engine captures **59.7%** of all hallucinations in the HaluEval-QA dataset.

## 4. Methodology: Layer Dissonance
The v6.4 engine focuses on **"Latent Trajectory Collapse."** When a model begins to hallucinate, the vector transformations between the middle and final layers exhibit a specific type of geometric turbulence (Dissonance) that is statistically absent during factual recall.

## 4. Predictive Risk Assessment for Unconstrained Outputs

Omega Engine identifies unauthorized content generation as a primary precursor to hallucinations. When a model produces information beyond the scope of the initial query (e.g., self-generated follow-up questions), the system captures the resulting increase in entropy and latent dissonance. Such "runaway" generation is flagged as high-risk, providing an early warning before the model commits to a definitive factual error.

## 6. Roadmap
- [ ] **Cross-Model Validation**: Verifying theoretical generalizability across different LLMs.
- [ ] **Training Efficiency**: Applying the theory to filter training data and reduce computational resource costs.

## 7. License / Contact
**License: All Rights Reserved (Proprietary)**
**(C) 2026 sibainu.**
* **Developer**: yubainu
* **YouTube**: [Demonstration Video](https://youtu.be/H1_zDC0SXQ8)
* **Contact**: yubainu98(at)gmail.com
* **NDA**: Available for technical briefs upon request.
