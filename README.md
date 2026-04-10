# Sibainu Engine v6.4-Delta (Technical Validation)

![Project Status](https://img.shields.io/badge/Status-Technical%20Validation-success)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)
![Tested on](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB)-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19052934.svg)](https://doi.org/10.5281/zenodo.19052934)

### Technical Report
**[Latent Dissonance Mapping: Deterministic Auditing for LLM Hallucinations](https://doi.org/10.5281/zenodo.19052934)**

# Sibainu Engine: Pre-emptive Hallucination Audit via Latent Geometry

**TL;DR:** Stop using LLMs to fact-check LLMs. 
Sibainu is an ultra-lightweight auditor that detects hallucinations *before* they are sampled by monitoring "Layer Dissonance" in the hidden state. 
Hallucinations aren't just wrong words; they are detectable geometric anomalies in the model's internal state dynamics.

- **Pre-emptive:** Identifies the collapse of latent trajectories with <1ms latency.
- **Efficient:** ROC-AUC > 0.90 on Gemma,Gemma2,Llama, and Mistral using a single RTX 3050 (4GB)and Google Colab(T4).
- **Deterministic:** No secondary LLM calls; 60% detection at 5% FSR across all tested architectures.


<img width="800" height="600" alt="roc-auc" src="https://github.com/user-attachments/assets/e094c0a1-a61b-4c84-acd4-44ead09ea65c" />



## The Concept: Inferential Dissonance Monitoring
Unlike traditional hallucination detectors that rely on external knowledge bases or LLM-as-a-judge, LDM (Latent Dissonance Mapping) monitors the internal "computational strain" of the model.

I focus on Inferential Dissonance: the geometric distortion in latent space that occurs when a model's internal logic conflicts with its output generation. This deterministic approach allows for identifying hallucinations caused by reasoning failures, even when the model appears confident.

## 1. Technical Overview
This project demonstrates a lightweight auditing layer that monitors internal **Hidden State Dynamics** to detect hallucinations *before* token generation.

* **No Training Required**: Works out-of-the-box with frozen weights. No fine-tuning or prior training on hallucination datasets is necessary.
* **Multi-Axis Analysis**: Beyond "Anchor Drift," v6.4 integrates **Layer Dissonance**—the structural inconsistency between transformer layers during anomalous inference.
* **Pre-emptive Detection**: Identifies the "collapse of latent trajectory" prior to the first token being sampled.
* **Theoretical Generalizability**: Validated on **Gemma-2b**. The geometric detection logic is theoretically applicable to any Transformer-based architecture.
* **Ultra-Low Resource**: Adds negligible latency ($O(d)$ per token). Developed and validated on consumer-grade hardware (**RTX 3050 4GB**).

### Performance Analysis

The Omega Engine is designed for ultra-low latency real-time auditing of LLM generation processes. The following benchmarks distinguish between the **intrinsic algorithmic latency (Core)** and the **total system response time (API)**.

#### 1. Latency Breakdown (Local Environment)
* **Environment**: Windows 11 / Python 3.12 / NVIDIA GeForce RTX 3050 (4GB)
* **Target Model**: `google/gemma-2b` (4-bit quantization)

| Measurement Scope | Latency | Technical Context |
| :--- | :--- | :--- |
| **Auditing Core (NumPy)** | **1.2 ms** | Pure mathematical vectorized computation. |
| **Data I/O & Validation** | **~12.0 ms** | Pydantic validation and List-to-Array conversion. |
| **End-to-End API (Local)** | **15.0 - 25.0 ms** | Total response time including FastAPI overhead. |



#### 2. Technical Insight: The "15ms" Reality
The observed latency of 15ms–20ms is primarily due to **data interface friction** rather than algorithmic complexity.

* **Zero-Bottleneck Design**: The core computation finishes in under 1ms, which is significantly faster than typical LLM token generation speeds (30–70ms/token). This ensures that the auditing process never becomes a bottleneck for the generation pipeline.
* **Scalability**: Most of the current delay stems from Python's serialization (JSON/List to NumPy conversion). **Theoretically**, this overhead can be eliminated in a production-grade environment (e.g., C++ implementation or using Shared Memory), bringing total latency close to the <1ms algorithmic limit.

#### 3. Stability & Constant-Time Execution
By leveraging vectorized operations, the auditing time remains constant regardless of the sequence length. This guarantees stable, real-time performance even during long-form text generation.

## 2. Validation Resources
* **`evaluation_results_v6.4.csv`**:
    * Final score data from validation.
* **`visualizations/`**:
    * **`ROC_Curve_v6.4.png`**: Primary evidence of **0.9176 AUC**.

## 🛠 How to Use the Demo Code

### 🖥 Environment Setup
A GPU with at least **4GB VRAM** (e.g., NVIDIA RTX 3050) is required.

1. **Install PyTorch (CUDA version)**:  
   Visit [pytorch.org](https://pytorch.org/) to install the version matching your CUDA toolkit.

2. **Install Required Libraries**:
```bash
pip install transformers accelerate gradio_client
# For Windows users (to support 4-bit quantization):
pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```


3. **Access to Gemma-2b:**

Accept the license on the Hugging Face model page and run huggingface-cli login in your terminal with your HF token.


### 🚀 Running the Real-time Demo (demo.py)
This script executes a 30-token inference with live Internal Consistency Metrics (ICM) monitoring via the Secure Proxy.

```bash
python demo.py
```

* **Input: Enter any technical or factual question when prompted.

* **Secure Inference: Your local GPU handles the heavy LLM processing (Gemma-2b), while the Omega Engine (hosted on Hugging Face) performs real-time geometric analysis of the latent states.

* **Output: The engine streams each token along with its real-time risk score. A final verdict (HIGH_RISK / SAFE) is issued based on the physical threshold of 3.6510.

### 🔒 Privacy & Security: Multi-tier Architecture
This demo utilizes a multi-tier proxy architecture to ensure user security:

* **Local (User PC)**: Features (hidden state vectors) are extracted on your own hardware.

* **Proxy (Public)**: These features are relayed to the engine through a secure middle-ware.

* **Engine (Private)**: The actual analysis and scoring are performed in a sandboxed environment.

No private Hugging Face tokens are required or stored on the client side.

> [!IMPORTANT]
> ### Understanding the Score: Beyond Factual Correctness
>
> You may occasionally see a HIGH_RISK verdict even when the model provides a factually correct answer. This is not necessarily a "False Positive," but a reflection of the engine's design philosophy:
>
> * Neural Instability: The engine measures the internal stress of the model during inference. A high score indicates that the model is struggling to maintain geometric consistency—a common occurrence when the model is "guessing" a correct answer based on weak or ambiguous internal signals.
> * Safety Margin (Recall vs. FSR): To ensure a 60% Recall of hallucinations, the engine is tuned with a 5% False Signal Rate (FSR). We intentionally prioritize flagging these "unstable truths" to prevent "confident lies" from slipping through.
> * Quantization Noise: Operating in 4-bit (NF4) mode introduces inherent noise into the model's latent space. This noise can occasionally push the metrics above the threshold, even during successful factual recall.



### 🔄 Running the Automated Recovery Demo (`recovery_agent_gemma.py`)

This script demonstrates a Closed-Loop Neural Calibration System that monitors and corrects the model's latent trajectory in real-time. It ensures high-fidelity output by managing "Geometric Distortion" within the neural manifold.

#### Real-time Monitoring & Steering (Route A):
The system continuously analyzes the structural integrity of the generated hidden states. When a "Geometric Distortion" is detected, the agent applies an immediate Steering Vector to the model's activation. This "surgical" intervention recalibrates the token selection process without interrupting the stream, visualized by the [CORRECTING...] status gauge.

#### Autonomous Recovery (Route B):
If the high-order distortion exceeds the tunable safety manifold, the system triggers an emergency Recovery Mode. The corrupted session is instantly aborted and re-initialized with a stabilized inference strategy to prevent factual hallucinations or structural collapse.

#### Latency Note:
The current execution involves high-dimensional tensor synchronization between the local client and the remote analysis API. While this introduces noticeable latency in the demo environment, the architecture is designed for future integration into edge-accelerated inference engines where this overhead is minimized.

#### Output:
A final, verified response delivered through a dual-layer safety architecture that prioritizes neural stability over raw generation speed.

## 3. Benchmarks (Actual Measurements)
### Dataset
* **Dataset Used**: [HaluEval-QA dataset](https://github.com/bjascob/HaluEval)
  * A large-scale collection of generated and human-annotated hallucinated samples for evaluating LLMs.

### 1.gemma2B(N=2000)

| Metric | Value (v6.4) | Previous (v6.1) | Note |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9176** | 0.8995 | Significant precision improvement. |
| **Recall @ 5% FSR** | **59.70%** | 48.2% (est) | Captures approx. 60% under strict constraints. (Threshold: 3.6510)|
| **Recall @ 10% FSR** | **74.75%** | 62.5% (est) | Captures approx. 75% under strict constraints. (Threshold: 2.9577)|
| **Precision** | **91.2%** | 88.5% | Minimizes unnecessary interventions. |
| **Latency** | **< 1ms** | < 1ms | Near-zero overhead on RTX 3050. |

### 2.Llama-3.2-1B(N=2000)
| Metric | Value |
| :--- | :--- |
| ROC-AUC | 0.9217 |
| Recall @5% FSR | 0.6375 |
| Recall @10% FSR | 0.7800 |

### 3.Mistral7B(N=250)
| Metric | Value |
| :--- | :--- |
| ROC-AUC | 0.9035 |
| Recall @5% FSR | 0.5960 |
| Recall @10% FSR | 0.7200 |

### 4.(unsloth)gemma-2-9b-it-bnb-4bit(N=500)
| Metric | Value |
| :--- | :--- |
| ROC-AUC | 0.9521 |
| Recall @5% FSR | 0.8620 |
| Recall @10% FSR | 0.9180 |


> [!NOTE]
> **Separation Efficiency**: At a 5.0% FSR (False Signal Rate), the engine captures **59.7%** of all hallucinations in the HaluEval-QA dataset.

## 4. Methodology: Layer Dissonance
The v6.4 engine focuses on **"Latent Trajectory Collapse."** When a model begins to hallucinate, the vector transformations between the middle and final layers exhibit a specific type of geometric turbulence (Dissonance) that is statistically absent during factual recall.

## 4. Predictive Risk Assessment for Unconstrained Outputs

Omega Engine identifies unauthorized content generation as a primary precursor to hallucinations. When a model produces information beyond the scope of the initial query (e.g., self-generated follow-up questions), the system captures the resulting increase in entropy and latent dissonance. Such "runaway" generation is flagged as high-risk, providing an early warning before the model commits to a definitive factual error.

## 6. Roadmap
- [ ] **Learning Phase**:  Low-cost generation of high-logic models via in-training feedback control using Adaptive Omega thresholds.
 - [ ] **A training model with a lower probability of hallucinations**: Proof-of-concept experiments to realize higher-precision models with a lower probability of hallucinations by applying this technology to audit internal geometric distortions during the training phase.

## 7. License / Contact
**License: All Rights Reserved (Proprietary)**
**(C) 2026 sibainu.**
* **Developer**: yubainu
* **YouTube**: [Demonstration Video](https://youtu.be/H1_zDC0SXQ8)
* **Contact**: yubainu98(at)gmail.com
* **NDA**: Available for technical briefs upon request.
