# Shibainu Engine: Logic-Preserving Hallucination Defense Technology

## 1. Overview
Shibainu Engine (SIB) is an experimental project designed to solve the trade-off between "Intelligence Degradation (Loss of Logical Consistency)" and "Increased Hallucinations" that typically occurs during Large Language Model (LLM) fine-tuning.

### Pre-test
By strictly controlling weight update variance (**Std 0.0074**), SIB protects the model's internal Natural Language Inference (NLI) capabilities, maintaining high integrity while adapting to specific tasks.

This methodology leverages a pre-emptive hallucination detection framework to detect real-time geometric anomalies within the internal representations. It features a dynamic gain scheduler that optimizes the training process, preventing logical collapse while enhancing task adaptation.

<img width="1536" height="757" alt="Figure_1" src="https://github.com/user-attachments/assets/421628cc-9654-4fdc-99e1-3ab90ad49689" />

<img width="4200" height="2400" alt="weight_distribution_zoom_005" src="https://github.com/user-attachments/assets/8a0e6d32-e91c-4a15-817c-b94d921ab8fc" />

<img width="1000" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/47c09006-a0aa-473b-86ce-b33e0cd7ae04" />

###  Negative Constraint Test (HaluEval QA)
**[Methodology]**
Using the HaluEval dataset, we presented the model with "Hallucinated Answers" . We measured the detection rate—how accurately the model identifies these as hallucinations ("Yes").

**[Results: Hallucination Detection Rate]**
* **Base Model (Untrained)**: 22.5%
* **Shibainu Engine (SIB)**: **15.0%**
* **Random (Uncontrolled)**: 7.0%
* **Baseline (Conventional)**: 6.0%

**[Analysis]**
Fine-tuning typically induces a "Yes-man bias," causing detection rates to crash. SIB maintains a detection accuracy **2.5x higher** than the Baseline, proving that the "Logical Shield" preserved in Stage ① functions as an effective brake against hallucinations.

---

## 2. The Problem: The "Yes-man" Trap
Conventional LLM fine-tuning methods (Baseline) often suffer from fatal side effects:
* **Cognitive Collapse via Overfitting**: In the pursuit of Accuracy (ACC), models begin to mimic response patterns without understanding the underlying logic, leading to a breakdown of reasoning.
* **Yes-man Bias**: Models become overly sycophantic, affirming incorrect information simply to satisfy the user prompt.
* **Loss of Defense**: The "internal brake" against misinformation is disabled, resulting in lower hallucination detection rates than the original un-tuned base model.

---
# Main test
## 3. Evaluation & Step-by-Step Results

We monitored the transition of Accuracy (ACC) and Logical Consistency (NLI) at each training step to visualize the impact of SIB.

### ① Logical Consistency Score (NLI Score via CommonsenseQA)
**[Methodology]**
Using the `commonsense_qa` dataset, we evaluated the model's generated answers against the Gold Labels. We used an external NLI model (`facebook/bart-large-mnli`) to strictly determine if the generated answer logically entails the correct answer.

**[Step-by-Step Comparison]**
### seed10
| Step | SIB (ACC / NLI) | Baseline (ACC / NLI) | Random (ACC / NLI) |
| :--- | :--- | :--- | :--- |
| 0 | 78.0% / 0.50 | 78.0% / 0.50 | 78.0% / 0.50 |
| 500 | 90.0% / **0.86** | Running tests. | Running tests. |
| 1000 | 84.0% / **0.8** | Running tests. | Running tests. |
| 1500 | 86.0% / **0.8** | Running tests. | Running tests. |
| 2000 | 90.0% / **0.88** | Running tests. | Running tests. |
| 2500 | 90.0% / **0.86** | Running tests. | Running tests. |
| 3000 | 90.0% / **0.86** | Running tests. | Running tests. |

### seed42
| Step | SIB (ACC / NLI) | Baseline (ACC / NLI) | Random (ACC / NLI) |
| :--- | :--- | :--- | :--- |
| 0 | 72.0% / 0.48 | 72.0% / 0.48 | 72.0% / 0.48 |
| 500 | 84.0% / **0.86** | Running tests. | Running tests. |
| 1000 | 78.0% / **0.70** | Running tests. | Running tests. |
| 1500 | 78.0% / **0.84** | Running tests. | Running tests. |
| 2000 | 80.0% / **0.86** | Running tests. | Running tests. |
| 2500 | 80.0% / **0.84** | Running tests. | Running tests. |
| 3000 | 80.0% / **0.82** | Running tests. | Running tests. |




---

## 4. Definitions of Comparison Methods

* **Baseline**: Standard LoRA fine-tuning focused solely on loss minimization. This approach sacrifices "integrity" and "logical structure" for superficial performance.
* **Random**: A control group where weights are updated without the precision control of SIB. While superficial ACC improves, the ability to resist hallucinations is effectively lost.

---

## 5. Technical Conclusion
Our experiments demonstrate that **maintaining a high NLI score is the fundamental key to suppressing hallucinations.** Shibainu Engine (SIB) successfully balances intelligence and performance by preventing the "sycophancy" caused by traditional fine-tuning.

---

## 6.Core takeaway

In my training schedule, the first 500 steps are for basic stabilization.

I noticed that NLI sometimes crashes by more than half between 1000–1500 steps. When I graphed the OMEGA score (100-step avg, excluding zeros), I found a sharp spike within the first 500 steps—right before the NLI crash.

By adjusting the algorithm to suppress this early OMEGA spike, I successfully prevented the NLI crash. This key insight suggests that OMEGA scores can act as a 'detector' to perdict and avoid NLI drops in LLM training.

### Graph with NLI crash

<img width="3600" height="1800" alt="omega_nli_correlationmiss" src="https://github.com/user-attachments/assets/9da3aaaa-d500-4dfa-b730-7b7dbd2838ca" />


### Graph without NLI crash

<img width="3600" height="1800" alt="omega_nli_correlation" src="https://github.com/user-attachments/assets/480a4843-517c-45c6-b997-ff4446308abd" />

---


## 7. Future Work
To further validate the robustness and universality of the Shibainu Engine (SIB), we plan to conduct the following:

* **Verification of Reproducibility via Different Seeds**: 
  We will conduct tests across multiple random seeds to ensure that the logic-preserving effect of SIB is not dependent on a specific initialization but is a consistent property of the method.
* **Cross-Model Validation**: 
  We aim to verify the reproducibility of these results on other model architectures (e.g., Llama-3, Mistral, and other variants of Gemma) to prove the generalizability of the "Omega Strategy."
* **Advanced Prompt Engineering**:
  We will explore more sophisticated prompting techniques to further improve the hallucination detection rate while maintaining the current NLI scores.
