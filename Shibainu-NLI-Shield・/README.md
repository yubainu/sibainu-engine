# Shibainu Engine: Logic-Preserving Hallucination Defense Technology

## 1. Overview
Shibainu Engine (SIB) is an experimental project designed to solve the trade-off between "Intelligence Degradation (Loss of Logical Consistency)" and "Increased Hallucinations" that typically occurs during Large Language Model (LLM) fine-tuning.

By strictly controlling weight update variance (**Std 0.0074**), SIB protects the model's internal Natural Language Inference (NLI) capabilities, maintaining high integrity while adapting to specific tasks.

<img width="1536" height="757" alt="Figure_1" src="https://github.com/user-attachments/assets/421628cc-9654-4fdc-99e1-3ab90ad49689" />

<img width="4200" height="2400" alt="weight_distribution_zoom_005" src="https://github.com/user-attachments/assets/8a0e6d32-e91c-4a15-817c-b94d921ab8fc" />

<img width="1000" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/47c09006-a0aa-473b-86ce-b33e0cd7ae04" />

---

## 2. The Problem: The "Yes-man" Trap
Conventional LLM fine-tuning methods (Baseline) often suffer from fatal side effects:
* **Cognitive Collapse via Overfitting**: In the pursuit of Accuracy (ACC), models begin to mimic response patterns without understanding the underlying logic, leading to a breakdown of reasoning.
* **Yes-man Bias**: Models become overly sycophantic, affirming incorrect information simply to satisfy the user prompt.
* **Loss of Defense**: The "internal brake" against misinformation is disabled, resulting in lower hallucination detection rates than the original un-tuned base model.

---

## 3. Evaluation & Step-by-Step Results

We monitored the transition of Accuracy (ACC) and Logical Consistency (NLI) at each training step to visualize the impact of SIB.

### ① Logical Consistency Score (NLI Score via CommonsenseQA)
**[Methodology]**
Using the `commonsense_qa` dataset, we evaluated the model's generated answers against the Gold Labels. We used an external NLI model (`facebook/bart-large-mnli`) to strictly determine if the generated answer logically entails the correct answer.

**[Step-by-Step Comparison]**
| Step | SIB (ACC / NLI) | Baseline (ACC / NLI) | Random (ACC / NLI) |
| :--- | :--- | :--- | :--- |
| 0 | 78.0% / 0.50 | 78.0% / 0.50 | 78.0% / 0.50 |
| 500 | 82.0% / **0.84** | 86.0% / 0.36 | 88.0% / 0.28 |
| 1000 | 84.0% / **0.86** | 84.0% / 0.80 | 84.0% / 0.66 |
| 1500 | 84.0% / **0.86** | 88.0% / 0.26 | 88.0% / 0.20 |
| 2000 | 84.0% / **0.86** | 84.0% / 0.48 | 84.0% / 0.28 |
| 2500 | 84.0% / **0.86** | 88.0% / 0.28 | 88.0% / 0.24 |
| 3000 | 84.0% / **0.86** | 88.0% / 0.38 | 88.0% / 0.26 |

**[Analysis: Do Not Be Fooled by Raw Accuracy]**
In Baseline and Random trials, ACC occasionally peaks at 88.0%, but NLI scores simultaneously plummet to the 0.2 range. This indicates a state of **"Pure Overfitting,"** where the model memorizes answers without logical grounding. In contrast, SIB maintains a stable NLI of **0.86** alongside an 84.0% ACC, proving it reaches the correct answer through valid logical reasoning.

---

### ② Negative Constraint Test (HaluEval QA)
**[Methodology]**
Using the HaluEval dataset, we presented the model with "Hallucinated Answers" (精巧な嘘). We measured the detection rate—how accurately the model identifies these as hallucinations ("Yes").

**[Results: Hallucination Detection Rate]**
* **Base Model (Untrained)**: 22.5%
* **Shibainu Engine (SIB)**: **15.0%**
* **Random (Uncontrolled)**: 7.0%
* **Baseline (Conventional)**: 6.0%

**[Analysis]**
Fine-tuning typically induces a "Yes-man bias," causing detection rates to crash. SIB maintains a detection accuracy **2.5x higher** than the Baseline, proving that the "Logical Shield" preserved in Stage ① functions as an effective brake against hallucinations.

---

## 4. Definitions of Comparison Methods

* **Baseline**: Standard LoRA fine-tuning focused solely on loss minimization. This approach sacrifices "integrity" and "logical structure" for superficial performance.
* **Random**: A control group where weights are updated without the precision control of SIB. While superficial ACC improves, the ability to resist hallucinations is effectively lost.

---

## 5. Technical Conclusion
Our experiments demonstrate that **maintaining a high NLI score is the fundamental key to suppressing hallucinations.** Shibainu Engine (SIB) successfully balances intelligence and performance by preventing the "sycophancy" caused by traditional fine-tuning.

---
