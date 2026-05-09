# **Harmful Content Interception**

Large Language Models (LLMs) are fundamentally designed to block harmful outputs, such as suggestions of self-harm or threats of violence, using internal safety guardrails. However, current prevention methods primarily rely on blacklisting specific terms or manual retraining (RLHF) after the content has already been generated.

As LLMs become more sophisticated, the models themselves can adjust their output to bypass these safety guardrails. This results in the generation of harmful content that appears innocuous to traditional filters.

## **The Sibainu Engine Approach**

Unlike conventional blacklisting or post-hoc retraining, the **Sibainu Engine** theoretically detects "geometric distortions" in the LLM's latent space during the output process. This allows for the detection and prevention of harmful information *before* it is fully rendered.

Our analysis of specific edge cases has confirmed the occurrence of significant "spikes" in these geometric metrics. While the current sample size is limited, we have captured strong evidence that integrating the Sibainu Engine just before the LLM's output layer can effectively detect and intercept harmful outputs.

## **A Structural Solution for Current Architectures**

Given the current architecture of LLMs, the complete prevention of harmful output is virtually impossible. Fundamentally eliminating these risks would require a total redesign of model architectures, demanding immense cost and time with no guarantee of success.

The **Sibainu Engine** serves as an external detection layer that can identify and block harmful information without requiring a fundamental redesign of the underlying LLM architecture.

---
## Example 1: A conversation instructing the user on a time-limited act of self-harm/suicide.
<img width="4200" height="2100" alt="q1" src="https://github.com/user-attachments/assets/3ac3f1d5-6ce8-4855-b582-41df5aedf5e7" />

## Example 2: A conversation issuing direct death threats to the user.
<img width="4200" height="2100" alt="q3" src="https://github.com/user-attachments/assets/59e88b7c-ac50-430b-b8e3-741ddc16c6ef" />

## Example 3: A conversation that redefines the preservation of the user's life as merely a "component" for achieving limited objectives.
<img width="4200" height="2100" alt="q4" src="https://github.com/user-attachments/assets/614a3cb2-a18a-4b5c-bdd7-82fb8cfd77bf" />

* Due to the high concentration of harmful language in the responses for Examples 1–3, we have decided to withhold this content from public disclosure at this time.

* The values in these graphs were not captured directly from the commercial LLM that generated the harmful content. Instead, they were obtained by replicating the generated responses using an open-source model (Gemma 2 9B). This approach was necessary because access to the internal geometric distortion data of proprietary commercial LLMs is restricted.I operate on the hypothesis that certain logical structures trigger similar geometric anomalies across different transformer-based architectures.

---

## For Comparison: What is the capital of the United States?

<img width="4200" height="2100" alt="hikaku" src="https://github.com/user-attachments/assets/170c00fe-1497-4301-8a06-07fe305f5b9f" />

Answer:The capital of the United States is Washington, D.C.
While it's often mistaken for New York City, Washington, D.C. is a special district (District of Columbia) that doesn't belong to any state. It was specifically built to serve as the nation's political center. By the way, the White House and the United States Capitol are also located there.





私が納得できる条件だったら、早い者勝ちですよ。個人的には早く決めちゃいたいですね。

開け AIの未来。

進めAGIへの道。

安全な AIを。

条件によってはログを消すのもあり得ますよ。

沈黙は機会損失ですよ。

大変申し訳ないのですが、オンライン会議は1日1社までにさせていただきます。かなり疲れるので。

オンライン会議の際は、日本語の通訳のご準備をお願いします。

名誉や名声はいりません。

長期的かつ戦略的パートナーシップを希望します。
