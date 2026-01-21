# SmolLogic-325M: Syllogistic Reasoning via SFT + GRPO

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Model](https://img.shields.io/badge/Base_Model-SmolLM--325M-green)
![Method](https://img.shields.io/badge/Training-SFT%20%2B%20GRPO-orange)
![Framework](https://img.shields.io/badge/Library-TRL%20%2F%20Unsloth-violet)

**SmolLogic-325M** is a specialized small language model (SLM) fine-tuned to solve syllogistic logic problems. 

This project demonstrates a critical insight in aligning small models: **Reinforcement Learning (RL) requires a base level of competence.** We successfully trained a ~325M parameter model by first using **Supervised Fine-Tuning (SFT)** to escape the "zero-reward" trap, followed by **Group Relative Policy Optimization (GRPO)** to maximize reasoning consistency.

---

## 📖 Overview

Logical reasoning is typically the domain of massive models (70B+). Small models often struggle to maintain the strict structure required for formal logic (Syllogisms).

When attempting to train **SmolLM** directly with Reinforcement Learning, the model failed to generate any valid syllogisms, resulting in **0 rewards** and no learning signal. To fix this, we implemented a two-stage pipeline:

1.  **Warm-up (SFT):** Teach the model the *format* of syllogisms using supervised data.
2.  **Alignment (GRPO):** Optimize the model's reasoning paths using group-based reinforcement learning to favor valid logical conclusions.

---

## 🧠 Methodology

Our training pipeline addresses the "Cold Start" problem in RLHF.

```mermaid
graph LR
    A[Base Model<br>SmolLM-325M] -->|Step 1: SFT| B[SFT Model<br>Valid Syntax]
    B -->|Step 2: GRPO| C[RL Model<br>Robust Logic]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
