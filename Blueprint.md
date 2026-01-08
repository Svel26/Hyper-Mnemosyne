# **Hyper-Mnemosyne: An Experimental Study in Hybrid Architectures**

## **1. Motivation: Exploring Architectural Convergence**

This project explores the intersection of State Space Models, Manifold-Constrained residuals, and Memory-augmented generation at a manageable scale.

Recent advancements in AI architecture—specifically the shift from pure Attention to State Space Models (Mamba), the stabilization of deep networks via geometric constraints (DeepSeek's mHC), and the integration of neural memory (Titans)—present new opportunities for efficient modeling. However, reproducing these results often requires massive compute clusters. "Hyper-Mnemosyne" is an attempt to adapt these concepts into a cohesive, small-scale research prototype (~150M parameters) that can be trained and studied on consumer hardware (e.g., an NVIDIA RTX 3090).

**Disclaimer:** This is an experimental adaptation of these concepts, not an official reproduction. The implementations are simplified for stability and scale.

---

## **2. Structural Foundation: mHC-Inspired Residual Mixing**

### **2.1 The Concept**

DeepSeek's "Manifold-Constrained Hyper-Connections" (mHC) proposes expanding the residual stream into multiple parallel branches to increase information flow capacity [1]. To prevent signal explosion, these branches are mixed via matrices constrained to the Birkhoff Polytope (doubly stochastic matrices).

### **2.2 Adaptation**

In this prototype, we implement a simplified version of mHC:

* **Parallel Branches**: The residual stream is split into 4 parallel "lanes".
* **Mixing**: We use a custom Triton kernel to perform Sinkhorn normalization, ensuring the mixing matrices typically remain stable.
* **Initialization**: Branches are initialized with noise to break symmetry, a critical fix for training dynamics at this scale.

---

## **3. Representational Objective: Auxiliary Latent Consistency (JEPA-Inspired)**

### **3.1 The Concept**

Joint-Embedding Predictive Architectures (JEPA) shift the learning objective from pixel/token reconstruction to latent space prediction [2]. The goal is to learn abstract representations that are invariant to surface-level noise.

### **3.2 Adaptation**

We integrate a "JEPA-inspired" auxiliary loss alongside the standard next-token prediction:

* **Hybrid Objective**: The model predicts the next token (Generative) AND predicts the latent state of a target view from a context view (Discriminative/Latent).
* **Stop-Gradient**: To prevent representation collapse (where all embeddings map to zero), we enforce a gradient stop on the target branch, a key insight from Siamese network literature.

---

## **4. Core Backbone: Mamba-2**

The majority of the network layers utilize **Mamba-2** blocks [3]. Mamba-2 leverages Structured State Space Duality (SSD) to achieve linear-time context scaling, making it significantly more efficient than Transformers for the "heavy lifting" of sequence modeling on consumer GPUs.

---

## **5. Memory Augmentation: Titans-Inspired Gated Residuals**

### **5.1 The Concept**

Google's Titans architecture proposes a "Neural Memory" module that learns at test-time to store user context [4].

### **5.2 Adaptation**

Given the instability of full meta-learning loops at this scale, we have implemented a **Simplified Gated Residual Memory**:

* **Mechanism**: A memory MLP processes the input and adds a gated signal back to the residual stream.
* **Training**: Instead of complex look-ahead meta-gradients, we currently train this module to minimize a simple reconstruction "surprise" loss or end-to-end task loss, serving as a long-term context buffer.

---

## **6. Optimization Strategy**

Training even a small model efficiently requires care. We utilize:

* **Muon Optimizer**: A moment-orthogonalized optimizer for 2D parameters, allowing for memory-efficient training [5].
* **Mixed Precision**: BFloat16 for the bulk of operations to maximize tensor core usage on Ampere GPUs.

---

## **7. Conclusion**

Hyper-Mnemosyne is a playground for architectural ideas. It is not intended to compete with Llama-3 or production models. Instead, it serves as a "petri dish" for verifying how deep signal propagation, state-space efficiency, and latent objectives interact in a constrained environment.

---

#### **References & Inspirations**

1. *DeepSeek-V3 Technical Report* (Concept: Multi-head Latent Attention/Connections). Note: mHC is an experimental interpretation of similar residual scaling ideas.
2. *Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023.* [arXiv](https://arxiv.org/abs/2301.08243)
3. *Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Mamba-2), 2024.* [arXiv](https://arxiv.org/abs/2405.21060)
4. *Behrouz et al., "Titans: Learning to Memorize at Test Time", 2024.* [arXiv](https://arxiv.org/abs/2412.03155)
5. *Shazeer, "Muon: purely momentum-orthogonalized optimization", 2024.* (Reference implementation adapted for PyTorch).
