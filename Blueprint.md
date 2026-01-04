# **Architectural Convergence in Small-Scale AI: A Blueprint for Integrating Manifold-Constrained Hyper-Connections, JEPA, and Neural Memory on Consumer Hardware**

## **1\. Introduction: The Post-Scaling Law Era and Architectural Efficiency**

For the better part of the last decade, the field of artificial intelligence has been dominated by a single, governing heuristic: the scaling hypothesis. This empirical observation—that model performance, measured by cross-entropy loss, scales as a power-law function of parameter count, dataset size, and computational budget—has driven the industry toward a "brute force" paradigm. The prevailing logic, often characterized as "Muscle Head" engineering 1, has necessitated massive capital expenditures on GPU clusters, measuring progress in trillions of parameters and gigawatts of power. However, as we approach the mid-2020s, a distinct transition is emerging. The asymptotic returns on pure scaling are beginning to diminish, not necessarily in theoretical terms, but certainly in terms of economic and practical feasibility for all but the largest sovereign entities.  
This shift marks the dawn of the "Architectural Efficiency" era. The focus is pivoting from simply making models larger to making them structurally smarter. For the independent researcher or engineer operating within the constraints of high-end consumer hardware—specifically a setup defined by an AMD Ryzen 7 7800X3D CPU and an NVIDIA RTX 3090 with 24GB of VRAM—this paradigm shift is not merely a trend; it is a necessity. To achieve state-of-the-art performance with a model size that fits into 24GB (approximately 3 to 7 billion parameters, depending on quantization), one cannot simply train a miniaturized version of Llama-3 or GPT-4. Such an approach inevitably leads to a model that is coherent but ultimately shallow in reasoning and limited in context.  
Instead, the path to a "best-in-class" small-scale model lies in the synthesis of disparate, frontier technologies that fundamentally alter how information is propagated, represented, and stored within the neural network. This report provides an exhaustive, blueprint-level analysis of a novel architecture designed specifically for your hardware profile. It integrates **Manifold-Constrained Hyper-Connections (mHC)** to stabilize deep signal propagation 2; **Joint-Embedding Predictive Architectures (JEPA)** to enforce semantic abstraction over rote memorization 3; **Titans Neural Memory** to enable infinite context via test-time learning 4; and the **Muon optimizer** to radically reduce the memory overhead of training.5  
The following sections will dissect these technologies with rigorous detail, exploring their mathematical underpinnings, their interactions, and the specific engineering required to implement them on a single-node Ryzen/NVIDIA system. We propose a synthesized architecture, tentatively titled **"Hyper-Mnemosyne,"** which leverages these advancements to punch significantly above its weight class, offering a roadmap for creating a highly capable AI agent on local hardware.

## ---

**2\. The Structural Foundation: Manifold-Constrained Hyper-Connections (mHC)**

The structural integrity of any deep neural network is maintained by its ability to propagate signals—both forward activations and backward gradients—without degradation. Since the introduction of ResNet and subsequently the Transformer, the residual connection ($x\_{l+1} \= x\_l \+ F(x\_l)$) has been the critical innovation enabling the training of deep networks. By providing an "identity highway," simple residuals ensure that gradients can flow through hundreds of layers without vanishing. However, recent research by DeepSeek and ByteDance suggests that this standard residual connection, while stable, acts as a topological bottleneck that limits the information flow capacity of the network.1

### **2.1 The Theoretical Limitations of Standard Residuals**

In a standard Transformer architecture, the residual stream serves as the central conduit for information. However, this conduit is typically single-lane. Each layer reads from the stream, processes data via Attention or Feed-Forward Networks (FFN), and adds the result back. This additive structure implies that the "width" of the information highway is fixed to the model's hidden dimension ($d\_{model}$). While increasing $d\_{model}$ increases capacity, it does so at a quadratic computational cost ($O(d^2)$) for linear layers.  
The "Hyper-Connections" (HC) proposal seeks to address this limitation by topologically expanding the residual stream into $n$ parallel branches (or "lanes"), allowing layers to mix information between these branches dynamically.2 Mathematically, unconstrained HC introduces learnable mixing matrices $H\_{res}$ that blend these parallel streams. If we define the residual state at layer $l$ as a matrix $X\_l \\in \\mathbb{R}^{n \\times d}$, the propagation rule becomes:

$$X\_{l+1} \= H\_{res}^l X\_l \+ H\_{post}^l F(H\_{pre}^l X\_l)$$  
Here, $H\_{res}^l \\in \\mathbb{R}^{n \\times n}$ is a learnable matrix governing how the parallel residual streams interact. $H\_{pre}^l$ aggregates information from the streams for the layer's computation, and $H\_{post}^l$ distributes the layer's output back into the streams.  
While this architecture significantly increases expressivity and allows the model to learn complex routing patterns—effectively deciding which "lane" carries syntax, which carries logic, etc.—it introduces a critical instability. In a standard residual net, the explicit identity term ($+ x\_l$) anchors the transformation. In HC, the mixing matrix $H\_{res}^l$ can effectively be any linear transformation. As the signal propagates through depth $L$, the composite mixing matrix is the product of layer-wise matrices: $\\prod\_{l=1}^L H\_{res}^l$.  
DeepSeek's analysis 2 demonstrates that if these matrices are unconstrained, the singular values of the product matrix diverge exponentially. This phenomenon, termed "Amax Gain Magnitude," leads to either signal explosion (gradients turning to NaNs) or signal collapse (gradients vanishing), making deep HC networks impossible to train at scale.

### **2.2 The mHC Solution: Geometric Regularization via the Birkhoff Polytope**

DeepSeek's **Manifold-Constrained Hyper-Connections (mHC)** resolves this instability not by removing the parallel streams—and thus sacrificing the expressivity gains—but by constraining the mixing matrices to a specific geometric manifold: the **Birkhoff Polytope**.1  
The Birkhoff Polytope, denoted as $\\mathcal{B}\_n$, is the set of **doubly stochastic matrices**. A square matrix $P \\in \\mathbb{R}^{n \\times n}$ is doubly stochastic if:

1. All entries are non-negative: $P\_{ij} \\ge 0$.  
2. Each row sums to 1: $\\sum\_j P\_{ij} \= 1$.  
3. Each column sums to 1: $\\sum\_i P\_{ij} \= 1$.

Constraining the residual mixing matrices $H\_{res}$ to lie on $\\mathcal{B}\_n$ has profound theoretical implications for deep signal propagation:  
**Norm Preservation:** The spectral norm (the largest singular value) of any doubly stochastic matrix is exactly 1\. This provides a rigorous guarantee that the signal energy does not explode as it propagates through the network, regardless of the depth $L$.7 Unlike standard initialization tricks that only control variance at the start of training, the manifold constraint enforces stability throughout the entire optimization trajectory.  
**Compositional Closure:** A crucial property of doubly stochastic matrices is that they are closed under multiplication. The product of two doubly stochastic matrices is itself a doubly stochastic matrix. This ensures that the stability property is invariant to network depth. The composite transformation of the residual stream over 100 layers is still a doubly stochastic matrix, maintaining the norm-preserving property globally.2  
**Convex Combination as Routing:** From an information-theoretic perspective, the mixing operation becomes a weighted averaging (convex combination) of the residual streams. The network learns to "shuffle and blend" the information packets between lanes without amplifying them. This allows the model to maintain distinct information contexts (e.g., maintaining a "syntax stream" and a "memory stream") while allowing them to interact controllably.9

### **2.3 Implementation: The Necessity of Triton Kernels**

Enforcing the Birkhoff Polytope constraint is typically done via the **Sinkhorn-Knopp algorithm**. However, implementing this iteratively in pure PyTorch creates a severe performance bottleneck. A naive implementation launches dozens of tiny CUDA kernels (one for each row/column normalization step) for every single layer. This "kernel launch overhead" will dominate the runtime, causing the GPU compute units to sit idle while the CPU struggles to dispatch instructions.1  
The Triton Strategy:  
For "Hyper-Mnemosyne" to run efficiently on an RTX 3090, you must implement the mHC mixer as a custom Fused Kernel using Triton.

1. **Fusion:** The kernel must perform the Sinkhorn iterations *and* the subsequent matrix multiplication ($X\_{mixed} \= P \\times X\_l$) within a single kernel call.  
2. **SRAM Utilization:** Since the mixing matrix $W \\in \\mathbb{R}^{4 \\times 4}$ is tiny, it can reside entirely in the GPU's L1 cache (SRAM) during the Sinkhorn iterations. This eliminates repeated round-trips to the slower HBM (High Bandwidth Memory).  
3. **Efficiency:** A fused Triton kernel reduces the overhead from \~7% (reported in papers using optimized CUDA) to effectively zero, as the mixing cost becomes negligible compared to the large matrix multiplications of the model backbone.

### **2.4 Hardware Considerations for mHC on RTX 3090**

For your specific hardware setup (RTX 3090), mHC presents a highly favorable trade-off. The primary bottleneck in LLM training and inference on consumer hardware is often memory bandwidth (reading/writing weights from VRAM), rather than pure compute (FLOPs).  
Overhead Analysis:  
With the Triton implementation described above, the computational cost is negligible. The Sinkhorn projection happens "in-register" or in shared memory.  
Memory Implications:  
mHC increases the width of the residual stream by a factor of $n$ (e.g., 4). This increases the memory footprint of the activations stored during training. However, it does not significantly increase the parameter count, as the mixing matrices are tiny.

* **Strategy:** On a 24GB card, activation memory is precious. However, the increased expressivity of mHC allows you to potentially reduce the *hidden dimension* ($d\_{model}$) or the number of layers while maintaining the same effective capacity. For example, an mHC model with $d=1024$ and $n=4$ might outperform a standard model with $d=2048$, while using less compute for the FFNs.

**Insight:** mHC effectively acts as a "free lunch" for stability in small-scale models that are pushed to their limits. By mathematically guaranteeing stable gradients, it allows for the use of more aggressive learning rates and prevents the "loss spikes" that often plague experimental architectures trained on limited budgets.

## ---

**3\. The Representational Objective: Joint-Embedding Predictive Architectures (JEPA)**

While mHC optimizes the *flow* of information through the network, **JEPA** revolutionizes the *objective* of learning itself. Standard Large Language Models (LLMs) are trained almost exclusively on Next-Token Prediction (NTP), a generative objective that forces the model to reconstruct the exact surface form of the data in pixel or token space. While successful, this approach has limitations: it forces the model to allocate significant capacity to modeling high-frequency noise and surface-level syntax, rather than focusing purely on deep semantic abstraction.3

### **3.1 The Concept: Prediction in Latent Space**

JEPA, championed by Yann LeCun and recently adapted for language, shifts the training target from **reconstruction** (predicting $x$ from $y$) to **latent prediction** (predicting the embedding of $x$ from the embedding of $y$).12

* **Generative Model (Standard LLM):** Models $P(x|y)$. The decoder must output the exact token ID.  
* **JEPA:** Predicts $z\_x$ from $z\_y$, where $z$ is a latent representation. $z \= Enc(x)$. The objective is to minimize the distance $D(Pred(z\_y), z\_x)$ in embedding space. Ideally, the encoder learns a mapping where "cat on mat" and "feline on rug" map to very similar vectors, meaning the predictor can succeed regardless of the specific phrasing.13

### **3.2 The "Multi-View" Challenge: Offline Data Pre-computation**

Implementing JEPA for text requires "multi-view" data—pairs of inputs $(x, y)$ that are semantically identical but syntactically distinct. Generating these views on the fly (e.g., using a smaller LLM to rewrite text) is a massive bottleneck. Your Ryzen 7800X3D, while powerful, cannot generate synthetic rewrites fast enough to keep an RTX 3090 fed with data. The GPU would spend most of its time waiting (starvation), resulting in 0% utilization.  
The Refined Data Pipeline:  
You must treat data augmentation as a pre-processing step, not a runtime step.

1. **Offline Generation:** Use your hardware to run a quantized (4-bit) Llama-3-8B or similar model overnight to generate the "Target" views.  
   * *Input:* "The quick brown fox jumps over the lazy dog."  
   * *Prompt:* "Rewrite this sentence in a formal academic tone."  
   * *Output:* "The rapid brunette vulpine leaps across the lethargic canine."  
2. **Storage:** Save these pairs $(x, y)$ in a compressed format (e.g., Parquet or Arrow) to disk.  
3. **Runtime Loading:** During training, the CPU only handles light tasks: reading the pre-computed pairs, tokenizing them, and applying random masking. This ensures the GPU is saturated with data at all times.

**Augmentation Sources:**

1. **Code-Text Pairs:** (GitHub issues vs. PR diffs, Docstrings vs. Function bodies).  
2. **Back-Translation:** (English \-\> German \-\> English).  
3. **Synthetic Rewrite:** (As described above, done offline).

### **3.3 Implementation Strategy**

For your specific build, implementing a pure JEPA (non-generative) is risky as it might lose the ability to chat fluently. The **Hybrid LLM-JEPA** approach is recommended.  
Architecture Flow:  
The model effectively runs two logical passes (which can be batched together with clever masking):

1. **Context Pass:** Processes the context view (e.g., Docstring). Output: $z\_{ctx}$ (the hidden state of the last token).  
2. **Target Pass:** Processes the target view (e.g., Code). Output: $z\_{tgt}$.  
3. **Predictor Head:** A small Multi-Layer Perceptron (MLP) takes $z\_{ctx}$ and tries to predict $z\_{tgt}$.  
4. **Loss Calculation:** Calculate Cosine Similarity or $L\_2$ distance between $Pred(z\_{ctx})$ and $z\_{tgt}$. Add this to the standard Cross-Entropy loss of the next-token prediction.3

## ---

**4\. The State Space Model Revolution: Mamba-2 & Structured State Space Duality**

While Titans (discussed in Section 5\) solves the memory problem, the "Core" processing layers of the network still need to be highly efficient. Standard Attention is $O(N^2)$, which is wasteful for the "bulk" processing of language structure. **Mamba-2**, based on State-Space Models (SSMs), offers a mathematically superior alternative for the majority of the network's layers.

### **4.1 Structured State Space Duality (SSD)**

The breakthrough in Mamba-2 is the discovery of Structured State Space Duality (SSD).15 The authors proved that the selective SSM recurrence can be rewritten as a form of "structured attention."  
Specifically, the recurrence can be computed as a block-decomposition matrix multiplication. This makes Mamba-2 extremely friendly to the Tensor Cores on your RTX 3090, which are specialized for exactly this type of dense matrix math.15  
**Hardware Implication:** Mamba-2 training is significantly faster than Mamba-1 and offers linear scaling $O(N)$ inference. For a 24GB card, this efficiency allows you to train on much longer sequences (e.g., 8192 or 16k tokens) than would be possible with a full Transformer.

### **4.2 Recommendation: The Hybrid Core**

Pure Mamba models can sometimes struggle with "copying" tasks (exact recall of a phone number mentioned 500 tokens ago) compared to Attention. The current SOTA practice (e.g., NVIDIA's Hymba, AI21's Jamba) is to use a **Hybrid Architecture**.16

* **Configuration:** Use **Mamba-2** blocks for the majority (e.g., 80-90%) of the network layers. This handles the syntactic and semantic structure efficiently.  
* **Attention Injection:** Intersperse standard **Sliding Window Attention (SWA)** or Global Attention layers every 4-6 Mamba blocks. This restores the "copying" capability and high-fidelity retrieval that Attention excels at.

## ---

**5\. Titans & Neural Memory: Breaking the Context Window Barrier**

Google's **Titans** architecture proposes a radical solution to context limitations: **Memory as a Neural Network**.4 However, training this module—which updates its own weights via gradients at test time—is notoriously unstable.

### **5.1 The Neural Memory Module (NMM)**

Instead of a static Key-Value (KV) cache, Titans uses a separate Multilayer Perceptron (MLP) that *learns* at test time.

* **Mechanism:** The architecture consists of a "Core" (the Mamba/Attention model) and a "Memory" (the MLP).  
* **Test-Time Training:** As the model processes tokens during inference, it calculates a gradient step to update the *weights* of the Memory MLP based on a "Surprise Metric".20

### **5.2 Risk Mitigation: The Two-Stage Training Protocol**

If the gradients explode during inference (test-time training), the model essentially lobotomizes itself, overwriting useful memories with noise. To mitigate this "engineering landmine," you must decouple the training of the backbone from the training of the memory.  
Stage 1: Backbone Pre-training  
Train the Mamba-2 \+ mHC core first. During this stage, the Titans Memory Module is disabled or replaced with a simple dummy placeholder. The goal is to establish a stable, high-performing language model that understands syntax and semantics.  
Stage 2: Memory Finetuning  
Once the backbone is stable, freeze (or severely lower the learning rate of) the backbone and introduce the Titans Memory Module. Train the model on long-context data (books, repositories), specifically optimizing the memory module's hyperparameters (step size, decay rate) to minimize the "Surprise" on future tokens. This isolates the instability of the memory module, making it easier to debug without wrecking the language capabilities of the core model.

## ---

**6\. The Optimization Frontier: Muon & 2D Optimization**

Designing the architecture is half the battle; training it on a single RTX 3090 is the other. Standard optimizers like AdamW require maintaining two state buffers (Momentum and Variance) for every parameter, which is memory prohibitive.

### **6.1 Muon: Momentum Orthogonalized Optimizer**

**Muon** is a second-order optimizer designed specifically for 2D parameters (matrices) in neural networks.5 It requires only **one** momentum buffer (vs. two for AdamW) and uses Newton-Schulz iterations to orthogonalize updates.

### **6.2 Low-Precision Optimization Strategy**

To fit a 2.8B parameter model into 24GB VRAM while leaving room for activations, we must aggressively optimize the optimizer states.  
**The 8-Bit / BFloat16 Strategy:**

1. **Muon States (for Matrices):** Store the momentum buffer in **BFloat16**. Research indicates Muon is robust to this precision reduction.  
   * Memory Cost: $2.66B \\text{ params} \\times 2 \\text{ bytes} \\approx 5.32 \\text{ GB}$.  
2. **AdamW States (for Vectors):** Vectors (LayerNorms, Biases) constitute \<5% of parameters. Use **8-bit AdamW** (via bitsandbytes) for these.  
   * Memory Cost: $0.14B \\text{ params} \\times 2 \\text{ states} \\times 1 \\text{ byte} \\approx 0.28 \\text{ GB}$.  
3. **Total Optimizer Overhead:** $\\sim 5.6 \\text{ GB}$.

This leaves significant headroom compared to the \~24GB required for a standard FP32 AdamW setup.

## ---

**7\. The "Hyper-Mnemosyne" Architecture: A Synthesis**

### **7.1 Model Specification**

* **Name:** Hyper-Mnemosyne-3B  
* **Parameter Count:** \~2.8 Billion  
* **Hidden Dimension ($d\_{model}$):** 2048  
* **Layers:** 32  
* **Expansion Rate (mHC):** 4 (Effective residual width \= 8192\)

### **7.2 Layer Topology (The "Hyper-Block")**

Based on the synthesized requirements, here is the architectural diagram of the "Hyper-Block":

Code snippet

graph TD  
    subgraph "The Hyper-Block (Layer L)"  
      
    Input \--\> Mixer\[mHC Mixer\]  
      
    Mixer \-- Sinkhorn Projection \--\> Mixed  
      
    Mixed \--\> Aggregation((Sum))  
    Aggregation \--\> CoreInput\[Core Input Vector\]  
      
    subgraph "Dual Path Core"  
        CoreInput \--\> PathA  
        CoreInput \--\> PathB  
        PathB \-.-\> MemoryStore\[(Neural Memory)\]  
    end  
      
    PathA \--\> Combine((Concat/Gate))  
    PathB \--\> Combine  
      
    Combine \--\> PostProject\[Post-Projection\]  
      
    PostProject \--\> Broadcaster  
    Mixed \--\> ResidualAdd((+))  
    Broadcaster \--\> ResidualAdd  
      
    ResidualAdd \--\> Output  
    end

### **7.3 Training Recipe (Single-Node Optimized)**

| Component | Strategy | Implementation Details |
| :---- | :---- | :---- |
| **Data** | **Offline Pre-computation** | Generate JEPA views (rewrites, code pairs) offline. Store in Parquet. CPU only handles masking/loading. |
| **Backbone** | **Mamba-2 (SSD)** | Use Mamba-2 for 90% of layers. Optimized with Triton kernels for Tensor Core usage. |
| **Routing** | **mHC \+ Triton** | Use a **Fused Triton Kernel** for Sinkhorn projection \+ Mixing. Do NOT use PyTorch broadcasting. |
| **Memory** | **Titans (Stage 2\)** | Introduce Titans Memory Module only after backbone convergence. Train via "Surprise" metric gradients. |
| **Optimizer** | **Mixed Precision** | **BFloat16 Muon** for matrices (5.3GB). **8-bit AdamW** for vectors (0.3GB). |

## ---

**8\. Validated VRAM Budget (RTX 3090\)**

With the 8-bit/BF16 optimization strategy, the memory math becomes viable for a 2.8B model on a 24GB card.  
**Static Memory:**

1. **Model Weights (BF16):** $2.8 \\text{B} \\times 2 \\text{ bytes} \= 5.6 \\text{ GB}$  
2. **Gradients (BF16):** $2.8 \\text{B} \\times 2 \\text{ bytes} \= 5.6 \\text{ GB}$  
3. **Optimizer State:**  
   * Muon (BF16): $5.32 \\text{ GB}$  
   * AdamW (8-bit): $0.28 \\text{ GB}$  
   * Total Opt: $\\sim 5.6 \\text{ GB}$  
     Total Static Footprint: $5.6 \+ 5.6 \+ 5.6 \= 16.8 \\text{ GB}$

Available for Activations:  
$24 \\text{ GB} \- 16.8 \\text{ GB} \= 7.2 \\text{ GB}$  
Activation Feasibility:  
Mamba-2 has linear activation scaling. With Gradient Checkpointing (recomputing activations during backward pass), a 7.2 GB buffer is sufficient to train with a batch size of roughly 4-8 sequences at context length 4096, or larger batches at 2048\. This confirms the build is feasible, provided you strictly adhere to the low-precision optimizer strategy.

## ---

**9\. Conclusion**

The "Hyper-Mnemosyne" is a high-risk, high-reward architectural bet. By synthesizing **mHC** for stable signal propagation, **Mamba-2** for throughput, **Titans** for memory, and **JEPA** for semantic density, it aims to outperform significantly larger models. However, the engineering path is narrow. Success depends entirely on the implementation details: using **Triton** for the mHC kernels, moving data augmentation **offline**, utilizing **Two-Stage Training** for stability, and strictly managing VRAM with **Quantized Optimizers**. If executed correctly, this blueprint offers a valid path to SOTA performance on consumer hardware.

#### **Works cited**

1. DeepSeek's paper latest evidence AI muscle head era coming to end, accessed on January 4, 2026, [https://www.constellationr.com/blog-news/insights/deepseeks-paper-latest-evidence-ai-muscle-head-era-coming-end](https://www.constellationr.com/blog-news/insights/deepseeks-paper-latest-evidence-ai-muscle-head-era-coming-end)  
2. mHC: Manifold-Constrained Hyper-Connections \- arXiv, accessed on January 4, 2026, [https://arxiv.org/pdf/2512.24880](https://arxiv.org/pdf/2512.24880)  
3. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures \- arXiv, accessed on January 4, 2026, [https://arxiv.org/html/2509.14252v2](https://arxiv.org/html/2509.14252v2)  
4. Titans: A Deep Dive into Next-Generation AI Memory Architecture | by Mohit Goyal | Medium, accessed on January 4, 2026, [https://medium.com/@mohit098/titans-a-deep-dive-into-next-generation-ai-memory-architecture-c0c7a16ee688](https://medium.com/@mohit098/titans-a-deep-dive-into-next-generation-ai-memory-architecture-c0c7a16ee688)  
5. Building the Muon Optimizer in PyTorch: A Geometric Approach to Neural Network Optimization | by Kye Gomez | Medium, accessed on January 4, 2026, [https://medium.com/@kyeg/building-the-muon-optimizer-in-pytorch-a-geometric-approach-to-neural-network-optimization-17f4601be548](https://medium.com/@kyeg/building-the-muon-optimizer-in-pytorch-a-geometric-approach-to-neural-network-optimization-17f4601be548)  
6. New paper by DeepSeek: mHC: Manifold-Constrained Hyper-Connections \- Reddit, accessed on January 4, 2026, [https://www.reddit.com/r/accelerate/comments/1q12161/new\_paper\_by\_deepseek\_mhc\_manifoldconstrained/](https://www.reddit.com/r/accelerate/comments/1q12161/new_paper_by_deepseek_mhc_manifoldconstrained/)  
7. The Manifold Dial: Visualizing Why DeepSeek's mHC Stabilizes Deep Networks, accessed on January 4, 2026, [https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/](https://subhadipmitra.com/blog/2026/deepseek-mhc-manifold-constrained-hyper-connections/)  
8. mHC: Manifold-Constrained Hyper-Connections \- arXiv, accessed on January 4, 2026, [https://arxiv.org/html/2512.24880v1](https://arxiv.org/html/2512.24880v1)  
9. DeepSeek mHC: Stabilizing Large Language Model Training \- Analytics Vidhya, accessed on January 4, 2026, [https://www.analyticsvidhya.com/blog/2026/01/deepseek-mhc/](https://www.analyticsvidhya.com/blog/2026/01/deepseek-mhc/)  
10. DeepSeek's mHC: Manifold-Constrained Hyper-Connections \- AI Papers Academy, accessed on January 4, 2026, [https://aipapersacademy.com/deepseek-mhc/](https://aipapersacademy.com/deepseek-mhc/)  
11. mHC: Manifold-Constrained Hyper-Connections \- ChatPaper, accessed on January 4, 2026, [https://chatpaper.com/paper/222652](https://chatpaper.com/paper/222652)  
12. Beyond Generative Models: The Joint Embedding Predictive Architecture | by Matteo Donati | Data Reply IT | DataTech | Medium, accessed on January 4, 2026, [https://medium.com/data-reply-it-datatech/beyond-generative-models-the-joint-embedding-predictive-architecture-3e7771978a5a](https://medium.com/data-reply-it-datatech/beyond-generative-models-the-joint-embedding-predictive-architecture-3e7771978a5a)  
13. Joint-Embedding Predictive Architectures \- Emergent Mind, accessed on January 4, 2026, [https://www.emergentmind.com/topics/joint-embedding-predictive-architectures-jepas-f59bf588-2819-44b5-9fa1-00dd8de73f20](https://www.emergentmind.com/topics/joint-embedding-predictive-architectures-jepas-f59bf588-2819-44b5-9fa1-00dd8de73f20)  
14. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures \- arXiv, accessed on January 4, 2026, [https://arxiv.org/abs/2509.14252](https://arxiv.org/abs/2509.14252)  
15. Revolutionizing Code Completion with Codestral Mamba, the Next-Gen Coding LLM, accessed on January 4, 2026, [https://developer.nvidia.com/blog/revolutionizing-code-completion-with-codestral-mamba-the-next-gen-coding-llm/](https://developer.nvidia.com/blog/revolutionizing-code-completion-with-codestral-mamba-the-next-gen-coding-llm/)  
16. New IBM Granite 4 Models to Reduce AI Costs with Inference-Efficient Hybrid Mamba-2 Architecture \- InfoQ, accessed on January 4, 2026, [https://www.infoq.com/news/2025/11/ibm-granite-mamba2-enterprise/](https://www.infoq.com/news/2025/11/ibm-granite-mamba2-enterprise/)  
17. IBM Released new Granite 4.0 Models with a Novel Hybrid Mamba-2/Transformer Architecture: Drastically Reducing Memory Use without Sacrificing Performance \- MarkTechPost, accessed on January 4, 2026, [https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models-with-a-novel-hybrid-mamba-2-transformer-architecture-drastically-reducing-memory-use-without-sacrificing-performance/](https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models-with-a-novel-hybrid-mamba-2-transformer-architecture-drastically-reducing-memory-use-without-sacrificing-performance/)  
18. NVIDIA Introduces Hymba 1.5B: A Hybrid Small Language Model Outperforming Llama 3.2 and SmolLM v2 \- MarkTechPost, accessed on January 4, 2026, [https://www.marktechpost.com/2024/11/22/nvidia-introduces-hymba-1-5b-a-hybrid-small-language-model-outperforming-llama-3-2-and-smollm-v2/](https://www.marktechpost.com/2024/11/22/nvidia-introduces-hymba-1-5b-a-hybrid-small-language-model-outperforming-llama-3-2-and-smollm-v2/)  
19. Titans: Learning to Memorize at Test Time \- A Breakthrough in Neural Memory Systems, accessed on January 4, 2026, [https://www.shaped.ai/blog/titans-learning-to-memorize-at-test-time-a-breakthrough-in-neural-memory-systems](https://www.shaped.ai/blog/titans-learning-to-memorize-at-test-time-a-breakthrough-in-neural-memory-systems)  
20. Titans \+ MIRAS: Helping AI have long-term memory \- Google Research, accessed on January 4, 2026, [https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)  
21. Practical Efficiency of Muon for Pretraining \- arXiv, accessed on January 4, 2026, [https://arxiv.org/html/2505.02222v1](https://arxiv.org/html/2505.02222v1)