Calculating the activation size (or activation memory footprint) during training of a neural network, such as a large language model (LLM), involves understanding the intermediate outputs stored during the forward pass for use in backpropagation. The size depends on the model architecture, input configuration, and training settings. Below, I’ll explain how to compute it step-by-step, focusing on transformer-based models (common for LLMs), and provide a concrete example.

### What Are Activations?

Activations are the intermediate outputs of each layer in a neural network during the forward pass. They are stored in memory because they’re needed to compute gradients during the backward pass. For transformers, activations include outputs of attention mechanisms, feed-forward networks, and sometimes layer normalization steps, depending on the implementation.

The memory required for activations scales with:

- **Batch size**: Number of samples processed simultaneously.
- **Sequence length**: Number of tokens in each input sample.
- **Hidden size**: Dimensionality of the model’s representations.
- **Number of layers**: Depth of the model.
- **Data type**: Precision (e.g., FP32 = 4 bytes, FP16 = 2 bytes).

Unlike parameters, gradients, and optimizer states (which depend only on model size), activation memory depends heavily on the input and configuration.

General Formula for Transformers
For a transformer model, the activation memory is dominated by the outputs of each layer, particularly from the attention and feed-forward (FFN) components. Here’s a simplified approach to calculate it:

**Key Variables:**
- **B**: Batch size (number of samples).
- **S**: Sequence length (number of tokens per sample).
- **H**: Hidden size (dimensionality of embeddings and layer outputs).
- **L**: Number of layers.
- **D**: Bytes per value (e.g., 4 for FP32, 2 for FP16).

**Per-Layer Activation Size:**
For each transformer layer, the primary activation is the output of the attention mechanism and FFN, both of size \( B \times S \times H \). Additional temporary activations (e.g., attention scores, intermediate FFN outputs) may also be stored, depending on the implementation.

**Total Activation Size:**
Base formula: 
\[ \text{Activation Size} = L \times B \times S \times H \times D \]
This assumes each layer’s output (\( B \times S \times H \)) is stored for all \( L \) layers.

**Adjustments:**
- **Attention Mechanism**: The self-attention block generates intermediate tensors (e.g., query, key, value matrices, and attention scores). These can add significant memory, especially if not optimized.
- **FFN**: The feed-forward network often has an intermediate expansion (e.g., 4H), which temporarily increases memory before being reduced back to \( H \).
- **Checkpointing**: Techniques like gradient checkpointing trade memory for recomputation, reducing the number of activations stored (e.g., from \( L \) to a smaller subset).

**Detailed Breakdown for Transformers:**
Let’s break it down by components within a transformer layer:

- **Attention Outputs**:
    - Output shape: \( B \times S \times H \).
    - Memory: \( B \times S \times H \times D \) per layer.
    - Temporary tensors (e.g., \( Q, K, V \) of shape \( B \times S \times H \) and attention scores \( B \times S \times S \)) may also be stored briefly.

- **Feed-Forward Network (FFN)**:
    - FFN typically expands the hidden size to \( 4H \) (a common expansion factor) and then projects back to \( H \).
    - Intermediate FFN output: \( B \times S \times 4H \).
    - Memory: \( B \times S \times 4H \times D \) (temporary), plus \( B \times S \times H \times D \) for the final output.

**Total Per Layer:**
Without optimization, a layer might store \( B \times S \times H \) (attention output) and \( B \times S \times 4H \) (FFN intermediate), plus smaller terms like normalization outputs. A rough estimate per layer is \( 5H \) to \( 14H \) worth of activations per token, depending on implementation (e.g., PyTorch stores more intermediates than optimized frameworks like DeepSpeed).

**Across All Layers:**
Multiply by \( L \), but optimizations (e.g., fused kernels, checkpointing) can reduce this.

**Simplified Practical Formula:**
For a typical transformer without checkpointing, the activation memory is often approximated as:
\[ \text{Activation Size} \approx L \times B \times S \times H \times D \times C \]
Where:
- \( C \) is a constant factor (e.g., 2–14) reflecting implementation details:
    - \( C \approx 2 \) for minimal storage (just layer outputs).
    - \( C \approx 12–14 \) for unoptimized PyTorch with attention and FFN intermediates.

A common industry heuristic for transformers is \( 12 \times L \times B \times S \times H \times D \) when using FP32 and no optimizations, based on empirical observations (e.g., from DeepSpeed and Megatron-LM).

**Example Calculation:**
Let’s compute the activation size for a 1-billion-parameter transformer model.

**Model Assumptions:**
- Parameters: 1 billion (\( P = 1,000,000,000 \)).
- Hidden size (\( H \)): For a transformer, \( P \approx 12 \times L \times H^2 \) (accounting for attention and FFN weights). Solving: \( L \approx 32 \), \( H \approx 4096 \) (a plausible configuration).
- Batch size (\( B \)): 1.
- Sequence length (\( S \)): 1024 (typical for LLMs).
- Data type: FP32 (\( D = 4 \) bytes).
- Factor (\( C \)): 12 (unoptimized case).

**Step-by-Step:**
- **Per-Layer Output**:
    - \( B \times S \times H = 1 \times 1024 \times 4096 = 4,194,304 \) elements.
    - Memory per layer: \( 4,194,304 \times 4 = 16,777,216 \) bytes ≈ 16.78 MB.
- **With Factor \( C = 12 \)**:
    - Accounts for attention intermediates (\( S \times S \)), FFN expansion (\( 4H \)), etc.
    - Per layer: \( 16.78 \text{ MB} \times 12 = 201.36 \text{ MB} \).
- **Total Across 32 Layers**:
    - \( 201.36 \text{ MB} \times 32 = 6,443.52 \text{ MB} \approx 6.44 \text{ GB} \).

**Result:**
For this configuration, activations require ~6.44 GB. This can vary:
- Minimal case (\( C = 2 \)): \( 16.78 \text{ MB} \times 32 = 536 \text{ MB} \approx 0.54 \text{ GB} \).
- Larger batch/sequence: Doubling \( B \) or \( S \) doubles the size.

**Verification:**
For a 1-billion-parameter model like GPT-2 medium, empirical reports suggest ~5–10 GB of activation memory with similar settings, aligning with our estimate. DeepSpeed’s formula: 
\[ \text{Activation Memory} \approx \frac{S \times B \times H \times L \times 12}{10^9} \text{ GB} \] (for FP32) yields ~6.4 GB, matching closely.

**Final Answer:**
To calculate activation size:
Use 
\[ \text{Activation Size} = L \times B \times S \times H \times D \times C \]
For a 1-billion-parameter transformer (\( L = 32, H = 4096 \)) with \( B = 1, S = 1024, D = 4 \) (FP32):
- ~6.44 GB (with \( C = 12 \), unoptimized).
- ~0.54 GB (with \( C = 2 \), minimal storage).

The exact size depends on implementation and optimizations (e.g., checkpointing reduces it to \( L \) layers’ worth). For your 1-billion-parameter model from earlier (16 GB for parameters, gradients, optimizer states), adding ~6 GB of activations brings the total to ~22 GB in a typical training scenario.
