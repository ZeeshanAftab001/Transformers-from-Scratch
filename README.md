# Transformers From Scratch

A **PyTorch implementation of the Transformer architecture from scratch**, built for learning and understanding how the core components of Transformers work internally.

This repository is **not intended to be a production-ready library**. Instead, it is a hands-on implementation of the fundamental building blocks introduced in the original Transformer architecture.

## 📌 Overview

Transformers are the foundation of many modern AI systems, including Large Language Models (LLMs), machine translation systems, and various sequence modeling architectures.

This project implements the core Transformer components using **Python and PyTorch**, without relying on high-level Transformer implementations.

The goal is to understand what happens inside a Transformer rather than simply using an existing implementation.

## 🧠 Components Implemented

The repository currently contains implementations of the following components:

```text
Transformers-from-Scratch/
│
├── feed_forward.py
├── input_embeddings.py
├── layer_norm.py
├── model.py
├── multi_head_attention.py
├── positional_encoding.py
└── README.md
```

### 1. Input Embeddings

`input_embeddings.py`

Converts token IDs into dense vector representations that can be processed by the Transformer.

```text
Token IDs
   ↓
Embedding Layer
   ↓
Dense Vectors
```

### 2. Positional Encoding

`positional_encoding.py`

Adds positional information to token embeddings.

Because Transformers process tokens in parallel rather than sequentially, positional information is required to tell the model where each token occurs in a sequence.

```text
Token Embeddings
       +
Positional Encoding
       ↓
Transformer Input
```

### 3. Multi-Head Attention

`multi_head_attention.py`

Implements the **Multi-Head Self-Attention** mechanism.

The attention mechanism allows the model to determine which tokens in a sequence are important to each other.

The implementation follows the core idea:

```text
Attention(Q, K, V)
    = softmax(QKᵀ / √dₖ)V
```

Multi-head attention allows the model to learn different relationships between tokens through multiple attention heads.

### 4. Feed-Forward Network

`feed_forward.py`

Implements the position-wise feed-forward network used inside Transformer blocks.

Conceptually:

```text
Input
  ↓
Linear Layer
  ↓
Activation
  ↓
Dropout
  ↓
Linear Layer
  ↓
Output
```

### 5. Layer Normalization

`layer_norm.py`

Implements Layer Normalization, which helps stabilize the activations during training.

The implementation is written manually to understand how normalization works rather than relying entirely on a pre-built Transformer module.

### 6. Transformer Model

`model.py`

Combines the implemented components into the overall Transformer architecture.

The general flow is:

```text
Input Tokens
     ↓
Input Embeddings
     ↓
Positional Encoding
     ↓
Multi-Head Attention
     ↓
Add & Norm
     ↓
Feed Forward Network
     ↓
Add & Norm
     ↓
Output
```

## 🏗️ Architecture

The implementation is based on the architecture introduced in the original Transformer paper:

> **Attention Is All You Need**
> Vaswani et al., 2017

The core architecture can be summarized as:

```text
                 Transformer
                      │
              Input Embeddings
                      │
              Positional Encoding
                      │
              ┌───────┴───────┐
              │               │
              ▼               │
       Multi-Head Attention   │
              │               │
          Add & Norm          │
              │               │
              ▼               │
       Feed Forward Network   │
              │               │
          Add & Norm          │
              │               │
              ▼               │
             Output           │
```

## 🛠️ Technologies

* **Python**
* **PyTorch**
* **Deep Learning**
* **Transformers**
* **Attention Mechanisms**

## 🎯 Purpose

The main purpose of this project is to understand the internal mathematics and implementation of Transformer models.

Instead of using something like:

```python
from transformers import ...
```

the goal is to implement the fundamental components manually using PyTorch.

This project helps explore concepts such as:

* Token embeddings
* Positional encoding
* Query, Key, and Value
* Self-attention
* Multi-head attention
* Scaling in attention
* Softmax attention weights
* Feed-forward networks
* Layer normalization
* Residual connections
* Transformer architecture

## 📚 Learning Reference

The implementation is primarily based on the original Transformer paper:

**Attention Is All You Need**

Vaswani et al., 2017.

The paper introduced the Transformer architecture and the self-attention mechanism that became the foundation for many modern NLP and generative AI models.

## 🚧 Project Status

This repository is currently a **work in progress**.

The implementation is being developed incrementally as I study and understand the internal components of Transformer architectures.

Planned improvements may include:

* [ ] Complete Transformer encoder
* [ ] Transformer decoder
* [ ] Masked self-attention
* [ ] Cross-attention
* [ ] Encoder-decoder architecture
* [ ] Training example
* [ ] Small language-model experiment
* [ ] Text generation
* [ ] Attention visualization
* [ ] Unit tests
* [ ] More detailed mathematical explanations

## 💡 Why From Scratch?

Modern frameworks make it extremely easy to use Transformer models, but abstraction can hide the mathematics and architecture underneath.

This project is an attempt to go one level deeper:

```text
Using Transformers
       ↓
Understanding Transformers
       ↓
Implementing Transformers
       ↓
Building with Transformers
```

The objective is not to create another Transformer library, but to **learn how Transformers actually work by implementing their core components myself**.

## 📁 Project Structure

| File                      | Description                        |
| ------------------------- | ---------------------------------- |
| `input_embeddings.py`     | Token embedding implementation     |
| `positional_encoding.py`  | Positional encoding implementation |
| `multi_head_attention.py` | Multi-head self-attention          |
| `feed_forward.py`         | Transformer feed-forward network   |
| `layer_norm.py`           | Layer normalization                |
| `model.py`                | Main Transformer model             |

## ⚠️ Disclaimer

This repository is primarily an **educational implementation**.

It is not intended to compete with optimized Transformer frameworks or production implementations such as PyTorch's built-in modules or Hugging Face Transformers.

The code prioritizes **clarity and understanding over optimization and production readiness**.

## ⭐ Future Goal

The long-term goal of this project is to progress from understanding the original Transformer architecture to implementing and experimenting with more advanced architectures used in modern AI systems.

---

**Built with PyTorch while learning Transformers from the ground up.**
