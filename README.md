# NOVA-BERT

**N**on-invasive Self-**A**ttention for Side Information Fusion in Sequential Recommendation

[![Paper](https://img.shields.io/badge/arXiv-2103.03578-b31b1b.svg)](https://arxiv.org/abs/2103.03578)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [NOVA Algorithm](#nova-algorithm)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [NOVA Model Usage](#nova-model-usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Introduction

NOVA-BERT implements the **NOVA (NOninVasive self-Attention)** mechanism for sequential recommendation, as proposed in:

> **Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation**
>
> Chang Liu, Xiaoguang Li, Guohao Cai, Zhenhua Dong, Hong Zhu, Lifeng Shang
>
> arXiv:2103.03578 (2021) | [Paper](https://arxiv.org/abs/2103.03578)

### The Problem

Traditional approaches to incorporating side information (e.g., item categories, brands, tags) into sequential recommendation models typically **directly fuse** this information into item embeddings:

```
item_embedding = item_embedding + category_embedding  # Direct fusion
```

This causes **information overwhelming** - the side information can dominate or distort the learned item representations, leading to suboptimal performance.

### The NOVA Solution

NOVA takes a fundamentally different approach: instead of modifying item embeddings, it uses side information to **improve the attention distribution**. This is **non-invasive** because:

1. Item embeddings remain unchanged
2. Side information only affects how items attend to each other
3. The model learns better attention patterns using auxiliary signals

## NOVA Algorithm

### Mathematical Formulation

#### Standard Self-Attention (BERT4Rec)

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

where:
  Q = X × W_Q    (Query projection)
  K = X × W_K    (Key projection)  
  V = X × W_V    (Value projection)
  X = item embeddings
  d = dimension per head
```

#### NOVA Self-Attention

```
NOVA_Attention(Q, K, V, Q_s, K_s) = softmax(
    Q × K^T / √d + λ × Q_s × K_s^T / √d
) × V

where:
  Q, K, V = projections from item embeddings (same as standard)
  Q_s = S × W_Qs   (Query projection from side info)
  K_s = S × W_Ks   (Key projection from side info)
  S = side information embeddings
  λ = side_info_weight (controls influence, default 0.5)
  
Key insight: V still comes from item embeddings only!
```

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    STANDARD ATTENTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Item Embeddings ──┬── Q ──┐                                   │
│                     ├── K ──┼── Attention Scores ── Softmax ──┐ │
│                     └── V ──┘                                  │ │
│                                                          Output │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    NOVA ATTENTION                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Item Embeddings ──┬── Q ──┐                                   │
│                     ├── K ──┼── Item Attention ──┐              │
│                     └── V ───────────────────────┼───┐          │
│                                                  │   │          │
│   Side Info ────────┬── Q_s ─┐                   │   │          │
│                     └── K_s ─┼── Side Attention ─┘   │          │
│                              │         ↓             │          │
│                              │    Combined Score     │          │
│                              │         ↓             │          │
│                              │      Softmax ─────────┼── Output │
│                                                      │          │
│   Note: V comes ONLY from item embeddings (non-invasive!)       │
└─────────────────────────────────────────────────────────────────┘
```

### Why Non-Invasive Works Better

| Approach | Method | Problem |
|----------|--------|---------|
| **Direct Fusion** | `emb = item_emb + side_emb` | Side info can overwhelm item signal |
| **Concatenation** | `emb = concat(item_emb, side_emb)` | Increases dimension, more parameters |
| **Gating** | `emb = gate * item_emb + (1-gate) * side_emb` | Still modifies item representation |
| **NOVA** | Side info affects attention only | Item embeddings preserved, better generalization |

## Installation

### Requirements

- Python 2.7+ or Python 3.6+
- TensorFlow 1.12+ (GPU recommended)
- CUDA compatible with TensorFlow
- NumPy, Six

### Install

```bash
git clone https://github.com/chenxingqiang/NOVA-BERT.git
cd NOVA-BERT
pip install -r requirements.txt
```

## Quick Start

### 1. Standard Mode (BERT4Rec)

```bash
./run_ml-1m.sh
```

### 2. NOVA Mode (with side information)

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/ml-1m.train.tfrecord \
    --test_input_file=./data/ml-1m.test.tfrecord \
    --vocab_filename=./data/ml-1m.vocab \
    --user_history_filename=./data/ml-1m.his \
    --checkpointDir=./checkpoints/ml-1m-nova \
    --signature=-nova \
    --bert_config_file=./bert_train/bert_config_ml-1m_64.json \
    --do_train=True \
    --do_eval=True \
    --batch_size=256 \
    --max_seq_length=200 \
    --num_train_steps=400000 \
    --learning_rate=1e-4 \
    --use_nova=True \
    --use_side_info=True \
    --side_info_vocab_size=100 \
    --side_info_weight=0.5
```

## NOVA Model Usage

### Python API

#### 1. Create NOVA Configuration

```python
import modeling

# Method 1: Create programmatically
config = modeling.NOVABertConfig(
    vocab_size=3420,                    # Number of items
    hidden_size=64,                     # Hidden dimension
    num_hidden_layers=2,                # Number of transformer layers
    num_attention_heads=2,              # Number of attention heads
    intermediate_size=256,              # Feed-forward dimension
    hidden_act="gelu",                  # Activation function
    hidden_dropout_prob=0.2,            # Hidden layer dropout
    attention_probs_dropout_prob=0.2,   # Attention dropout
    max_position_embeddings=200,        # Max sequence length
    # NOVA-specific parameters
    use_side_info=True,                 # Enable side information
    side_info_vocab_size=100,           # Number of categories/tags
    side_info_embedding_size=64,        # Side info embedding dim
    side_info_weight=0.5                # λ: side info influence weight
)

# Method 2: Load from JSON file
config = modeling.NOVABertConfig.from_json_file("bert_config.json")
```

#### 2. Create NOVA Model

```python
import tensorflow as tf

# Prepare inputs
batch_size = 32
seq_length = 200

input_ids = tf.placeholder(tf.int32, [batch_size, seq_length])      # Item IDs
input_mask = tf.placeholder(tf.int32, [batch_size, seq_length])     # Attention mask
side_info_ids = tf.placeholder(tf.int32, [batch_size, seq_length])  # Category IDs

# Create NOVA model
model = modeling.NOVABertModel(
    config=config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    side_info_ids=side_info_ids,        # Side information (optional)
    use_one_hot_embeddings=False
)

# Get outputs
sequence_output = model.get_sequence_output()   # [batch, seq, hidden]
embedding_table = model.get_embedding_table()   # [vocab_size, hidden]
side_info_emb = model.get_side_info_embedding() # [batch, seq, hidden] or None
```

#### 3. Use NOVA Attention Layer Directly

```python
# For custom architectures
context = modeling.nova_attention_layer(
    from_tensor=query_tensor,           # [batch*seq, hidden]
    to_tensor=key_value_tensor,         # [batch*seq, hidden]
    side_info_tensor=side_embeddings,   # [batch*seq, side_hidden] (optional)
    attention_mask=mask,                # [batch, seq, seq]
    num_attention_heads=2,
    size_per_head=32,
    attention_probs_dropout_prob=0.1,
    side_info_weight=0.5,               # λ parameter
    do_return_2d_tensor=True
)
```

#### 4. Use NOVA Transformer

```python
# Full NOVA transformer stack
outputs = modeling.nova_transformer_model(
    input_tensor=embeddings,            # [batch, seq, hidden]
    side_info_tensor=side_embeddings,   # [batch, seq, hidden] (optional)
    attention_mask=mask,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    side_info_weight=0.5,
    do_return_all_layers=True
)
```

## API Reference

### Classes

#### `NOVABertConfig`

Configuration class for NOVA-BERT model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | required | Item vocabulary size |
| `hidden_size` | int | 768 | Hidden layer dimension |
| `num_hidden_layers` | int | 12 | Number of transformer layers |
| `num_attention_heads` | int | 12 | Number of attention heads |
| `intermediate_size` | int | 3072 | Feed-forward layer size |
| `hidden_act` | str | "gelu" | Activation function |
| `hidden_dropout_prob` | float | 0.1 | Hidden layer dropout |
| `attention_probs_dropout_prob` | float | 0.1 | Attention dropout |
| `max_position_embeddings` | int | 512 | Maximum sequence length |
| `use_side_info` | bool | False | Enable side information |
| `side_info_vocab_size` | int | 0 | Side info vocabulary size |
| `side_info_embedding_size` | int | 64 | Side info embedding dimension |
| `side_info_weight` | float | 0.5 | Weight (λ) for side info attention |

#### `NOVABertModel`

NOVA-enhanced BERT model for sequential recommendation.

| Method | Returns | Description |
|--------|---------|-------------|
| `get_sequence_output()` | Tensor [B, S, H] | Final hidden states |
| `get_all_encoder_layers()` | List of Tensors | All layer outputs |
| `get_embedding_output()` | Tensor [B, S, H] | Input embeddings |
| `get_embedding_table()` | Tensor [V, H] | Item embedding table |
| `get_side_info_embedding()` | Tensor or None | Side info embeddings |

### Functions

#### `nova_attention_layer`

```python
nova_attention_layer(
    from_tensor,                    # Query source tensor
    to_tensor,                      # Key/Value source tensor
    side_info_tensor=None,          # Side information tensor (optional)
    attention_mask=None,            # Attention mask
    num_attention_heads=1,          # Number of heads
    size_per_head=512,              # Dimension per head
    query_act=None,                 # Query activation
    key_act=None,                   # Key activation
    value_act=None,                 # Value activation
    attention_probs_dropout_prob=0.0,
    initializer_range=0.02,
    do_return_2d_tensor=False,
    batch_size=None,
    from_seq_length=None,
    to_seq_length=None,
    side_info_weight=0.5            # λ for side info influence
)
```

#### `nova_transformer_model`

```python
nova_transformer_model(
    input_tensor,                   # Input embeddings [B, S, H]
    side_info_tensor=None,          # Side info embeddings [B, S, H]
    attention_mask=None,            # Attention mask [B, S, S]
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    intermediate_act_fn=gelu,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    do_return_all_layers=False,
    side_info_weight=0.5            # λ for side info influence
)
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_nova` | bool | False | Enable NOVA model |
| `--use_side_info` | bool | False | Use side information in attention |
| `--side_info_vocab_size` | int | 0 | Size of side info vocabulary |
| `--side_info_embedding_size` | int | 64 | Side info embedding dimension |
| `--side_info_weight` | float | 0.5 | Weight (λ) for side info attention |

## Configuration

### Standard BERT4Rec Config

```json
{
  "vocab_size": 3420,
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "intermediate_size": 256,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "attention_probs_dropout_prob": 0.2,
  "max_position_embeddings": 200,
  "type_vocab_size": 2,
  "initializer_range": 0.02
}
```

### NOVA Config (with side information)

```json
{
  "vocab_size": 3420,
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "intermediate_size": 256,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "attention_probs_dropout_prob": 0.2,
  "max_position_embeddings": 200,
  "type_vocab_size": 2,
  "initializer_range": 0.02,
  "use_side_info": true,
  "side_info_vocab_size": 100,
  "side_info_embedding_size": 64,
  "side_info_weight": 0.5
}
```

### Hyperparameter Guidelines

| Parameter | Recommendation | Notes |
|-----------|----------------|-------|
| `side_info_weight` | 0.3 - 0.7 | Start with 0.5, tune based on validation |
| `side_info_embedding_size` | 32 - 128 | Usually same as hidden_size or smaller |
| `side_info_vocab_size` | Depends on data | Number of unique categories/tags |

## Datasets

### Supported Datasets

| Dataset | Items | Users | Interactions | File |
|---------|-------|-------|--------------|------|
| ML-1M | 3,416 | 6,040 | 999,611 | `data/ml-1m.txt` |
| ML-20M | 26,744 | 138,493 | 20,000,263 | `data/ml-20m.zip` |
| Beauty | 12,101 | 22,363 | 198,502 | `data/beauty.txt` |
| Steam | 13,047 | 334,730 | 3,693,591 | `data/steam.txt` |

### Data Format

#### Basic Format (user-item interactions)

```
user_id item_id
user_id item_id
...
```

Each line represents one interaction, sorted by timestamp per user.

#### Side Information Format

Create an additional mapping file for side information:

```
item_id category_id
1 5
2 3
3 5
...
```

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain at K |
| **Hit@K** | Hit Rate at K (1 if target in top-K, else 0) |
| **MRR** | Mean Reciprocal Rank |

### Expected Results

Results on ML-1M dataset (hidden_size=64, 2 layers):

| Model | NDCG@10 | Hit@10 |
|-------|---------|--------|
| BERT4Rec | ~0.40 | ~0.60 |
| NOVA-BERT | ~0.42 | ~0.62 |

*Note: Results may vary based on hyperparameters and random seeds.*

## Project Structure

```
NOVA-BERT/
├── modeling.py              # Model implementations
│   ├── BertConfig           # Standard BERT config
│   ├── BertModel            # Standard BERT model
│   ├── NOVABertConfig       # NOVA config with side info params
│   ├── NOVABertModel        # NOVA-enhanced BERT model
│   ├── nova_attention_layer # NOVA attention mechanism
│   └── nova_transformer_model # NOVA transformer stack
├── run.py                   # Training and evaluation
├── optimization.py          # AdamW optimizer
├── gen_data_fin.py          # Data preprocessing
├── vocab.py                 # Vocabulary management
├── util.py                  # Utilities
├── bert_train/              # Config files
├── data/                    # Datasets
├── requirements.txt         # Dependencies
└── run_*.sh                 # Training scripts
```

## Citation

If you use this code, please cite:

```bibtex
@article{liu2021noninvasive,
  title={Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation},
  author={Liu, Chang and Li, Xiaoguang and Cai, Guohao and Dong, Zhenhua and Zhu, Hong and Shang, Lifeng},
  journal={arXiv preprint arXiv:2103.03578},
  year={2021}
}

@inproceedings{sun2019bert4rec,
  title={BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
  author={Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  booktitle={CIKM},
  pages={1441--1450},
  year={2019}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- NOVA paper authors for the non-invasive attention mechanism
- Google AI Language Team for the original BERT implementation
- BERT4Rec authors for sequential recommendation with BERT
