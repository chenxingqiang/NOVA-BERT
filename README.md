# NOVA-BERT

**N**on-invasive Self-Attention for Side Information Fusion in Sequential Recommendation

This repository implements NOVA-BERT, which enhances the BERT4Rec architecture with the NOVA (NOninVasive self-Attention) mechanism for incorporating side information (e.g., item categories, tags) into sequential recommendations.

## Paper Reference

This implementation is based on:

> **Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation**
>
> Liu, Chang; Li, Xiaoguang; Cai, Guohao; Dong, Zhenhua; Zhu, Hong; Shang, Lifeng
>
> arXiv:2103.03578 (2021)
>
> https://arxiv.org/abs/2103.03578

## Key Innovation: NOVA Mechanism

Traditional approaches directly fuse side information into item embeddings, which can cause **information overwhelming**. NOVA takes a different approach:

- **Non-invasive**: Side information affects the **attention distribution**, not the item embeddings
- **Better Attention**: Uses side info to generate more informative attention weights
- **Preserved Embeddings**: Item embeddings remain intact, preserving their learned representations

### How NOVA Works

```
Standard Attention:
    attention = softmax(Q_item · K_item^T)

NOVA Attention:
    attention = softmax(Q_item · K_item^T + λ · Q_side · K_side^T)
```

Where:
- `Q_item, K_item`: Query and Key from item embeddings
- `Q_side, K_side`: Query and Key from side information (categories, tags, etc.)
- `λ (lambda)`: Weight controlling side information influence (default: 0.5)

## Features

- **NOVA Attention Mechanism**: Non-invasive side information fusion
- **Bidirectional Self-Attention**: BERT's masked language model for sequences
- **Multi-head Attention**: Configurable attention heads
- **Position Embeddings**: Learned positional information
- **Backward Compatible**: Falls back to standard BERT4Rec when no side info

## Requirements

- Python 2.7+ or Python 3.6+
- TensorFlow 1.12+ (GPU version recommended)
- CUDA compatible with TensorFlow 1.12
- NumPy
- Six

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
NOVA-BERT/
├── modeling.py          # BERT + NOVA model architectures
│   ├── BertModel        # Standard BERT model
│   ├── NOVABertModel    # NOVA-enhanced BERT model
│   ├── nova_attention_layer()     # NOVA attention implementation
│   └── nova_transformer_model()   # NOVA transformer
├── optimization.py      # Adam optimizer with weight decay
├── run.py              # Main training script (supports --use_nova)
├── gen_data_fin.py     # Data generation with multiprocessing
├── vocab.py            # Vocabulary management
├── util.py             # Utility functions
├── bert_train/         # Model configuration files
├── data/               # Dataset files
└── run_*.sh            # Training scripts
```

## Quick Start

### Standard BERT4Rec Mode

```bash
./run_ml-1m.sh
```

### NOVA Mode with Side Information

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/ml-1m.train.tfrecord \
    --test_input_file=./data/ml-1m.test.tfrecord \
    --vocab_filename=./data/ml-1m.vocab \
    --user_history_filename=./data/ml-1m.his \
    --checkpointDir=./checkpoints/ml-1m-nova \
    --signature=-nova \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_ml-1m_64.json \
    --batch_size=256 \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --num_train_steps=400000 \
    --learning_rate=1e-4 \
    --use_nova=True \
    --use_side_info=True \
    --side_info_vocab_size=100 \
    --side_info_weight=0.5
```

## NOVA-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_nova` | Enable NOVA model | False |
| `--use_side_info` | Use side information in attention | False |
| `--side_info_vocab_size` | Size of side info vocabulary (e.g., # categories) | 0 |
| `--side_info_embedding_size` | Embedding dimension for side info | 64 |
| `--side_info_weight` | Weight (λ) for side info attention | 0.5 |

## Configuration

### Standard Configuration (`bert_config_ml-1m_64.json`)

```json
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}
```

### NOVA Configuration (with side info)

```json
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420,
  "use_side_info": true,
  "side_info_vocab_size": 100,
  "side_info_embedding_size": 64,
  "side_info_weight": 0.5
}
```

## Model Classes

### NOVABertModel

```python
import modeling

# Create NOVA config
config = modeling.NOVABertConfig(
    vocab_size=32000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    use_side_info=True,
    side_info_vocab_size=100,
    side_info_weight=0.5
)

# Create NOVA model
model = modeling.NOVABertModel(
    config=config,
    is_training=True,
    input_ids=input_ids,           # [batch, seq_len] item ids
    input_mask=input_mask,         # [batch, seq_len] mask
    side_info_ids=category_ids     # [batch, seq_len] category ids
)

# Get outputs
sequence_output = model.get_sequence_output()
embedding_table = model.get_embedding_table()
```

### nova_attention_layer

```python
# Direct use of NOVA attention
context = modeling.nova_attention_layer(
    from_tensor=query_tensor,
    to_tensor=key_value_tensor,
    side_info_tensor=side_info_embeddings,  # Non-invasive side info
    attention_mask=mask,
    num_attention_heads=2,
    size_per_head=32,
    side_info_weight=0.5
)
```

## Supported Datasets

| Dataset | Description | File |
|---------|-------------|------|
| ML-1M | MovieLens 1 Million ratings | `data/ml-1m.txt` |
| ML-20M | MovieLens 20 Million ratings | `data/ml-20m.zip` |
| Beauty | Amazon Beauty products | `data/beauty.txt` |
| Steam | Steam video games | `data/steam.txt` |

## Evaluation Metrics

- **NDCG@K** (Normalized Discounted Cumulative Gain): K = 1, 5, 10
- **Hit@K** (Hit Rate): K = 1, 5, 10
- **MRR** (Mean Reciprocal Rank)

## Data Format

### Basic Format (user-item pairs)

```
user_id item_id
user_id item_id
...
```

### With Side Information

To use NOVA with side information, prepare an additional file mapping items to categories:

```
item_id category_id
item_id category_id
...
```

## Citation

If you use this code, please cite both papers:

```bibtex
@article{liu2021noninvasive,
  title={Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation},
  author={Liu, Chang and Li, Xiaoguang and Cai, Guohao and Dong, Zhenhua and Zhu, Hong and Shang, Lifeng},
  journal={arXiv preprint arXiv:2103.03578},
  year={2021}
}

@inproceedings{sun2019bert4rec,
  author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
  booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  series = {CIKM '19},
  year = {2019},
  pages = {1441--1450},
  publisher = {ACM}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- NOVA paper authors for the non-invasive attention mechanism
- Original BERT implementation by Google AI Language Team
- BERT4Rec paper authors for sequential recommendation with BERT
