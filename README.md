# NOVA-BERT

A BERT-based Sequential Recommendation System built on the BERT4Rec architecture for sequential recommendation tasks.

## Overview

NOVA-BERT applies the bidirectional self-attention mechanism from BERT to model user behavior sequences for next-item prediction. This implementation supports multiple datasets and provides a complete pipeline for data preprocessing, training, and evaluation.

## Features

- **Bidirectional Self-Attention**: Leverages BERT's masked language model approach for sequential recommendation
- **Multi-head Attention**: Configurable attention heads for capturing different aspects of user preferences
- **Position Embeddings**: Learns positional information of items in sequences
- **Masked Item Prediction**: Predicts masked items in user sequences for training
- **Flexible Configuration**: Supports various hidden sizes (64, 128, 256) and model architectures

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
├── modeling.py          # BERT model architecture
├── optimization.py      # Adam optimizer with weight decay
├── run.py              # Main training and evaluation script
├── gen_data_fin.py     # Data generation with multiprocessing
├── gen_data.py         # Alternative data generation script
├── vocab.py            # Vocabulary management
├── util.py             # Utility functions for data processing
├── bert_train/         # Model configuration files
│   ├── bert_config_*_64.json
│   ├── bert_config_*_128.json
│   └── bert_config_*_256.json
├── data/               # Dataset files
│   ├── ml-1m.txt       # MovieLens 1M dataset
│   ├── ml-20m.zip      # MovieLens 20M dataset
│   ├── beauty.txt      # Amazon Beauty dataset
│   └── steam.txt       # Steam games dataset
└── run_*.sh            # Training scripts for each dataset
```

## Supported Datasets

| Dataset | Description | File |
|---------|-------------|------|
| ML-1M | MovieLens 1 Million ratings | `data/ml-1m.txt` |
| ML-20M | MovieLens 20 Million ratings | `data/ml-20m.zip` |
| Beauty | Amazon Beauty product reviews | `data/beauty.txt` |
| Steam | Steam video game platform | `data/steam.txt` |

## Quick Start

### 1. Run with MovieLens-1M Dataset

```bash
./run_ml-1m.sh
```

This script performs two steps:
1. **Data Generation**: Creates masked training data from user sequences
2. **Model Training**: Trains the NOVA-BERT model and evaluates performance

### 2. Manual Execution

#### Step 1: Generate Training Data

```bash
python -u gen_data_fin.py \
    --dataset_name=ml-1m \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --mask_prob=1.0 \
    --dupe_factor=10 \
    --masked_lm_prob=0.2 \
    --prop_sliding_window=0.5 \
    --signature="-default" \
    --pool_size=10
```

#### Step 2: Train the Model

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/ml-1m-default.train.tfrecord \
    --test_input_file=./data/ml-1m-default.test.tfrecord \
    --vocab_filename=./data/ml-1m-default.vocab \
    --user_history_filename=./data/ml-1m-default.his \
    --checkpointDir=./checkpoints/ml-1m \
    --signature=-default-64 \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_ml-1m_64.json \
    --batch_size=256 \
    --max_seq_length=200 \
    --max_predictions_per_seq=40 \
    --num_train_steps=400000 \
    --num_warmup_steps=100 \
    --learning_rate=1e-4
```

## Configuration

### Model Hyperparameters

Configuration files are located in `bert_train/`. Example configuration (`bert_config_ml-1m_64.json`):

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

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_size` | Dimension of hidden layers | 64/128/256 |
| `num_hidden_layers` | Number of transformer layers | 2 |
| `num_attention_heads` | Number of attention heads | 2 |
| `intermediate_size` | Feed-forward layer size | 256 |
| `max_position_embeddings` | Maximum sequence length | 200 |
| `hidden_dropout_prob` | Dropout rate for hidden layers | 0.2 |
| `attention_probs_dropout_prob` | Dropout rate for attention | 0.2 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_seq_length` | Maximum sequence length | 200 |
| `batch_size` | Training batch size | 256 |
| `learning_rate` | Initial learning rate | 1e-4 |
| `num_train_steps` | Total training steps | 400000 |
| `num_warmup_steps` | Warmup steps | 100 |
| `masked_lm_prob` | Probability of masking items | 0.2 |
| `dupe_factor` | Data duplication factor | 10 |

## Evaluation Metrics

The model is evaluated using the following metrics:
- **NDCG@K** (Normalized Discounted Cumulative Gain): K = 1, 5, 10
- **Hit@K** (Hit Rate): K = 1, 5, 10
- **MRR** (Mean Reciprocal Rank)

## Data Format

Input data files should be in the format:
```
user_id item_id
user_id item_id
...
```

Each line represents a user-item interaction, sorted by timestamp for each user.

## Extending to New Datasets

1. Prepare your data in the required format (`user_id item_id` per line)
2. Place the data file in `./data/your_dataset.txt`
3. Create a configuration file in `./bert_train/bert_config_your_dataset_64.json`
4. Update the `vocab_size` in the config based on your item count
5. Run the training pipeline

## Citation

This implementation is based on the BERT4Rec paper:

```bibtex
@inproceedings{Sun:2019:BSR:3357384.3357895,
  author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
  booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  series = {CIKM '19},
  year = {2019},
  isbn = {978-1-4503-6976-3},
  location = {Beijing, China},
  pages = {1441--1450},
  numpages = {10},
  url = {http://doi.acm.org/10.1145/3357384.3357895},
  doi = {10.1145/3357384.3357895},
  acmid = {3357895},
  publisher = {ACM},
  address = {New York, NY, USA}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Original BERT implementation by Google AI Language Team
- BERT4Rec paper authors for the sequential recommendation approach
