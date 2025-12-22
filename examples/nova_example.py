#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NOVA-BERT Example Usage

This script demonstrates how to use the NOVA-BERT model programmatically.
NOVA (NOninVasive self-Attention) uses side information to improve attention
distribution without directly modifying item embeddings.

Reference: https://arxiv.org/abs/2103.03578
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import modeling


def create_nova_config():
    """Create a NOVA-BERT configuration."""
    config = modeling.NOVABertConfig(
        vocab_size=1000,                    # Number of items
        hidden_size=64,                     # Hidden dimension
        num_hidden_layers=2,                # Number of transformer layers
        num_attention_heads=2,              # Number of attention heads
        intermediate_size=256,              # Feed-forward dimension
        hidden_act="gelu",                  # Activation function
        hidden_dropout_prob=0.1,            # Hidden layer dropout
        attention_probs_dropout_prob=0.1,   # Attention dropout
        max_position_embeddings=50,         # Max sequence length
        type_vocab_size=2,
        initializer_range=0.02,
        # NOVA-specific parameters
        use_side_info=True,                 # Enable side information
        side_info_vocab_size=20,            # Number of categories
        side_info_embedding_size=64,        # Side info embedding dim
        side_info_weight=0.5                # λ: side info influence weight
    )
    return config


def build_nova_model(config, batch_size=4, seq_length=10):
    """Build a NOVA-BERT model for demonstration."""
    
    # Create placeholders for inputs
    input_ids = tf.placeholder(tf.int32, [batch_size, seq_length], name="input_ids")
    input_mask = tf.placeholder(tf.int32, [batch_size, seq_length], name="input_mask")
    side_info_ids = tf.placeholder(tf.int32, [batch_size, seq_length], name="side_info_ids")
    
    # Create NOVA-BERT model
    model = modeling.NOVABertModel(
        config=config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        side_info_ids=side_info_ids,
        use_one_hot_embeddings=False
    )
    
    return model, input_ids, input_mask, side_info_ids


def demonstrate_nova_attention():
    """Demonstrate the NOVA attention mechanism."""
    
    print("=" * 60)
    print("NOVA-BERT Demonstration")
    print("=" * 60)
    
    # Create configuration
    config = create_nova_config()
    print("\n1. NOVA Configuration:")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Num layers: {config.num_hidden_layers}")
    print(f"   - Num attention heads: {config.num_attention_heads}")
    print(f"   - Use side info: {config.use_side_info}")
    print(f"   - Side info vocab size: {config.side_info_vocab_size}")
    print(f"   - Side info weight (λ): {config.side_info_weight}")
    
    # Build model
    print("\n2. Building NOVA-BERT model...")
    tf.reset_default_graph()
    
    batch_size = 4
    seq_length = 10
    
    model, input_ids, input_mask, side_info_ids = build_nova_model(
        config, batch_size, seq_length
    )
    
    # Get model outputs
    sequence_output = model.get_sequence_output()
    embedding_table = model.get_embedding_table()
    side_info_embedding = model.get_side_info_embedding()
    
    print(f"   - Sequence output shape: {sequence_output.shape}")
    print(f"   - Embedding table shape: {embedding_table.shape}")
    if side_info_embedding is not None:
        print(f"   - Side info embedding shape: {side_info_embedding.shape}")
    
    # Count parameters
    total_params = sum([
        tf.reduce_prod(v.shape).numpy() 
        for v in tf.trainable_variables()
    ] if hasattr(tf.reduce_prod(tf.trainable_variables()[0].shape), 'numpy') else [0])
    
    print(f"\n3. Model Statistics:")
    print(f"   - Number of trainable variables: {len(tf.trainable_variables())}")
    
    # Show NOVA-specific variables
    print("\n4. NOVA-specific Variables:")
    for var in tf.trainable_variables():
        if 'side_info' in var.name:
            print(f"   - {var.name}: {var.shape}")
    
    print("\n" + "=" * 60)
    print("NOVA-BERT model built successfully!")
    print("=" * 60)


def compare_attention_mechanisms():
    """Compare standard attention vs NOVA attention."""
    
    print("\n" + "=" * 60)
    print("Attention Mechanism Comparison")
    print("=" * 60)
    
    print("""
    STANDARD ATTENTION:
    ==================
    attention_scores = Q × K^T / √d
    attention_probs = softmax(attention_scores)
    output = attention_probs × V
    
    where Q, K, V all come from item embeddings.
    
    
    NOVA ATTENTION:
    ==============
    item_attention = Q_item × K_item^T / √d
    side_attention = Q_side × K_side^T / √d
    
    combined_scores = item_attention + λ × side_attention
    attention_probs = softmax(combined_scores)
    output = attention_probs × V_item
    
    Key differences:
    1. Side information creates additional attention signals
    2. V (value) still comes only from item embeddings
    3. λ controls how much side info influences attention
    4. Item embeddings are NOT modified (non-invasive)
    
    
    WHY NON-INVASIVE IS BETTER:
    ==========================
    Traditional: item_emb = item_emb + category_emb  (invasive)
    - Can cause information overwhelming
    - Category signal may dominate item signal
    - Harder to learn good item representations
    
    NOVA: attention = f(item_emb) + λ × f(category_emb)  (non-invasive)
    - Item embeddings remain pure
    - Side info only affects attention weights
    - Better generalization
    """)


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Run demonstrations
    demonstrate_nova_attention()
    compare_attention_mechanisms()
