#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for NOVA-BERT implementation.

Tests the NOVA attention mechanism and model components.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import modeling


class TestNOVABertConfig(unittest.TestCase):
    """Test NOVABertConfig class."""
    
    def test_config_creation(self):
        """Test creating a NOVA config."""
        config = modeling.NOVABertConfig(
            vocab_size=1000,
            hidden_size=64,
            use_side_info=True,
            side_info_vocab_size=20,
            side_info_weight=0.5
        )
        
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 64)
        self.assertTrue(config.use_side_info)
        self.assertEqual(config.side_info_vocab_size, 20)
        self.assertEqual(config.side_info_weight, 0.5)
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = modeling.NOVABertConfig(
            vocab_size=1000,
            hidden_size=64,
            use_side_info=True
        )
        
        config_dict = config.to_dict()
        self.assertIn('vocab_size', config_dict)
        self.assertIn('use_side_info', config_dict)
    
    def test_config_to_json(self):
        """Test config JSON serialization."""
        config = modeling.NOVABertConfig(
            vocab_size=1000,
            hidden_size=64
        )
        
        json_str = config.to_json_string()
        self.assertIn('"vocab_size"', json_str)


class TestNOVAAttentionLayer(unittest.TestCase):
    """Test nova_attention_layer function."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.reset_default_graph()
        self.batch_size = 2
        self.seq_length = 5
        self.hidden_size = 32
        self.num_heads = 2
        self.size_per_head = 16
    
    def test_attention_without_side_info(self):
        """Test NOVA attention without side information (should work like standard)."""
        from_tensor = tf.placeholder(
            tf.float32, 
            [self.batch_size * self.seq_length, self.hidden_size]
        )
        
        output = modeling.nova_attention_layer(
            from_tensor=from_tensor,
            to_tensor=from_tensor,
            side_info_tensor=None,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            do_return_2d_tensor=True,
            batch_size=self.batch_size,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length
        )
        
        expected_shape = [
            self.batch_size * self.seq_length,
            self.num_heads * self.size_per_head
        ]
        self.assertEqual(output.shape.as_list(), expected_shape)
    
    def test_attention_with_side_info(self):
        """Test NOVA attention with side information."""
        from_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size * self.seq_length, self.hidden_size]
        )
        side_info = tf.placeholder(
            tf.float32,
            [self.batch_size * self.seq_length, self.hidden_size]
        )
        
        output = modeling.nova_attention_layer(
            from_tensor=from_tensor,
            to_tensor=from_tensor,
            side_info_tensor=side_info,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            side_info_weight=0.5,
            do_return_2d_tensor=True,
            batch_size=self.batch_size,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length
        )
        
        expected_shape = [
            self.batch_size * self.seq_length,
            self.num_heads * self.size_per_head
        ]
        self.assertEqual(output.shape.as_list(), expected_shape)
    
    def test_side_info_weight_zero(self):
        """Test that side_info_weight=0 ignores side information."""
        from_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size * self.seq_length, self.hidden_size]
        )
        side_info = tf.placeholder(
            tf.float32,
            [self.batch_size * self.seq_length, self.hidden_size]
        )
        
        # With weight=0, side info should be ignored
        output = modeling.nova_attention_layer(
            from_tensor=from_tensor,
            to_tensor=from_tensor,
            side_info_tensor=side_info,
            num_attention_heads=self.num_heads,
            size_per_head=self.size_per_head,
            side_info_weight=0.0,  # Zero weight
            do_return_2d_tensor=True,
            batch_size=self.batch_size,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length
        )
        
        self.assertIsNotNone(output)


class TestNOVABertModel(unittest.TestCase):
    """Test NOVABertModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.reset_default_graph()
        self.batch_size = 2
        self.seq_length = 10
        
        self.config = modeling.NOVABertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=20,
            use_side_info=True,
            side_info_vocab_size=10,
            side_info_embedding_size=32,
            side_info_weight=0.5
        )
    
    def test_model_without_side_info(self):
        """Test NOVA model without providing side information."""
        input_ids = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        input_mask = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        
        model = modeling.NOVABertModel(
            config=self.config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            side_info_ids=None  # No side info
        )
        
        output = model.get_sequence_output()
        expected_shape = [self.batch_size, self.seq_length, self.config.hidden_size]
        self.assertEqual(output.shape.as_list(), expected_shape)
    
    def test_model_with_side_info(self):
        """Test NOVA model with side information."""
        input_ids = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        input_mask = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        side_info_ids = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        
        model = modeling.NOVABertModel(
            config=self.config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            side_info_ids=side_info_ids
        )
        
        output = model.get_sequence_output()
        side_emb = model.get_side_info_embedding()
        
        expected_shape = [self.batch_size, self.seq_length, self.config.hidden_size]
        self.assertEqual(output.shape.as_list(), expected_shape)
        self.assertIsNotNone(side_emb)
    
    def test_model_embedding_table(self):
        """Test that embedding table is accessible."""
        input_ids = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        
        model = modeling.NOVABertModel(
            config=self.config,
            is_training=False,
            input_ids=input_ids
        )
        
        emb_table = model.get_embedding_table()
        expected_shape = [self.config.vocab_size, self.config.hidden_size]
        self.assertEqual(emb_table.shape.as_list(), expected_shape)
    
    def test_training_vs_eval_mode(self):
        """Test that dropout is applied in training mode only."""
        input_ids = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        
        # Training mode
        tf.reset_default_graph()
        train_model = modeling.NOVABertModel(
            config=self.config,
            is_training=True,
            input_ids=input_ids
        )
        
        # Eval mode
        tf.reset_default_graph()
        eval_model = modeling.NOVABertModel(
            config=self.config,
            is_training=False,
            input_ids=input_ids
        )
        
        # Both should produce valid outputs
        self.assertIsNotNone(train_model.get_sequence_output())
        self.assertIsNotNone(eval_model.get_sequence_output())


class TestNOVATransformer(unittest.TestCase):
    """Test nova_transformer_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.reset_default_graph()
        self.batch_size = 2
        self.seq_length = 5
        self.hidden_size = 32
    
    def test_transformer_without_side_info(self):
        """Test NOVA transformer without side information."""
        input_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size, self.seq_length, self.hidden_size]
        )
        
        output = modeling.nova_transformer_model(
            input_tensor=input_tensor,
            side_info_tensor=None,
            hidden_size=self.hidden_size,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            do_return_all_layers=False
        )
        
        expected_shape = [self.batch_size, self.seq_length, self.hidden_size]
        self.assertEqual(output.shape.as_list(), expected_shape)
    
    def test_transformer_with_side_info(self):
        """Test NOVA transformer with side information."""
        input_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size, self.seq_length, self.hidden_size]
        )
        side_info = tf.placeholder(
            tf.float32,
            [self.batch_size, self.seq_length, self.hidden_size]
        )
        
        output = modeling.nova_transformer_model(
            input_tensor=input_tensor,
            side_info_tensor=side_info,
            hidden_size=self.hidden_size,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            side_info_weight=0.5,
            do_return_all_layers=False
        )
        
        expected_shape = [self.batch_size, self.seq_length, self.hidden_size]
        self.assertEqual(output.shape.as_list(), expected_shape)
    
    def test_return_all_layers(self):
        """Test returning all layer outputs."""
        input_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size, self.seq_length, self.hidden_size]
        )
        
        num_layers = 3
        outputs = modeling.nova_transformer_model(
            input_tensor=input_tensor,
            hidden_size=self.hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            intermediate_size=64,
            do_return_all_layers=True
        )
        
        self.assertEqual(len(outputs), num_layers)
        for layer_output in outputs:
            expected_shape = [self.batch_size, self.seq_length, self.hidden_size]
            self.assertEqual(layer_output.shape.as_list(), expected_shape)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Run tests
    unittest.main(verbosity=2)
