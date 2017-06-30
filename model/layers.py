# Neural-network layers for the SRL model
import numpy as np
import tensorflow as tf


def embed_inputs(raw_inputs,
                 vocab_size,
                 embed_size,
                 name='embed',
                 embeddings=None):
    with tf.variable_scope(name):
        if embeddings is None:
            shape = (vocab_size, embed_size)
            word_embeddings = tf.get_variable(
                'embeddings',
                shape=(vocab_size, embed_size),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            
            # First row should always be zeros
            zeros = tf.zeros((1, embed_size), dtype=tf.float32)
            embeddings = tf.concat([zeros, word_embeddings], axis=0)
            
        inputs = tf.nn.embedding_lookup(embeddings, raw_inputs)
        return inputs

def batch_matmul(x, W):
    """
    x: (batch_size, d0, d1)
    W: (d1, d2)
    returns xW: (batch_size, d0, d2)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    d0 = shape[1]
    d1 = shape[2]
    d2 = tf.shape(W)[1]
    _x = tf.reshape(x, [-1, d1])
    _xW = tf.matmul(_x, W, a_is_sparse=True)
    xW = tf.reshape(_xW, [batch_size, d0, d2])
    return xW
    

def forward_lstm(inputs, lstm_cell, init_state):
    """
    inputs shape: (num_steps, batch_size, embed_size)
    lstm_cell is a function (from model.lstm_cells) with signature
      lstm_cell(state, input) -> new_state
    init_state is the initial state of the lstm cell
    state = tf.stack([c, h]), shape: (2, batch_size, state_size)
    """
    final_states = tf.scan(lstm_cell, inputs, initializer=init_state)
    top_states = tf.unstack(final_states, axis=1)[-1]
    _, outputs = tf.unstack(top_states, axis=1)
    return outputs


def backward_lstm(inputs, lstm_cell, init_state):
    r_inputs = tf.reverse(inputs, axis=(1,))
    r_outputs = forward_lstm(r_inputs, lstm_cell, init_state)
    outputs = tf.reverse(r_outputs, axis=(0,))
    return outputs


def bi_lstm(inputs, lstm_cell, init_state):
    with tf.variable_scope('BiLSTM'):
        with tf.variable_scope('forward'):
            f_outputs = forward_lstm(inputs, lstm_cell, init_state)
        with tf.variable_scope('backward'):
            b_outputs = backward_lstm(inputs, lstm_cell, init_state)
        outputs = tf.concat([f_outputs, b_outputs], axis=2)
        return outputs
