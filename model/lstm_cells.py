# LSTM functions
# Implementations based on:
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#   https://github.com/karpathy/char-rnn
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
from __future__ import division

import numpy as np
import tensorflow as tf
import collections


def make_lstm_cell(input_size, state_size, batch_size):
    """
    Returns an lstm cell function and the zero state for the lstm.
    The cell function takes an lstm state and an input tensor as input
    and returns the next lstm state. An lstm state is a tensor with
    shape (2, state_size), so that tf.unstack(state) = [c_state, h_state].
    """

    # Concatenate all the parameters into one tensor to batch operations
    Wx_shape = (input_size, 4 * state_size)
    Wh_shape = (state_size, 4 * state_size)
    W_init = tf.random_uniform_initializer(-0.08, 0.08)
    b_shape = (4 * state_size,)

    # Initialize forget biases to 1.0 and all other biases to 0
    # (order is input gate, forget gate, candidate gate, output gate)
    init_b = ([0 for _ in range(state_size)] +
              [1 for _ in range(state_size)] +
              [0 for _ in range(2 * state_size)])
    b_init = tf.constant_initializer(init_b)

    # Define the lstm function
    def cell(state, x):
        Wx = tf.get_variable("Wx", shape=Wx_shape, initializer=W_init)
        Wh = tf.get_variable("Wh", shape=Wh_shape, initializer=W_init)
        b = tf.get_variable("b", shape=b_shape, initializer=b_init)

        c_prev, h_prev = tf.unstack(state)

        # Do all the linear combinations in one batch then split
        x_sum = tf.matmul(x, Wx)
        h_sum = tf.matmul(h_prev, Wh)
        all_sums = x_sum + h_sum + b

        s1, s2, s3, s4 = tf.split(all_sums, 4, axis=1)

        # i = input gate, f = forget gate, cn = candidate gate, o = output gate
        i = tf.sigmoid(s1)
        f = tf.sigmoid(s2)
        cn = tf.tanh(s3)
        o = tf.sigmoid(s4)

        c_new = f * c_prev + i * cn
        h_new = o * tf.tanh(c_new)

        return tf.stack([c_new, h_new])

    # Define the zero state
    c_init = tf.zeros((batch_size, state_size), dtype=tf.float32)
    h_init = tf.zeros((batch_size, state_size), dtype=tf.float32)
    zero_state = tf.stack([c_init, h_init])

    return cell, zero_state


def make_stacked_lstm_cell(input_size,
                           state_size,
                           batch_size,
                           num_layers,
                           dropout):
    """
    Returns an lstm cell function and the zero state for the lstm.
    The cell function takes an lstm state and an input tensor as input
    and returns the next lstm state. An lstm state is a tensor with
    shape (num_layers, 2, state_size), so that
      tf.unstack(state) = [layer0_state, layer1_state, ...]
      tf.unstack(layer_state) = [c_state, h_state]
    """
    
    init_cells = [make_lstm_cell(input_size, state_size, batch_size)]
    for _ in xrange(num_layers - 1):
        init_cells.append(make_lstm_cell(state_size, state_size, batch_size))

    cells, zero_states = zip(*init_cells)

    def cell(state, x):
        states = tf.unstack(state)
        cur_input = x
        new_states = []
        for i, cell in enumerate(cells):
            with tf.variable_scope("cell_%d" % i) as scope:
                cur_state = states[i]
                new_state = cell(cur_state, cur_input)
                new_states.append(new_state)
                _, h = tf.unstack(new_state)
                cur_input = tf.nn.dropout(h, keep_prob=dropout)
        return tf.stack(new_states)

    zero_state = tf.stack(zero_states)

    return cell, zero_state

