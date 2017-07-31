# LSTM functions
# Implementations based on:
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#   https://github.com/karpathy/char-rnn
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
from __future__ import division

import numpy as np
import tensorflow as tf
import collections


class LSTMCell:
    def set_dropout_mask(self, dropout):
        # dropout mask to apply recurrent dropout to lstm state
        ones = tf.ones(self.zero_state.shape, dtype=tf.float32)
        self.dropout_mask = tf.nn.dropout(ones, keep_prob=dropout)

    
    def __init__(self, input_size, state_size, batch_size, dropout=1.0):
        self.Wx_shape = (input_size, 4 * state_size)
        self.Wh_shape = (state_size, 4 * state_size)
        self.W_init = tf.random_uniform_initializer(-0.08, 0.08)

        self.b_shape = (4 * state_size,)
        init_b = ([0 for _ in range(state_size)] +
                  [1 for _ in range(state_size)] +
                  [0 for _ in range(2 * state_size)])
        self.b_init = tf.constant_initializer(init_b)

        # Define the zero state
        c_init = tf.zeros((batch_size, state_size), dtype=tf.float32)
        h_init = tf.zeros((batch_size, state_size), dtype=tf.float32)
        self.zero_state = tf.stack([c_init, h_init])

        self.set_dropout_mask(dropout)

        
    def __call__(self, state, x):
        Wx = tf.get_variable("Wx", shape=self.Wx_shape,
                             initializer=self.W_init)
        Wh = tf.get_variable("Wh", shape=self.Wh_shape,
                             initializer=self.W_init)
        b = tf.get_variable("b", shape=self.b_shape,
                            initializer=self.b_init)

        c_prev, h_prev = tf.unstack(state)

        # Do all the linear combinations in one batch and then split
        x_sum = tf.matmul(x, Wx, name='x_sum')
        h_sum = tf.matmul(h_prev, Wh, name='h_sum')
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


    def scan(self, inputs, dropout, init_state=None):
        if init_state is None:
            init_state = self.zero_state
        self.set_dropout_mask(dropout)
        final_states = tf.scan(self.__call__, inputs, initializer=init_state)
        _, outputs = tf.unstack(final_states, axis=1)
        return outputs


class LSTM:
    def __init__(self, input_size, state_size, batch_size,
                 num_layers, dropout):
        self.cells = [LSTMCell(input_size, state_size, batch_size)]
        for _ in xrange(num_layers - 1):
            self.cells.append(LSTMCell(state_size, state_size, batch_size))
        self.zero_state = tf.stack([cell.zero_state for cell in self.cells])
        self.dropout = dropout

    def __call__(self, inputs, init_state=None):
        if init_state is None:
            init_state = self.zero_state
        init_states = tf.unstack(init_state)
        next_inputs = inputs
        for i, cell in enumerate(self.cells):
            with tf.variable_scope('lstm_%d' % i):
                outputs = cell.scan(next_inputs, init_states[i])
                next_inputs = tf.nn.dropout(outputs, keep_prob=self.dropout)
        return next_inputs
        
        
class BiLSTM:
    def __init__(self, input_size, state_size, batch_size,
                 num_layers, dropout):
        self.cells = [LSTMCell(input_size, state_size, batch_size)]
        for _ in xrange(num_layers - 1):
            self.cells.append(LSTMCell(2 * state_size, state_size, batch_size))
        self.zero_state = tf.stack([cell.zero_state for cell in self.cells])
        self.dropout = dropout
        self.noise_shape = (1, batch_size, 2 * state_size)

    def __call__(self, inputs, init_state=None):
        if init_state is None:
            init_state = self.zero_state
        init_states = tf.unstack(init_state)
        next_inputs = inputs

        for i, cell in enumerate(self.cells):
            with tf.variable_scope('bilstm_%d' % i):
                with tf.variable_scope('forward'):
                    f_outputs = cell.scan(
                        next_inputs, self.dropout, init_states[i])
                with tf.variable_scope('backward'):
                    r_inputs = tf.reverse(next_inputs, axis=(0,))
                    rb_outputs = cell.scan(
                        r_inputs, self.dropout, init_states[i])
                    b_outputs = tf.reverse(rb_outputs, axis=(0,))
                outputs = tf.concat([f_outputs, b_outputs], axis=2)
                next_inputs = tf.nn.dropout(outputs,
                                            keep_prob=self.dropout,
                                            noise_shape=self.noise_shape)
        return next_inputs

    

def make_stacked_bilstm(input_size,
                        state_size,
                        batch_size,
                        num_layers,
                        dropout,
                        seq_lengths):
    init_cells = [make_lstm_cell(input_size, state_size, batch_size)]
    for _ in xrange(num_layers - 1):
        init_cells.append(
            make_lstm_cell(state_size * 2, state_size, batch_size))

    cells, zero_states = zip(*init_cells)

    def bilstm_fn(inputs, init_state):
        init_states = tf.unstack(init_state)
        next_inputs = inputs
        for i, cell in enumerate(cells):
            with tf.variable_scope("bilstm_%d" % i) as scope:
                outputs = bilstm(next_inputs, cell, init_states[i])
                next_inputs = tf.nn.dropout(outputs, keep_prob=dropout)
        return next_inputs

    zero_state = tf.stack(zero_states)

    return bilstm_fn, zero_state
    

    


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

        # Do all the linear combinations in one batch and then split
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
    

def forward_lstm(inputs, lstm_cell, init_state, keep_prob=1.0):
    """
    inputs shape: (num_steps, batch_size, embed_size)
    lstm_cell is a function (from model.lstm_cells) with signature
      lstm_cell(state, input) -> new_state
    init_state is the initial state of the lstm cell
    state = tf.stack([c, h]), shape: (2, batch_size, state_size)
    final_state: (seq_length, 2, batch_size, state_size)
    outputs: (seq_length, batch_size, state_size)
    Only returns the outputs (the 'h' part of the state)
    """
    final_states = tf.scan(lstm_cell, inputs, initializer=init_state)
    _, outputs = tf.unstack(final_states, axis=1)
    return outputs


def bilstm(inputs, lstm_cell, init_state):
    with tf.variable_scope('forward'):
        f_outputs = forward_lstm(inputs, lstm_cell, init_state)
    with tf.variable_scope('backward'):
        r_inputs = tf.reverse(inputs, axis=(0,))
        rb_outputs = forward_lstm(r_inputs, lstm_cell, init_state)
        b_outputs = tf.reverse(rb_outputs, axis=(0,))
    outputs = tf.concat([f_outputs, b_outputs], axis=2)
    return outputs


def make_stacked_bilstm(input_size,
                        state_size,
                        batch_size,
                        num_layers,
                        dropout,
                        seq_lengths):
    init_cells = [make_lstm_cell(input_size, state_size, batch_size)]
    for _ in xrange(num_layers - 1):
        init_cells.append(
            make_lstm_cell(state_size * 2, state_size, batch_size))

    cells, zero_states = zip(*init_cells)

    def bilstm_fn(inputs, init_state):
        init_states = tf.unstack(init_state)
        next_inputs = inputs
        for i, cell in enumerate(cells):
            with tf.variable_scope("bilstm_%d" % i) as scope:
                outputs = bilstm(next_inputs, cell, init_states[i])
                next_inputs = tf.nn.dropout(outputs, keep_prob=dropout)
        return next_inputs

    zero_state = tf.stack(zero_states)

    return bilstm_fn, zero_state


def make_stacked_tf_bilstm(input_size,
                           state_size,
                           batch_size,
                           num_layers,
                           dropout,
                           seq_lengths):
    cells = [tf.contrib.rnn.LSTMCell(state_size) for _ in xrange(num_layers)]
    zero_states = [cell.zero_state(batch_size, tf.float32) for cell in cells]

    def bilstm_fn(inputs, zero_states):
        next_inputs = inputs
        for i, cell in enumerate(cells):
            with tf.variable_scope("bilstm_%d" % i) as scope:
                output, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    inputs=next_inputs,
                    dtype=tf.float32,
                    time_major=True,
                    sequence_length=seq_lengths)
                outputs = tf.concat(output, 2)
                next_inputs = tf.nn.dropout(outputs, keep_prob=dropout)
        return next_inputs

    return bilstm_fn, zero_states
                    
        
