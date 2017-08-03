# Neural-network layers for the SRL model
import numpy as np
import tensorflow as tf


def get_word_embeddings(idx_to_word, fn='data/embeddings/sskip.100.vectors'):
    """
    Load word embeddings from a file and store them in an embedding matrix
    indexed by the indices in idx_to_word.
    """
    word_to_vec = {}
    with open(fn, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            word_to_vec[word] = vec
    num_words = max(idx_to_word.keys()) + 1
    embed_size = len(word_to_vec.values()[0])
    embeddings = np.zeros((num_words, embed_size), dtype=np.float32)
    unk = np.random.randn(embed_size)
    for idx, word in idx_to_word.iteritems():
        if word in word_to_vec:
            embeddings[idx, :] = word_to_vec[word]
        elif idx > 0:
            embeddings[idx, :] = unk
    return embeddings


def embed_inputs(raw_inputs,
                 vocab_size,
                 embed_size,
                 reserve_zero=True,
                 name='embed',
                 embeddings=None):
    with tf.variable_scope(name):
        if embeddings is None:
            shape = (vocab_size, embed_size)
            embeddings = tf.get_variable(
                'embeddings',
                shape=(vocab_size, embed_size),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            
            # If reserve_zero, make sure first row is always zeros
            if reserve_zero:
                zeros = tf.zeros((1, embed_size), dtype=tf.float32)
                embeddings = tf.concat([zeros, embeddings], axis=0)
            
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

def fully_connected_layer(
        inputs,
        output_size,
        W_init=tf.orthogonal_initializer(),
        b_init=tf.constant_initializer(0.0),
        nonlinearity=None,
        name='fully_connected'):
    """
    inputs: (batch_size, d0, d1)
    """
    with tf.variable_scope(name):
        input_size = tf.shape(inputs)[2]
        W = tf.get_variable(
            'W',
            shape=(input_size, output_size),
            initializer=W_init,
            dtype=tf.float32)
        b = tf.get_variable(
            'b',
            shape=(output_size,),
            initializer=b_init,
            dtype=tf.float32)
        result = batch_matmul(x, W) + b
        if nonlinearity is not None:
            result = nonlinearity(result)
        return result

def word_dropout(words, freqs, alpha, unk_idx):
    """
    Replace words with UNK token with probability
      alpha / (freq(word) + alpha)
    """

    # Replace dropout words with 0 token
    probs = 1 - (alpha / (tf.cast(freqs, tf.float32) + alpha))
    probs += tf.random_uniform(tf.shape(probs), dtype=tf.float32)
    mask = tf.cast(tf.floor(probs), tf.int32)
    words *= mask

    # Add unk tokens
    unk_mask = -1 * (mask - 1)
    words += unk_idx * unk_mask

    return words
    
