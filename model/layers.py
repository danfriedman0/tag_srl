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
    for idx, word in idx_to_word.iteritems():
        if word in word_to_vec:
            embeddings[idx, :] = word_to_vec[word]
        elif idx > 0:
            embeddings[idx, :] = np.random.randn(embed_size)
    return embeddings


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
