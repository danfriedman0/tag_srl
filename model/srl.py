# Syntax-agnostic Semantic Role Labeling (based on Marcheggiani et al, 2017)
#   https://arxiv.org/pdf/1701.02593.pdf
#   https://github.com/diegma/neural-dep-srl
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from model import layers, data_loader, lstm_cells


class SRL_Model(object):
    def __init__(self, vocabs, args):
        batch_size = args.batch_size
        words_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        pos_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        lemmas_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        pred_idx_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        pred_pos_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))

        # Word representation

        ## Trainable word embeddings
        word_embeddings = layers.embed_inputs(
            raw_inputs=words_placeholder,
            vocab_size=vocabs['words'].size,
            embed_size=args.word_embed_size,
            name='word_embedding')

        ## Pretrained word embeddings
        pretr_word_vectors = data_loader.get_word_embeddings(
            vocabs['words'].idx_to_word)
        pretr_embed_size = pretr_word_vectors.shape[1]
        pretr_word_embeddings = layers.embed_inputs(
            raw_inputs=words_placeholder,
            vocab_size=vocabs['words'].size,
            embed_size=pretr_embed_size,
            name='pretr_word_embedding')

        ## POS embeddings
        pos_embeddings = layers.embed_inputs(
            raw_inputs=pos_placeholder,
            vocab_size=vocabs['pos'].size,
            embed_size=args.pos_embed_size,
            name='pos_embedding')

        ## Lemma embeddings for predicates (0's for non-predicates)
        lemma_embeddings = layers.embed_inputs(
            raw_inputs=lemmas_placeholder,
            vocab_size=vocabs['lemmas'].size,
            embed_size=args.lemma_embed_size,
            name='lemma_embedding')

        ## Binary flags to mark the predicate
        seq_length = tf.shape(words_placeholder)[1]
        pred_markers = tf.expand_dims(tf.one_hot(pred_pos_placeholder,
                                                 seq_length,
                                                 dtype=tf.float32), -1)

        ## Concatenate everything on the last dimension
        inputs = tf.concat([word_embeddings,
                            pretr_word_embeddings,
                            pos_embeddings,
                            lemma_embeddings,
                            pred_markers], axis=2)

        ## (num_steps, batch_size, embed_size)
        lstm_inputs = tf.transpose(inputs, perm=[1,0,2])

        input_size = (args.word_embed_size +
                      pretr_embed_size +
                      args.pos_embed_size +
                      args.lemma_embed_size + 1)


        # BiLSTM
        lstm_cell, zero_state = lstm_cells.make_stacked_lstm_cell(
            input_size=input_size,
            state_size=args.state_size,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            dropout=args.dropout)

        lstm_outputs = layers.bi_lstm(lstm_inputs, lstm_cell, zero_state)
        outputs = tf.transpose(lstm_outputs, perm=[1, 0, 2])


        # Projection

        ## Get the output state corresponding to the predicate in each sentence
        indices = tf.stack([tf.range(batch_size, dtype=tf.int32),
                            pred_pos_placeholder], axis=1)
        pred_outputs = tf.gather_nd(outputs, indices)

        ## Concatenate predicate state with all the other output states
        tiled_pred_outputs = tf.tile(tf.expand_dims(pred_outputs, 1),
                                     (1, seq_length, 1))
        combined_outputs = tf.concat([outputs, tiled_pred_outputs], axis=2)

        ## Compose role and (output) pred embeddings to get projection weights
        ## (see section 2.4.3 of Marcheggiani et al 2017)
        num_roles = vocabs['roles'].size
        lstm_output_size = args.state_size * 2

        role_embeddings = tf.get_variable(
            'role_embeddings',
            shape=(num_roles, args.role_embed_size),
            dtype=tf.float32)
        pred_embeddings = layers.embed_inputs(
            raw_inputs=pred_idx_placeholder,
            vocab_size=vocabs['lemmas'].size,
            embed_size=args.output_lemma_embed_size,
            name='output_lemma_embedding')

        with tf.variable_scope('projection'):
            # Multiply roles and frames separately then add the results
            U_role = tf.get_variable(
                'U_role',
                shape=(args.role_embed_size, lstm_output_size),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            U_pred = tf.get_variable(
                'U_pred',
                shape=(args.output_lemma_embed_size, lstm_output_size),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            W_role = tf.matmul(role_embeddings, U_role)
            W_pred = tf.matmul(pred_embeddings, U_pred)

            # Need to generate batch_size W's...
            W_role_tiled = tf.tile(tf.expand_dims(W_role, 0),
                                   (batch_size, 1, 1))
            W_pred_tiled = tf.tile(tf.expand_dims(W_pred, 1),
                                   (1, num_roles, 1))
            W = tf.transpose(tf.nn.relu(W_role_tiled + W_pred_tiled),
                             perm=[0,2,1])

            # (batch_size,seq_len,out_size)*(batch_size,out_size,num_roles)
            #   = (batch_size,seq_len,num_roles)
            logits = tf.matmul(outputs, W)
            predictions = tf.nn.softmax(logits)


        # Loss op and optimizer
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_placeholder,
            logits=logits)
        loss = tf.reduce_mean(cross_ent)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)


        # Add everything to the model
        self.words_placeholder = words_placeholder
        self.pos_placeholder = pos_placeholder
        self.lemmas_placeholder = lemmas_placeholder
        self.pred_idx_placeholder = pred_idx_placeholder
        self.pred_pos_placeholder = pred_pos_placeholder
        self.labels_placeholder = labels_placeholder
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op
