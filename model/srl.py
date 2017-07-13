# Syntax-agnostic Semantic Role Labeling (based on Marcheggiani et al, 2017)
#   https://arxiv.org/pdf/1701.02593.pdf
#   https://github.com/diegma/neural-dep-srl
from __future__ import print_function
from __future__ import division

import sys
import numpy as np
import tensorflow as tf

from model import layers, lstm
from util.data_loader import batch_producer


class SRL_Model(object):
    def __init__(self, vocabs, args):
        self.args = args
        batch_size = args.batch_size

        # Input placeholders
        ## Inputs are shaped (batch_size, seq_length) unless the input
        ##   applies to the whole sentence (e.g. predicate)
        ## words: sequences of word ids
        ## pos: predicted parts of speech
        ## lemmas: lemma ids for the predicates, 0's for the other words
        ## preds: integer id of the predicate in the sentence
        ## preds_idx: the index (position) of the predicate in the sentence
        ## labels: semantic role label for each word in the sequence
        ## labels_mask: mask invalid arg labels (given the predicate)
        ## stags_placeholder: a UD supertag for each word
        ## use_dropout_placeholder: 0.0 or 1.0, whether or not to use dropout
        words_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        pos_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        lemmas_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        preds_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        preds_idx_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        labels_mask_placeholder = tf.placeholder(
            tf.float32, shape=(batch_size, vocabs['labels'].size))
        stags_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        use_dropout_placeholder = tf.placeholder(tf.float32, shape=())

        # Word representation

        ## Trainable word embeddings
        word_embeddings = layers.embed_inputs(
            raw_inputs=words_placeholder,
            vocab_size=vocabs['words'].size,
            embed_size=args.word_embed_size,
            name='word_embedding')

        ## Pretrained word embeddings
        pretr_word_vectors = layers.get_word_embeddings(
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

        word_features = [word_embeddings, pretr_word_embeddings,
                         pos_embeddings, lemma_embeddings]

        ## UD supertag embeddings
        if args.use_stags:
            stag_embeddings = layers.embed_inputs(
                raw_inputs=stags_placeholder,
                vocab_size=vocabs['stags'].size,
                embed_size=args.stag_embed_size,
                name='stag_embedding')
            word_features.append(stag_embeddings)
         
        ## Binary flags to mark the predicate
        seq_length = tf.shape(words_placeholder)[1]
        pred_markers = tf.expand_dims(tf.one_hot(preds_idx_placeholder,
                                                 seq_length,
                                                 dtype=tf.float32),
                                      axis=-1)
        word_features.append(pred_markers)
        
        ## Concatenate all the word features on the last dimension
        inputs = tf.concat(word_features, axis=2)

        ## (num_steps, batch_size, embed_size)
        ## num_steps has to be first because LSTM scans over the 1st dimension
        lstm_inputs = tf.transpose(inputs, perm=[1,0,2])

        input_size = (args.word_embed_size +
                      pretr_embed_size +
                      args.pos_embed_size +
                      args.lemma_embed_size + 1)
        if args.use_stags:
            input_size += args.stag_embed_size
        

        # BiLSTM
        dropout = 1.0 - (1.0 - args.dropout) * use_dropout_placeholder
        bilstm, zero_state = lstm.make_stacked_bilstm(
            input_size=input_size,
            state_size=args.state_size,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            dropout=dropout)

        lstm_outputs = bilstm(lstm_inputs, zero_state)
        outputs = tf.transpose(lstm_outputs, perm=[1, 0, 2])


        # Projection

        ## Get the output state corresponding to the predicate in each sentence
        ## outputs is shaped (batch_size, seq_length, output_size)
        ## so pred_outputs is (batch_size, output_size)
        indices = tf.stack([tf.range(batch_size, dtype=tf.int32),
                            preds_idx_placeholder], axis=1)
        pred_outputs = tf.gather_nd(outputs, indices)

        ## Concatenate predicate state with all the other output states
        tiled_pred_outputs = tf.tile(tf.expand_dims(pred_outputs, 1),
                                     (1, seq_length, 1))
        combined_outputs = tf.concat([outputs, tiled_pred_outputs], axis=2)

        ## Compose role and (output) pred embeddings to get projection weights
        ## (see section 2.4.3 of Marcheggiani et al 2017)
        num_roles = vocabs['labels'].size
        lstm_output_size = args.state_size * 4 # (2 LSTMs for word, 2 for pred)

        with tf.variable_scope('projection'):
            # Get embedding for labels (roles) and predicates
            ## (num_roles, r_embed_size)
            role_embeddings = tf.get_variable(
                'role_embeddings',
                shape=(num_roles, args.role_embed_size),
                dtype=tf.float32)
            ## (batch_size, p_embed_size)
            pred_embeddings = layers.embed_inputs(
                raw_inputs=preds_placeholder,
                vocab_size=vocabs['lemmas'].size,
                embed_size=args.output_lemma_embed_size,
                name='output_lemma_embedding')

            # Tile the embeddings:
            # need role_embeddings to be (batch_size, num_roles, r_embed_size)
            # and pred_embeddings to be (batch_size, num_roles, p_embed_size)
            # so we can concatenate
            tiled_role_embeddings = tf.tile(tf.expand_dims(role_embeddings, 0),
                                            (batch_size, 1, 1))
            tiled_pred_embeddings = tf.tile(tf.expand_dims(pred_embeddings, 1),
                                            (1, num_roles, 1))
            rp_embeddings = tf.concat([tiled_pred_embeddings,
                                       tiled_role_embeddings], axis=2)

            # Compose embeddings into projection matrix W
            # W: (batch_size, lstm_output_size, num_roles)            
            rp_size = args.role_embed_size + args.output_lemma_embed_size
            U = tf.get_variable(
                'U',
                shape=(rp_size, lstm_output_size),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            b = tf.get_variable(
                'b',
                shape=(lstm_output_size),
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)
            activation = tf.nn.relu(layers.batch_matmul(rp_embeddings, U) + b)
            W = tf.transpose(activation, perm=[0, 2, 1])
            
            # (batch_size, seq_len, out_size)*(batch_size, out_size, num_roles)
            # = (batch_size, seq_len, num_roles)
            logits = tf.matmul(combined_outputs, W)


        # Mask the roles that can't be assigned (given the predicate)
        # labels_mask_placeholder: (batch_size, num_roles)
        # So tile the masks and multiply
        if args.restrict_labels:
            masks_tiled = tf.tile(tf.expand_dims(labels_mask_placeholder, 1),
                                  (1, seq_length, 1))
            logits = tf.multiply(logits, masks_tiled)

        predictions = tf.nn.softmax(logits)


        # Loss op and optimizer
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_placeholder,
            logits=logits)
        loss = tf.reduce_mean(cross_ent)

        ## Clip gradients (https://stackoverflow.com/a/36501922)
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                       for grad, var in gvs]
        train_op = optimizer.apply_gradients(clipped_gvs)


        # Add everything to the model
        self.words_placeholder = words_placeholder
        self.pos_placeholder = pos_placeholder
        self.lemmas_placeholder = lemmas_placeholder
        self.preds_placeholder = preds_placeholder
        self.preds_idx_placeholder = preds_idx_placeholder
        self.labels_placeholder = labels_placeholder
        self.labels_mask_placeholder = labels_mask_placeholder
        self.stags_placeholder = stags_placeholder
        self.use_dropout_placeholder = use_dropout_placeholder
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op


    def run_training_batch(self, session, batch):
        """
        A batch contains input tensors for words, pos, lemmas, preds,
          preds_idx, and labels (in that order)
        Runs the model on the batch (through train_op if train=True)
        Returns the loss
        """
        (words, pos, lemmas, preds, preds_idx,
         labels, labels_mask, stags) = batch
        feed_dict = {
            self.words_placeholder: words,
            self.pos_placeholder: pos,
            self.lemmas_placeholder: lemmas,
            self.preds_placeholder: preds,
            self.preds_idx_placeholder: preds_idx,
            self.labels_placeholder: labels,
            self.labels_mask_placeholder: labels_mask,
            self.stags_placeholder: stags,
            self.use_dropout_placeholder: 1.0
        }
        fetches = [self.loss, self.train_op]
        loss, _ = session.run(fetches, feed_dict=feed_dict)
        return loss


    def run_testing_batch(self, session, batch):
        """
        A batch contains input tensors for words, pos, lemmas, preds,
          preds_idx, and labels (in that order)
        Runs the model on the batch (through train_op if train=True)
        Returns loss and also predicted argument labels.
        """
        (words, pos, lemmas, preds, preds_idx,
         labels, labels_mask, stags) = batch        
        feed_dict = {
            self.words_placeholder: words,
            self.pos_placeholder: pos,
            self.lemmas_placeholder: lemmas,
            self.preds_placeholder: preds,
            self.preds_idx_placeholder: preds_idx,
            self.labels_placeholder: labels,
            self.stags_placeholder: stags,
            self.use_dropout_placeholder: 0.0
        }
        fetches = [self.loss, self.predictions]
        loss, probabilities = session.run(fetches, feed_dict=feed_dict)
        return loss, probabilities
    

    def run_training_epoch(self, session, vocabs, fn):
        batch_size = self.args.batch_size
        total_loss = 0
        num_batches = 0
        total_batches = sum(1 for _ in batch_producer(batch_size, vocabs, fn))

        for i, (_, batch) in enumerate(
                batch_producer(batch_size, vocabs, fn, train=True)):
            loss = self.run_training_batch(session, batch)
            total_loss += loss
            num_batches += 1
            if i % 10 == 0:
                avg_loss = total_loss / num_batches
                msg = '\r{}/{}    loss: {}'.format(i, total_batches, avg_loss)
                sys.stdout.write(msg)
                sys.stdout.flush()
        print('\n')
                
        return total_loss / num_batches


    def run_testing_epoch(self, session, vocabs, fn, fn_sys):
        batch_size = self.args.batch_size
        total_loss = 0
        num_batches = 0
        total_batches = sum(1 for _ in batch_producer(batch_size, vocabs, fn))

        predicted_sents = []
        for i, (sents, batch) in enumerate(
                batch_producer(batch_size, vocabs, fn, train=False)):
            batch_loss, probabilities = self.run_testing_batch(session, batch)
            total_loss += batch_loss
            num_batches += 1

            # Add the predictions to the sentence objects for later evaluation
            for sent, probs in zip(sents, probabilities):
                sent.add_predictions(probs, vocabs['labels'],
                                     restrict_labels=self.args.restrict_labels)
                # (sent.parent is the complete sentence, as opposed to the
                # predicate-specific sentence)
                if sent.parent not in predicted_sents:
                    predicted_sents.append(sent.parent)
            
            if i % 10 == 0:
                avg_loss = total_loss / num_batches
                msg = '\r{}/{}    loss: {}'.format(i, total_batches, avg_loss)
                sys.stdout.write(msg)
                sys.stdout.flush()
        print('\n')

        # Write the predictions to a file for evaluation
        with open(fn_sys, 'w') as f:
            for sent in predicted_sents:
                f.write(str(sent) + '\n')
        print('Wrote predictions to', fn_sys)
    
        return total_loss / num_batches    
