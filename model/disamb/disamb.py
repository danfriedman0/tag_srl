# LSTM model for predicate detection
# Similar to He et al, 2017
from __future__ import print_function
from __future__ import division

import sys
import numpy as np
import tensorflow as tf

from model import layers, lstm
from util.data_loader import disamb_batch_producer
from util.conll_io import get_lemma_to_preds


class Redirect(object):
    def __init__(self):    
        self.stdout = sys.stdout
    def write(self, s):
        pass


class DisambModel(object):
    def __init__(self, vocabs, args):
        self.args = args
        batch_size = args.batch_size

        # Input placeholders
        words_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        pos_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        lemmas_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
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
            args.language,
            vocabs['words'].idx_to_word)
        with tf.variable_scope('pretr_word_embedding'):
            pretr_word_embeddings = tf.nn.embedding_lookup(
                pretr_word_vectors, words_placeholder)

        ## POS embeddings
        pos_embeddings = layers.embed_inputs(
            raw_inputs=pos_placeholder,
            vocab_size=vocabs['pos'].size,
            embed_size=args.pos_embed_size,
            name='pos_embedding')

        word_features = [word_embeddings, pretr_word_embeddings,
                         pos_embeddings]

        ## Supertag embeddings
        if args.use_stags:
            stag_embeddings = layers.embed_inputs(
                raw_inputs=stags_placeholder,
                vocab_size=vocabs['stags'].size,
                embed_size=args.stag_embed_size,
                name='stag_embedding')
            word_features.append(stag_embeddings)

        ## Concatenate all the word features on the last dimension
        inputs = tf.concat(word_features, axis=2)
        input_size = inputs.shape[2]

        
        # BiLSTM

        ## (num_steps, batch_size, embed_size)
        ## num_steps has to be first because LSTM scans over the 1st dimension
        lstm_inputs = tf.transpose(inputs, perm=[1,0,2])

        ## use_dropout_placeholder is 0 or 1, so this just turns dropout
        ## on or off
        dropout = 1.0 - (1.0 - args.dropout) * use_dropout_placeholder
        recurrent_dropout = (1.0 - (1.0 - args.recurrent_dropout) *
                             use_dropout_placeholder)
        
        bilstm = lstm.BiLSTM(
            cell=lstm.HighwayLSTMCell,
            input_size=input_size,
            state_size=args.state_size,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        
        lstm_outputs = bilstm(lstm_inputs)

        ## Transpose back to (batch_size, num_steps, embed_size)
        outputs = tf.transpose(lstm_outputs, perm=[1, 0, 2])


        # Projection
        with tf.variable_scope('projection'):
            lstm_output_size = args.state_size * 2
            num_labels = vocabs['predicates'].size
            W = tf.get_variable(
                'W',
                shape=(lstm_output_size, num_labels),
                initializer=tf.orthogonal_initializer(),
                dtype=tf.float32)
            logits = layers.batch_matmul(outputs, W)
            if args.restrict_labels:
                label_masks = self.get_label_masks(vocabs, args.language)
                mask = tf.nn.embedding_lookup(label_masks,
                                              lemmas_placeholder)
                logits = tf.multiply(logits, mask)
            predictions = tf.nn.softmax(logits)

        
        # Loss op and optimizer
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_placeholder,
            logits=logits)
        loss = tf.reduce_mean(cross_ent)
        optimizer = tf.train.AdamOptimizer()

        ## compute_gradients prints some of the split gradients to stdout
        ## for whatever reason, so capture that here
        redirect = Redirect()
        sys.stdout = redirect
        gvs = optimizer.compute_gradients(loss)
        sys.stdout = redirect.stdout

        ## Clip gradients (https://stackoverflow.com/a/36501922)        
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                       for grad, var in gvs]
        train_op = optimizer.apply_gradients(clipped_gvs)
        
        
        # Add everything to the model
        self.words_placeholder = words_placeholder
        self.pos_placeholder = pos_placeholder
        self.lemmas_placeholder = lemmas_placeholder
        self.labels_placeholder = labels_placeholder
        self.stags_placeholder = stags_placeholder
        self.use_dropout_placeholder = use_dropout_placeholder
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op

        self.training_batches = None
        self.testing_batches = None


    def batch_to_feed(self, batch):
        words, pos, lemmas, labels, stags = batch
        feed_dict = {
            self.words_placeholder: words,
            self.pos_placeholder: pos,
            self.lemmas_placeholder: lemmas,
            self.labels_placeholder: labels,
            self.stags_placeholder: stags,
        }
        return feed_dict
        
    def run_training_batch(self, session, batch):
        """
        A batch contains input tensors for words, pos, lemmas, preds,
          preds_idx, and labels (in that order)
        Runs the model on the batch (through train_op if train=True)
        Returns the loss
        """
        feed_dict = self.batch_to_feed(batch)
        feed_dict[self.use_dropout_placeholder] = 1.0
        fetches = [self.loss, self.train_op]

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        
        loss, _ = session.run(fetches, feed_dict=feed_dict)
        # loss, _ = session.run(fetches,
        #                       feed_dict=feed_dict,
        #                       options=options,
        #                       run_metadata=run_metadata)
        
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(chrome_trace)
        
        return loss

    
    def run_testing_batch(self, session, batch):
        """
        A batch contains input tensors for words, pos, lemmas, preds,
          preds_idx, and labels (in that order)
        Runs the model on the batch (through train_op if train=True)
        Returns loss and also predicted argument labels.
        """
        feed_dict = self.batch_to_feed(batch)
        feed_dict[self.use_dropout_placeholder] = 0.0
        fetches = [self.loss, self.predictions]
        loss, probabilities = session.run(fetches, feed_dict=feed_dict)
        return loss, probabilities
    

    def run_training_epoch(self, session, vocabs, fn_txt, fn_stags, language):
        batch_size = self.args.batch_size
        total_loss = 0
        num_batches = 0

        if self.training_batches is None:
            print('Loading training batches...')
            self.training_batches = [batch for batch in disamb_batch_producer(
                batch_size, vocabs, fn_txt, fn_stags, language, train=True)]
            print('Loaded {} training batches'.format(
                len(self.training_batches)))
        total_batches = len(self.training_batches)
        
        for i, (_, batch) in enumerate(self.training_batches):
            loss = self.run_training_batch(session, batch)
            total_loss += loss
            num_batches += 1
            if i % 10 == 0:
                avg_loss = total_loss / num_batches
                batch_size = len(batch[0][1])
                msg = '\r{}/{}    loss: {}    batch_size: {}'.format(
                    i, total_batches, avg_loss, batch_size)
                sys.stdout.write(msg)
                sys.stdout.flush()
        print('\n')

        return total_loss / num_batches


    def run_testing_epoch(self, session, vocabs, fn_txt, fn_stags,
                          fn_sys, language, fill_all=False):
        batch_size = self.args.batch_size
        total_loss = 0
        num_batches = 0

        if self.testing_batches is None:
            print('Loading testing batches...')
            self.testing_batches = [batch for batch in disamb_batch_producer(
                batch_size, vocabs, fn_txt, fn_stags, language, train=False)]
            self.gold_predicates = []
            for sents, _ in self.testing_batches:
                for sent in sents:
                    self.gold_predicates += sent.predicates
            print('Loaded {} testing batches.'.format(
                len(self.testing_batches)))
        total_batches = len(self.testing_batches)

        # lemma_to_preds = get_lemma_to_preds(
        #     'data/{}/conll09/train.txt'.format(language))
        lemma_to_preds = None
        
        predicted_predicates = []
        predicted_sents = []
        f_out = open(fn_sys, 'w')
        for i, (sents, batch) in enumerate(self.testing_batches):
            batch_loss, probabilities = self.run_testing_batch(session, batch)
            total_loss += batch_loss            
            num_batches += 1

            for sent, probs in zip(sents, probabilities):
                if not fill_all:
                    probs[:, 0] = 0.0
                # pred_ids = np.argmax(probs, axis=1)
                # predictions = vocabs['predicates'].decode_sequence(pred_ids)
                predictions = sent.add_predicted_predicates(
                    probs, vocabs['predicates'],
                    fill_all=fill_all, lemma_to_preds=lemma_to_preds)
                f_out.write('\n'.join(predictions) + '\n\n')
                predicted_predicates += predictions

            if i % 10 == 0:
                avg_loss = total_loss / num_batches
                msg = '\r{}/{}    loss: {}'.format(
                    i, total_batches, avg_loss)
                sys.stdout.write(msg)
                sys.stdout.flush()
        print('\n')
        self.test_batches = num_batches
        f_out.close()

        # Get labeled and unlabeled F1 scores
        lf1, uf1 = self.get_f1(predicted_predicates, self.gold_predicates)
    
        return total_loss / num_batches, lf1, uf1


    def get_f1(self, predicted, gold):
        """
        predicted and gold should both be lists of strings,
         '_' means no prediction
        returns labeled_f1, unlabeled_f1
        """
        correct_labeled = 0
        correct_unlabeled = 0
        num_predicted = 0
        num_gold = 0

        for p, g in zip(predicted, gold):
            if p != '_':
                num_predicted += 1
                if p == g:
                    correct_labeled += 1
                    correct_unlabeled += 1
                elif g != '_':
                    correct_unlabeled += 1
                
            if g != '_':
                num_gold += 1
        
        if num_predicted == 0 or num_gold == 0:
            return 0, 0

        labeled_precision = correct_labeled / num_predicted
        labeled_recall = correct_labeled / num_gold
        labeled_f1 = (2 * labeled_precision * labeled_recall /
                      (labeled_precision + labeled_recall))

        unlabeled_precision = correct_unlabeled / num_predicted
        unlabeled_recall = correct_unlabeled / num_gold
        unlabeled_f1 = (2 * unlabeled_precision * unlabeled_recall /
                        (unlabeled_precision + unlabeled_recall))

        return labeled_f1, unlabeled_f1
        
                
    def get_label_masks(self, vocabs, language):
        """
        Returns a matrix mapping lemmas to allowable predicates.
        masks[i] is a vector of size `num_predicates` that contains
          1's in all entries of predicates that are possible for lemma i and
          0's everywhere else.
        """
        fn = 'data/{}/conll09/train.txt'.format(language)
        lemma_to_preds = get_lemma_to_preds(fn)
        masks = np.zeros((vocabs['lemmas'].size, vocabs['predicates'].size),
                         dtype=np.float32)
        for i, lemma in vocabs['lemmas'].idx_to_word.iteritems():
            if lemma in lemma_to_preds:
                preds = lemma_to_preds[lemma]
                idxs = vocabs['predicates'].encode_sequence(preds)
                for j in idxs:
                    masks[i][j] = 1.0
            else:
                masks[i, :] = 0.0 # Allow everything
        return masks
