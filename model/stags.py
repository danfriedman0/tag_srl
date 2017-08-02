# Functions for working with supertags
import numpy as np
import tensorflow as tf
from model import layers

def get_model1_embeddings(vocab, embed_size):
    """
    Given a stag vocabulary (from `vocab.py`), return
    `stag_embeddings`: maps each supertag to a feature-based embedding:
      [head_embedding; head_direction, left_dependents, right_dependents].
    head_embedding is a trainable embedding of size `embed_size`.
    The other features are binary.
    """
    fn = 'data/stags/model1.txt'
    lines = open(fn, 'r').read().split('\n')
    stag_to_str = dict([line.split(' ') for line in lines if line != ''])

    head_rels = set([s.split('/')[0] for s in stag_to_str.values()])
    num_rels = len(head_rels)
    rel_to_idx = {r: i for i, r in enumerate(head_rels)}
    
    def str_to_feats(s):
        # HEAD_REL/DIR+L_R
        # DIR, L, and R are optional.
        # Returns (head_id, binary_feature_vector)
        parts = s.split('+')
        head_parts = parts[0].split('/')
        head_rel = head_parts[0]
        if len(head_parts) > 1:
            head_dir = head_parts[1]
        else:
            head_dir = None
        l = 0
        r = 0
        if len(parts) > 1:
            subparts = parts[1].split('_')
            if len(subparts) > 1:
                l = 1
                r = 1
            else:
                l = 1 if subparts[0] == 'L' else 0
                r = 1 if subparts[0] == 'R' else 0
        head_rel = rel_to_idx[head_rel]
        bin_feats = np.zeros(4, dtype=np.int32)        
        if head_dir == 'L':
            bin_feats[0] = 1
        if head_dir == 'R':
            bin_feats[1] = 1
        bin_feats[2] = l
        bin_feats[2] = r
        return head_rel, bin_feats

    # Fill a column vector for head_rel ids and a matrix of binary features.
    # Each row maps a supertag to a head_rel/binary feature vector
    head_rels = np.zeros((vocab.size,), dtype=np.int32)
    bin_feats = np.zeros((vocab.size, 4), dtype=np.int32)
    for idx, stag in vocab.idx_to_word.iteritems():
        if stag in stag_to_str:
            head_rel, bin_feat = str_to_feats(stag_to_str[stag])
            head_rels[idx] = head_rel
            bin_feats[idx, :] = bin_feat

    # Get trainable embeddings for head relations
    head_embeddings = layers.embed_inputs(
        raw_inputs=head_rels,
        vocab_size=num_rels,
        embed_size=embed_size,
        name='head_rel_embedding')

    # Concatenate head embeddings and binary features
    stag_embeddings = tf.concat([head_embeddings, bin_feats], axis=1)

    return stag_embeddings
        
        
        
    
