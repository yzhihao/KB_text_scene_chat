from __future__ import division
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xinit
import abc
import util


class QAbase(object):
    """
    Base class for Question Ansering
    """

    def __init__(self, entity_vocab_size,entity_1_vocab_size, embedding_size, hops=3,
                 question_encoder='lstm', use_peepholes=True, load_pretrained_model=False,
                 load_pretrained_vectors=False, pretrained_entity_vectors=None, verbose=False):

        self.entity_vocab_size = entity_vocab_size
        self.entity_1_vocab_size=entity_1_vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = embedding_size
        self.question_encoder = question_encoder
        self.use_peepholes = use_peepholes
        self.hops = hops

        """Common Network parameters"""
        # projection
        self.W = tf.get_variable("W", shape=[self.embedding_size, 2 * self.embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(tf.zeros([2 * self.embedding_size]), name="b")

        self.W1 = tf.get_variable("W1", shape=[2 * self.embedding_size, self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b1 = tf.Variable(tf.zeros([self.embedding_size]), name="b1")
        # weights for each hop of the memory network
        self.R = [tf.get_variable('R{}'.format(h), shape=[2 * self.embedding_size, 2 * self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer()) for h in range(self.hops)]
        self.attn_weights_all_hops = []
        # with tf.device('/cpu:0'):
        # embedding layer
        initializer_op = None
        trainable = False
        if load_pretrained_model:
            if verbose:
                print(
                    'Load pretrained model is set to {0} and hence entity_lookup_table trainable is set to {0}'.format(
                        load_pretrained_model))
                trainable = True
        if load_pretrained_vectors:
            if verbose:
                print('pretrained entity & word embeddings available. Initializing with them.')
            assert (pretrained_entity_vectors is not None)
            initializer_op = tf.constant_initializer(pretrained_entity_vectors)
        else:
            if verbose:
                print('No pretrained entity & word embeddings available. Learning entity embeddings from scratch')
                trainable = True
            initializer_op = tf.contrib.layers.xavier_initializer()

        self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                   shape=[self.entity_vocab_size - 1, self.embedding_size],
                                                   dtype=tf.float32,
                                                   initializer=initializer_op, trainable=trainable)

        self.entity_1_lookup_table = tf.get_variable("entity_lookup_table_1",
                                                   shape=[self.entity_1_vocab_size - 1, self.embedding_size],
                                                   dtype=tf.float32,
                                                   initializer=initializer_op, trainable=trainable)

        # dummy memory is set to -inf, so that during softmax for attention weight, we correctly
        # assign these slots 0 weight.
        self.entity_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.entity_lookup_table_extended = tf.concat([self.entity_lookup_table, self.entity_dummy_mem],0)
        self.entity_1_lookup_table_extended = tf.concat([self.entity_1_lookup_table, self.entity_dummy_mem], 0)


        # for encoding question
        with tf.variable_scope('q_forward'):
            self.q_fw_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size)
        with tf.variable_scope('q_backward'):
            self.q_bw_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size)



    def get_key_embedding(self, *args, **kwargs):
        raise NotImplementedError

    def get_value_embedding(self, val_mem):
        # each is [B, max_num_slots, D]
        val_embedding = tf.nn.embedding_lookup(self.entity_1_lookup_table_extended, val_mem, name="val_embedding")
        return val_embedding

    def seek_attention(self, question_embedding, key, value, C, mask):
        """ Iterative attention. """
        for h in range(self.hops):
            expanded_question_embedding = tf.expand_dims(question_embedding, 1)
            # self.key*expanded_question_embedding [B, M, 2D]; self.attn_weights: [B,M]
            attn_logits = tf.reduce_sum(key * expanded_question_embedding, 2)
            attn_logits = tf.where(mask, attn_logits, C)# exter
            self.attn_weights = tf.nn.softmax(attn_logits)
            self.attn_weights_all_hops.append(self.attn_weights)
            # self.p = tf.Print(attn_weights, [attn_weights], message='At hop {}'.format(h), summarize=10)
            # attn_weights_reshape: [B, M, 1]
            attn_weights_reshape = tf.expand_dims(self.attn_weights, -1)
            # self.value * attn_weights_reshape:[B, M, D]; self.attn_value:[B, D]
            attn_value = tf.reduce_sum(value * attn_weights_reshape, 1)
            # attn_value_proj : [B, 2D]
            # attn_value_proj = tf.nn.relu(tf.add(tf.matmul(attn_value, self.W), self.b))
            attn_value_proj = tf.add(tf.matmul(attn_value, self.W), self.b)
            sum = question_embedding + attn_value_proj
            # question_embedding: [B, 2D]
            question_embedding = tf.matmul(sum, self.R[h])
        return question_embedding




    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class text_round(QAbase):
    """
    Class for KB Question Answering
    TODO(rajarshd): describe input/output behaviour
    """

    def __init__(self, relation_vocab_size,
                 key_encoder='concat', **kwargs):
        super(text_round, self).__init__(**kwargs)
        self.key_encoder = key_encoder
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        self.relation_lookup_table = tf.get_variable("relation_lookup_table", shape=[self.relation_vocab_size - 1,
                                                                                     self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

        self.relation_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.relation_lookup_table = tf.concat([self.relation_lookup_table, self.relation_dummy_mem],0)

    def get_key_embedding(self, entity):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, D]
        e1_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, entity, name="e1_embedding")


        # r_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, relation, name="r_embedding")
        #
        # # key shape is [B, max_num_slots, 2D]
        # if self.key_encoder == 'concat':
        #     key = tf.concat( [e1_embedding, r_embedding],2)
        # else:
        #     raise NotImplementedError
        return e1_embedding

    def __call__(self,e1, value,ques,a_w):
        # split memory and get corresponding embeddings

        #e2=tf.concat([e1, e1],-1)
        C = tf.ones_like(e1, dtype='float32') * -1000
        mask = tf.not_equal(e1, self.entity_vocab_size - 1)
        key = self.get_key_embedding(e1)
        key=tf.reshape(tf.matmul(tf.reshape(key,[-1,self.embedding_size]),a_w),[64,-1,2*self.embedding_size])#make it shape

        attn_ques = self.seek_attention(ques, key, value, C, mask)

        # output embeddings - share with entity lookup table
        # B = tf.slice(self.entity_lookup_table, [0, 0], [1789936, -1])
        B = self.entity_1_lookup_table_extended
        # project down
        model_answer = tf.add(tf.matmul(attn_ques, self.W1), self.b1)  # model_answer: [B, D]
        #logits = tf.matmul(model_answer, B, transpose_b=True, name='ent_mul_manzil')  # scores: [B, num_entities]
        return model_answer






class TextKBQA(QAbase):
    """
    Class for QA with Text+KB
    """

    def __init__(self, relation_vocab_size,
                 key_encoder='concat',
                 join='concat2',
                 separate_key_lstm=False, **kwargs):
        super(TextKBQA, self).__init__(**kwargs)
        self.join = join
        self.key_encoder = key_encoder
        self.separate_key_lstm = separate_key_lstm
        self.relation_vocab_size = relation_vocab_size

        """Specialized Network parameters"""
        # projection
        self.relation_lookup_table = tf.get_variable("relation_lookup_table", shape=[self.relation_vocab_size - 1,
                                                                                     self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

        self.relation_dummy_mem = tf.constant(0.0, shape=[1, self.embedding_size], dtype='float32')

        self.relation_lookup_table_extended = tf.concat( [self.relation_lookup_table, self.relation_dummy_mem],0)

    def get_key_embedding(self, entity, relation):
        """TODO(rajarshd): describe various options"""
        # each is [B, max_num_slots, D]
        e1_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_extended, entity, name="e1_embedding")
        r_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, relation, name="r_embedding")

        # key shape is [B, max_num_slots, 2D]
        if self.key_encoder == 'concat':
            key = tf.concat( [e1_embedding, r_embedding],2)
        else:
            raise NotImplementedError
        return key

    def __call__(self, memory, text_key, key_len, val_mem, ques):
        # split memory and get corresponding embeddings
        e1, r, e2 = tf.unstack(memory, axis=2)
        kb_C = tf.ones_like(e1, dtype='float32') * -1000
        kb_mask = tf.not_equal(e1, self.entity_vocab_size - 1)
        kb_value = self.get_value_embedding(e2)

        # key_mem is [B, max_num_mem, max_key_len]
        # key_len is [B, max_num_mem]
        # val_mem is [B, max_num_mem]
        text_C = tf.ones_like(val_mem, dtype='float32') * -1000
        text_mask = tf.not_equal(val_mem, 0)
        text_value = self.get_value_embedding(val_mem)
        kb_key = self.get_key_embedding(e1, r)
        #kb_key, text_key = self.get_key_embedding(e1, r, key_mem, key_len)
        #ques = self.get_question_embedding(question, question_lengths)

        # get attention on retrived informations based on the question
        # kb_attn_ques = self.seek_attention(ques, kb_key, kb_value, kb_C, kb_mask)  # [B, 2D]
        # text_attn_ques = self.seek_attention(ques, text_key, text_value, text_C, text_mask)  # [B, 2D]

        #if self.join == 'batch_norm':
        mean_kb_key, var_kb_key = tf.nn.moments(kb_key, axes=[0,1])
        mean_kb_value, var_kb_value = tf.nn.moments(kb_value, axes=[0,1])
        mean_text_key, var_text_key = tf.nn.moments(kb_key, axes=[0,1])
        mean_text_value, var_text_value = tf.nn.moments(kb_value, axes=[0,1])
        text_key = tf.nn.batch_normalization(text_key, mean_text_key, var_text_key, mean_kb_key, var_kb_key, 1e-8)
        text_value = tf.nn.batch_normalization(text_value, mean_text_value, var_text_value, mean_kb_value, var_kb_value, 1e-8)

        merged_key = tf.concat([kb_key, text_key],1)
        merged_value = tf.concat([kb_value, text_value],1 )
        merged_C = tf.concat([kb_C, text_C],1 )
        merged_mask = tf.concat([kb_mask, text_mask],1)

        # get attention on retrived informations based on the question
        attn_ques = self.seek_attention(ques, merged_key, merged_value, merged_C, merged_mask)  # [B, 2D]
        model_answer = tf.add(tf.matmul(attn_ques, self.W1), self.b1)  # model_answer: [B, D]

        # output embeddings - share with entity lookup table
        B = self.entity_1_lookup_table_extended
        logits = tf.matmul(model_answer, B, transpose_b=True, name='ent_mul_manzil')  # scores: [B, num_entities]
        return logits