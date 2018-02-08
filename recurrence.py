import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer as xinit
from KBQA import TextKBQA,text_round

def gru(num_units):
    return tf.contrib.rnn.GRUCell(num_units)

def gru_n(num_units, num_layers):
    
    # https://github.com/tensorflow/tensorflow/issues/8191
    return tf.contrib.rnn.MultiRNNCell(
                    [gru(num_units) for _ in range(num_layers)],
                    state_is_tuple=True
                    )


def get_variables(n, shape, name='W'):
    return (tf.get_variable(name+str(i), dtype=tf.float32, shape=shape)
               for i in range(n))


'''
    Uni-directional RNN

    [usage]
    cell_ = gru_n(hdim, 3)
    outputs, states = uni_net(cell = cell_,
                             inputs= inputs_emb,
                             init_state= cell_.zero_state(batch_size, tf.float32),
                             timesteps = L)
'''

def uni_net(cell, inputs, init_state, timesteps, time_major=False, scope='uni_net_0',memory=False):
    # convert to time major format
    if not time_major:
        inputs_tm = tf.transpose(inputs, [1, 0, 2])
    # collection of states and outputs
    states, outputs = [init_state], []

    d = cell.state_size[0]
    num_layers = len(cell.state_size)

    with tf.variable_scope(scope):
       # output,all_state = cell(inputs_tm, states[-1])
        for i in range(timesteps):
            # if memory==True and i ==0:
            #     tf.get_variable_scope().reuse_variables()
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs_tm[i], states[-1])
            outputs.append(output)
            states.append(state)

    states = tf.reshape(tf.transpose(tf.stack(states[1:]), [2, 0, 1, 3]), [-1, d*num_layers])

    embedding_size=300
    Ws = tf.get_variable('Ws', dtype=tf.float32, shape=[num_layers*d, d],initializer=xinit())
    Ws_last_emb = tf.get_variable('Ws_last_emb', dtype=tf.float32, shape=[d, embedding_size], initializer=xinit())
    states = tf.reshape(tf.matmul(states, Ws), [-1, timesteps, d])
    last_state=states[:,-1]
    last_state = tf.reshape(tf.matmul(last_state, Ws_last_emb), [-1, embedding_size])
    return tf.stack(outputs), states,last_state


'''
    Bi-directional RNN

    [usage]
    (states_f, states_b), outputs = bi_net(cell_f= gru_n(hdim,3),
                                        cell_b= gru_n(hdim,3),
                                        inputs= inputs_emb,
                                        batch_size= batch_size,
                                        timesteps=L,
                                        scope='bi_net_5',
                                        num_layers=3,
                                        project_outputs=True)
'''
def bi_net(cell_f, cell_b, inputs, batch_size, timesteps, 
           scope= 'bi_net',
           project_outputs=False,memory=False):

    with tf.variable_scope('fwd'):
        # forward
        _, states_f,last_state_f = uni_net(cell_f,
                              inputs,
                              cell_f.zero_state(batch_size, tf.float32),
                              timesteps,
                              memory = memory)

    with tf.variable_scope('bwd'):
        # backward
        inputs1=tf.reverse(inputs, axis=[1])
        _, states_b ,last_state_b= uni_net(cell_b,
                              inputs1,
                              cell_b.zero_state(batch_size, tf.float32),
                              timesteps,
                                memory = memory)
    
    outputs = None
    # outputs
    #  TODO : fix dimensions
    # if project_outputs:
    #     states = tf.concat([states_f, states_b], axis=-1)
    #
    #     if len(states.shape) == 4 and num_layers:
    #         states = tf.reshape(tf.transpose(states, [-2, 0, 1, -1]), [-1, hdim*2*num_layers])
    #         Wo = tf.get_variable('/Wo', dtype=tf.float32, shape=[num_layers*2*hdim, hdim])
    #     elif len(states.shape) == 3:
    #         states = tf.reshape(tf.transpose(states, [1, 0, 2]), [-1, hdim*2])
    #         Wo = tf.get_variable('/Wo', dtype=tf.float32, shape=[2*hdim, hdim])
    #     else:
    #         print('>> ERR : Unable to handle state reshape')
    #         return None
    #
    #     outputs = tf.reshape(tf.matmul(states, Wo), [batch_size, timesteps, hdim])

    return (states_f, states_b), outputs,inputs1,(last_state_f,last_state_b)


'''
    Attention Mechanism

    based on "Neural Machine Translation by Jointly Learning to Align and Translate"
        https://arxiv.org/abs/1409.0473

    [usage]
    ci = attention(enc_states, dec_state, params= {
        'Wa' : Wa, # [d,d]
        'Ua' : Ua, # [d,d]
        'Va' : Va  # [d,1]
        })
    shape(enc_states) : [B, L, d]
    shape(dec_state)  : [B, d]
    shape(ci)         : [B,d]

'''
def attention(enc_states, dec_state, params, d, timesteps):

    #d:enc_states  ?,12,512  dec_state   ?*256
    Wa, Ua = params['Wa'], params['Ua']
    # s_ij -> [B,L,d]
    a = tf.tanh(tf.expand_dims(tf.matmul(dec_state, Wa), axis=1) + 
            tf.reshape(tf.matmul(tf.reshape(enc_states,[-1, 2*d]), Ua), [-1, timesteps, d]))
    Va = params['Va'] # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(a, [-1, d]), Va), [-1, timesteps]))
    # c_i -> weighted sum of encoder states
    return tf.reduce_sum(enc_states*tf.expand_dims(scores, axis=-1), axis=1) # [B, d]    


'''
    Attentive Decoder

    [usage]
    dec_outputs, dec_states = attentive_decoder(enc_states,
                                    tf.zeros(dtype=tf.float32, shape=[B,d]),
                                    batch_size=B,timesteps=L,feed_previous=True,
                                    inputs = inputs)
    shape(enc_states) : [B, L, d]
    shape(inputs) : [[B, d]] if feed_previous else [L, B, d]





def attention(self, h_t, encoder_hs,params):
    # scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
    #                    self.W_a) + self.b_a), self.v_a)
    #          for h_s in tf.split(0, self.max_size, encoder_hs)]
    # scores = tf.squeeze(tf.pack(scores), [2])

    W_c, b_c = params['Wa'], params['Ua']
    scores = tf.reduce_sum(tf.mul(encoder_hs, h_t), 2)
    a_t = tf.nn.softmax(tf.transpose(scores))
    a_t = tf.expand_dims(a_t, 2)
    c_t = tf.batch_matmul(tf.transpose(encoder_hs, perm=[1, 2, 0]), a_t)
    c_t = tf.squeeze(c_t, [2])
    h_tld = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), W_c) + b_c)

    return h_tld
'''

def orthogonal_initializer(shape,name=None,scale = 1.0):
  #https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
  scale = 1.0
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape) #this needs to be corrected to float32
  return tf.Variable(scale * q[:shape[0], :shape[1]],name=name,trainable=True, dtype=tf.float32)

class GRUcell(object):

    def __init__(self, D_input, D_cell, initializer, L2=False, init_h=None):

        #self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer
        self.type = 'gru'

        '''
        if init_h is None:
          # If init_h is not provided, initialize it
          # the shape of init_h is [n_samples, D_cell]
          self.init_h = tf.matmul(self.incoming[0,:,:], tf.zeros([self.D_input, self.D_cell]))
          self.previous = self.init_h
        '''

        self.rgate = self.Gate()
        self.ugate = self.Gate()
        self.cell = self.Gate()

        self.W_x = tf.concat(values=[self.rgate[0], self.ugate[0], self.cell[0]], axis=1)#D_input*(3*D_cell)
        self.W_h = tf.concat(values=[self.rgate[1], self.ugate[1], self.cell[1]], axis=1)
        self.W_c = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=1)
        #self.b = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=0)
        #self.b_c = tf.concat(values=[self.rgate[4], self.ugate[4], self.cell[4]], axis=0)
        self.Uo = self.initializer([self.D_cell, self.D_input])
        self.Vo = self.initializer([self.D_input, self.D_input])
        self.Co = self.initializer([self.D_cell, self.D_input])

        if L2:
          self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    def Gate(self, bias = 0.001):
        # Since we will use gate multiple times, let's memory_code a class for reusing
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        Wc = self.initializer([self.D_cell, self.D_cell])
        #b  = tf.Variable(tf.constant(bias, shape=[self.D_cell]),trainable=True)
        #b_c= tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx,Wh,Wc

    def Slice_W(self, x, n):
        # split W's after computing
        return x[:, n*self.D_cell:(n+1)*self.D_cell]

    def Step(self,current_x,prev_h,ci):

        # + self.b,+self.b_c
        Wx = tf.matmul(current_x, self.W_x)
        Wc = tf.matmul(ci, self.W_c)
        Wh = tf.matmul(prev_h, self.W_h)


        r = tf.sigmoid(self.Slice_W(Wx, 0) + self.Slice_W(Wh, 0)+self.Slice_W(Wc, 0))

        u = tf.sigmoid(self.Slice_W(Wx, 1) + self.Slice_W(Wh, 1)+self.Slice_W(Wc, 1))

        c = tf.tanh(self.Slice_W(Wx, 2)+r*self.Slice_W(Wc, 2)+ r*self.Slice_W(Wh, 2))

        current_h = (1-u)*prev_h + u*c

        output = tf.matmul(current_h,self.Uo) + tf.matmul( current_x,self.Vo) + tf.matmul(ci,self.Co)

        return output,current_h

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    probs = tf.nn.softmax(out_layer)
    prediction = tf.argmax(probs, axis=-1)
    prediction_fan=tf.argmin(probs, axis=-1)
    return prediction,prediction_fan

def attentive_decoder(memory,enc_states, batch_size,
                      d, timesteps,
                        entity_1_vocab_size,
                     entity_vocab_size,
                      relation_vocab_size,
                      inputs = [],
                      scope='attentive_decoder_0',
                      num_layers=1,
                      feed_previous=False,
                       rnn_bcell=None,
                       att_params=None,
                       emb_mat=None,
                        triple_entity_i2w=None,
                        emb_enc_inputs=None,
                        last_state=None,
                        text_last_state_all=None,
                        memory_text_value=None,
                        text_last_state_round_all= None,
                        memory_round_key=None
                      ):

    ##############################################################################################################
    outputs,outputs_ture = [inputs[0]],[inputs[0]] # include GO token as init input
    states=[rnn_bcell.zero_state(batch_size, tf.float32)]
    ci1=None
    output1=None
    state1=None
    logits_list=[]
    with tf.variable_scope(scope):
        for i in range(timesteps):
            # if i ==0:
            #     input_=tf.zeros(dtype=tf.float32, shape=[batch_size,d])
            # else:
            input_ = outputs[-1] if feed_previous else inputs[i]
            ci = attention(enc_states, states[-1][-1], att_params, d, timesteps)

            if i > 0:
                tf.get_variable_scope().reuse_variables()
            a_w = tf.get_variable('a_w', shape=[300, 2 * 300], dtype=tf.float32,
                                  initializer=xinit())
            #########-----TextKBQA-------########################
            KBmodel = TextKBQA(entity_1_vocab_size=entity_1_vocab_size, entity_vocab_size=entity_vocab_size,
                               relation_vocab_size=relation_vocab_size,
                               embedding_size=300)
            entity_output = KBmodel(memory, text_last_state_all, 10, memory_text_value, last_state)
            probs = tf.nn.softmax(entity_output)
            predict_op = tf.argmax(probs, 1, name="predict_op")
            #########-----TextKBQA-------########################

            #########-----text_round-------########################
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope,reuse=True):
                text_model = text_round(entity_1_vocab_size=entity_1_vocab_size, entity_vocab_size=entity_vocab_size,
                                   relation_vocab_size=relation_vocab_size,
                                   embedding_size=300)
                #last_state_tra=tf.matmul(last_state, tra)
                output_round = text_model(memory_round_key, text_last_state_round_all,last_state,a_w)
                #########-----text_round-------#######################

                emb_inputs_tmp = tf.concat([input_, ci], axis=1)
                emb_inputs = tf.concat([emb_inputs_tmp, output_round], axis=1)

            ##########gen_encoder############
            output, state = rnn_bcell(emb_inputs, states[-1])
            ##########gen_encoder############

            #########----MLP--------#########
            MLP_x = tf.concat([emb_inputs, states[-1][-1]], axis=1)
            prediction_zhen, prediction_fan = multilayer_perceptron(MLP_x, att_params['MLP_weights'],
                                                                    att_params['MLP_biases'])
            #########----MLP--------#########


            #output=tf.reshape(output, [-1, 300])
            if i==0:
                ci1=ci
                state1=state
                output1=output
            outputs_ture.append(output)
            logits=tf.matmul(output, att_params['Wo']) + att_params['bo']
            #logits = tf.reshape(proj_outputs, [batch_size, L, vocab_size])

            probs = tf.nn.softmax(logits)
            prediction = tf.argmax(probs, axis=-1)

            sum = tf.add(tf.multiply(prediction_zhen, prediction), tf.multiply(prediction_fan, predict_op))

            tra = tf.constant([[1]*60544])# voc_size
            tra = tf.cast(tra, dtype=tf.float32)
            prediction_zhen_f = tf.reshape(tf.cast(prediction_zhen, dtype=tf.float32),[batch_size,1])
            prediction_fan_f = tf.reshape(tf.cast(prediction_fan, dtype=tf.float32),[batch_size,1])
            prediction_zhen_logit = tf.matmul(prediction_zhen_f, tra)
            prediction_fan_logit = tf.matmul(prediction_fan_f, tra)
            sum_logit = tf.add(tf.multiply(prediction_zhen_logit, logits), tf.multiply(prediction_fan_logit, entity_output))
            logits_list.append(sum_logit)
            output= tf.nn.embedding_lookup(emb_mat,sum)
            outputs.append(output)
            states.append(state)


        #states_bm = tf.transpose(tf.stack(states[-1][1:]), [1, 0, 2])
        #outputs_bm = tf.transpose(tf.stack(outputs_ture[1:]), [1, 0, 2])
        outputs_bm = tf.transpose(tf.stack(logits_list[:]), [1, 0, 2])
    return outputs_bm,ci1,state1,output1
