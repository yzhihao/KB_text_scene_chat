#!/usr/bin/python
#-*-coding:utf-8-*-
import math
import os

import numpy as np
from  text_data_util import get_text_index_list
from  get_round_text import get_round_text_index_list
import data_analyze
from recurrence import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class AttentiveSeq2seq():

    def __init__(self, L,vocab_size_text,vocab_size_q,vocab_size_a,w2i_a,w2i_q,triple_entity_i2w,triple_raletion_i2w,enc_hdim=512, dec_hdim=512):

        tf.reset_default_graph()

        # placeholders
        self.max_memory_len=10
        self.inputs = tf.placeholder(tf.int32, shape=[None,L], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None,L], name='targets')
        self.memory_text = tf.placeholder(tf.int32, shape=[None,None,L], name='memory_text')

        self.memory_round_text = tf.placeholder(tf.int32, shape=[None, None, L], name='memory_round_text')
        self.memory_round_key = tf.placeholder(tf.int32, [None, None], name="memory_round_key")

        self.memory=tf.placeholder(tf.int32, [None, None, 3], name="memory")
        self.memory_text_value = tf.placeholder(tf.int32, [None, None], name="memory_text_value")

        self.go_token = tf.reshape(tf.fill(tf.shape(self.inputs[:,0]), w2i_a['input_GO__']), [-1,1])
        self.decoder_inputs = tf.concat( values=[self.go_token, self.targets[:, 0:]], axis=1)
        self.training = tf.placeholder(tf.bool, name='is_training')
        batch_size = tf.shape(self.inputs)[0] # infer batch size

        # embedding
        embedding_size=300
        self.emb_mat_a = tf.get_variable('emb', shape=[vocab_size_a, embedding_size], dtype=tf.float32,
                                 initializer=xinit())
        self.emb_mat_q = tf.get_variable('emb_q', shape=[vocab_size_q, embedding_size], dtype=tf.float32,
                                       initializer=xinit())

        self.emb_mat_text = tf.get_variable('emb_text', shape=[vocab_size_text, embedding_size], dtype=tf.float32,
                                         initializer=xinit())


        self.emb_enc_inputs = tf.nn.embedding_lookup(self.emb_mat_q, self.inputs)
        self.emb_memory_text= tf.nn.embedding_lookup(self.emb_mat_text, self.memory_text)

        self.emb_memory_round_text = tf.nn.embedding_lookup(self.emb_mat_text, self.memory_round_text)

        emb_dec_inputs = tf.nn.embedding_lookup(self.emb_mat_a, self.decoder_inputs)

        cell_f=gru_n(num_layers=1, num_units=enc_hdim)
        cell_b=gru_n(num_layers=1, num_units=enc_hdim)

        # encoder self.
        with tf.variable_scope('encoder'):
            (self.estates_f, self.estates_b), _,self.inputs1,(last_state_f,last_state_b) = bi_net(cell_f,cell_b,
                                               self.emb_enc_inputs,
                                               batch_size=batch_size,
                                               timesteps=L,
                                              )
        # encoder states
        self.estates= tf.concat([self.estates_f, self.estates_b], axis=-1)
        self.last_state = tf.concat([last_state_f, last_state_b], axis=-1)

        #########################get_text_memory_key#########################
        self.emb_memory_text = tf.transpose(self.emb_memory_text, [1, 0, 2,3], name='max_len_major')
        with tf.variable_scope('get_text_memory'):
            text_last_state_list=[]
            for i in range(self.max_memory_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                self.emb_memory_text_one=self.emb_memory_text[i]
                (self.estates_f, self.estates_b), _,self.inputs1,(last_state_f,last_state_b) = bi_net(cell_f,cell_b,
                                                       self.emb_memory_text_one,
                                                       batch_size=batch_size,
                                                       timesteps=L,memory=True
                                                      )
                last_state_one = tf.concat([last_state_f, last_state_b], axis=-1)
                text_last_state_list.append(last_state_one)

            text_last_state_all=tf.transpose(tf.stack(text_last_state_list[:]), [1, 0, 2]) #[B,MAX_len_nenory,2D]
        #########################get_text_memory_key#########################

        #########################get_text_round_value############################
        self.emb_memory_round_text = tf.transpose(self.emb_memory_round_text, [1, 0, 2, 3], name='max_len_major')
        self.a_w = tf.get_variable('a_w', shape=[2 * embedding_size, embedding_size], dtype=tf.float32,initializer=xinit())
        with tf.variable_scope('get_text_memory'):
            text_last_state_round_list = []
            for i in range(self.max_memory_len):
                if i >= 0:
                    tf.get_variable_scope().reuse_variables()
                self.emb_memory_round_text_one = self.emb_memory_round_text[i]
                (self.estates_f, self.estates_b), _, self.inputs1, (last_state_f, last_state_b) = bi_net(cell_f, cell_b,
                                                                                                         self.emb_memory_round_text_one,
                                                                                                         batch_size=batch_size,
                                                                                                         timesteps=L,
                                                                                                         memory=True
                                                                                                         )
                last_emb_memory_round_text_one = tf.concat([last_state_f, last_state_b], axis=-1)
                last_emb_memory_round_text_one=tf.matmul(last_emb_memory_round_text_one,self.a_w)
                text_last_state_round_list.append(last_emb_memory_round_text_one)
            text_last_state_round_all = tf.transpose(tf.stack(text_last_state_round_list[:]), [1, 0, 2])  #[B,MAX_len_nenory,D]
        ##########################get_text_round_value############################



        # convert decoder inputs to time_major format
        self.emb_dec_inputs = tf.transpose(emb_dec_inputs, [1,0,2], name='time_major')

        with tf.variable_scope('decoder') as scope:
            rnn_bcell=gru_n(num_layers=1
                            , num_units=enc_hdim)

            self.Wo = tf.get_variable('Wo', shape=[dec_hdim, vocab_size_a], dtype=tf.float32,
                                      initializer=xinit())
            self.bo = tf.get_variable('bo', shape=[vocab_size_a], dtype=tf.float32,
                                      initializer=xinit())
            Ua = tf.get_variable('att2', shape=[2*dec_hdim, dec_hdim], dtype=tf.float32)
            Wa = tf.get_variable('att1', shape=[dec_hdim, dec_hdim], dtype=tf.float32)
            Va = tf.get_variable('Va', shape=[dec_hdim, 1], dtype=tf.float32)
            #Wa_ques = tf.get_variable('Wa_ques', shape=[15*2*dec_hdim, 2*dec_hdim], dtype=tf.float32)
            #question_lengths = tf.constant([64], name="question_lengths")

            ############################################################################
            n_input = enc_hdim + 2*enc_hdim + embedding_size+embedding_size
            n_hidden_1=256
            n_classes=2
            MLP_weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
            }
            MLP_biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
            #############################################################################


            att_params = {
                'Wa': Wa, 'Ua': Ua, 'Va': Va,'Wo':self.Wo,'bo':self.bo,'embedding_size':embedding_size,'MLP_weights':MLP_weights,'MLP_biases':MLP_biases

            }

            self.decoder_logits_inf, self.ci_ ,self.state1,self.output1= attentive_decoder(self.memory,self.estates, batch_size, dec_hdim, L,len(w2i_a), len(triple_entity_i2w),len(triple_raletion_i2w),
                                                                                 inputs=self.emb_dec_inputs,
                                                                                 feed_previous=True,
                                                                                 rnn_bcell=rnn_bcell,
                                                                                 att_params=att_params,
                                                                                emb_mat=self.emb_mat_a,
                                                                                triple_entity_i2w=triple_entity_i2w,
                                                                                emb_enc_inputs=self.emb_enc_inputs,
                                                                                last_state=self.last_state,
                                                                                text_last_state_all=text_last_state_all,memory_text_value=self.memory_text_value,
                                                                                text_last_state_round_all=text_last_state_round_all,memory_round_key=self.memory_round_key
                                                                                 )
            tf.get_variable_scope().reuse_variables()

            self.decoder_logits,self.ci,self.state1_,self.output1_= attentive_decoder(self.memory,self.estates, batch_size, dec_hdim, L,
                                                                                      len(w2i_a),
                                                                                      len(triple_entity_i2w),len(triple_raletion_i2w),
                                                 inputs=self.emb_dec_inputs,rnn_bcell=rnn_bcell,att_params=att_params,emb_mat=self.emb_mat_a,

                                                                                      triple_entity_i2w=triple_entity_i2w,
                                                                                        emb_enc_inputs = self.emb_enc_inputs,
                                                                                      last_state=self.last_state,
                                                                                      text_last_state_all=text_last_state_all,memory_text_value=self.memory_text_value,
                                                                                        text_last_state_round_all = text_last_state_round_all,memory_round_key=self.memory_round_key
                                                                                      )

        self.logits=self.decoder_logits
        logits_val = self.decoder_logits_inf
        # self.logits=tf.reshape(self.proj_outputs, [batch_size, L, vocab_size])
        #
        # logits_val = tf.reshape(self.proj_outputs_inf, [batch_size, L, vocab_size])

        self.probs_val=tf.nn.softmax(self.logits)
        self.prediction_val = tf.argmax(self.probs_val, axis=-1)
        # probabilities
        self.probs = tf.nn.softmax(logits_val)


        # calculate loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = self.logits,
                        labels = self.targets)

        cross_entropy_val= tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_val,
            labels=self.targets)

        # apply mask  * padding_mask
        masked_cross_entropy = cross_entropy

        # average across sequence, batch
        self.loss = tf.reduce_mean(masked_cross_entropy)

        self.loss_val = tf.reduce_mean(cross_entropy_val)

        # optimization
        #optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads = optimizer.compute_gradients(self.loss)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients

        self.train_op = optimizer.apply_gradients(grads)


        # inference
        self.prediction = tf.argmax(self.probs, axis=-1)

        # attach session to object
        config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config_proto)
        #self.sess = tf.Session()

        self.global_step = tf.Variable(0, trainable=False)

        self.saver = tf.train.Saver(tf.all_variables())


    #  Get sequence lengths
    def seq_len(self, t):
        return tf.reduce_sum(tf.cast(t>0, tf.int32), axis=1)

    def train(self, batch_size, epochs, trainset):

        idx_q, idx_a,dialogue_triple_index,Validate_idx_q,Validate_idx_a,Validate_triple_index,dialogue_text_index,Validate_text_index,dialogue_text_value_index,Validate_text_value_index ,\
                dialogue_text_index_round, Validate_text_index_round, dialogue__round_index, Validate__round_index= trainset

        B = batch_size

        self.sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            avg_loss,avg_loss_val = 0.,0.
            for j in range(len(idx_q)//B):
                decoder_outputs,inputs___,output1,ci_,decoder_inputs,_, l,l_val,probs= self.sess.run([self.decoder_logits,self.inputs,self.output1,self.ci_,self.decoder_inputs,self.train_op, self.loss,self.loss_val,self.decoder_logits], feed_dict = {
                    self.inputs : idx_q[j*B:(j+1)*B],
                    self.targets : idx_a[j*B:(j+1)*B],
                    self.memory: dialogue_triple_index[j * B:(j + 1) * B],
                    self.memory_text: dialogue_text_index[j * B:(j + 1) * B],
                    self.memory_text_value: dialogue_text_value_index[j * B:(j + 1) * B],

                    self.memory_round_text: dialogue_text_index_round[j * B:(j + 1) * B],
                    self.memory_round_key: dialogue__round_index[j * B:(j + 1) * B]
                })
                avg_loss += l
                avg_loss_val += l_val
                if j and j%60==0:

                    perplexity = math.exp(avg_loss / 60) if avg_loss / 60 < 300 else float('inf')
                    print('{}:{}(all in a epochs:{}) : avg_loss_val:{},avg_loss:{};perplexity:{}'.format(i, j, len(idx_q) // B, avg_loss_val / 60,avg_loss / 60,
                                                                                perplexity))
                    avg_loss, avg_loss_val = 0.,0.

                if j and j%600==0:


                    avg_loss = 0.
                    length = len(Validate_idx_q) // B
                    outputs = []
                    outputs_eval = []
                    for j1 in range(length):
                        l = model.sess.run(model.loss, feed_dict={
                            model.inputs: Validate_idx_q[j1 * B:(j1 + 1) * B],
                            model.targets: Validate_idx_a[j1 * B:(j1 + 1) * B],
                            model.memory: Validate_triple_index[j1 * B:(j1 + 1) * B],
                            model.memory_text: Validate_text_index[j1 * B:(j1 + 1) * B],
                            model.memory_text_value: Validate_text_value_index[j1 * B:(j1 + 1) * B],


                            model.memory_round_text: Validate_text_index[j1 * B:(j1 + 1) * B],
                            model.memory_round_key: Validate_text_value_index[j1 * B:(j1 + 1) * B]

                        })
                        avg_loss += l
                        estates,prediction_result,prediction_result_eval,the_ci,the_decoder_input = model.sess.run([self.estates,model.prediction_val,model.prediction,model.ci,model.emb_dec_inputs],
                                                                                                                   feed_dict={
                                                                                                                            model.inputs: Validate_idx_q[j1 * B:(j1 + 1) * B],
                                                                                                                            model.targets: Validate_idx_a[j1 * B:(j1 + 1) * B],
                                                                                                                            model.memory: Validate_triple_index[j1 * B:(j1 + 1) * B],
                                                                                                                           model.memory_text: Validate_text_index[j1 * B:(j1 + 1) * B],
                                                                                                                           model.memory_text_value: Validate_text_value_index[j1 * B:(j1 + 1) * B],


                                                                                                                       model.memory_round_text: Validate_text_index[j1 * B:(j1 + 1) * B],
                                                                                                                       model.memory_round_key: Validate_text_value_index[j1 * B:(j1 + 1) * B]
                                                                                                                                })


                        for line in prediction_result:
                            output = []
                            for i3 in line: #data_utils.EOS_ID ,data_utils.UNK_ID
                                # if int(i3) == 3:
                                #     break
                                # if int(i3) == 1:
                                #     continue
                                # else:
                                output.append(i3)
                            if len(output) == 0:
                                output_sentence = str('不知道')
                                #print '不知道'
                            else:
                                output_sentence = ' '.join([str(idx2w_a_infer[i3]) for i3 in output])
                            outputs.append(output_sentence)

                        for line_eval in prediction_result_eval:
                            output_eval = []
                            for i3_eval in line_eval:
                                # if int(i3_eval) == 3:
                                #     break
                                # if int(i3_eval) == 1:
                                #     continue
                                # else:
                                output_eval.append(i3_eval)
                            if len(output_eval) == 0:
                                output_sentence_eval = str('不知道')
                                # print '不知道'
                            else:
                                output_sentence_eval = ' '.join([str(idx2w_a_infer[i3_eval]) for i3_eval in output_eval])
                            outputs_eval.append(output_sentence_eval)

                    fl = open('/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/model_Validate_eval.txt', 'w')
                    for kj_eval in outputs_eval:
                        fl.write(str(kj_eval))
                        fl.write("\n")
                    fl.close()


                    fl = open('/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/model_Validate.txt', 'w')
                    for kj in outputs:
                        fl.write(str(kj))
                        fl.write("\n")
                    fl.close()

                    perplexity = math.exp(avg_loss / (len(Validate_idx_q) // B))
                    print('all is {};and the perplexity is {}'.format(j, perplexity))
                    avg_loss = 0.


            model_dir='/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/nn_model'
            checkpoint_path = os.path.join(model_dir, "model.ckpt")
            model.saver.save(self.sess, checkpoint_path, global_step=i)


    def dev(self, triple_ture_test, batch_size, trainset):
        dia_entity=[]
        dia_entity_mun=[]
        for one_dia in  triple_ture_test:
            dia_one_entity = []
            for i in range(len(one_dia)):
                if one_dia[i][0]==0:
                    dia_entity_mun.append(i)
                    break
                else:
                   # dia_one_entity.append(one_dia[i][0])
                    dia_one_entity.append(one_dia[i][2])
            dia_entity.append(dia_one_entity)


        model_dir = "/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/nn_model"
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(model.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            model.sess.run(tf.initialize_all_variables())

        test_idx_q, test_idx_a,idx_q, idx_a = trainset
        B = batch_size
        avg_loss = 0.
        length = len(test_idx_q) // B
        outputs = []
        outputs_eval = []
        for j1 in range(length):

            prediction_result_eval, the_ci, the_decoder_input = model.sess.run(
                [model.prediction, model.ci, model.emb_dec_inputs], feed_dict={
                    model.inputs: test_idx_q[j1 * B:(j1 + 1) * B],
                    model.targets: test_idx_a[j1 * B:(j1 + 1) * B],
                    model.memory: triple_ture_test[j1 * B:(j1 + 1) * B]
                })


            for line_eval in prediction_result_eval:
                output_eval = []
                for i3_eval in line_eval:
                    if int(i3_eval) == 3:
                        break
                    if int(i3_eval) == 1:
                        continue
                    else:
                        output_eval.append(i3_eval)
                if len(output_eval) == 0:
                    output_sentence_eval = str('不知道')
                else:
                    output_sentence_eval = ' '.join([str(idx2w_a_infer[i3_eval]) for i3_eval in output_eval])
                outputs.append(output_eval)
                outputs_eval.append(output_sentence_eval)

        dia_entity_ture_mun = []
        ret=0
        ret_exit=0
        ret_pre=0
        for i in range(len(outputs)):
            for word in outputs[i]:
                if word in dia_entity[i]:
                    ret=ret+1
            if ret!=0:
                ret_exit=ret_exit+1
            if ret==len(dia_entity[i]):
                ret_pre = ret_pre + 1
            dia_entity_ture_mun.append(ret)
            ret=0
        lenout = len(outputs)


        fl = open('/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/model_test_eval.txt', 'w')
        for kj_eval in outputs_eval:
            fl.write(str(kj_eval))
            fl.write("\n")
        fl.write("all is :"+str(lenout)+';ret_exit is:'+str(ret_exit)+';ret_pre is:'+str(ret_pre))
        fl.write("\n")
        fl.close()



#from test_model_data.data_util import *




if __name__ == '__main__':

    metadata, idx_q, idx_a,dialogue_triple_index,Validate_idx_q,Validate_idx_a,Validate_triple_index=data_analyze.load_data('/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/')


   # print ret
    idx2w_a_infer = metadata['idx2w_a_infer']
    w2idx_a_infer = metadata['w2idx_a_infer']
    i2w_q = metadata['idx2w_q']
    w2i_q = metadata['w2idx_q']
    triple_ture_test = metadata['triple_ture_test']
    triple_entity_i2w = metadata['i2w_entity']
    triple_raletion_i2w = metadata['i2w_relation']
    triple_raletion_i2w=np.array(triple_raletion_i2w)
    a1=triple_ture_test[-10:-1]
    print triple_ture_test[-1]

    output_sentence_eval = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[10]])
    output_sentence_eval1 = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[11]])
    output_sentence_eval2 = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[12]])
    output_sentence_eval3 = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[13]])
    output_sentence_eval4 = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[14]])
    output_sentence_eval5 = ' '.join([str(i2w_q[i3_eval]) for i3_eval in idx_q[15]])


    #i2w, w2i, Validate_idx_q, Validate_idx_a, idx_q, idx_a = get_data()
    L = len(idx_q[0])
    vocab_size_a = len(w2idx_a_infer)
    vocab_size_q = len(w2i_q)

    metadata_text, dialogue_text_index, Validate_text_index, text_ture_test=get_text_index_list.load_data('/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/text_data_util/')
    w2i_text=metadata_text['word2index_text']
    vocab_size_text = len(w2i_text)
    Validate_text_value_index=metadata_text['Validate_text_value_index']
    text_value_ture_test=metadata_text['text_value_ture_test']
    dialogue_text_value_index=metadata_text['dialogue_text_value_index']



    metadata_text_round, dialogue_text_index_round, Validate_text_index_round, text_ture_test_round = get_round_text_index_list.load_data(
        '/home/evan/PycharmProjects/ncm-adv-master/KB_text_round_Chat/get_round_text/')
    Validate__round_index = metadata_text_round['Validate_text_value_index']
    text__round_test = metadata_text_round['text_value_ture_test']
    dialogue__round_index = metadata_text_round['dialogue_text_value_index']

    model = AttentiveSeq2seq(L, vocab_size_text,vocab_size_q,vocab_size_a,w2idx_a_infer,w2i_q,triple_entity_i2w,triple_raletion_i2w)
    # begin training


    #model.dev(triple_ture_test, batch_size=64, trainset=(idx_q, idx_a, Validate_idx_q, Validate_idx_a))
    model.train(batch_size=64, epochs=60, trainset=(idx_q, idx_a,dialogue_triple_index,Validate_idx_q,Validate_idx_a,Validate_triple_index,
                                                    dialogue_text_index,Validate_text_index,dialogue_text_value_index,Validate_text_value_index,
                                                    dialogue_text_index_round, Validate_text_index_round, dialogue__round_index,Validate__round_index
                                                    ))
