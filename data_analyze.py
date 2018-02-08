#!/usr/bin/python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



triple=[]
entity=set()
relation=set()
import nltk
import itertools
import re
import numpy as np
import pickle

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 0
        }

UNK = '<UNK>'
EOS = '<EOS>'
GO  = 'input_GO__'

def filter_line(line, whitelist):
    line = re.sub(u'[-()`～！@#￥%……&×（=+·{＂」}「、；”】：’”，《【。》/？）——!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\']', '',
                  str((line.encode("utf-8"))).decode())
    line = re.sub(u'  ', ' ',
                  str((line.encode("utf-8"))).decode())
    return line




def filter_data(lines_post, lines_respose,triple_dialogue):
    filtered_q, filtered_a = [], []
    filtered_triple_dialogue=[]
    for i in range(len(lines_post)):
        qlen, alen = len(lines_post[i].split(' ')), len(lines_respose[i].split(' '))
        if qlen >= limit['minq'] and qlen < limit['maxq']:
            if alen >= limit['mina'] and alen < limit['maxa'] - 1:
                filtered_q.append(lines_post[i])
                filtered_a.append(lines_respose[i])
                filtered_triple_dialogue.append(triple_dialogue[i])
    raw_data_len = len(lines_post)
    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a,filtered_triple_dialogue


import copy
def index_(tokenized_sentences, vocab_size,entity_word=None):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab=[]
    for w in set(itertools.chain(*tokenized_sentences)):
        if freq_dist[w] >=3:
            a=(w,freq_dist[w])
            vocab.append(a)
    if entity_word!=None:
        index2word =  [ x[0] for x in vocab ]
        index2word_tmp = list()
        index2word_tmp = copy.copy(index2word)
        for i in range(len(index2word)):
            if index2word[i] in entity_word and index2word[i] in index2word_tmp:
                index2word_tmp.remove(index2word[i])
        index2word=['<PAD>','<UNK>', 'input_GO__','<EOS>']+entity_word+index2word_tmp
    else:
        index2word =   ['<PAD>', '<UNK>', 'input_GO__', '<EOS>']+[x[0] for x in vocab]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

import jieba


def pad_seq(seq, lookup, maxlen, q=True):
    indices = []
    for word in seq:
        #        word = unicode(word, "utf-8")
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if not q:
        return indices + [lookup[EOS]] + [0] * (maxlen - len(seq) - 1)

    return indices + [0] * (maxlen - len(seq))

def zero_pad(dialogue_triple,qtokenized, atokenized,  w2idx_q,w2idx_a):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = []
    idx_a = []
    dialogue_triple_filter=[]
    for i in range(data_len):
        dialogue_one_triple=dialogue_triple[i]
        q_indices = pad_seq(qtokenized[i], w2idx_q, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx_a, limit['maxa'], q = False)
        ret=0
        ret1 =0
        for i_ in range(len(a_indices)):
            if a_indices[0]==0 and a_indices[1]==0:
                ret1=1
                break
            if a_indices[i_]==1:
                ret=ret+1
            if a_indices[i_]==3:
                if ret*1.0/(i_+1)>=0.5:
                    ret1=1
                    break
        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        if ret1==1 :
            continue
        idx_q.append(np.array(q_indices))
        idx_a.append(np.array(a_indices))
        dialogue_triple_filter.append(dialogue_one_triple)

    np.array(idx_q)
    np.array(idx_a)

    return idx_q, idx_a,dialogue_triple_filter


def zero_pad_triple(filtered_triple_dialogue, w2i_entity,w2i_entity_1, w2i_relation):
    dialogue_triple_index = []
    max_len_triple=0
    for i1 in  filtered_triple_dialogue:
        ret=len(i1)
        if max_len_triple<ret:
            max_len_triple=ret

    for i in filtered_triple_dialogue:
        triple_one_dia = []
        for tri in i:
            triple_one_dia.append([w2i_entity[tri[0]], w2i_relation[tri[1]], w2i_entity_1[tri[2]]])
        triple_one_dia=triple_one_dia+[[w2i_entity['<dummy_entity>'],w2i_relation['<dummy_relation>'],w2i_entity_1['<dummy_entity>']]] * (max_len_triple - len(triple_one_dia))
        dialogue_triple_index.append(triple_one_dia)

    return dialogue_triple_index, max_len_triple


def getKB():
    f_rtple = open('/home/evan/PycharmProjects/ncm-adv-master/baseline/data/cqa_triple_origina_v0', 'r')
    f_rtple = open('/home/evan/PycharmProjects/ncm-adv-master/baseline/data/cqa_triple_origina_v0', 'r')

    DIR_PATH = 'E:/'

    triple = []
    entity = set()
    entity_1 = set()
    relation = set()
    index_dialogue=0
    triple_dialogue={}
    question_list = []
    answer_list = []
    for line in f_rtple.readlines()[1:-1]:
        line = line.strip()
        line = line.split('\t')
        ################################################################
        question = line[0].split(':')[1]
        sentence = jieba.cut(question, cut_all=False)
        sentence = (' '.join(sentence)).encode("utf-8")
        question_list.append(sentence)

        answer = line[1].split(':')[1]
        sentence = jieba.cut(answer, cut_all=False)
        sentence = (' '.join(sentence)).encode("utf-8")
        answer_list.append(sentence)
        ################################################################

        if len(line)>5:
            ret = 0
            for i in line[2:]:
                if i.split(':')[0] == str('triple0'):
                    begin = line.index(i)
                    break

            for i in line[begin:]:
                if ret % 3 == 0:
                    entity1 = i.split(':')[1]
                    entity.add(entity1)
                if ret % 3 == 1:
                    relation1 = i
                    relation.add(relation1)
                if ret % 3 == 2:
                    entity2 = i
                    entity_1.add(entity2)
                ret += 1
                if ret % 3 == 0:
                    triple.append([entity1, relation1, entity2])
                    if index_dialogue not in triple_dialogue.keys():
                        triple_dialogue[index_dialogue]=[]
                        triple_dialogue[index_dialogue].append([entity1, relation1, entity2])
                    else:
                        triple_dialogue[index_dialogue].append([entity1, relation1, entity2])
        else:
            entity1 = line[2].split(':')[1]
            entity.add(entity1)
            relation1 = line[3]
            relation.add(relation1)
            entity2 = line[4]
            entity_1.add(entity2)
            triple_dialogue[index_dialogue] = []
            triple_dialogue[index_dialogue].append([entity1, relation1, entity2])


        index_dialogue=index_dialogue+1
        print index_dialogue

    print('\n>> all')
    freq_dist = nltk.FreqDist(list(entity))
    vocab_entity = []
    for w in set(entity):
        a = (w, freq_dist[w])
        vocab_entity.append(a)

    i2w_entity = ['<dummy_entity>'] + [x[0] for x in vocab_entity]
    #i2w_entity = [x[0] for x in vocab_entity]
    w2i_entity = dict([(w, i) for i, w in enumerate(i2w_entity)])


    print('\n>> all')
    freq_dist = nltk.FreqDist(list(entity_1))
    vocab_entity = []
    for w in set(entity_1):
        a = (w, freq_dist[w])
        vocab_entity.append(a)

    i2w_entity_1 = ['<dummy_entity>'] + [x[0] for x in vocab_entity]
    #i2w_entity = [x[0] for x in vocab_entity]
    w2i_entity_1 = dict([(w, i) for i, w in enumerate(i2w_entity_1)])



    freq_dist = nltk.FreqDist(list(relation))
    vocab_relation = []
    for w in set(relation):
        a = (w, freq_dist[w])
        vocab_relation.append(a)
    i2w_relation = ['<dummy_relation>'] + [x[0] for x in vocab_relation]
    w2i_relation = dict([(w, i) for i, w in enumerate(i2w_relation)])
    # triple_index = []
    #
    # for i in triple:
    #     E_e1 = w2i_entity[i[0]]
    #     E_e2 = w2i_entity[i[2]]
    #     E_r = w2i_relation[i[1]]
    #     triple_index.append([E_e1, E_r, E_e2])

    return question_list,answer_list,i2w_entity, w2i_entity,i2w_entity_1,w2i_entity_1, i2w_relation, w2i_relation,triple_dialogue



def changetest_data(i2w_a,idx2w_q,triple_test):
    test_idx_q = np.load('/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/' + 'test_idx_q.npy')
    test_idx_a = np.load('/home/evan/PycharmProjects/ncm-adv-master/KB_Chat/' + 'test_idx_a.npy')

    test_idx_q_ture=[]
    test_idx_a_ture=[]
    test_idx_q_str=[]
    triple_test_ture=[]

    for i in range(len(test_idx_q)):
        if str(test_idx_q[i].tolist()) in test_idx_q_str:
            continue
        test_idx_q_str.append(str(test_idx_q[i].tolist()))
        test_idx_q_ture.append(test_idx_q[i])
        test_idx_a_ture.append(test_idx_a[i])
        triple_test_ture.append(triple_test[i])

    triple_test_ture=triple_test_ture[0:64*40]
    test_idx_q = test_idx_q_ture[0:64*40]
    test_idx_a = test_idx_a_ture[0:64*40]

    np.save('test_idx_q.npy', test_idx_q)
    np.save('test_idx_a.npy', test_idx_a)
    test_qlines=[]
    test_alines=[]
    for line_eval in test_idx_q:
        output_eval = []
        for i3_eval in line_eval:
            if int(i3_eval) == 3 or int(i3_eval) == 0:
                break
            else:
                output_eval.append(i3_eval)
        if len(output_eval) == 0:
            output_sentence_eval = str('不知道')
        else:
            output_sentence_eval = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in output_eval])
        test_qlines.append(output_sentence_eval)

    for line_eval in test_idx_a:
        output_eval = []
        for i3_eval in line_eval:
            if int(i3_eval) == 3:
                break
            else:
                output_eval.append(i3_eval)
        if len(output_eval) == 0:
            output_sentence_eval = str('不知道')
        else:
            output_sentence_eval = ' '.join([str(i2w_a[i3_eval]) for i3_eval in output_eval])
        test_alines.append(output_sentence_eval)

    fl = open('/home/evan/PycharmProjects/ncm-adv-master/data_test.txt', 'w')
    for i in test_alines:
        fl.write(str(i))
        fl.write("\n")
    fl.close()

    fl = open('/home/evan/PycharmProjects/ncm-adv-master/data_test_query.txt', 'w')
    for i in test_qlines:
        fl.write(str(i))
        fl.write("\n")
    fl.close()
    return triple_test_ture

def get_pair():

    # question_list=[]
    # answer_list=[]
    # for line in f_rtple.readlines()[1:-1]:
    #     line = line.strip()
    #     line = line.split('\t')
    #     ret=0
    #     question=line[0].split(':')[1]
    #     sentence = jieba.cut(question, cut_all=False)
    #     sentence = (' '.join(sentence)).encode("utf-8")
    #     question_list.append(sentence)
    #
    #     answer=line[1].split(':')[1]
    #     sentence = jieba.cut(answer, cut_all=False)
    #     sentence = (' '.join(sentence)).encode("utf-8")
    #     answer_list.append(sentence)

    print('\n>> getKB')
    question_list,answer_list,i2w_entity, w2i_entity, i2w_entity_1,w2i_entity_1,i2w_relation, w2i_relation, triple_dialogue = getKB()
    lines_respose = [filter_line(line, EN_WHITELIST) for line in answer_list]

    lines_post = [filter_line(line, EN_WHITELIST) for line in question_list ]

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')

    # qlines=
    # alines=filter_data(lines_respose)

    qlines, alines,filtered_triple_dialogue = filter_data(lines_post, lines_respose,triple_dialogue)

    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = []
    atokenized = []
    for wordlist in qlines:
        wordlist_ = str(wordlist).split(' ')
        while '' in wordlist_:
            wordlist_.remove('')
        qtokenized.append(wordlist_)

    for wordlist in alines:
        wordlist_ = str(wordlist).split(' ')
        while '' in wordlist_:
            wordlist_.remove('')
        atokenized.append(wordlist_)
    VOCAB_SIZE=1000
    idx2w_q, w2idx_q, freq_dist_q = index_(qtokenized, vocab_size=VOCAB_SIZE)
    #idx2w_a, w2idx_a, freq_dist_a = index_(atokenized, vocab_size=VOCAB_SIZE)
    idx2w_a_infer, w2idx_a_infer, freq_dist_idx2w_a_infer = index_(atokenized, vocab_size=VOCAB_SIZE,entity_word=i2w_entity_1)

    idx_q, idx_a, dialogue_triple_filter = zero_pad(filtered_triple_dialogue,qtokenized, atokenized, w2idx_q, w2idx_a_infer)
    dialogue_triple_index,max_len_triple=zero_pad_triple(dialogue_triple_filter, w2i_entity,w2idx_a_infer,w2i_relation)
    print('\n >> Zero Padding')

    ret=0

    for i in range(len(idx_a)):
        if (idx_a[i][0]==1 and idx_a[i][1]==3 )or(idx_a[i][0]==0 and idx_a[i][1]==0) :
            ret=ret+1

    print 'ret' + str(ret)

    output_sentence_eval = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[10]])
    output_sentence_eval1 = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[11]])
    output_sentence_eval2 = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[12]])
    output_sentence_eval3 = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[13]])
    output_sentence_eval4 = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[14]])
    output_sentence_eval5 = ' '.join([str(idx2w_q[i3_eval]) for i3_eval in idx_q[15]])

    # output_sentence_evala = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[10]])
    # output_sentence_evala1 = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[11]])
    # output_sentence_evala2 = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[12]])
    # output_sentence_evala3 = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[13]])
    # output_sentence_evala4 = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[14]])
    # output_sentence_evala5 = ' '.join([str(idx2w_a[i3_eval]) for i3_eval in idx_a[15]])

    test_len = 64 * 80
    test_idx_q = idx_q[0:test_len]
    test_idx_a = idx_a[0:test_len]
    Validate_idx_q = idx_q[test_len:2 * test_len]
    Validate_idx_a = idx_a[test_len:2 * test_len]
    Validate_qlines = qlines[test_len:2 * test_len]
    Validate_alines = alines[test_len:2 * test_len]
    test_qlines = qlines[0:test_len]
    test_alines = alines[0:test_len]
    idx_q = idx_q[2 * test_len:-1]
    idx_a = idx_a[2 * test_len:-1]
    triple_triple_index = dialogue_triple_index[2 * test_len:-1]
    Validate_triple_index = dialogue_triple_index[test_len:2 * test_len]
    test_triple_index=dialogue_triple_index[0:test_len]

    print('\n >> Save numpy arrays to disk')

    fl = open('/home/evan/PycharmProjects/ncm-adv-master/data_Validate.txt', 'w')
    for i in Validate_alines:
        fl.write(str(i))
        fl.write("\n")
    fl.close()

    fl = open('/home/evan/PycharmProjects/ncm-adv-master/data_Validate_query.txt', 'w')
    for i in Validate_qlines:
        fl.write(str(i))
        fl.write("\n")
    fl.close()

    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    np.save('dialogue_triple_index.npy', triple_triple_index)

    np.save('test_idx_q.npy', test_idx_q)
    np.save('test_idx_a.npy', test_idx_a)

    np.save('Validate_idx_q.npy', Validate_idx_q)
    np.save('Validate_idx_a.npy', Validate_idx_a)
    np.save('Validate_triple_index.npy', Validate_triple_index)



    # let us now save the necessary dictionaries
    print('\n >>  changetest_data')
    triple_ture_test=changetest_data(idx2w_a_infer, idx2w_q, test_triple_index)
    metadata = {
        'w2idx_q': w2idx_q,
        'idx2w_q': idx2w_q,
        'limit': limit,
        'freq_dist_q': freq_dist_q,
        'idx2w_a_infer':idx2w_a_infer,
        'w2idx_a_infer': w2idx_a_infer,

        'i2w_entity':i2w_entity,
        'w2i_entity': w2i_entity,
        'i2w_relation': i2w_relation,
        'w2i_relation': w2i_relation,
        'triple_ture_test':triple_ture_test

    }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

#get_pair()





def load_data(PATH,test=False):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    dialogue_triple_index= np.load(PATH + 'dialogue_triple_index.npy')
    if test:
        idx_q = np.load(PATH + 'test_idx_q.npy')
        idx_a = np.load(PATH + 'test_idx_a.npy')

    Validate_idx_q = np.load(PATH + 'Validate_idx_q.npy')
    Validate_idx_a = np.load(PATH + 'Validate_idx_a.npy')
    Validate_triple_index = np.load(PATH + 'Validate_triple_index.npy')

    return metadata, idx_q, idx_a,dialogue_triple_index,Validate_idx_q,Validate_idx_a,Validate_triple_index