#!/usr/bin/python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import re
import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle




EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''


FILENAME_post = '/home/evan/PycharmProjects/ncm-adv-master/data_util/stc_weibo_train_post'
FILENAME_respose = '/home/evan/PycharmProjects/ncm-adv-master/data_util/stc_weibo_train_response'


_DIGIT_RE = re.compile(r"\d{3,}")
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

limit = {
        'maxq' : 12,
        'minq' : 0,
        'maxa' : 12,
        'mina' : 3
        }

UNK = 'unk'
EOS = 'EOS'
GO  = 'input_GO__'


PAD_ID = 0
GO_ID = 1
UNK_ID = 1
EOS_ID=-1

VOCAB_SIZE = 60000




def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    line=re.sub(u'[-()`～！@#￥%……&×（=+·{＂」}「、；”】：’”，《【。》/？）——!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\']', '', str((line.encode("utf-8"))).decode())
    line = re.sub(u'  ', ' ',
                  str((line.encode("utf-8"))).decode())
    return line





def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size+1)
    del vocab[0]
    # index2word
    index2word =  ['_']+[GO] +[UNK] + [ x[0] for x in vocab ] +  [EOS]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist




'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(lines_post,lines_respose):

    filtered_q, filtered_a = [], []
    for i in range(len(lines_post)):
        qlen, alen = len(lines_post[i].split(' ')), len(lines_respose[i].split(' '))
        if qlen >= limit['minq'] and qlen < limit['maxq']:
            if alen >= limit['mina'] and alen < limit['maxa']-1:
                filtered_q.append(lines_post[i])
                filtered_a.append(lines_respose[i])

    raw_data_len = len(lines_post)
    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a




def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], q = False)

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen,  q = True):
    indices = []
    for word in seq:
#        word = unicode(word, "utf-8")
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    if not q:
        return indices + [lookup[EOS]] + [0]*(maxlen-len(seq)-1)
    
    return indices + [0]*(maxlen - len(seq))


def process_data():

    print('\n>> Read lines from file')

    lines_post = read_lines(filename=FILENAME_post)
    lines_respose = read_lines(filename=FILENAME_respose)



    print('\n:: Sample from read(p) lines')
    print(lines_post[121:125])

    print('\n:: Sample from read(p_respode) lines')
    print(lines_respose[121:125])

    print('\n>> Filter lines')
    lines_respose = [filter_line(line, EN_WHITELIST) for line in lines_respose]

    lines_post = [filter_line(line, EN_WHITELIST) for line in lines_post]


    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')

    #qlines=
    #alines=filter_data(lines_respose)

    qlines, alines = filter_data(lines_post,lines_respose)


    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))


    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized=[]
    atokenized=[]
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

    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

  #  vocab_path = os.path.join("/home/evan/PycharmProjects/ncm-adv-master/data", "vocab%d.in" % VOCAB_SIZE)  # 'tf_seq2seq_chatbot/data/vocab20000.in'

  #  idx2w, w2idx, freq_dist = create_vocabulary(vocab_path,qtokenized + atokenized, VOCAB_SIZE)


    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    output_sentence_eval = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[10]])
    output_sentence_eval1 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[11]])
    output_sentence_eval2 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[12]])
    output_sentence_eval3 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[13]])
    output_sentence_eval4 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[14]])
    output_sentence_eval5 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_q[15]])

    output_sentence_evala = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[10]])
    output_sentence_evala1 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[11]])
    output_sentence_evala2 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[12]])
    output_sentence_evala3 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[13]])
    output_sentence_evala4 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[14]])
    output_sentence_evala5 = ' '.join([str(idx2w[i3_eval]) for i3_eval in idx_a[15]])

    test_len=64*80
    test_idx_q = idx_q[0:test_len]
    test_idx_a = idx_a[0:test_len]
    Validate_idx_q = idx_q[test_len:2*test_len]
    Validate_idx_a = idx_a[test_len:2 * test_len]

    Validate_qlines= qlines[test_len:2 * test_len]
    Validate_alines= alines[test_len:2 * test_len]
    test_qlines = qlines[0:test_len]
    test_alines=alines[0:test_len]
    idx_q=idx_q[2*test_len:-1]
    idx_a=idx_a[2*test_len:-1]


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


    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    np.save('test_idx_q.npy', test_idx_q)
    np.save('test_idx_a.npy', test_idx_a)

    np.save('Validate_idx_q.npy', Validate_idx_q)
    np.save('Validate_idx_a.npy', Validate_idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH,test=False):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    if test:
        idx_q = np.load(PATH + 'test_idx_q.npy')
        idx_a = np.load(PATH + 'test_idx_a.npy')

    Validate_idx_q = np.load(PATH + 'Validate_idx_q.npy')
    Validate_idx_a = np.load(PATH + 'Validate_idx_a.npy')

    return metadata, idx_q, idx_a,Validate_idx_q,Validate_idx_a


if __name__ == '__main__':
    process_data()
    #get_vocabulary()
