import  pickle
import nltk
import itertools
import copy
import numpy as np

UNK = '<UNK>'
EOS = '<EOS>'
GO  = 'input_GO__'



def load_data(PATH):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    dialogue_triple_index= np.load(PATH + 'dialogue_triple_index.npy')

    test_idx_q = np.load(PATH + 'test_idx_q.npy')
    test_idx_a = np.load(PATH + 'test_idx_a.npy')

    Validate_idx_q = np.load(PATH + 'Validate_idx_q.npy')
    Validate_idx_a = np.load(PATH + 'Validate_idx_a.npy')
    Validate_triple_index = np.load(PATH + 'Validate_triple_index.npy')

    return metadata, idx_q, idx_a,dialogue_triple_index,Validate_idx_q,Validate_idx_a,Validate_triple_index,test_idx_a,test_idx_q

def pad_seq(seq, lookup,maxlen=20):
    indices = []
    if isinstance(seq, str):
        seq=seq.split(' ')
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))

def zero_pad(qtokenized,  w2idx_q):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    text_ture_list=[]
    for one_diaogue in qtokenized:
        idx_q = []
        for one_text in one_diaogue:
            q_indices = pad_seq(one_text, w2idx_q)
            idx_q.append(np.array(q_indices))
        while len(idx_q)!=10:
            q_indices = pad_seq([], w2idx_q)
            idx_q.append(np.array(q_indices))
        text_ture_list.append(idx_q)
    np.array(text_ture_list)
    return text_ture_list

def index_(tokenized_sentences):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab=[]
    for w in set(itertools.chain(*tokenized_sentences)):
        if freq_dist[w] >=3:
            a=(w,freq_dist[w])
            vocab.append(a)
    index2word =   ['<PAD>', '<UNK>', 'input_GO__', '<EOS>']+[x[0] for x in vocab]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def get_dialogue_text():
    with open('/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/text_data_util/triple_text_ture_yet.pkl', 'rb') as f:
        metadata_ = pickle.load(f)

    triple_text_ture = metadata_['triple_text_ture']
    triple = metadata_['triple']
    metadata, idx_q, idx_a, dialogue_triple_index, Validate_idx_q, Validate_idx_a, Validate_triple_index, test_idx_a, test_idx_q = load_data(
        '/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/')
    idx2w_a_infer = metadata['idx2w_a_infer']
    w2idx_a_infer = metadata['w2idx_a_infer']
    i2w_q = metadata['idx2w_q']
    w2i_q = metadata['w2idx_q']
    triple_ture_test = metadata['triple_ture_test']
    triple_entity_i2w = metadata['i2w_entity']
    triple_raletion_i2w = metadata['i2w_relation']
    triple_raletion_i2w = np.array(triple_raletion_i2w)
    a1 = triple_ture_test[-10:-1]

    dialogue_text_value_index=[]
    dialogue_text_index = []
    ret=0
    for triple1 in dialogue_triple_index:
        ret=ret+1
        dialogue_text_index_one = []
        dialogue_text_value_index_one=[]
        for one_triple in triple1:
            if one_triple[0] != 0:
                index = list(triple).index([triple_entity_i2w[one_triple[0]], triple_raletion_i2w[one_triple[1]],idx2w_a_infer[one_triple[2]]])
                for one_text in triple_text_ture[index]:
                    dialogue_text_index_one.append(one_text)
                    dialogue_text_value_index_one.append(one_triple[2])

        while len(dialogue_text_index_one) < 10:
            dialogue_text_index_one.append([])
        while len(dialogue_text_index_one) > 10:
            dialogue_text_index_one.pop()

        while len(dialogue_text_value_index_one) < 10:
            dialogue_text_value_index_one.append(w2idx_a_infer['<dummy_entity>'])
        while len(dialogue_text_value_index_one) > 10:
            dialogue_text_value_index_one.pop()

        dialogue_text_value_index.append(np.array(dialogue_text_value_index_one))
        dialogue_text_index.append(dialogue_text_index_one)
        print str(len(dialogue_triple_index))+':'+str(ret)


    text_ture_test = []
    text_value_ture_test = []
    ret = 0
    for triple1 in triple_ture_test:
        ret = ret + 1
        dialogue_text_index_one = []
        dialogue_text_value_index_one=[]
        for one_triple in triple1:
            if one_triple[0] != 0:
                index = triple.index([triple_entity_i2w[one_triple[0]], triple_raletion_i2w[one_triple[1]],
                                     idx2w_a_infer[one_triple[2]]])
                for one_text in triple_text_ture[index]:
                    dialogue_text_index_one.append(one_text)
                    dialogue_text_value_index_one.append(one_triple[2])


        while len(dialogue_text_index_one) < 10:
            dialogue_text_index_one.append([])
        while len(dialogue_text_index_one) > 10:
            dialogue_text_index_one.pop()

        while len(dialogue_text_value_index_one) < 10:
            dialogue_text_value_index_one.append(w2idx_a_infer['<dummy_entity>'])
        while len(dialogue_text_value_index_one) > 10:
            dialogue_text_value_index_one.pop()

        text_value_ture_test.append(np.array(dialogue_text_value_index_one))

        text_ture_test.append(dialogue_text_index_one)
        print str(len(triple_ture_test)) + ':' + str(ret)


    Validate_text_index = []
    Validate_text_value_index = []
    ret = 0
    for triple1 in Validate_triple_index:
        ret = ret + 1
        dialogue_text_index_one = []
        dialogue_text_value_index_one=[]

        for one_triple in triple1:
            if one_triple[0] != 0:
                index = triple.index([triple_entity_i2w[one_triple[0]], triple_raletion_i2w[one_triple[1]],
                                     idx2w_a_infer[one_triple[2]]])
                for one_text in triple_text_ture[index]:
                    dialogue_text_index_one.append(one_text)
                    dialogue_text_value_index_one.append(one_triple[2])

        while len(dialogue_text_index_one) < 10:
            dialogue_text_index_one.append([])
        while len(dialogue_text_index_one) > 10:
            dialogue_text_index_one.pop()

        while len(dialogue_text_value_index_one) < 10:
            dialogue_text_value_index_one.append(w2idx_a_infer['<dummy_entity>'])
        while len(dialogue_text_value_index_one) > 10:
            dialogue_text_value_index_one.pop()

        Validate_text_value_index.append(np.array(dialogue_text_value_index_one))

        Validate_text_index.append(dialogue_text_index_one)


        print str(len(Validate_triple_index)) + ':' + str(ret)

    return Validate_text_index,text_ture_test,dialogue_text_index,Validate_text_value_index,text_value_ture_test,dialogue_text_value_index






def get_index():
    with open('/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/text_data_util/triple_text_ture_yet.pkl', 'rb') as f:
        metadata = pickle.load(f)

    triple_text_ture = metadata['triple_text_ture']
    triple = metadata['triple']

    list_word=[]
    for i in triple_text_ture.keys():
        if len(triple_text_ture[i]) != 0:
            for one_text in triple_text_ture[i]:
                wordlist_ = str(one_text).split(' ')
                while '' in wordlist_:
                    wordlist_.remove('')
                list_word.append(wordlist_)

    Validate_text_index, text_ture_test, dialogue_text_index,Validate_text_value_index,text_value_ture_test,dialogue_text_value_index=get_dialogue_text()
    index2word, word2index, freq_dist=index_(list_word)

    dialogue_text_index=zero_pad(dialogue_text_index,word2index)
    Validate_text_index = zero_pad(Validate_text_index, word2index)
    text_ture_test = zero_pad(text_ture_test, word2index)
    np.save('dialogue_text_index.npy', dialogue_text_index)
    np.save('Validate_text_index.npy', Validate_text_index)
    np.save('text_ture_test.npy', text_ture_test)

    metadata_text = {
        'index2word_text': index2word,
        'word2index_text': word2index,
        'Validate_text_value_index': Validate_text_value_index,
        'text_value_ture_test': text_value_ture_test,
        'dialogue_text_value_index': dialogue_text_value_index,
    }

    # write to disk : data control dictionaries
    with open('metadata_text.pkl', 'wb') as f:
        pickle.dump(metadata_text, f)



#get_index()


def load_data(PATH):
    with open(PATH + 'metadata_text.pkl', 'rb') as f:
        metadata_text = pickle.load(f)
    text_ture_test = np.load(PATH + 'text_ture_test.npy')
    Validate_text_index = np.load(PATH + 'Validate_text_index.npy')
    dialogue_text_index = np.load(PATH + 'dialogue_text_index.npy')

    return metadata_text, dialogue_text_index,Validate_text_index,text_ture_test

# metadata_text, dialogue_text_index, Validate_text_index, text_ture_test=load_data('/home/evan/PycharmProjects/ncm-adv-master/KB_text_Chat/text_data_util/')
# w2i_text=metadata_text['word2index_text']
# i2w_text = metadata_text['index2word_text']
# vocab_size_text = len(w2i_text)
# Validate_text_value_index=metadata_text['Validate_text_value_index']
# text_value_ture_test=metadata_text['text_value_ture_test']
# dialogue_text_value_index=metadata_text['dialogue_text_value_index']
#
# output_sentence_eval = ' '.join([str(i2w_text[i3_eval]) for i3_eval in list(dialogue_text_value_index[1][0])])
# output_sentence_eval1 = ' '.join([str(i2w_text[i3_eval]) for i3_eval in list(dialogue_text_value_index[2][1])])
# print (str(output_sentence_eval).decode('string_escape'))
# print (str(output_sentence_eval1).decode('string_escape'))