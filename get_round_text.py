#!/usr/bin/python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import pickle

f_rtple = open('/home/evan/PycharmProjects/ncm-adv-master/baseline/data/cqa_triple_origina_v0', 'r')

DIR_PATH = 'E:/'



def get_entity():
    triple = list()
    entity = set()
    relation = set()
    for line in f_rtple.readlines()[1:-1]:
        line = line.strip()
        line = line.split('\t')
        ret = 0

        for i in line[2:]:
            if i.split(':')[0]==str('triple0'):
                begin=line.index(i)
                break

        for i in line[begin:]:
            try:
                if ret%3==0:
                    entity1 = i.split(':')[1]
                    entity.add(entity1)
                if ret%3==1:
                    relation1 = i
                    relation.add(relation1)
                if ret%3==2:
                    entity2 = i
                    entity.add(entity2)
                ret+=1
                if ret%3==0:
                    if [entity1,relation1,entity2] not in triple:
                        triple.append([entity1,relation1,entity2])
            except:
                print '21321'
    return list(entity),triple






def get_text():
    filepath = '/home/evan/PycharmProjects/ncm-adv-master/baseline/data/text/'
    # file_list = []
    pathDir = os.listdir(filepath)
    # for allDir in pathDir:
    #     child = os.path.join('%s%s' % (filepath, allDir))
    #     file_list.append(child)

    text_dirt = {}
    for allDir in pathDir:
        file_text = os.path.join('%s%s' % (filepath, allDir))
        fo = open(file_text, "r")
        text = fo.readline()
        if text == '':
            print ''
            continue
        dict_key = allDir.split('.txt')[0]
        text_dirt[str(dict_key)] = text.split('。')[0:-1]
        fo.close()

    import jieba
    entity,triple=get_entity()

    entity_text={}
    ret = 0
    result=0
    all=len(triple)
    for i in entity:
        entity_text[ret]=[]
        if i in text_dirt.keys() :
            for  sen in  text_dirt[i]:
                sentence = jieba.cut(sen, cut_all=False)
                sentence = (' '.join(sentence)).encode("utf-8")
                sentence = sentence.split()
                #if i[0] in sentence and i[2] in sentence:
                entity_text[ret].append(sentence)

        if len(entity_text[ret])!=0:
            result=result+1
        ret=ret+1
        print str(ret)+'all is'+str(all)
        print 'result' + str(result)

    metadata = {
        'triple_text': entity_text,
        'entity':entity,
        'triple': triple

    }

    # write to disk : data control dictionaries
    with open('triple_text.pkl', 'wb') as f:
        pickle.dump(metadata, f)

#get_text()
# with open('metadata.pkl', 'rb') as f:
#     metadata = pickle.load(f)
#
# triple_text=metadata['triple_text']
# entity,triple=get_entity()
# metadata = {
#     'triple_text': triple_text,
#     'triple':triple
# }
#
# with open('triple_text.pkl', 'wb') as f:
#     pickle.dump(metadata, f)


import re
def filter_line(line):
    line = re.sub(u'[-()`～！@#￥%……&×（=+·{＂」}「、；”】：’”，《【。》/？）——!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\']', '',
                  str((line.encode("utf-8"))).decode())
    line = re.sub(u'  ', ' ',
                  str((line.encode("utf-8"))).decode())
    return line


def get_triple():
    ret = 0
    all_text=0
    with open('triple_text.pkl', 'rb') as f:
        metadata = pickle.load(f)
    triple_text=metadata['triple_text']
    entity = metadata['entity']
    triple = metadata['triple']
    text_ture=None
    text_1=None
    triple_text_ture={}
    for i in triple_text.keys():
        triple_text_ture[i]=[]
        if i==1694:
            print ''
        for text in triple_text[i]:
            all_text = all_text + 1
            line = ' '.join([str(i3_eval) for i3_eval in text])
            line_spilt_list=line.split('，')
            for i1 in  line_spilt_list:
                try:
                    i1 = re.sub(
                        u'\（.*' + str(triple[i][2]) + '.*' + str(triple[i][0]) + '.*?\）|\（.*' + str(
                            triple[i][0]) + '.*' + str(triple[i][2]) + '.*?\）',
                        str(triple[i][0]) + ' ' + str(triple[i][2]),
                        str((i1.encode("utf-8"))).decode())
                    i1 = re.sub(u'\（.*' + str(triple[i][2]) + '.*?\）', str(triple[i][2]),
                                str((i1.encode("utf-8"))).decode())
                    i1 = re.sub(u'\（.*' + str(triple[i][0]) + '.*?\）', str(triple[i][0]),
                                str((i1.encode("utf-8"))).decode())
                    i1 = re.sub(u'\（.*?\）', '',
                                str((i1.encode("utf-8"))).decode())
                    i1 = re.sub(u'  ', ' ',
                                str((i1.encode("utf-8"))).decode())
                except:
                    pass
                if entity[int(i)] in i1.split(' ') and len(i1.split(' '))!=1 :
                    text_1=i1
                if text_1!=None:
                    text_ture=text_1
                    text_1=None
                    break

            if text_ture!=None:
                line=filter_line(text_ture)
                line_list=line.split(' ')
                while '' in line_list:
                    line_list.remove('')
                if len(line_list)>20:
                    ret=ret+1
                    print str(ret)+'all:'+str(all_text)+'i:'+str(i)
                if len(line_list) <=20:
                    triple_text_ture[i].append(' '.join([str(i3_eval) for i3_eval in line_list]))
                text_ture=None

    metadata = {
        'triple_text_ture': triple_text_ture,
        'triple':triple,
        'entity':entity
    }

    with open('triple_text_ture.pkl', 'wb') as f:
        pickle.dump(metadata, f)

#get_triple()

def get():
    with open('triple_text_ture.pkl', 'rb') as f:
        metadata = pickle.load(f)

    triple_text_ture = metadata['triple_text_ture']
    triple = metadata['triple']
    entity = metadata['entity']
    ret=0
    ret1=0
    for i in triple_text_ture.keys():
        if ret<len(triple_text_ture[i]):
            ret=len(triple_text_ture[i])
        if len(triple_text_ture[i])>5:
            while len(triple_text_ture[i])>5:
                triple_text_ture[i].pop()
            ret1=ret1+1

    print str(ret)+':1的有'+str(ret1)+'总：'+str(i)

    metadata = {
        'triple_text_ture': triple_text_ture,
        'triple':triple,
        'entity': entity

    }

    with open('triple_text_ture.pkl', 'wb') as f:
        pickle.dump(metadata, f)
get()
    #总：59449, 0：51615  1:6062 2：1275 3：315 4:137 5:24：

