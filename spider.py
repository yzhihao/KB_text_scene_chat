#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2016-10-26 13:52:57
# Project: test_education

from pyspider.libs.base_handler import *
import re
import time
f_rtple = open('/home/evan/PycharmProjects/ncm-adv-master/baseline/data/cqa_triple_origina_v0', 'r')

DIR_PATH = 'E:/'

triple=[]
entity=set()
relation=set()

def get_entity():

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
                    triple.append([entity1,relation1,entity2])
            except:
                print 1

# 主类
class Handler(BaseHandler):
    def __init__(self):
        self.deal = Deal()

    crawl_config = {
    }

    @every(minutes=24 * 60)
    def on_start(self):#
        for i in entity:
            self.crawl('https://baike.baidu.com/item/'+i, callback=self.index_page)


    @config(age=10 * 24 * 60 * 60)
    def index_page(self, response):
        title = response.doc('#title .left h1').text()
        name = title.split(':')[0]
        price = response.doc('.price').text()
        developer = response.doc('#title .left h2 ').text()
        app_url = response.save['app_url']
        app_url = app_url.split('?')[0]
        imgs = response.doc('#left-stack .lockup a .artwork meta').items()
        app_id = app_url.split('id')[1]

        dir_path = self.deal.mkDir(app_id)
        dbimgpath = ''
        if dir_path:
            imgs = response.doc('#left-stack .lockup a .artwork meta').items()
            for img in imgs:
                url = img.attr.content
                if url:
                    extension = self.deal.getExtension(url)
                    file_name = app_id + '.' + extension
                    dbimgpath = dir_path + '/' + file_name
                    self.crawl(img.attr.content, callback=self.save_img,
                               save={'dir_path': dir_path, 'file_name': file_name}, validate_cert=False)
            self.detail_page(app_id, name, dbimgpath, app_url, developer, price, 0)

    def save_img(self, response):
        content = response.content
        dir_path = response.save['dir_path']
        file_name = response.save['file_name']
        file_path = dir_path + '/' + file_name
        self.deal.saveImg(content, file_path)



# 图片保存的until类
import os


class Deal:
    def __init__(self):
        self.path = DIR_PATH
        if not self.path.endswith('/'):
            self.path = self.path + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def mkDir(self, path):
        path = path.strip()
        dir_path = self.path + path
        exists = os.path.exists(dir_path)
        if not exists:
            os.makedirs(dir_path)
            return dir_path
        else:
            return dir_path

    def saveImg(self, content, path):
        f = open(path, 'wb')
        f.write(content)
        f.close()

    def saveBrief(self, content, dir_path, name):
        file_name = dir_path + "/" + name + ".txt"
        f = open(file_name, "w+")
        f.write(content.encode('utf-8'))

    def getExtension(self, url):
        extension = url.split('.')[-1]
        return extension
