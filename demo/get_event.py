#!/bin/env python2.7
# encoding=utf-8
import sys,os,json
import urllib2
import requests
import codecs
reload(sys)
sys.setdefaultencoding("utf-8")
from urllib import quote
from urllib2 import *
sentence = sys.argv[1]
def get_post(body):
    url = "http://127.0.0.1:8889/IEDemo"
    headers = {'content-type': 'application/json'}
    res = requests.post(url,body,headers=headers)
    return res.text 

def post_zeus(query):
    q_d = {}
    q_d['query'] = query
    return get_post(json.dumps(q_d))

def get_tag(query):
    if '?' in query:
        query = query.replace('?','')
    if '[' in query:
        query = query.replace('[','')
    if ']' in query:
        query = query.replace(']','')
    sentence = query
    host_name = 'http://11.251.194.202:5555/tag'
    sentence = [i for i in unicode(sentence, 'utf-8')]
    sentence = "\1".join(sentence)
    host_name  = host_name + '/' + sentence
    f =  urlopen(host_name)
    data = f.read()
    res =  data.decode('utf-8')
    return res

def decode_result(res,query):
    res = json.loads(res)
    if True:
        sen = res['text']
        tags = res['spo_list']
        print "<br>"
        old_offset = 0
        event_type = tags[0]['predicate']
        print("Event:")
        print(event_type)
        print('<br>')
        for item in tags:
            mention  = item['mention']
            label = item['type']
            start = item['offset']
            offset = start
            end  = start + len(mention) 
            print sen[old_offset:offset]
            old_offset = end 
            if label == u'accident':
                print '<font color=red>'+ mention +'('+label+')</font>'
            if label == u'object':
                print '<font color=blue>'+mention+'('+label+')</font>'
            if label == u'participant':
                print '<font color=purple>'+mention+'('+label+')</font>'
            if label == u'time':
                print '<font color=green>'+mention+'('+label+')</font>'
            if label == u'location':
                print '<font color=pink>'+mention+'('+label+')</font>'
            if label == u'denoter':
                print '<font color=red>'+mention+'('+label+')</font>'
    print sen[old_offset:]

res = post_zeus(sentence)
decode_result(res,sentence)

