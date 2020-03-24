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
    #url = "http://11.251.194.202:5000/IEDemo"
    url = "http://127.0.0.1:8890/IEDemo"
    headers = {'content-type': 'application/json'}
    res = requests.post(url,body,headers=headers)
    return res.text 

j1 = '''   <script>
    var triples = [
        '''

j2 = '''
        ];

    var svg = d3.select("#svg-body").append("svg")
                .attr("width", 800)
                .attr("height", 600)
                ;

    var force = d3.layout.force().size([800, 600]);
    var graph = triplesToGraph(triples);

    update();

  </script>
'''
def post_zeus(query):
    q_d = {}
    q_d['query'] = query
    return get_post(json.dumps(q_d))

def decode_result(res,query):
    res = json.loads(res)
    tags = res['links']
    triple = "" 
    for item in tags:
        t = item['source']
        r = item['value']
        h = item['target']
        triple += '{subject:"' + h + '",     predicate:"' + r + '",     object:"' + t + '"},' 
        print(h+" "+r+" "+t+"</br>")
    triple = triple[:-1]
    print(j1+triple+j2)

def decode_openue_result(res,query):
    res = json.loads(res)
    triple = "" 
    if True:
        tags = res['spo_list']
        for item in tags:
            t = item['object']
            r = item['predicate']
            h = item['subject']
            if h == u'感染' or h == u'病毒':
                continue
            triple += '{subject:"' + h + '",     predicate:"' + r + '",     object:"' + t + '"},' 
            print(h+" "+r+" "+t+"</br>")
    triple = triple[:-1]
    print(j1+triple+j2)
    print('<br>')
    print("关系预测时间: "+str(res['0']['rel_time'])+'ms<br>')
    print("实体预测时间: "+str(res['0']['ent_time'])+'ms<br>')
    print("全部预测时间: "+str(res['0']['total_time'])+'ms<br>')
    prob = [str(x) for x in res['class_prob']]
    print("关系概率: "+" ".join(prob)+'<br>')

#sentence = "2014年， 8月17日傍晚香港警方公布的当日由维多利亚公园出发参与反占中保普选大游行的人数2014年反对派七一游行时警方公布的最高峰数字是98600人2014年8月17日晚大联盟公布初步统计数字反占中游行参与者为19.3万人3"
res = post_zeus(sentence)
decode_openue_result(res,sentence)

