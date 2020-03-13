import sys
import json
f1 = open(sys.argv[1])
label = {}
for l in f1:
    spo = json.loads(l.strip())['spo_list']
    if len(spo) != 0:
        for item in spo:
            p = item['event']
            if p not in label:
                label[p] = 1
            else:
                label[p] +=1
label_list = []
for k,v in label.items():
    print(k)
    print(v)
    label_list.append(k)
    
print(label_list)

