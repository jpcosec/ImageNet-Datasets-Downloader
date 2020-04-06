import pandas as pd
classes=[i.split(":")[-1] for i in open("Imagenet_classes").read().replace("}","").split('\n')]
classes=[i.replace("'",'') for i in classes]
data=pd.read_csv("classes_in_imagenet.csv")
"""
s1 = pd.Series(classes)
for i in data['class_name']:
...     s1.str.contains(i)
...     
"""


unsplit=[]
for i in classes:
     for j in i.split(","):
        unsplit.append(j.strip())

ids=data[data.isin(unsplit)['class_name']]['synid']
f=open("dowloader.sh",'w')

f.write("""python ./downloader.py \
    -data_root ./test1 \
    -use_class_list True \
    -class_list %s \
    -images_per_class 5 \
    -multiprocessing_workers 24 \ 
"""%" ".join([i for i in ids.values]))

f.close()