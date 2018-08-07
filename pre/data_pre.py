import pandas as pd
from collections import Counter
import re
import jieba
import numpy as np

ff = pd.read_csv('../videos.csv', encoding='utf-8-sig')
pat ='[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）；;“”：《》-]+'

id_list,title_list, poi_list = [], [], []
for i in range(len(ff)):
    new_title = jieba.cut(re.sub(pat, '', ff['title'][i]))
    new_poi = ff['poi'][i]

    if new_poi is not np.nan and len(ff['title'][i]) > 0:
        id_list.append(ff['video_id'][i])
        title_list.append(','.join(new_title))
        poi_list.append(new_poi)

print('saving')
pd.DataFrame({'video_id':id_list,'title': title_list, 'poi': poi_list}).to_csv('../videos_train.csv', ',', encoding='utf-8', index=False)
print('done!')

'''
title,poi
"面包,吃,不,掉,3,招,保持,面包,新鲜,好味","饮食小窍门,保鲜,面包,饮食小窍门,保鲜"
"一款,让,你,告别,花痴,的,软件,白,百合,都,能,识别","软件,移动端,软件,移动端,形色,产品推荐,图像识别"
"吉他,精品,教程,原来,课,前,准备,有,这么,多,知识点,真是,涨,知识,了","音乐教学,音乐科普,乐器教学,乐理教学,音乐教学,乐器教学,乐理教学,音乐科普,吉他,调弦"
'''