import pandas as pd
import numpy as np

ff = pd.read_csv('d:/data/voice/voice_cut.csv',encoding='utf-8-sig')

voice_list = []
poi_list = []
for l in range(len(ff['voice'])):
    voice = eval(ff['voice'][l])
    voice_list.append(','.join(voice))

    if ff['poi'][l] is np.nan:
        poi_list.append('')
    else:
        poi = ff['poi'][l].replace(' ','').split(',')
        poi_list.append(','.join(sorted(list(set(poi)))))


pd.DataFrame({'id':ff['id'],'title':ff['title'],'poi':poi_list,'voice':voice_list}).to_csv('d:/data/voice/voice_cut_.csv',encoding='utf-8',index=False)