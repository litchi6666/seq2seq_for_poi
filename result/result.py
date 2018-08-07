import pandas as pd

title = pd.read_csv('title_to_poi.csv', encoding='utf-8-sig')
voice = pd.read_csv('voice_to_poi.csv', encoding='utf-8-sig')

dict_title, dict_voice = {},{}

for i in range(len(title)):
    dict_title[title['id'][i]] = title['poi'][i]


for i in range(len(voice)):
    dict_voice[voice['id'][i]] = voice['predict_poi'][i]


msg = pd.read_csv('videos.csv',encoding='utf-8-sig')

id_list, title_list, poi_list,title_to_poi,voice_to_poi = [],[],[],[],[]
for i in range(len(msg)):
    id = msg['video_id'][i]
    title = msg['title'][i]
    poi = msg['poi'][i]
    if id in dict_title or id in dict_voice:
        id_list.append(id)
        title_list.append(title)
        poi_list.append(poi)
        title_to_poi.append(dict_title.get(id,''))
        voice_to_poi.append(dict_voice.get(id,''))

r = pd.DataFrame({'id':id_list,'title':title_list,'poi':poi_list,'title_to_poi': title_to_poi, 'voice_to_poi': voice_to_poi})
r.to_csv('result.csv',columns=['id','title','poi','title_to_poi','voice_to_poi'],index=False,encoding='gbk')