import pandas as pd
import jieba.analyse

voice_file = 'D:\\workspace\\worm\\video_voice.csv'
voice = pd.read_csv(voice_file, encoding='utf-8-sig', header=0)

# poi_file = 'D:\\data\\videos\\videos.csv'
# poi = pd.read_csv(poi_file, encoding='utf-8-sig', header=0)

new_voice = 'D:\\workspace\\worm\\video_new_voice_tags.csv'

ff = open(new_voice,'w',encoding='utf-8')
for i in range(len(voice['id'])):
    id = str(voice['id'][i]).replace('"','')
    title = voice['title'][i]
    voice_text = voice['voice'][i]

    voice_text = voice_text.replace('嗯','')
    voice_text = voice_text.replace('啊', '')
    voice_text = voice_text.replace('噢', '')
    voice_text = voice_text.replace('吧', '')
    voice_text = voice_text.replace('哎', '')
    voice_text = voice_text.replace('唉', '')
    voice_text = voice_text.replace('(', '')
    voice_text = voice_text.replace(')', '')

    tags = jieba.analyse.textrank(voice_text)

    t =','.join(tags)

    if len(voice_text) > 100:
        ff.write(id+',"'+title+'","'+t+'"\n')
ff.close()

