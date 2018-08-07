import pandas as pd
import jieba
import jieba.posseg

to_cut_file = 'D:\\data\\voice\\video_voice_clean.csv'
to_cut = pd.read_csv(to_cut_file,encoding='utf-8-sig',header=0)

poi_file = 'D:\\data\\videos\\videos.csv'
poi = pd.read_csv(poi_file, encoding='utf-8-sig', header=0)

voice_cut = 'D:\\data\\voice\\voice_cut_n_v.csv'

def stop_words():
    '''获取并且返回停用词表'''
    stopwords = []
    with open('../stopwords.txt', 'r', encoding='utf-8-sig') as ff:
        lines = ff.readlines()
        for l in lines:
            stopwords.append(l.strip())
    print(len(stopwords))
    return stopwords


stopwords = stop_words()

video_ids = poi['video_id'].tolist()


id_list,title_list,poi_list,voice_cut_list = [],[],[],[]

for i in range(len(to_cut['id'])):
    try:
        id = to_cut['id'][i]
        voice = to_cut['voice'][i]

        index = video_ids.index(id)
        title = poi['title'][index]
        poi_ = poi['poi'][index]

        voice_tok = jieba.posseg.cut(voice)

        id_list.append(id)
        title_list.append(title)
        poi_list.append(poi_)

        # 去停用词语
        tok_list = []
        for w in voice_tok:
            if w not in stopwords:
                if w.flag[0] == 'n' or w.flag[0] == 'v':  # 仅保留动词和名词
                    tok_list.append(w.word)
        voice_cut_list.append(tok_list)

        if i%100 == 0:
            print(i)
    except:
        print("==============",i)
save = pd.DataFrame({'id':id_list,'title':title_list,'poi':poi_list,'voice':voice_cut_list},columns=['id','title',
                                                                                                     'poi','voice'])
save.to_csv(voice_cut,',',encoding='utf-8')

