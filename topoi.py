import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
import jieba.analyse
from jieba.analyse.tfidf import TFIDF


def load_model():
    print('loading data')
    idf, _ = TFIDF().idf_loader.get_idf()
    model = Word2Vec.load('D:\\data\\wiki_embedding\\model\\wiki.zh.model')
    with open('file/stopwords.txt', 'r', encoding='utf-8-sig') as sw:
        stop_words = [line.strip() for line in sw.readlines()]
    print('load success')

    return idf, model,stop_words


def simi(word1, word2, model):
    try:
        value = 0.0
        if word1 in model and word2 in model:
            value = model.similarity(word1, word2)
        return value
    except:
        print(word1,word2)
        return 0.0


def comput_biggst_tf_idf(sentence, idf, number=10, stp_ws=[]):
    '''

    :param sentence:
    :param idf:
    :param number:
    :param stp_ws:
    :return:
    '''
    words = sentence.split(',')
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0.0) + 1.0

    totle = sum(freq.values())
    dict_weights = {}
    for w in freq.keys():
        if w in stp_ws:  # 去停用词
            continue
        weight = idf.get(w, 0.) * freq[w] / totle
        dict_weights[w] = weight

    if len(dict_weights) < 1:
        return []

    sorted_pair = sorted(dict_weights.items(), key=lambda x: x[1], reverse=True)
    new_words, _ = zip(*sorted_pair)

    number = min(len(words), number)
    return new_words[:number]  # 返回去了停用词的关键词


def list_clean(sentence_list):
    words_dict = {}
    iterator = 0
    for w in sentence_list:
        w = w.strip()
        if w not in words_dict:
            words_dict[w] = iterator
            iterator += 1
    sorted_w = sorted(words_dict.items(), key=lambda x: x[1], reverse=False)
    words, _ = list(zip(*sorted_w))
    words_dict.clear()
    return words


def process():
    ff = pd.read_csv('file/voice_cut_n_v.csv', encoding='utf-8-sig')
    # ff = pd.read_csv('file/test.csv', encoding='utf-8-sig')
    idf, model, stop_words = load_model()

    id_list, poi_list, title_list, predicted_list = [], [], [], []

    for i in range(len(ff['id'])):
    # for i in range(100):
        id = ff['id'][i]
        poi = ff['poi'][i]

        title = ff['title'][i]
        voice = ff['voice'][i]

        if title is np.nan or voice is np.nan:
            continue
        # 提取关键词
        important_title_words = comput_biggst_tf_idf(','.join(jieba.cut(title)), idf, 5,stop_words)
        important_poi_words = comput_biggst_tf_idf(voice, idf, 10, stop_words)

        if len(important_title_words) < 1:
            predicted_poi = important_poi_words
        elif len(important_poi_words) < 1:
            predicted_poi = important_title_words
        else:
            simi_dict = {}
            # 计算相似度
            for w1 in important_title_words:
                for w2 in important_poi_words:
                    simi_value = simi(w1,w2,model)
                    simi_dict[(w1,w2)] = simi_value

            # 相似度排序
            sorted_pairs = sorted(simi_dict.items(), key=lambda x: x[1], reverse=True)
            big_pairs, _ = zip(*sorted_pairs)

            # 去重
            important_words = []
            for tuple_p in big_pairs:
                important_words.append(tuple_p[0])
                important_words.append(tuple_p[1])
            # 去重
            predicted_poi = list_clean(important_words)

        # 输出
        id_list.append(id)
        title_list.append(title)
        poi_list.append(poi)
        predicted_list.append(','.join(predicted_poi))
        if i > 0 and i % 100 == 0:
            print(i)

    new_ff = pd.DataFrame({'id': id_list, 'title': title_list, 'poi': poi_list, 'predict': predicted_list})
    new_ff.to_csv('naive_poi.csv', encoding='utf-8', index=False, columns=['id', 'title', 'poi', 'predict'])


if __name__ == '__main__':
    process()
