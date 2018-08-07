import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec

Embedding_size = 200


def word_vocab(column=0):
    """词到idx，idx到词的映射
    return idx
    :param column: 0:title，1:poi
    :return:两个映射的字典，并且去除某些使用频率比较低的词语
    """
    if column == 0:
        col = 'title'
    elif column == 1:
        col = 'poi'
    else:
        print('wrong column')

    # file = pd.read_csv('../file/videos_train.csv', encoding='utf-8-sig')
    file = pd.read_csv('../file/new_videos_1.csv', encoding='utf-8-sig')
    counter = {}

    special_tags = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    max_length = 0
    for line in file[col]:
        words = line.split(',')
        for w in words:
            counter[w] = counter.get(w, 0) + 1
        if len(words) > max_length:
            max_length = len(words)
    print(max_length)

    '''删除某些词频比较低的词语
    删除词频小于某个频次的词语后还剩：
    sorce：原始:91303  '<5':23272 '<3':33690 '<2':46031
    target：原始:124472 '<5':14081 '<3':23153 '<2':37066
    '''
    keys = list(counter.keys())
    for key in keys:
        if counter[key] < 5:
            del counter[key]

    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    words_to_index = dict(zip(words, range(4, len(words)+4)))

    for i in range(len(special_tags)):
        words_to_index[special_tags[i]] = i

    index_to_words = {}

    for key, value in words_to_index.items():
        index_to_words[value] = key

    return index_to_words, words_to_index


def get_word_embedding_matrix(words_to_index):
    """
    将{’abc‘：1}的字典转化为二维数组，【词语个数，词向量维度】
    :param words_to_index: 字典，词语到idx的映射
    :return: 形状为【词语个数，词向量维度】的numpy二维数组，
    """
    w2c_model = Word2Vec.load('D:/model/w2v/wiki.zh.model')
    print('word2vec model load success')
    static_embedding = np.zeros([len(words_to_index), Embedding_size])
    for word, idx in words_to_index.items():
        if word in w2c_model:
            word_vec = w2c_model[word]
        else:
            word_vec = 0.2 * np.random.random(Embedding_size) - 0.1
            # 不在模型中的词语随机返回一个词向量
        static_embedding[idx, :] = word_vec

    # PAD标签的词向量重置为0
    pad_id = words_to_index['<PAD>']
    static_embedding[pad_id, :] = np.zeros(Embedding_size)

    return static_embedding


if __name__ == '__main__':
    idx_to_words, words_to_idx = word_vocab(1)

    print(len(words_to_idx))
    print(words_to_idx['<PAD>'])

    # static_embedding = get_word_embedding_matrix(words_to_idx)
    #
    # print(static_embedding[:10,:])
