from pre.data_utils import word_vocab
import pandas as pd


def to_word(word_idx, words_list):
    return words_list[word_idx]


def stop_words():
    '''获取并且返回停用词表'''
    stopwords = []
    with open('../file/stopwords.txt', 'r', encoding='utf-8-sig') as ff:
        lines = ff.readlines()
        for l in lines:
            stopwords.append(l.strip())
    return stopwords


def sentence_to_idx(sentence, word_vocab, stopwords, max_length=25, is_source=True):
    """
    1.返回一个句子的编码，输入的文本为str 为一个句子分好词的结果，如"面包,吃,不,掉,3,招,保持,面包,新鲜,好味"
    2.去掉停用词
    3.如果不是源文本，即目标文本，需要在结尾加上<EOS>
    4.如果文本超过最大长度需要截断文本，低于最大长度需要将文本以<PAD>补齐
    :param sentence:
    :param word_vocab:
    :param max_length:
    :param is_target:special_tags = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    :return:
    """
    sts_idx = []
    '''--2--'''
    for word in sentence.split(','):
        if word not in stopwords:
            sts_idx.append(word_vocab.get(word, 1))
    '''--3---'''
    if not is_source:
        sts_idx.append(word_vocab.get('<EOS>'))
    '''--4---'''
    if len(sts_idx) > max_length:
        return sts_idx[:max_length]
    else:
        sts_idx = sts_idx + [word_vocab.get('<PAD>')] * (max_length - len(sts_idx))
        return sts_idx


def get_sts2idx():
    stopwords = stop_words()

    file = pd.read_csv('../file/videos_train.csv', ',', encoding='utf-8-sig')
    file = pd.read_csv('../file/new_videos_1.csv', ',', encoding='utf-8-sig')

    # 对源句子进行编码转换
    source_sts2idx = []
    _, words_idx = word_vocab(0)
    for line in file['title']:
        source_sts2idx.append(sentence_to_idx(line, words_idx, stopwords))

    # 对目标句子进行编码转换
    target_sts2idx = []
    _, words_idx = word_vocab(1)
    for line in file['poi']:
        target_sts2idx.append(sentence_to_idx(line, words_idx, stopwords, is_source=False))

    return source_sts2idx, target_sts2idx


if __name__ == '__main__':
    special_tags = '<PAD>,<UNK>,<GO>,<EOS>'
    idx_to_words, words_to_idx = word_vocab(0)
    print(sentence_to_idx(special_tags, words_to_idx))
    print(sentence_to_idx("教,你,如何,打造,80,年代,Party,专用,妆容", words_to_idx, is_source=False))

    # source_sts2idx, target_sts2idx = get_sts2idx()
    # print(source_sts2idx[:20])

    print(idx_to_words[5])