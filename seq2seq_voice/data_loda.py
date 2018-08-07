import pandas as pd
import numpy as np


def word_vocab(file):
    """词到idx，idx到词的映射
    return idx
    :param column: 0:title，1:poi
    :return:两个映射的字典，并且去除某些使用频率比较低的词语
    """
    special_tags = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    # max_source_length = {}

    counter_source = {}
    counter_target = {}

    for i in range(len(file['id'])):

        source = file['voice'][i]
        target = file['poi'][i]

        if source is np.nan or target is np.nan:
            continue

        voice_list = source.split(',')
        # length = len(voice_list)
        # max_source_length[length] = max_source_length.get(length, 0) + 1

        for word in voice_list:
            counter_source[word] = counter_source.get(word, 0) + 1

        for word in target.split(','):
            counter_target[word] = counter_target.get(word, 0) + 1

    # max_source_length = sorted(max_source_length.items(),key=lambda x:x[1],reverse=True)
    # for k,v in max_source_length:
    #     print('%s,%s' % (k, v))
    #
    # print(counter_source['做'])
    '''删除某些词频比较低的词语
    删除词频小于某个频次的词语后还剩：
    source 的平均长度为189 因此seq2seq模型的source中使用的lstm个数为200
    target 的最大长度为26 target中使用的lstm个数为25
    
    sorce：原始:91303  '<5':23272 '<3':33690 '<2':46031    
    target：原始:124472 '<5':14081 '<3':23153 '<2':37066
    '''
    deletNum = 5
    keys = list(counter_source.keys())
    for key in keys:
        if counter_source[key] < deletNum:
            del counter_source[key]

    keys = list(counter_target.keys())
    for key in keys:
        if counter_target[key] < deletNum:
            del counter_target[key]

    '''
    构建词序索引
    '''
    # source
    count_pairs = sorted(counter_source.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    source_words_to_index = dict(zip(words, range(4, len(words)+4)))

    for i in range(len(special_tags)):
        source_words_to_index[special_tags[i]] = i

    source_index_to_words = {}

    for key, value in source_words_to_index.items():
        source_index_to_words[value] = key

    # target
    count_pairs = sorted(counter_target.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    target_words_to_index = dict(zip(words, range(4, len(words) + 4)))

    for i in range(len(special_tags)):
        target_words_to_index[special_tags[i]] = i

    target_index_to_words = {}

    for key, value in target_words_to_index.items():
        target_index_to_words[value] = key
    """return """
    print('词-索引关系加载成功')
    return source_words_to_index, target_words_to_index, source_index_to_words, target_index_to_words


def sentence_to_idx(sentence, word_vocab, max_length=200, is_source=True):
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
    if is_source:  # source 即voice文本 是list的形式保存的str
        for word in sentence.split(','):
            sts_idx.append(word_vocab.get(word, 1))
    else:  # target是逗号分割的字符串
        for word in sentence.split(','):
            sts_idx.append(word_vocab.get(word, 1))
        sts_idx.append(word_vocab.get('<EOS>'))
    '''--4---'''
    if len(sts_idx) > max_length:
        return sts_idx[:max_length]
    else:
        sts_idx = sts_idx + [word_vocab.get('<PAD>')] * (max_length - len(sts_idx))
        return sts_idx


def get_sts2idx(file, source_words_to_index, target_words_to_index):
    source_sts2idx, target_sts2idx = [], []
    for i in range(len(file['voice'])):
        if file['voice'][i] is not np.nan and file['poi'][i] is not np.nan:
            # 编码装换
            source_sts2idx.append(sentence_to_idx(file['voice'][i], source_words_to_index))
            target_sts2idx.append(sentence_to_idx(file['poi'][i], target_words_to_index, max_length=25, is_source=False))
    print('句子编码加载成功')
    return source_sts2idx, target_sts2idx


if __name__=="__main__":
    '''运行实例'''
    cut_file = 'd:/workspace/worm/voice_cut.csv'
    file = pd.read_csv(cut_file, ',', encoding='utf-8-sig')
    source_words_to_index, target_words_to_index, _, _ = word_vocab(file)

    source_sts2idx, target_sts2idx = get_sts2idx(file,source_words_to_index,target_words_to_index)

    print(len(source_sts2idx))
    print(len(target_sts2idx))

    print(source_sts2idx[1])
    print(target_sts2idx[1])



