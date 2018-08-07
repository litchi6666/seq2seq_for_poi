import re
import jieba.analyse
import jieba

print(list(str(-100)))

abc = '怎么的什么破了，地下三层停车位停得满满的我都'\
      '绕快没油了才抢到一个车位，是你车技差听不进去，'\
      '找不到停车位很正常。到2016年年底全中国有超过2'\
      '亿辆汽车，保守估计停车位缺口超过5000万个，这可比3'\
      '000万娶不到老婆的剩男还厉害，就说北京560万辆车停车费'\
      '只有300万个，对了，里面还有有110万个是非法运营的，'\
      '找不到正规停车场就停路边被我算了一笔账，平地下一个月一'\
      '千停路边贴条一次罚200那么多车停着总不能天天来贴条，'\
      '就算我倒霉一筹备贴一次，一个月才800还省200，呢'\
      '停车费贵也很正常，道路交通有公益性，停车场又没有'\
      '公益性，本来就不该多show，就该用价格控制你们这种'\
      '占用城市空间资源的行为。我开车就闭嘴，想第一时'\
      '间看这么多有趣的视频和漫画，快来关注是之车学院呀。'
# import jieba.posseg
# togs = jieba.posseg.cut(abc)
# for word in togs:
#     if word.flag[0] == 'n' or word.flag[0]=='v':
#         print(word.word,word.flag)
ids = '单品搭配 ,上装,单品搭配,上装,穿搭元素,高级风,女性'.replace(' ','').split(',')
ids = sorted(list(set(ids)),reverse=True)
print(','.join(ids))
#
# pat ='[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）；;“”]+'
#
# tags = jieba.analyse.textrank('怎么的什么破了，地下三层停车位停得满满的我都'
#                               '绕快没油了才抢到一个车位，是你车技差听不进去，'
#                               '找不到停车位很正常。到2016年年底全中国有超过2'
#                               '亿辆汽车，保守估计停车位缺口超过5000万个，这可比3'
#                               '000万娶不到老婆的剩男还厉害，就说北京560万辆车停车费'
#                               '只有300万个，对了，里面还有有110万个是非法运营的，'
#                               '找不到正规停车场就停路边被我算了一笔账，平地下一个月一'
#                               '千停路边贴条一次罚200那么多车停着总不能天天来贴条，'
#                               '就算我倒霉一筹备贴一次，一个月才800还省200，呢'
#                               '停车费贵也很正常，道路交通有公益性，停车场又没有'
#                               '公益性，本来就不该多show，就该用价格控制你们这种'
#                               '占用城市空间资源的行为。我开车就闭嘴，想第一时'
#                               '间看这么多有趣的视频和漫画，快来关注是之车学院呀。',)
#
#
# print(len(str))
#
# print(','.join(tags))
#
# def stop_words():
#     '''获取并且返回停用词表'''
#     stopwords = []
#     with open('/stopwords.txt', 'r', encoding='utf-8-sig') as ff:
#         lines = ff.readlines()
#         for l in lines:
#             stopwords.append(l.strip())
#     print(len(stopwords))
#     return stopwords
#
# stop = stop_words()
# print(stop[1:10])
#
# if '了' in stop:
#     print('了')