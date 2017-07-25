import jieba.posseg as pseg
import time

TRAIN_INPUT = 'data/develop.data'

def segment(input,output):
    outlist = []
    with open(input, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            lst = pseg.lcut(line)
            outlist.append(lst)
            # print(lst)
            # break

if __name__ == '__main__':
    pass
    # init = time.time()
    # segment(TRAIN_INPUT,None)
    # print(time.time() - init)
    # with open('data/test1',encoding='utf-8') as input:
    #     a,b,c = eval(input.readline())
    #     print(a[0][0])
    # from jieba import analyse
    #
    # tfidf = analyse.extract_tags
    # textrank = analyse.textrank
    #
    # text = '雨刮器包含哪些部件？'
    #
    # keys = tfidf(text)
    # k = textrank(text)
    # print(keys)
    # print(k)



