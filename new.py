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
    init = time.time()
    segment(TRAIN_INPUT,None)
    print(time.time() - init)