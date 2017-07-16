import requests
from DataReader import DataReader
import jieba.posseg as pseg
import time
import word2vec


TRAIN_DATA = 'data/training.data'
DEV_DATA = 'data/develop.data'
TEST_DATA = 'data/test.data'


class Segmentation:
    def __init__(self, type):
        self.type = type
        self.url = "http://api.ltp-cloud.com/analysis/"
        self.args = {
            'api_key' : 'U123B0P3O7aPJczpltzpJqboORHrGhaKPpXpcF1l',
            'text' : '',
            'pattern' : 'pos',
            'format' : 'json'
        }
        self.word_lst = []

    def segment(self, text):
        res = []
        if self.type == 'jieba':
            res = pseg.lcut(text)
            for i in range(len(res)):
                res[i] = (res[i].word, res[i].flag)
        elif self.type == 'yuyanyun':
            self.args['text'] = text
            res = requests.post(self.url, self.args)
            json = res.json()[0][0]
            res = []
            for i in json:
                res.append((i['cont'], i['pos']))
        else:
            print('unknow segment type!')
        for i in res:
            self.word_lst.append(i[0])
        return res

class DataProcess:
    """
        self.qu : 分完词的问题列表，每个词是个元组，包括词和磁性
        self.wr_ans : 分完词的错误答案列表
        self.cor_ans : 分完词的正确答案列表
    """
    def __init__(self,seg_type='jieba', data=DEV_DATA):
        self.word_file = data.split('.')[0]+'.word'
        self.bin_file = data.split('.')[0]+'.bin'
        self.seg = Segmentation(seg_type)
        self.dr = DataReader(data)
        self.dr.filt()
        self.qu = []
        self.wr_ans = []
        self.cor_ans = []

    def seg_word(self):
        print('开始分词...')
        for i in range(len(self.dr)):
            if self.seg.type == 'yuyanyun':
                time.sleep(0.0051)
            self.qu.append(self.seg.segment(self.dr.question[i]))
            cor = [self.seg.segment(self.dr.cor_ans[i][j]) for j in range(len(self.dr.cor_ans[i]))]
            self.cor_ans.append(cor)
            wr = [self.seg.segment(self.dr.wr_ans[i][j]) for j in range(len(self.dr.wr_ans[i]))]
            self.wr_ans.append(wr)
        print('分词成功')
        out = open(self.word_file, 'w', encoding='utf-8')
        for w in self.seg.word_lst:
            out.write(w+'\n')
        out.close()
        print('词典文件写入成功')

    def word_vector(self):
        print('开始转化词向量')
        word2vec.word2vec(self.word_file, self.bin_file, binary=0, verbose=True)
        print('词向量转化完成')


if __name__ == '__main__':
    dp = DataProcess(data=DEV_DATA)
    dp.seg_word()
    dp.word_vector()



