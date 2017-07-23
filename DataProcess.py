import requests
from DataReader import DataReader
import jieba.posseg as pseg
import time
import word2vec


TRAIN_DATA = 'data/training.data'
DEV_DATA = 'data/develop.data'
TEST_DATA = 'data/test.data'

now = time.time()


class Segmentation:
    def __init__(self, type, filetr_x=True):
        self.type = type
        self.filter_x = filetr_x
        self.url = "http://api.ltp-cloud.com/analysis/"
        self.args = {
            'api_key' : 'U123B0P3O7aPJczpltzpJqboORHrGhaKPpXpcF1l',
            'text' : '',
            'pattern' : 'pos',
            'format' : 'json'
        }
        self.max_len = 0

    def segment(self, text):
        res = []
        if self.type == 'jieba':
            lcut = pseg.lcut(text)
            for i in range(len(lcut)):
                if self.filter_x and lcut[i].flag == 'x':
                    pass
                else:
                    res.append((lcut[i].word, lcut[i].flag))
            self.max_len = self.max_len if len(res) < self.max_len else len(res)

        elif self.type == 'yuyanyun':
            self.args['text'] = text
            res = requests.post(self.url, self.args)
            json = res.json()[0][0]
            res = []
            for i in json:
                res.append((i['cont'], i['pos']))
        else:
            print('unknow segment type!')
        return res




class DataProcess:
    """
        self.qu : 分完词的问题列表，每个词是个元组，包括词和词性
        self.wr_ans : 分完词的错误答案列表
        self.cor_ans : 分完词的正确答案列表
    """
    def __init__(self,seg_type='jieba', data=DEV_DATA, fill='FUCK', needfill=True):
        self.word_file = data.split('.')[0]+'.word'
        self.bin_file = data.split('.')[0]+'.bin'
        self.seg = Segmentation(seg_type)
        self.dr = DataReader(data)
        self.dr.filt()
        self.qu = []
        self.wr_ans = []
        self.cor_ans = []
        self.qu_vec = []
        self.wr_ans_vec = []
        self.cor_ans_vec = []
        self.max_len = 0
        self.fill = fill
        self.needfill = needfill


    def seg_word(self):
        print('%.2f:开始分词...'%(time.time()-now))
        for i in range(len(self.dr)):
            if self.seg.type == 'yuyanyun':
                time.sleep(0.0051)
            self.qu.append(self.seg.segment(self.dr.question[i]))
            cor = [self.seg.segment(self.dr.cor_ans[i][j]) for j in range(len(self.dr.cor_ans[i]))]
            self.cor_ans.append(cor)
            wr = [self.seg.segment(self.dr.wr_ans[i][j]) for j in range(len(self.dr.wr_ans[i]))]
            self.wr_ans.append(wr)
        print('%.2f:分词成功'%(time.time()-now))
        self.max_len = self.seg.max_len
        if self.needfill:
            self.fill_data()
            print('%.2f:数据填充完成...'%(time.time()-now))
        print('最大长度：%s' % self.max_len)
        print('%.2f:词典文件写入成功'%(time.time()-now))

    def word_vector(self):
        print('%.2f:开始转化词向量'%(time.time()-now))
        word2vec.word2vec(self.word_file, self.bin_file, binary=1, verbose=False)
        print('%.2f:词向量转化完成'%(time.time()-now))

    def gene_wordvec(self):
        print('%.2f:开始填充词向量...'%(time.time()-now))
        model = word2vec.load(self.bin_file)

        for i in range(len(self.qu)):
            lst = []
            for j in range(len(self.qu[i])):
                if self.qu[i][j][0] in model:
                    lst.append(model[self.qu[i][j][0]])
                else:
                    lst.append(model[self.fill])
            self.qu_vec.append(lst)

            lst = []
            for j in range(len(self.cor_ans[i])):
                try:
                    lst.append([model[self.cor_ans[i][j][k][0]]
                                if self.cor_ans[i][j][k][0] in model
                                else model[self.fill]
                                for k in range(len(self.cor_ans[i][j]))])
                except KeyError:
                    print(self.cor_ans[i][j])
            self.cor_ans_vec.append(lst)

            lst = []
            for j in range(len(self.wr_ans[i])):
                lst.append([model[self.wr_ans[i][j][k][0]]
                            if self.wr_ans[i][j][k][0] in model
                            else model[self.fill]
                            for k in range(len(self.wr_ans[i][j]))])
            self.wr_ans_vec.append(lst)
        print('%.2f:词向量填充完成'%(time.time()-now))

    def fill_data(self):
        with open(self.word_file, 'w', encoding='utf-8') as out:
            for i in range(len(self.qu)):
                if len(self.qu[i]) < self.max_len:
                    self.qu[i] += [(self.fill, 'null')] * (self.max_len - len(self.qu[i]))
                for j in range(len(self.qu[i])):
                    out.write('%s\n'%self.qu[i][j][0])
            for i in range(len(self.wr_ans)):
                for j in range(len(self.wr_ans[i])):
                    if len(self.wr_ans[i][j]) < self.max_len:
                        self.wr_ans[i][j] += [(self.fill, 'null')] * (self.max_len - len(self.wr_ans[i][j]))
                    for k in range(len(self.wr_ans[i][j])):
                        out.write('%s\n'%self.wr_ans[i][j][k][0])
            for i in range(len(self.cor_ans)):
                for j in range(len(self.cor_ans[i])):
                    if len(self.cor_ans[i][j]) < self.max_len:
                        self.cor_ans[i][j] += [(self.fill, 'null')] * (self.max_len - len(self.cor_ans[i][j]))
                    for k in range(len(self.cor_ans[i][j])):
                        out.write('%s\n' % self.cor_ans[i][j][k][0])


if __name__ == '__main__':
    dp = DataProcess(data=DEV_DATA)
    dp.seg_word()
    dp.word_vector()
    dp.gene_wordvec()
    # dp.save_vec()
    # print(len(dp.qu_vec[0]), dp.qu_vec[0][0].shape)
    # model = word2vec.load('data/develop.bin')
    # print(model['FUCK'])