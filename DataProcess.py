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
        self.word_lst = []
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
        for i in res:
            self.word_lst.append(i[0])
        return res


class DataProcess:
    """
        self.qu : 分完词的问题列表，每个词是个元组，包括词和词性
        self.wr_ans : 分完词的错误答案列表
        self.cor_ans : 分完词的正确答案列表
    """
    def __init__(self,seg_type='jieba', data=DEV_DATA, fill='NAN', needfill=True):
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
        print('%s:开始分词...'%str(time.time()-now))
        for i in range(len(self.dr)):
            if self.seg.type == 'yuyanyun':
                time.sleep(0.0051)
            self.qu.append(self.seg.segment(self.dr.question[i]))
            cor = [self.seg.segment(self.dr.cor_ans[i][j]) for j in range(len(self.dr.cor_ans[i]))]
            self.cor_ans.append(cor)
            wr = [self.seg.segment(self.dr.wr_ans[i][j]) for j in range(len(self.dr.wr_ans[i]))]
            self.wr_ans.append(wr)
        print('%s:分词成功'%str(time.time()-now))

        if self.needfill:
            self.seg.word_lst.append(self.fill)
            self._fill_data()
            print('%s:数据填充完成...'%str(time.time()-now))

        out = open(self.word_file, 'w', encoding='utf-8')
        for w in self.seg.word_lst:
            out.write(w+'\n')
        out.close()
        self.max_len = self.seg.max_len
        print('%s:词典文件写入成功'%str(time.time()-now))

    def word_vector(self):
        print('%s:开始转化词向量'%str(time.time()-now))
        word2vec.word2vec(self.word_file, self.bin_file, binary=1, verbose=False)
        print('%s:词向量转化完成'%str(time.time()-now))

    def gene_wordvec(self):
        print('%s:开始填充词向量...'%str(time.time()-now))
        model = word2vec.load(self.bin_file)
        for i in range(len(self.qu)):
            lst = []
            for j in range(len(self.qu[i])):
                lst.append(model[self.qu[i][j][0]])
            self.qu_vec.append(lst)

            lst = []
            for j in range(len(self.cor_ans[i])):
                try:
                    lst.append([model[self.cor_ans[i][j][k][0]] for k in range(len(self.cor_ans[i][j]))])
                except KeyError:
                    print(self.cor_ans[i][j])
            self.cor_ans_vec.append(lst)

            lst = []
            for j in range(len(self.wr_ans[i])):
                lst.append([model[self.wr_ans[i][j][k][0]] for k in range(len(self.wr_ans[i][j]))])
            self.wr_ans_vec.append(lst)
        print('%s:词向量填充完成'%str(time.time()-now))

    def _fill_data(self):
        for i in range(len(self.qu)):
            if len(self.qu[i]) < self.max_len:
                self.qu[i] += [(self.fill,'null') for j in range(len(self.qu[i]), self.max_len)]
        for i in range(len(self.wr_ans)):
            for j in range(len(self.wr_ans[i])):
                if len(self.wr_ans[i][j]) < self.max_len:
                    self.wr_ans[i][j] += [(self.fill,'null') for j in range(len(self.wr_ans[i][j]), self.max_len)]
        for i in range(len(self.cor_ans)):
            for j in range(len(self.cor_ans[i])):
                if len(self.cor_ans[i][j]) < self.max_len:
                    self.cor_ans[i][j] += [(self.fill,'null') for j in range(len(self.cor_ans[i][j]), self.max_len)]

if __name__ == '__main__':
    dp = DataProcess(data=DEV_DATA)
    dp.seg_word()
    dp.word_vector()
    dp.gene_wordvec()