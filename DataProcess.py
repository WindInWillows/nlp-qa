import sys
import requests
from DataReader import DataReader
import jieba.posseg as pseg
import time
import word2vec
from jieba import analyse
import numpy

tfidf = analyse.extract_tags
textrank = analyse.textrank


TRAIN_DATA = 'data/training.data'
DEV_DATA = 'data/develop.data'
TEST_DATA = 'data/test.data'
STOP_WORD = 'data/stop_word'

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
        # self.max_len = 0

    def segment(self, text):
        res = []
        if self.type == 'jieba':
            lcut = pseg.lcut(text)
            for i in range(len(lcut)):
                if self.filter_x and lcut[i].flag == 'x':
                    pass
                else:
                    res.append((lcut[i].word, lcut[i].flag))
            # self.max_len = self.max_len if len(res) < self.max_len else len(res)

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

        self.qu_vec : 词向量
        self.cor_ans_vec : 正确答案词向量列表
        self.wr_ans_vec : 错误答案词向量列表
    """
    def __init__(self,seg_type='jieba', data=DEV_DATA, fill='NULL', needfill=True):
        self.base_fname = data.split('.')[0]
        self.word_file = data.split('.')[0]+'.word'
        self.bin_file = data.split('.')[0]+'.bin'
        self.seg_file = data.split('.')[0] + '.seg'
        self.vec_file = data.split('.')[0] + '.vec'

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
        self.fill_len = 0
        self.fill = fill
        # self.needfill = needfill

        self.freq = [0] * 1050

        self.stop_word = self._load_stop_word()


    def _load_stop_word(self):
        lst = []
        with open(STOP_WORD, encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                lst.append(line.split('\n')[0])
        return set(lst)

    def seg_word(self):
        print('%.2f:开始分词...'%(time.time()-now))
        with open(self.seg_file, 'w', encoding='utf-8') as fout:
            for i in range(len(self.dr)):
                if self.seg.type == 'yuyanyun':
                    time.sleep(0.0051)
                qu = self.seg.segment(self.dr.question[i])
                self.qu.append(qu)
                cor = [self.seg.segment(self.dr.cor_ans[i][j]) for j in range(len(self.dr.cor_ans[i]))]
                self.cor_ans.append(cor)
                wr = [self.seg.segment(self.dr.wr_ans[i][j]) for j in range(len(self.dr.wr_ans[i]))]
                self.wr_ans.append(wr)
                fout.write('%s,%s,%s\n'%(str(qu),str(cor),str(wr)))
        print('%.2f:分词成功'%(time.time()-now))


    def load_seg(self):
        with open(self.seg_file, encoding='utf-8') as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                qu, cor, wr = eval(line)
                lst = [len(qu)]
                self.freq[len(qu)] += 1
                for i in cor:
                    lst.append(len(i))
                    self.freq[len(i)] += 1
                self.qu.append(qu)
                self.cor_ans.append(cor)
                self.wr_ans.append(wr)
                self.max_len = max(max(lst), self.max_len)
        self.fill_len = self.get_fill_length()

    def get_fill_length(self):
        s = sum(self.freq)
        t = 0
        for i in range(len(self.freq)):
            t += self.freq[i]
            if t/s > 0.98:
                return i

    def save_word(self):
        with open(self.word_file, 'w', encoding='utf-8') as f:
            for i in range(len(self.qu)):
                lst = [w[0] for w in self.qu[i]]
                if len(lst) > self.fill_len:
                    lst = lst[:self.fill_len]
                else:
                    lst = lst + [self.fill] * (self.fill_len - len(lst))

                for j in range(len(self.wr_ans[i])):
                    tmp = [w[0] for w in self.wr_ans[i][j]]
                    if len(tmp) > self.fill_len:
                        tmp = tmp[:self.fill_len]
                    else:
                        tmp += [self.fill] * (self.fill_len - len(tmp))
                    lst += tmp
                for j in range(len(self.cor_ans[i])):
                    tmp = [w[0] for w in self.cor_ans[i][j]]
                    if len(tmp) > self.fill_len:
                        tmp = tmp[:self.fill_len]
                    else:
                        tmp += [self.fill] * (self.fill_len - len(tmp))
                    lst += tmp
                f.write('%s\n'%('\n'.join(lst)))

    def word_vector(self):
        print('%.2f:开始转化词向量'%(time.time()-now))
        word2vec.word2vec(self.word_file, self.bin_file, binary=1, verbose=False)
        print('%.2f:词向量转化完成'%(time.time()-now))

    def gene_wordvec(self):
        print('%.2f:开始填充词向量...'%(time.time()-now))
        model = word2vec.load(self.bin_file)
        # f = open(self.vec_file,'w')
        for i in range(len(self.qu)):
            cur = time.time()
            qlst = []
            for j in range(min(len(self.qu[i]), self.fill_len)):
                if self.qu[i][j][0] in model:
                    qlst.append(model[self.qu[i][j][0]])
                else:
                    qlst.append(model[self.fill])
            qlst += [model[self.fill]] * (self.fill_len-len(self.qu[i]))
            self.qu_vec.append(qlst)

            clst = []
            for j in range(len(self.cor_ans[i])):
                lst = []
                for k in range(min(len(self.cor_ans[i][j]), self.fill_len)):
                    if self.cor_ans[i][j][k][0] in model:
                        lst.append(model[self.cor_ans[i][j][k][0]])
                    else:
                        lst.append(model[self.fill])
                lst += [model[self.fill]] * (self.fill_len - len(self.cor_ans[i][j]))
                clst.append(lst)
            self.cor_ans_vec.append(clst)

            wlst = []
            for j in range(len(self.wr_ans[i])):
                lst = []
                for k in range(min(len(self.wr_ans[i][j]), self.fill_len)):
                    if self.wr_ans[i][j][k][0] in model:
                        lst.append(model[self.wr_ans[i][j][k][0]])
                    else:
                        lst.append(model[self.fill])
                lst += [model[self.fill]] * (self.fill_len - len(self.wr_ans[i][j]))
                wlst.append(lst)
            self.wr_ans_vec.append(wlst)
            # f.write('%s,%s,%s\n'%(str(qlst), str(clst), str(wlst)))

            res = '\r%.2f'%(i/len(self.qu)*100)
            sys.stdout.write(res)
        # f.close()
        print('%.2f:词向量填充完成，写入文件完成'%(time.time()-now))

    def fix(self,qu,ans):
        qus = ''.join([i[0] for i in qu])
        keys = textrank(qus)
        weis = [1/i for i in range(1, len(keys)+1)]
        m = 0
        for pa in ans:
            wei = 0
            lst = [i[0] for i in pa]
            for i in range(len(keys)):
                wei += weis[i] * (0 if keys[i] in lst else 1)
            m = max(m, wei)
        return m

    def test(self):
        total = len(self.qu)
        count = 0
        for i in range(len(self.qu)):
            cor_max = self.fix(self.qu[i], self.cor_ans[i])
            wr_max = self.fix(self.qu[i], self.wr_ans[i])

            if wr_max < cor_max:
                count += 1
        print(count/total)


    def fill_data(self):
        with open(self.word_file, 'w', encoding='utf-8') as out:
            for i in range(len(self.qu)):
                if len(self.qu[i]) < self.fill_len:
                    self.qu[i] += [(self.fill, 'null')] * (self.fill_len - len(self.qu[i]))
                for j in range(len(self.qu[i])):
                    out.write('%s\n'%self.qu[i][j][0])
            for i in range(len(self.wr_ans)):
                for j in range(len(self.wr_ans[i])):
                    if len(self.wr_ans[i][j]) < self.fill_len:
                        self.wr_ans[i][j] += [(self.fill, 'null')] * (self.fill_len - len(self.wr_ans[i][j]))
                    for k in range(len(self.wr_ans[i][j])):
                        out.write('%s\n'%self.wr_ans[i][j][k][0])
            for i in range(len(self.cor_ans)):
                for j in range(len(self.cor_ans[i])):
                    if len(self.cor_ans[i][j]) < self.fill_len:
                        self.cor_ans[i][j] += [(self.fill, 'null')] * (self.fill_len - len(self.cor_ans[i][j]))
                    for k in range(len(self.cor_ans[i][j])):
                        out.write('%s\n' % self.cor_ans[i][j][k][0])


if __name__ == '__main__':
    dp = DataProcess(data=TRAIN_DATA)
    base = time.time()
    dp.load_seg()
    # dp.test()
    # dp.save_word()
    # dp.word_vector()
    dp.gene_wordvec()
    print(type(dp.qu_vec[0][0]), dp.qu_vec[0][0])
    # model = word2vec.load(dp.bin_file)

