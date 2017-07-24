from jieba import analyse

tfidf = analyse.extract_tags
textrank = analyse.textrank

TRAIN_DATA = 'data/training.data'
DEV_DATA = 'data/develop.data'


class DataReader:
    def __init__(self, file):
        self.train_file = open(file, encoding='utf-8')
        self.question = []
        self.cor_ans = []
        self.wr_ans = []

    def filt(self):
        pre = ''
        corlst = []
        wrlst = []
        while True:
            flag = False
            line = self.train_file.readline()
            if not line:
                if not flag:
                    self.cor_ans.append(corlst)
                    self.wr_ans.append(wrlst)
                break
            lst = line.split('\t')
            if pre != lst[0]:
                flag = True
                pre = lst[0]
                self.question.append(pre)
                if len(corlst) != 0 or len(wrlst) != 0:
                    self.cor_ans.append(corlst)
                    self.wr_ans.append(wrlst)
                corlst = []
                wrlst = []
            if int(lst[-1]) == 1:
                corlst.append(lst[1])
            elif int(lst[-1]) == 0:
                wrlst.append(lst[1])
            else:
                print(lst[-1])

    def __len__(self):
        return len(self.question)

    def show(self):
        print(self.wr_ans[:100])
        # pass

    def fix(self,qu,ans):
        keys = tfidf(qu)
        keys2 = textrank(qu)
        klen = min(len(keys2), len(keys))
        # for i in range(klen):

        m = 0
        for i in range(len(ans)):
            t = 0
            for k in range(len(keys)):
                t += 1 / (k + 1) * (1 if keys[k] in ans[i] else 0)
            for k in range(len(keys2)):
                t += 1 / (k + 1) * (1 if keys2[k] in ans[i] else 0)
            m = max(t, m)
        return m

    def test(self):
        count = 0
        for i in range(len(self.question)):
            cor = self.fix(self.question[i], self.cor_ans[i])
            wr = self.fix(self.question[i], self.wr_ans[i])
            if wr < cor:
                count += 1
        print(count/len(self.question))


if __name__ == '__main__':
    dr = DataReader(TRAIN_DATA)
    dr.filt()
    dr.test()





