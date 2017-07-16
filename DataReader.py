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


if __name__ == '__main__':
    dr = DataReader(DEV_DATA)
    dr.filt()
    dr.show()

