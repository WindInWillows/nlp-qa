from aip import AipNlp
from DataReader import DataReader
import time
import json
import re

AIP_ID = 'c68c52f38cf1454cbea41543450fc4c2'
API_KEY = 'G9hCennFyiZReL9uoGcAmwYD'
SECRET_KEY = '9DAuD67SV3vClE9hN92b4SHIGyUzdMDG'

aip = AipNlp(AIP_ID, API_KEY, SECRET_KEY)

dr = DataReader('data/develop.data')
dr.filt()

# s = re.sub('\'', '\"',"{'a':'b'}")
# res = json.loads(s, encoding='utf-8')
# print(res['a'])
#
cor_count = wr_count = 0
for i in range(len(dr.question)):
    wr_max = 0
    for j in range(len(dr.wr_ans[i])):
        time.sleep(0.1)
        res = eval(str(aip.simnet(dr.question[i], dr.wr_ans[i][j])))
        wr_max = max(res['score'], wr_max)
    time.sleep(0.1)
    res = eval(str(aip.simnet(dr.question[i], dr.cor_ans[i][0])))
    if res['score'] < wr_max:
        wr_count += 1
    else:
        cor_count += 1
    print(cor_count+wr_count, cor_count/ (cor_count+wr_count))

