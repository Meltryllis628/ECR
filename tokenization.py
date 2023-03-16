import pandas as pd
import torch
from ltp import LTP
import re
import sys

train = pd.read_csv("test_split.csv").loc[:,["Dialogue_ID", "Utterance" ,"Emotion"]]
ltp = LTP("LTP/small")

ltp.to("cuda")


def process_bar(num, total):
    rate = float(float(num)/total)
    ratenum = int(100*rate)
    r = '\r[{}{}]{}% {} Lines done.'.format('*'*ratenum,' '*(100-ratenum), ratenum, num)
    sys.stdout.write(r)
    sys.stdout.flush()

i,n =0,100
for i in range(n):

    process_bar(i+1,n)

train["Utterance"] = train["Utterance"].apply((lambda x: re.sub('[\s，。]','',x)))
speakers = pd.read_csv("speakers.txt", header = None)
for speaker in speakers:
    ltp.add_words(words=speaker)

size = len(train.index)

train["Tokenization"] = ""
for i in range(len(train.index)):
    train.loc[i, "Tokenization"] = str(ltp.pipeline(train.loc[i, "Utterance"], tasks = ["cws"])["cws"])
    process_bar(i, size)

train.to_csv("test_tokenized.csv")