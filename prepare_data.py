#!/usr/bin/env python

import pandas as pd

SRCTRAINFILE='yahoo_answers_csv/train.csv'
SRCVALIDATIONFILE='yahoo_answers_csv/test.csv'

DSTTRAINFILE='comprehend-train.csv'
DSTVALIDATIONFILE='comprehend-test.csv'

MAXITEM=100000

trainFrame = pd.read_csv(SRCTRAINFILE, header=None)

for i in range(1, 11):
    num = len(trainFrame[trainFrame[0] == i])
    dropnum = num - MAXITEM
    indextodrop = trainFrame[trainFrame[0] == i].sample(n=dropnum).index
    trainFrame.drop(indextodrop, inplace=True)

trainFrame[0] = trainFrame[0].apply({
                    1:'SOCIETY_AND_CULTURE',
                    2:'SCIENCE_AND_MATHEMATICS',
                    3:'HEALTH',
                    4:'EDUCATION_AND_REFERENCE',
                    5:'COMPUTERS_AND_INTERNET',
                    6:'SPORTS',
                    7:'BUSINESS_AND_FINANCE',
                    8:'ENTERTAINMENT_AND_MUSIC',
                    9:'FAMILY_AND_RELATIONSHIPS',
                    10:'POLITICS_AND_GOVERNMENT'
                }.get)
trainFrame['document'] = trainFrame[trainFrame.columns[1:]].apply(
    lambda x: ' \\n '.join(x.dropna().astype(str)),
    axis=1
)
trainFrame.drop([1, 2, 3], axis=1, inplace=True)
trainFrame['document'] = trainFrame['document'].str.replace(',', '&#44;')
trainFrame.to_csv(path_or_buf=DSTTRAINFILE,
                  header=False,
                  index=False,
                  escapechar='\\',
                  doublequote=False,
                  quotechar='"')


validationFrame = pd.read_csv(SRCVALIDATIONFILE, header=None)
validationFrame['document'] = validationFrame[validationFrame.columns[1:]].apply(
    lambda x: ' \\n '.join(x.dropna().astype(str)),
    axis=1
)
validationFrame.drop([0, 1, 2, 3], axis=1, inplace=True)
validationFrame['document'] = validationFrame['document'].str.replace(',', '&#44;')
validationFrame.to_csv(path_or_buf=DSTVALIDATIONFILE,
                       header=False,
                       index=False,
                       escapechar='\\',
                       doublequote=False,
                       quotechar='"')
