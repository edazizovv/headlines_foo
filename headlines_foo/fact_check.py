from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

import os
import pandas
# tokenizer = AutoTokenizer.from_pretrained("elozano/bert-base-cased-clickbait-news")

# model = AutoModelForSequenceClassification.from_pretrained("elozano/bert-base-cased-clickbait-news")

sentence = "I like you. I love you"
# encoded = tokenizer(sentence, return_tensors="pt")
# embedded = model(encoded)


model = "elozano/bert-base-cased-clickbait-news"
tokenizer = "elozano/bert-base-cased-clickbait-news"

# model = "valurank/distilroberta-clickbait"
# tokenizer = "valurank/distilroberta-clickbait"

from transformers import TextClassificationPipeline

clickbait_pipe = TextClassificationPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(
                model
            ),
            tokenizer=AutoTokenizer.from_pretrained(tokenizer),
        )


d = './data/'
headlines = []
for f in os.listdir(d):
    with open(d + f, 'rb') as fp:
        data_item = pickle.load(fp)
        keys = list(data_item.keys())
        headlines += keys

data = pandas.DataFrame(data={'headline': headlines})


results = clickbait_pipe(data['headline'].values.tolist())
statuses = [x['label'] for x in results]
scores = [x['score'] for x in results]

data['headline_status'] = statuses
data['headline_score'] = scores

