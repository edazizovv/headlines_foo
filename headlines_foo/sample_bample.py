try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

import os
import pandas
from simcse import SimCSE
from scipy.spatial.distance import cosine

d = './data/'
all_in = {'text': []}
to_compare = {'headline': [], 'item': [], 'previous': [], 'j': []}
for f in os.listdir(d):
    with open(d + f, 'rb') as fp:
        data_item = pickle.load(fp)
        keys = list(data_item.keys())
        values = list(data_item.values())
        all_in['text'] += keys
        for j in range(len(keys)):
            for item in values[j].split('\n'):
                all_in['text'].append(item)
                to_compare['headline'].append(keys[j])
                to_compare['item'].append(item)
                if j == 0:
                    to_compare['previous'].append(item)
                else:
                    to_compare['previous'].append(to_compare['headline'][-2])
                to_compare['j'].append(j)

all_in = pandas.DataFrame(all_in)
to_compare = pandas.DataFrame(to_compare)

embedder_kwg = {'model_name_or_path': "princeton-nlp/sup-simcse-bert-base-uncased"}
embedder = SimCSE(**embedder_kwg)

emb_code = 'simcse_one'
# """

# embedded = all_in['text'].apply(func=embedder.encode).values
embedded = embedder.encode(all_in['text'].values.tolist()).numpy()
embeddings = {all_in['text'].values[j]: embedded[j] for j in range(all_in.shape[0])}


with open(emb_code, 'wb') as fp:
    pickle.dump(embeddings, fp, protocol=pickle.HIGHEST_PROTOCOL)
# """
"""
with open(emb_code, 'rb') as fp:
    embeddings = pickle.load(fp)
"""
"""
started = False
texted = {'headline': [], 'text': []}
heads = []
nexts = []
for i, row in to_compare.iterrows():
    to_head = cosine(embeddings[row['headline']], embeddings[row['item']])
    to_next = cosine(embeddings[row['previous']], embeddings[row['item']])
    heads.append(to_head)
    nexts.append(nexts)

distances = pandas.DataFrame(data={'heads': heads, 'nexts': nexts})
"""