try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

import os
import pandas

d = './data/'
all_in = {'text': []}
to_compare = {'headline': [], 'item': []}
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

all_in = pandas.DataFrame(all_in)
to_compare = pandas.DataFrame(to_compare)

d = './data/'
headlines = []
for f in os.listdir(d):
    with open(d + f, 'rb') as fp:
        data_item = pickle.load(fp)
        keys = list(data_item.keys())
        headlines += keys

data = pandas.DataFrame(data={'headline': headlines})

from shock_pods.models.catapults import Ballista


#
if __name__ == '__main__':

    run_code = None
    save_embeddings = False

    topic_estimator = 'lda_gensim_single'
    # topic_estimator = 'lda_gensim_ensemble'

    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.5, 'offset': 1}
    topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.5, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.5, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 1, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 1, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 1, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 1, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 1, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 1, 'offset': 10}

    topic_estimator_n = 10

    ballista = Ballista(run_code=run_code,
                        topic_estimator=topic_estimator, topic_estimator_kwg=topic_estimator_kwg,
                        topic_estimator_n=topic_estimator_n,
                        save_embeddings=save_embeddings)

    data = data.rename(columns={'headline': 'text'})

    data, topics = ballista.project(data=data, text_field='text')
    # scored = ballista.score(data=data, scorer=mcc_adopted)

    # data[data['category'] == '2']['topic'].value_counts()


"""
with open('data_1665927816.p', 'rb') as fp:
    data_1 = pickle.load(fp)

with open('data_1665929638.p', 'rb') as fp:
    data_2 = pickle.load(fp)

with open('data_1665931469.p', 'rb') as fp:
    data_3 = pickle.load(fp)
"""


"""
i = 1

sample_head = list(data.items())[i][0]
sample_text = list(data.items())[i][1]

from transformers import pipeline

qa_model = pipeline("question-answering")
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
result = qa_model(question = sample_head, context = sample_text)
"""