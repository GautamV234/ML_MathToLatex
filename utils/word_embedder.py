from collections import Counter
from itertools import combinations
from math import log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, norm
from string import punctuation


class WordEmbedder():
    def __init__(self,vec_size = 10):
        self.vec_size = vec_size
        self.words_dataset = "HN_posts_year_to_Sep_26_2016.csv"
        self.punctrans = str.maketrans(dict.fromkeys(punctuation))


# df.head()

    def tokenize(self,title):
        x = title.lower() # Lowercase
        x = x.encode('ascii', 'ignore').decode() # Keep only ascii chars.
        x = x.translate(self.punctrans) # Remove punctuation
        return x.split() # Return tokenized.

    def generate_embeddings(self,):

        df = pd.read_csv(self.words_dataset, usecols=['title'])
        texts_tokenized = df['title'].apply(self.tokenize)

        cx = Counter()
        cxy = Counter()
        for text in texts_tokenized:
            for x in text:
                cx[x] += 1
            for x, y in map(sorted, combinations(text, 2)):
                cxy[(x, y)] += 1

        # print('%d tokens before' % len(cx))
        min_count = (1 / 1000) * len(df)
        max_count = (1 / 50) * len(df)
        for x in list(cx.keys()):
            if cx[x] < min_count or cx[x] > max_count:
                del cx[x]
        # print('%d tokens after' % len(cx))
        # print('Most common:', cx.most_common()[:25])

        for x, y in list(cxy.keys()):
            if x not in cx or y not in cx:
                del cxy[(x, y)]

        x2i, i2x = {}, {}
        for i, x in enumerate(cx.keys()):
            x2i[x] = i
            i2x[i] = x

        sx = sum(cx.values())
        sxy = sum(cxy.values())

        pmi_samples = Counter()
        data, rows, cols = [], [], []
        for (x, y), n in cxy.items():
            rows.append(x2i[x])
            cols.append(x2i[y])
            data.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
            pmi_samples[(x, y)] = data[-1]
        PMI = csc_matrix((data, (rows, cols)))

        U, _, _ = svds(PMI, k=self.vec_size)
        norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
        U /= np.maximum(norms, 1e-7)

        return x2i, U

