import nltk
import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

import config
from tokens2index import tokens2index, index2tokens

# a crowd of people, a group of people,  five men, a boy,a girl
sentence_o = ["Eine Menschenmenge", 'Eine Gruppe von Menschen', 'Fünf Männer', 'Ein Junge', 'ein Mädchen']
sentence = [nltk.word_tokenize(i.lower(), language="german") for i in sentence_o]
print(sentence)
sentence_index = [[config.BOS_IDX] + tokens2index(i, 'de') + [config.EOS_IDX] for i in sentence]

max_len = max(len(i) for i in sentence_index)

for i in range(len(sentence_index)):
    sentence_index[i] = sentence_index[i] + [config.PAD_IDX] * (max_len - len(sentence_index[i]))

print(sentence_index)
sentence_index = torch.tensor(sentence_index).to(config.device)
model = torch.load('attention.model').to(config.device)

output, (hidden, cell) = model.encoder(sentence_index)
hidden = hidden.squeeze(0)
hidden = hidden.cpu().detach().numpy()
print(hidden)


from sklearn.decomposition import PCA
import numpy as np


pca = PCA(n_components=2)
pca.fit(hidden)
print(pca.transform(hidden))

data = pca.transform(hidden)

plt.scatter(data[:, 0], data[:, 1])

for i in range(data.shape[0]):
    plt.annotate(sentence_o[i], (data[i,0], data[i,1]))

plt.show()
