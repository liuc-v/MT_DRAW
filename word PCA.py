import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

import config
from tokens2index import tokens2index, index2tokens


# 一 'ein', 'einem', 'eine', 'einer', 'eines', 衣服衬衫 'hemd', 'oberteil',
word = ['ein', 'einem', 'in', 'eine', ',', 'und', 'mit', 'auf', 'mann', 'einer', 'der', 'frau', 'die', 'zwei', 'einen', 'im', 'an', 'von', 'sich', 'dem', 'mädchen', 'junge', 'vor', 'zu', 'steht', 'männer', 'sitzt', 'hund', 'den', 'straße', 'während', 'gruppe', 'hält', 'spielt', 'das', 'hemd', 'personen', 'über', 'drei', 'eines', 'frauen', 'blauen', 'neben', 'ist', 'kind', 'roten', 'weißen', 'stehen', 'sitzen', 'menschen', 'am', 'aus', 'spielen', 'durch', 'bei', 'geht', 'trägt', 'fährt', 'wasser', 'um', 'kinder', 'kleines', 'person', 'macht', 'springt', 'kleiner', 'schwarzen', 'entlang', 'leute', 'gehen', 'etwas', 'mehrere', 'seinem', 'großen', 'oberteil', 'jungen', 'hand', 'grünen', 'läuft', 'sind', 'für', 'hintergrund', 'fahrrad', 'freien', 'jacke', 'luft', 'strand', 'ball', 'hat', 'anderen', 'schaut', 'junger', 'kleidung', 'hinter', 'sie']

index = tokens2index(word, 'de')
index = torch.tensor(index).to(config.device)

model = torch.load('attention.model')
model = model.to(config.device)

word_emb = model.encoder.embedding(index)
word_emb = word_emb.cpu().detach().numpy()
print(word_emb.shape)


from sklearn.decomposition import PCA
import numpy as np


pca = PCA(n_components=2)
pca.fit(word_emb)
print(pca.transform(word_emb))

data = pca.transform(word_emb)

plt.scatter(data[:, 0], data[:, 1])

for i in range(data.shape[0]):
    plt.annotate(word[i], (data[i,0], data[i,1]))

plt.show()

# 英文部分
# 'orange', 'red', 'blue', 'pink', 'green', 'white', 'black', 'yellow', 'brown', 'a', 'one', 'two', 'three', 'large', 'small', 'young', 'little', 'in', 'on', 'by', 'from', 'for', 'behind', 'over', 'running', 'walking', 'jumping', 'sitting', 'standing', 'people', 'men', 'person', 'women', 'children'

# word = ['a', '.', 'in', 'the', 'on', 'man', 'is', 'and', 'of', 'with', 'woman', ',', 'two', 'are', 'to', 'people', 'at', 'an', 'wearing', 'young', 'white', 'shirt', 'black', 'his', 'while', 'blue', 'men', 'sitting', 'girl', 'red', 'boy', 'dog', 'standing', 'playing', 'group', 'street', 'down', 'walking', 'front', 'her', 'holding', 'one', 'water', 'three', 'by', 'women', 'green', 'little', 'up', 'for', 'child', 'looking', 'outside', 'as', 'large', 'through', 'yellow', 'children', 'brown', 'person', 'from', 'their', 'ball', 'hat', 'into', 'small', 'next', 'other', 'dressed', 'some', 'out', 'over', 'building', 'riding', 'running', 'near', 'jacket', 'another', 'around', 'sidewalk', 'field', 'orange', 'crowd', 'beach', 'stands', 'pink', 'sits', 'jumping', 'behind', 'table', 'grass', 'background', 'snow', 'bike', 'stand', 'city']
word = ['orange', 'red', 'blue', 'pink', 'green', 'white', 'black', 'yellow', 'brown', 'a', 'one', 'two', 'three', 'large', 'small', 'young', 'little', 'in', 'on', 'by', 'from', 'for', 'behind', 'over', 'running', 'walking', 'jumping', 'sitting', 'standing', 'people', 'men', 'person', 'women', 'children']

index = tokens2index(word, 'en')
index = torch.tensor(index).to(config.device)

model = torch.load('attention.model')
model = model.to(config.device)

word_emb = model.decoder.embedding(index)
word_emb = word_emb.cpu().detach().numpy()
print(word_emb.shape)


from sklearn.decomposition import PCA
import numpy as np


pca = PCA(n_components=2)
pca.fit(word_emb)
print(pca.transform(word_emb))

data = pca.transform(word_emb)

plt.scatter(data[:, 0], data[:, 1])

for i in range(data.shape[0]):
    plt.annotate(word[i], (data[i,0], data[i,1]), fontsize=20)

plt.show()
