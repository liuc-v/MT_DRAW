import pickle


with open('./vocab/SRC_VOCAB.pkl', 'rb') as f:
    src_vocab = pickle.load(f)

with open('./vocab/TRG_VOCAB.pkl', 'rb') as f:
    trg_vocab = pickle.load(f)


def tokens2index(tokens, ln):
    if ln == 'de':
        return [src_vocab.get_stoi()[token] for token in tokens]
    if ln == 'en':
        return [trg_vocab.get_stoi()[token] for token in tokens]


def index2tokens(index, ln):
    if ln == 'de':
        return [src_vocab.get_itos()[i] for i in index]
    if ln == 'en':
        return [trg_vocab.get_itos()[i] for i in index]