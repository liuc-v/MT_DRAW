import nltk


def load_file(file_path):
    german_sentences = []
    english_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            a = line.split(" ||| ")
            german_sentences.append(nltk.word_tokenize(a[0].lower(), language="german"))
            english_sentences.append(nltk.word_tokenize(a[1].lower(), language="english"))
    return german_sentences, english_sentences
