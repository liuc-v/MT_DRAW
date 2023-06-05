import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

import config
from tokens2index import tokens2index, index2tokens


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear((hidden_dim * 2), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # 不使用偏执向量

    def forward(self, decoder_hidden, encoder_outputs, mask):  # mask用来屏蔽padding(batch_size, seq_len)
        # hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
       #  attention = torch.zeros(batch_size, seq_len).to(config.device)
        decoder_hidden = decoder_hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy)
        # for i in range(seq_len):  # 这里如果这样算会很慢
        #     energy = torch.tanh(self.attn(torch.cat((decoder_hidden[0], encoder_outputs[:, i]), dim=1)))
        #     attention[:, i] = self.v(energy)[:, 0]

        attention = attention.squeeze(2).masked_fill(mask == 1, -1e10)
        return f.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        embedded_seq = self.dropout(self.embedding(input_seq))
        output, (hidden, cell) = self.lstm(embedded_seq)
        return output, (hidden, cell)

    def init_hidden_cell(self, batch_size):  # 可以不使用
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(config.hidden_dim)

    def forward(self, trg_seq, hidden, cell, encoder_outputs, mask):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        attention = self.attention(hidden, encoder_outputs, mask)
        attention = attention.unsqueeze(1)
        weighted = torch.bmm(attention, encoder_outputs)
        embedded = self.dropout(self.embedding(trg_seq))  # embedded: (batch_size, 1, embedding_dim),这里的1是因为我们是一个单词一个单词传入的

        new_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.lstm(new_input, (hidden, cell))
        # output: (batch_size, 1, hidden_size)  # output有1这个维度是因为output与输入序列长度有关
        # hidden: (num_layers, batch_size, hidden_size)

        fc_output = self.fc(output.squeeze(1))
        return fc_output, (hidden, cell), attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio):
        batch_size = input_seq.size(0)
        src_len = input_seq.size(1)
        target_len = target_seq.size(1)   # 得到target_len是为了控制解码器输出的长度
        target_vocab_size = self.decoder.fc.out_features

        mask = torch.zeros(batch_size, src_len).to(config.device)
        mask = mask.masked_fill(input_seq == config.PAD_IDX, 1)

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_output, (hidden, cell) = self.encoder(input_seq)
        # encoder_output: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        input_seq = target_seq[:, 0]   # 取出开始符
        for t in range(1, target_len):
            output, (hidden, cell), attention = self.decoder(input_seq.unsqueeze(1), hidden, cell, encoder_output, mask)
            # output: (batch_size, output_size)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            prediction = output.argmax(1)
            input_seq = target_seq[:, t] if teacher_force else prediction
        return outputs


word = ['a', 'an', 'are', 'is', 'man', 'woman', 'on', 'at', 'in']
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

plt.scatter(pca.transform(word_emb)[:, 0], pca.transform(word_emb)[:, 1])

for i in range(data.shape[0]):
    plt.annotate(word[i], (data[i,0], data[i,1]))

plt.show()
