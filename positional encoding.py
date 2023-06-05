from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import matplotlib.pyplot as plt
import config


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        fig, ax = plt.subplots()
        im = ax.imshow(pe.numpy())
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Position')
        plt.show()

        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


pe = PositionalEncoding(256, 100, 0.1)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=config.device)) == 1).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)
    return mask


def create_mask(src, trg):
    src_len = src.shape[1]
    trg_len = trg.shape[1]

    trg_mask = generate_square_subsequent_mask(trg_len)
    src_mask = torch.zeros((src_len, src_len), device=config.device).type(torch.bool)

    src_padding_mask = (src == config.PAD_IDX)
    trg_padding_mask = (trg == config.PAD_IDX)
    return src_mask, trg_mask, src_padding_mask, trg_padding_mask


class TransformerModel(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, src_vocab_size, trg_vocab_size, d_model, d_forward, n_head, dropout, device):
        super(TransformerModel, self).__init__()
        self.encoder_tok_embedding = nn.Embedding(src_vocab_size, d_model)
       # self.encoder_pos_embedding = nn.Embedding(config.max_len, d_model)

        self.decoder_tok_embedding = nn.Embedding(trg_vocab_size, d_model)
      #  self.decoder_pos_embedding = nn.Embedding(config.max_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, config.max_len, dropout)
        self.device = device
        self.transformer = nn.Transformer(d_model=d_model,nhead=n_head, num_encoder_layers=encoder_layers, num_decoder_layers=decoder_layers, dim_feedforward=d_forward, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg)

        batch_size = src.shape[0]
        src_len = src.shape[1]
        trg_len = trg.shape[1]
     #   src_pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
     #   trg_pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

     #   src_emb = self.encoder_pos_embedding(src_pos) + self.dropout(self.encoder_tok_embedding(src) * math.sqrt(config.d_model))
     #   trg_emb = self.decoder_pos_embedding(trg_pos) + self.dropout(self.decoder_tok_embedding(trg) * math.sqrt(config.d_model))

        # 对src和tgt进行编码
        src_emb = self.encoder_tok_embedding(src)
        trg_emb = self.decoder_tok_embedding(trg)
        # 给src和tgt的token增加位置信息
        src_emb = self.positional_encoding(src_emb)
        trg_emb = self.positional_encoding(trg_emb)

        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, src_padding_mask)

        return self.generator(outs)

    def encoder(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        src_pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src_emb = self.encoder_pos_embedding(src_pos) + self.encoder_tok_embedding(src) * math.sqrt(config.d_model)
        return self.transformer.encoder(src_emb)

    def decoder(self, trg, memory):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src_emb = self.decoder_pos_embedding(trg_pos) + self.decoder_tok_embedding(trg) * math.sqrt(config.d_model)

        return self.transformer.decoder(src_emb, memory)
