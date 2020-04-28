import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__()
        print(kwargs)
        num_sublayers = kwargs["num_sublayers"]
        self.model_dim = kwargs["model_dim"]
        self.num_heads = kwargs["num_heads"]
        self.vocab_size = kwargs["vocab_size"]
        # self.num_encoder_layers = kwargs["num_encoder_layers"]
        embedding_size = kwargs["embedding_size"]


        # encoder
        # decoder
        self.enc_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.dec_embedding = nn.Embedding(self.vocab_size, self.model_dim)

        self.encoder_stack = nn.ModuleList()
        self.decoder_stack = nn.ModuleList()

        for i in range(0, num_sublayers):
            self.encoder_stack.append(TransformerLayer(model_dim=self.model_dim, vocab_size=self.vocab_size, num_heads=self.num_heads, embedding_size=embedding_size))
            self.decoder_stack.append(TransformerLayer(model_dim=self.model_dim, vocab_size=self.vocab_size, num_heads=self.num_heads, embedding_size=embedding_size, is_decoder=True))
        self.logits = nn.Linear(self.model_dim, self.vocab_size)


    def forward(self, enc_inputs):
        # print("Enc input: ", enc_inputs.size())
        # print("Dec input: ", dec_inputs.size())

        # packed_enc_embedding = pack_padded_sequence(enc_embedding, enc_lengths, batch_first=True, enforce_sorted=False)
        # packed_dec_embedding = pack_padded_sequence(dec_embedding, decoder_lengths, batch_first=True, enforce_sorted=False)
        enc_output = self.enc_embedding(enc_inputs)

        for i, encoder in enumerate(self.encoder_stack):
            enc_output = encoder(enc_output)

        # dec_output = self.dec_embedding(dec_inputs)
        # for i, decoder in enumerate(self.decoder_stack):
        #     dec_output = decoder(dec_output, enc_output)


        return self.logits(enc_output)

def attention(q, k, v, kdim):
    attention = torch.matmul(q, k.transpose(1,2))

    attention /= torch.sqrt(torch.tensor(kdim).float())

    maxlen = attention.size(1)
    x_lens = torch.LongTensor(torch.arange(maxlen))
    mask = torch.arange(maxlen)[None, :] > x_lens[:, None]

    attention.masked_fill_(mask, -1 * float("Inf"))

    # softmax
    attention = F.softmax(attention, dim=-1)

    # matmul
    attention = torch.matmul(attention, v)

    return attention


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, model_dim, kdim):
        super(SingleHeadAttentionLayer, self).__init__()
        self.ql = nn.Linear(model_dim, kdim, bias=False)
        self.kl = nn.Linear(model_dim, kdim, bias=False)
        self.vl = nn.Linear(model_dim, kdim, bias=False)
        self.kdim = kdim

    def forward(self, q, k, v):
        q = self.ql(q)
        k = self.kl(k)
        v = self.vl(v)

        attn = attention(q, k, v, self.kdim)
        return attn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        kdim = model_dim // num_heads

        self.layer_stack = nn.ModuleList([])

        for i in range(num_heads):
            self.layer_stack.append(SingleHeadAttentionLayer(model_dim, kdim))
        self.linear = nn.Linear(kdim*num_heads, model_dim)

    def forward(self, q, k, v):

        batch_size = q.size(0)
        attns = []
        for i, layer in enumerate(self.layer_stack):
            output = layer(q, k, v)
            attns.append(output)

        cat = torch.cat(attns, dim=2)
        return self.linear(cat)

class TransformerLayer(nn.Module):
    def __init__(self, model_dim, vocab_size, num_heads=1, embedding_size=100, is_decoder=False):
        super(TransformerLayer, self).__init__()

        self.is_decoder = is_decoder
        # input embeddings
        # positional encodings
        # compute q, k, v
        # 2 ffs
        # softmax
        self.embedding = nn.Embedding(vocab_size, model_dim)
        # self.pos_encoding =
        self.mh_attn = MultiHeadAttentionLayer(model_dim=model_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(model_dim)
        if (self.is_decoder):
            self.mh_attn_2 = MultiHeadAttentionLayer(model_dim=model_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


        self.ff = nn.Linear(model_dim, model_dim)
        self.ff2 = nn.Linear(model_dim, model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)




    def forward(self, inputs, enc_output=None):

        attn = self.mh_attn(inputs, inputs, inputs)
        # print("Attn size:", attn.size())
        # print("Encoding size:", pos_encoding.size())

        add_norm = self.norm(attn + inputs)

        if (self.is_decoder):
            q2 = add_norm
            v2 = enc_output
            k2 = enc_output

            attn = self.mh_attn_2(q2, v2, k2)
            add_norm = self.norm2(attn + add_norm)

        ff = self.ff2(F.relu(self.ff(add_norm)))

        add_norm_ff = self.norm_ff(ff + add_norm)

        return add_norm_ff
