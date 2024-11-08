import torch
import torch.nn as nn
import torch.nn.functional as F


# using template from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# tutorial about positional encoding: https://kikaben.com/transformers-positional-encoding/
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        
        #div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term = 10000 ** ( (2 * torch.arange(0, d_model) ) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        for i in range(max_len):
            if i % 2 == 0:    
                pe[i, 0, :] = torch.sin(position[i] / div_term)
            else:
                pe[i, 0, :] = torch.cos(position[i] / div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x #self.dropout(x)


# sidenote: understanding skip-connections: https://theaisummer.com/skip-connections/
class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads)
        self.ln = nn.LayerNorm(embedding_dim, eps=1e-12, elementwise_affine=True)
        
    def forward(self, x: torch.Tensor, is_causal: bool=False):
        sequence_len = list(x.size())[0]
        
        if is_causal:
            attn_output, attn_output_weights = self.mha(query=x, key=x, value=x, is_causal=True, \
                                                    attn_mask=nn.Transformer.generate_square_subsequent_mask(sequence_len))
        else:
            attn_output, attn_output_weights = self.mha(query=x, key=x, value=x)

        x = x + attn_output # skip-connection
        x = self.ln(x)
                
        return x, attn_output, attn_output_weights


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int): #must be same as encoder
        super().__init__()
        
        self.masked_mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads)
        self.ln = nn.LayerNorm(embedding_dim, eps=1e-12, elementwise_affine=True)

        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads)
        
        
    def forward(self, x: torch.Tensor, query: torch.Tensor=None, key: torch.Tensor=None):
        sequence_len = list(x.size())[0]
        
        attn_output, attn_output_weights = self.masked_mha(query=x, key=x, value=x, is_causal=True, \
                                                attn_mask=nn.Transformer.generate_square_subsequent_mask(sequence_len))#, is_causal=True)

        x = x + attn_output # skip-connection
        x = self.ln(x)
        
        if query is None or key is None: # only for debugging
            attn_output, attn_output_weights = self.mha(query=x, key=x, value=x)
        else:
            attn_output, attn_output_weights = self.mha(query=query, key=key, value=x)
        
        x = x + attn_output # skip-connection
        x = self.ln(x)
        
        return x, attn_output, attn_output_weights


class DecoderOnly(nn.Module):
    def __init__(self, n_decoders: int, alphabet_size: int, embedding_dim: int, max_len: int, n_heads_decoder: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, max_len=max_len+2)
        self.input_embedding = nn.Embedding(alphabet_size+3, embedding_dim) # +3 for start, stop, padding symbol
        
        self.decoders = {"D{}".format(x): Decoder(embedding_dim=embedding_dim, n_heads=n_heads_decoder) for x in range(n_decoders)}
        
        self.output_fnn = nn.Linear(in_features=embedding_dim, out_features=alphabet_size+3) # +2 for start and stop
        self.gelu = torch.nn.GELU()
        
        self.dropout = nn.Dropout(0.2)
        self.softmax_output = nn.Softmax(dim=-1)
        
        self.attention_output_layer = nn.Identity() 
        self.attention_weight_layer = nn.Identity() 
        self.embedding_output_layer = nn.Identity() 
        #self.tgt_embedding_output_layer = nn.Identity()

    def forward(self, src: torch.Tensor):
        x = self.input_embedding(src)
        #src_embedding_out = self.input_embedding(x)
        x = self.pos_encoding(x)

        for decoder in self.decoders.values():
          x, attention_output, attention_weights = decoder(x)
          x = self.dropout(x)

        attention_output = self.attention_output_layer(attention_output)
        attention_weights = self.attention_weight_layer(attention_weights)
                
        x = self.gelu(self.output_fnn(x))
        x = self.softmax_output(x)
        return x


# sidenote: understanding skip-connections: https://theaisummer.com/skip-connections/
class AuTransformer(nn.Module):
    def __init__(self, n_encoders: int, n_decoders: int, alphabet_size: int, embedding_dim: int, max_len: int, 
                 n_heads_encoder: int = 3, n_heads_decoder: int = 3):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, max_len=max_len+2)
        self.input_embedding = nn.Embedding(alphabet_size+3, embedding_dim) # +3 for start, stop, padding symbol
        
        self.encoders = {"E{}".format(x): Encoder(embedding_dim=embedding_dim, n_heads=n_heads_encoder) for x in range(n_encoders)}
        self.decoders = {"D{}".format(x): Decoder(embedding_dim=embedding_dim, n_heads=n_heads_decoder) for x in range(n_decoders)}
        
        self.output_fnn = nn.Linear(in_features=embedding_dim, out_features=alphabet_size+3) # +2 for start and stop
        self.gelu = torch.nn.GELU()
        
        self.dropout = nn.Dropout(0.2)
        self.softmax_output = nn.Softmax(dim=-1)
        
        self.attention_output_layer = nn.Identity() 
        self.attention_weight_layer = nn.Identity() 
        self.src_embedding_output_layer = nn.Identity() 
        #self.tgt_embedding_output_layer = nn.Identity() 

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        x = self.input_embedding(src)
        src_embedding_out = self.src_embedding_output_layer(x)
        x = self.pos_encoding(x)

        for encoder in self.encoders.values():
          x, attention_output, attention_weights = encoder(x)
          x = self.dropout(x)

        attention_output = self.attention_output_layer(attention_output)
        attention_weights = self.attention_weight_layer(attention_weights)
        
        tgt_embedding = self.input_embedding(tgt)
        tgt_embedding = self.pos_encoding(tgt_embedding)
        #tgt_embedding_out = self.tgt_embedding_output_layer(tgt_embedding) 

        for decoder in self.decoders.values():
          x, _, _ = decoder(x=tgt_embedding, query=x, key=x)
          x = self.dropout(x)
        
        x = self.gelu(self.output_fnn(x))
        x = self.softmax_output(x)
        return x
    
# sidenote: understanding skip-connections: https://theaisummer.com/skip-connections/
class AuAcceptor(nn.Module):
    """Takes as input a transformer, and adds a binary classifier layer on top of the encoder
    (can be easily tweaked so that it can take the decoder output instead of encoder input).
    """
    def __init__(self, transformer_model, alphabet_size: int, embedding_dim: int, maxlen_of_sequence: int, freeze_transformer: bool=True):
        super().__init__()

        self.pos_encoding = transformer_model.pos_encoding
        self.input_embedding = transformer_model.input_embedding
        
        self.encoders = transformer_model.encoders
        if freeze_transformer:
            for encoder in self.encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False

        self.decoders = transformer_model.decoders
        if freeze_transformer:
            for decoder in self.decoders.values():
                for param in decoder.parameters():
                    param.requires_grad = False
        
        self.dropout = nn.Dropout(0.2)

        # if you want to use whole sequence for output 
        self.hidden_layer = nn.Linear(in_features=embedding_dim * maxlen_of_sequence, out_features=1)
        # if you only want to use last hidden representation for output
        #self.hidden_layer = nn.Linear(in_features=embedding_dim, out_features=1)
        
        self.attention_output_layer = nn.Identity() 
        self.attention_weight_layer = nn.Identity() 
        self.src_embedding_output_layer = nn.Identity() 
        #self.tgt_embedding_output_layer = nn.Identity()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, last_index: int):
        x = self.input_embedding(src)
        src_embedding_out = self.src_embedding_output_layer(x)
        x = self.pos_encoding(x)

        for encoder in self.encoders.values():
          x, attention_output, attention_weights = encoder(x)
          x = self.dropout(x)

        attention_output = self.attention_output_layer(attention_output)
        attention_weights = self.attention_weight_layer(attention_weights)
        
        # *** uncomment the next lines to use the decoder as well *** 
        #tgt_embedding = self.input_embedding(tgt)
        #tgt_embedding = self.pos_encoding(tgt_embedding)
        #tgt_embedding_out = self.tgt_embedding_output_layer(tgt_embedding) 

        #for decoder in self.decoders.values():
        #  x = decoder(x=tgt_embedding, query=x, key=x)
        #  x = self.dropout(x)

        # ------------- this block if you want to use whole sequence -------------
        x = torch.permute(x, [1, 0, 2])
        x = torch.flatten(x, 1, 2)
        x = F.sigmoid(self.hidden_layer(x))
        # ------------- end of block -------------

        # ------------- this block if you want to use only last hidden representation -------------
        #x_shape = list(x.size()) # seq_length, batchsize, hidden_dim
        #x_select = torch.zeros(( x_shape[1], x_shape[2] )).float()
        #for i in range(x_shape[1]):
        #    x_select[i] = x[last_index[i], i]
        #x = F.sigmoid(self.hidden_layer(x_select))
        # ------------- end of block -------------

        return x
    