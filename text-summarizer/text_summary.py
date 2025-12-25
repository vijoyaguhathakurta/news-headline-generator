
import numpy as np
import pandas as pd
import contractions
import re
import tensorflow as tf
import pickle

# THE TRANSFORMER MODEL

def preprocess(x):
  # expand contractions
  x= x.apply(lambda y: contractions.fix(y))
  # remove html tags
  x= x.apply(lambda y: re.compile('<.*?>').sub(r'',y))
  # remove url
  x= x.apply(lambda y: re.compile(r'https?://\S+|www\.\S+').sub(r'',y))
  # remove 's
  x=  x.apply(lambda y: re.sub(r"'s\b","",y))
  # remove '
  x= x.apply(lambda y: re.sub("'",'', y))
  # add end and start token
  x= x.apply(lambda y: 'sos ' + y + ' eos')
  return x

#positional encoding layer
def positional_encoding(length, depth):
  i = depth/2
  positions = np.arange(length)[:, np.newaxis]
  i = np.arange(i)[np.newaxis, :]
  angle_rates = 1 / (10000**(2*i/depth))
  angle_rads = positions * angle_rates
  pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
  return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHead_Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHead_Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def scaled_dot_product_attention(self,q, k, v, mask):
      matmul_qk = tf.matmul(q, k, transpose_b=True)
      dk = tf.cast(tf.shape(k)[-1], tf.float32)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
      if mask is not None:
        scaled_attention_logits += (mask * -1e9)
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
      output = tf.matmul(attention_weights, v)
      return output, attention_weights
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights
def feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  ])
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHead_Attention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layernorm_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_ffn = tf.keras.layers.LayerNormalization()
        self.dropout_mha = tf.keras.layers.Dropout(rate)
        self.dropout_ffn = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout_mha(attn_output, training=training)
        out1 = self.layernorm_mha(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out2 = self.layernorm_ffn(out1 + ffn_output)
        return out2
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_mha = MultiHead_Attention(d_model, num_heads)
        self.cross_mha = MultiHead_Attention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layernorm_masked_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_cross_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_ffn = tf.keras.layers.LayerNormalization()
        self.dropout_masked_mha = tf.keras.layers.Dropout(rate)
        self.dropout_cross_mha = tf.keras.layers.Dropout(rate)
        self.dropout_ffn = tf.keras.layers.Dropout(rate)
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        masked_attn, masked_attn_weights = self.masked_mha(x, x, x, look_ahead_mask)
        masked_attn = self.dropout_masked_mha(masked_attn, training=training)
        out1 = self.layernorm_masked_mha(masked_attn + x)
        cross_attn, cross_attn_weights = self.cross_mha(enc_output, enc_output, out1, padding_mask)
        cross_attn = self.dropout_cross_mha(cross_attn, training=training)
        out2 = self.layernorm_cross_mha(cross_attn + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm_ffn(ffn_output + out2)
        return out3, masked_attn_weights, cross_attn_weights
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            # This constructs a dynamic key for accessing attention weights within a specific decoder layer.
            # The {} is a placeholder that will be replaced with the actual value of i+1.
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
def accuracy_function(label, pred):
  label = tf.cast(label, pred.dtype)
  match = label == pred
  mask = label != 0
  match = match & mask
  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def evaluate(input, output):
  # loading tokenizer
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  num_layers = 3  # original=6
  d_model = 128 # original=512
  dff = 512 # original=2048
  num_heads = 4 # original=8
  dropout_rate = 0.1 # original= 0.1
  VOCAB_SIZE=len(tokenizer.word_index)+1
  ENCODER_LEN = 100
  DECODER_LEN = 20
  # Building the transformer model 
  transformer = Transformer(
    num_layers= num_layers,
    d_model= d_model,
    num_heads= num_heads,
    dff= dff,
    input_vocab_size= VOCAB_SIZE,
    target_vocab_size= VOCAB_SIZE,
    rate= dropout_rate)
  inp=tf.constant([[  4,   174,  2600,  8781,    62,   688,    55,  2337,  6709,
          15,    37,   463,    17,     7,   236,   231,   612,  5369,
           9,   582,   177,  6882,   114,    20,   234,     5,    98,
           5,    27,  6882,    37,    32,   568,   178,   127,   582,
        1632,    30,  2234,    19,    20,    31,    26,  4016,   236,
          37,   112,   669,   231,   127,   582,    20,  4016,     5,
         127,  3450,    99,    19,    20,   127,   612,    38, 23731,
        2462,     9,  1007,  2600,    48,     3,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0]])
  tar_inp=tf.constant([[   4,   37,  463,   17,    7,  236,  231,    2,  612, 5369, 2600,
       8781,    3,    0,    0,    0,    0,    0,    0,    0]])
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
  # loading the saved weights
  transformer.load_weights('checkpoints')
  for i in range(1,DECODER_LEN):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, output)
        predictions, _ = transformer(input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id.shape[0]==1 and predicted_id == tokenizer.word_index['eos']:
           return output
        output = tf.concat([output, predicted_id], axis=-1)
  return output
def summarize(txt):
   # loading tokenizer
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  num_layers = 3  # original=6
  d_model = 128 # original=512
  dff = 512 # original=2048
  num_heads = 4 # original=8
  dropout_rate = 0.1 # original= 0.1
  VOCAB_SIZE=len(tokenizer.word_index)+1
  ENCODER_LEN = 100
  DECODER_LEN = 20
  txt= preprocess(pd.Series(txt))
  txt= tokenizer.texts_to_sequences(txt)
  txt= tf.keras.utils.pad_sequences(txt, maxlen= ENCODER_LEN, truncating='post', padding='post')
  txt= tf.cast(txt, dtype=tf.int32)
  out= tf.cast([[tokenizer.word_index['sos']]],dtype=tf.int32)
  p= evaluate(txt,out)
  p=p.numpy()
  p= tokenizer.sequences_to_texts(p)
  s=p[0][4:]
  return s