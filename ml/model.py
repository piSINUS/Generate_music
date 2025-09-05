import music21 as m21
from keras import layers, models
import numpy as np
import tensorflow as tf

file = "final_project/cs1-1pre.mid"
example_score = m21.converter.parse(file).chordify()

def create_dataset(elements):
    ds = (
        tf.data.Dataset.from_tensor_slices(elements)
        .batch(batch_size=42, drop_remainder=True)
        .shuffle(1000)
    )

    vectorize_layer = layers.TextVectorization(
        sandardize  = None, output_mode="int"
    )

    vectorize_layer.adapt(ds)
    vocab =  vectorize_layer.get_vocabulary()

    return ds, vectorize_layer, vocab

notes_seq_ds, notes_vectorize_layer, notes_vocab = create_dataset(notes)
durations_seq_ds, durations_vectorize_layer, durations_vocab = create_dataset(durations)
seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(selfl,vocab_size, embed_dim):
        