import tensorflow as tf

from tensorboard.plugins import projector

import os

from LanguageModel import *
from Training import *



def main():
    
    log_dir = "./visualize"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    words = load_data()
    
    dict_word_token = {}
   
    cnt_tokens = 0
    for word in words:
     
        if word not in dict_word_token:
            dict_word_token[word] = cnt_tokens
            cnt_tokens += 1

    vocab_size = len(dict_word_token)


    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for word, token in dict_word_token.items():
            f.write(f"{word}\n")


    #
    # Initialize model
    #
    lm = LanguageModel(vocab_size, embedding_size)

    # build via call
    x = tf.zeros(shape=(1, window_size))
    lm(x)

    lm.load_weights(f"./saved_models/trained_weights_20").expect_partial()
    lm.summary()

    embedding_layer = lm.layer_list[0]
    weights = tf.Variable(embedding_layer.get_weights()[0][1:])
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

