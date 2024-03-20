import tensorflow as tf

import os

from LanguageModel import *
from Training import *
from Logging import *

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


    dict_token_word = {token: word for word, token in dict_word_token.items()}


    #
    # Initialize model
    #
    lm = LanguageModel(vocab_size, embedding_size)

    # build via call
    x = tf.zeros(shape=(1, window_size))
    lm(x)

    lm.load_weights(f"./saved_models/trained_weights_20").expect_partial()
    lm.summary()


    print(generate_text(lm, "Harry Potter", dict_word_token, dict_token_word, 10))
   
    print(generate_text(lm, "Dementors", dict_word_token, dict_token_word, 10))

    print(generate_text(lm, "dumbledore", dict_word_token, dict_token_word, 10))
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

