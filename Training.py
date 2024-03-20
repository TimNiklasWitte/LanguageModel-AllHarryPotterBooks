import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime
import numpy as np

from LanguageModel import *
from Logging import *


NUM_EPOCHS = 20
BATCH_SIZE = 64

window_size = 30
embedding_size = 256
num_sampled_negative_classes = 32


def load_data():
    with open("./data/Harry_Potter_all_books_preprocessed.txt") as file:
        content = file.read()

    content = content.lower()
    content = content.replace(" !", " ! ")
    content = content.replace(" ?", " ? ")
    content = content.replace(" .", " . ")
    content = content.strip()

    words = content.split(" ")

    words = [word.strip() for word in words]
    words = [word for word in words if len(words) > 2]
    return words


def dataset_generator():

    words = load_data()
    
    dict_word_token = {}
   
    cnt_tokens = 0
    for word in words:
     
        if word not in dict_word_token:
            dict_word_token[word] = cnt_tokens
            cnt_tokens += 1

    for word_pos in range(len(words) - window_size - 1):

        input_words = words[word_pos:word_pos + window_size]
        target_words = words[word_pos+1:word_pos + window_size+1]

        input_tokens = [dict_word_token[input_word] for input_word in input_words]
        target_tokens = [dict_word_token[target_word] for target_word in target_words]


        yield np.array(input_tokens), np.array(target_tokens)
        


def main():

    words = load_data()
    
    dict_word_token = {}
   
    cnt_tokens = 0
    for word in words:
     
        if word not in dict_word_token:
            dict_word_token[word] = cnt_tokens
            cnt_tokens += 1


    dict_token_word = {token: word for word, token in dict_word_token.items()}

  
    vocab_size = len(dict_word_token)

     
    dataset = tf.data.Dataset.from_generator(
                    dataset_generator,
                    output_signature=(
                            tf.TensorSpec(shape=(window_size,), dtype=tf.uint16),
                            tf.TensorSpec(shape=(window_size,), dtype=tf.uint16)
                        )
                )
    
    dataset = dataset.apply(prepare_data)

   
    train_dataset = dataset.skip(750)
    test_dataset = dataset.take(750)

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)
    


    #
    # Initialize model
    #
    lm = LanguageModel(vocab_size, embedding_size)

    # build via call
    x = tf.zeros(shape=(1, window_size))
    lm(x)
    lm.summary()


    # text = generate_text(lm, "harry potter", dict_word_token, dict_token_word, generated_text_length=20)
    # print(text)
    # return 

    #
    # Train and test loss/accuracy
    #
    print(f"Epoch 0")
  
    log(train_summary_writer, lm, train_dataset, test_dataset, dict_word_token, dict_token_word, num_sampled_negative_classes, epoch=0)
     
    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for input_sequence, target_sequence in tqdm.tqdm(train_dataset, position=0, leave=True): 
            lm.train_step(input_sequence, target_sequence, num_sampled_negative_classes)

        log(train_summary_writer, lm, train_dataset, test_dataset, dict_word_token, dict_token_word, num_sampled_negative_classes, epoch)

        # Save model (its parameters)
        if epoch % 5 == 0:
            lm.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")



def prepare_data(dataset):

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(50000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")