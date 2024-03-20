
import tensorflow as tf
import numpy as np

def log(train_summary_writer, lm, train_dataset, test_dataset, dict_word_token, dict_token_word, num_sampled_negative_classes, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        lm.test_step(train_dataset.take(500), num_sampled_negative_classes)

    #
    # Train
    #
    train_loss = lm.metric_loss.result()
    lm.metric_loss.reset_states()


    #
    # Test
    #

    lm.test_step(test_dataset, num_sampled_negative_classes)

    test_loss = lm.metric_loss.result()
    lm.metric_loss.reset_states()


    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"test_loss", test_loss, step=epoch)


    #
    # Output
    #
    print(f"train_loss: {train_loss}")
    print(f"test_loss: {test_loss}")


    texts = [
        "Harry Potter had always known he was different",
        "It was nearing midnight and the corridors of Hogwarts were ominously quiet",
        "The Forbidden Forest whispered secrets that only the bravest dared to explore",
        "Hermione Granger poured over her books in the library determined to solve the latest mystery",
        "Dobby the house elf appeared suddenly his large eyes wide with worry",
        "Quidditch practice was in full swing with Gryffindor and Slytherin teams battling fiercely for the Quaffle"
    ]
    
    for text_id, init_text in enumerate(texts):
        text = generate_text(lm, init_text, dict_word_token, dict_token_word, 30)
      
        with train_summary_writer.as_default():
            tf.summary.text(name=str(text_id), data = text, step=epoch)



def generate_text(lm, init_text, dict_word_token, dict_token_word, generated_text_length):

    vocab_size = len(dict_word_token)

    init_text = init_text.lower()

    init_text = init_text.replace(" !", " ! ")
    init_text = init_text.replace(" ?", " ? ")
    init_text = init_text.replace(" .", " . ")
    init_text = init_text.strip()

    words = init_text.split(" ")
    words = [word.strip() for word in words]
    words = [word for word in words if len(words) >= 1]

    init_tokens = [dict_word_token[word] for word in words]

    init_tokens = tf.convert_to_tensor(init_tokens, dtype=tf.uint16)
    # add batch dim
    init_tokens = tf.expand_dims(init_tokens, axis=0)
    
    y, states = lm(init_tokens, initial_states=None, return_states=True)
    vocab_distri = y[0, -1, :] # remove batch dim, consider only last output
    vocab_distri = vocab_distri.numpy()

    token = np.random.choice(vocab_size, p=vocab_distri)

    generated_tokens = [token]

    token =  tf.convert_to_tensor(token)
    # add batch dim
    token = tf.expand_dims(token, axis=0)
    token = tf.expand_dims(token, axis=1)

    
    for i in range(generated_text_length):

        y, states = lm(token, initial_states=states, return_states=True)
        vocab_distri = y[0, 0, :] # remove batch dim and remove seq dim
        vocab_distri = vocab_distri.numpy()
       
        token = np.random.choice(vocab_size, p=vocab_distri)

        generated_tokens.append(token)

        token =  tf.convert_to_tensor(token)
        # add batch dim and seq dim
        token = tf.expand_dims(token, axis=0)
        token = tf.expand_dims(token, axis=1)


    generated_words = [dict_token_word[token] for token in generated_tokens]
    generate_text = " ".join(generated_words)

    text = f"{init_text} | {generate_text}"
    
    return text
