import tensorflow as tf
import tqdm

class LanguageModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size):
        super(LanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.last_units = 50
        self.layer_list = [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size),
            tf.keras.layers.LSTM(units=100, return_sequences=True, return_state=True),
            tf.keras.layers.LSTM(units=self.last_units, return_sequences=True, return_state=True)
        ]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
   

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    def build(self, input_shape):

        self.w_score = self.add_weight(
            shape=(self.vocab_size, self.last_units), 
            initializer="random_normal", 
            trainable=True,
            name='w_score'
        )

        self.b_score = self.add_weight(
            shape=(self.vocab_size, ), 
            initializer="random_normal", 
            trainable=True,
            name='b_score'
        )


        super(LanguageModel, self).build(input_shape)
        

    @tf.function
    def call(self, x, initial_states=None, return_states=False):


        if return_states:
            x, states = self.get_activation_last_layer(x, initial_states=initial_states, return_states=return_states)
        else:
            x = self.get_activation_last_layer(x)

        y = x @ tf.transpose(self.w_score, perm=(1,0))  + self.b_score
      
        y = tf.nn.softmax(y, axis=-1)

        if return_states:
            return y, states
        
        else:
            return y
    

    @tf.function
    def get_activation_last_layer(self, x, initial_states=None, return_states=False):

        states = []
        for idx, layer in enumerate(self.layer_list):

            if isinstance(layer, tf.keras.layers.LSTM):
                
                if initial_states:
                    initial_state = initial_states[idx - 1]
                else:
                    initial_state = None 

                x, final_hidden_state, final_cell_state = layer(x, initial_state=initial_state)
                states.append([final_hidden_state, final_cell_state])
            else:
                x = layer(x)

        if return_states:
            return x, states 
        else:
            return x

    @tf.function
    def train_step(self, input_sequence, target_sequence, num_sampled_negative_classes):

        batch_size = tf.shape(input_sequence)[0]
        seq_len = tf.shape(input_sequence)[1]
   
        
        target_sequence = tf.reshape(target_sequence, shape=(batch_size*seq_len, 1))
        
        with tf.GradientTape() as tape:
            x = self.get_activation_last_layer(input_sequence)

            x = tf.reshape(x, shape=(batch_size*seq_len, self.last_units))

            loss = tf.nn.nce_loss(
                        weights=self.w_score,                      # [vocab_size, embed_size]
                        biases=self.b_score,                       # [vocab_size]
                        labels=target_sequence,                    # [batch_size, 1]
                        inputs=x,                                  # [batch_size, embed_size]
                        num_sampled=num_sampled_negative_classes,  # negative sampling: number 
                        num_classes=self.vocab_size,
                        num_true=1                                 # positive sample
                    )

            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)



    def test_step(self, dataset, num_sampled_negative_classes):
          
        self.metric_loss.reset_states()

        print("Testing")
        for input_sequence, target_sequence in tqdm.tqdm(dataset, position=0, leave=True):
            batch_size = tf.shape(input_sequence)[0]
            seq_len = tf.shape(input_sequence)[1]

            x = self.get_activation_last_layer(input_sequence)

            x = tf.reshape(x, shape=(batch_size*seq_len, self.last_units))
            target_sequence = tf.reshape(target_sequence, shape=(batch_size*seq_len, 1))
            loss = tf.nn.nce_loss(
                        weights=self.w_score,                      # [vocab_size, embed_size]
                        biases=self.b_score,                       # [vocab_size]
                        labels=target_sequence,                    # [batch_size, 1]
                        inputs=x,                                  # [batch_size, embed_size]
                        num_sampled=num_sampled_negative_classes,  # negative sampling: number 
                        num_classes=self.vocab_size,
                        num_true=1                                 # positive sample
                    )

            loss = tf.reduce_mean(loss)

            self.metric_loss.update_state(loss)

