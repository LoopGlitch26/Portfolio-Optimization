import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from sklearn import preprocessing

class Actor:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.__build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def __build_model(self):
        input_state = Input(shape=(self.state_size,))
        dense1 = Dense(64, activation='relu')(input_state)
        dense2 = Dense(64, activation='relu')(dense1)
        dense3 = Dense(64, activation='relu')(dense2)
        action_probs = Dense(self.action_size, activation='softmax')(dense3)
        self.model = Model(inputs=input_state, outputs=action_probs)

    def select_action(self, state):
        state = np.array(state).reshape(1, self.state_size)
        action_probs = self.model.predict(state).flatten()
        return action_probs

    def update_policy(self, states, actions, advantages):
        states = [np.array(x).flatten().tolist() for x in states]
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)

        with tf.GradientTape() as tape:
            action_probs = self.model(states, training=True)
            selected_action_probs = tf.reduce_sum(action_probs * actions, axis=1)
            log_probs = tf.math.log(selected_action_probs)
            loss = -tf.reduce_mean(log_probs * advantages)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
