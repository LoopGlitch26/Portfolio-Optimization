import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense


class StockEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.state_size = data.shape[1] + 3 # stock data + balance + net_worth + stock_owned
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = 0
        self.stock_owned = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.hstack((self.data.iloc[self.current_step].values, [self.balance, self.net_worth, self.stock_owned]))
        return obs

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        # print(current_price)
        #print(self.net_worth)
        prev_net_worth = self.net_worth
        if action == 0: # Buy
            if current_price <= self.balance:
                self.stock_owned += 1
                self.balance -= current_price
                self.net_worth = self.balance + self.stock_owned * current_price
        elif action == 1: # Sell
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.balance += current_price
                self.net_worth = self.balance + self.stock_owned * current_price
        elif action == 2:
            self.net_worth = self.balance + self.stock_owned * current_price
        print(self.balance,self.stock_owned,current_price)

        # Go to the next day
        self.current_step += 1

        # Calculate reward
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        print(reward)

        # Check if done
        done = (self.current_step == 100 - 1)

        return (self._get_observation(), reward, done)

class ActorCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.learning_rate = 0.001

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(32, activation='relu')(state_input)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(self.action_size, activation='softmax')(dense2)

        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(32, activation='relu')(state_input)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(1, activation='linear')(dense2)

        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def train(self, state, action, reward, next_state, done):
        target = np.zeros((1, 1))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.gamma * next_value - value
            target[0][0] = reward + self.gamma * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def act(self, state):
        probabilities = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action

def train_model(env, actor_critic, episodes=100):
    total_rewards = []
    i = 0
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size]).astype(np.float32)
        done = False
        total_reward = 0

        while not done:
            action = actor_critic.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size]).astype(np.float32)
            actor_critic.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode : {episode}, Total Reward : {total_reward}")
        actor_critic.actor.save(f"actor{i}.h5")
        actor_critic.critic.save(f'critic{i}.h5')
        i+=1

    print(total_rewards)
    plt.plot(total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def main():
    df = pd.read_csv('new_dataset.csv')
    groups = list(set(df['Symbol']))
    df.head()
    grouped_df = df.groupby('Symbol')
    #data = pd.read_csv('new_datset.csv')
    data = grouped_df.get_group(groups[0])
    data = data[['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50']]
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    env = StockEnv(data)
    state_size = env.state_size
    action_size = 3 # Buy or Sell

    actor_critic = ActorCritic(state_size=state_size, action_size=action_size)
    train_model(env=env, actor_critic=actor_critic)

if __name__ == '__main__':
    main()
