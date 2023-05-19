# -*- coding: utf-8 -*-

import numpy as np
import random
import pandas as pd
#from google.colab import drive
import matplotlib.pyplot as plt


from .Actor import Actor
from .Critic import Critic
#drive.mount('/content/drive/')

#csv_path = '/content/drive/MyDrive/SDP/'
df = pd.read_csv('new_dataset.csv')
groups = list(set(df['Symbol']))
df.head()

grouped_df = df.groupby('Symbol')
dt_start_index = random.randint(100, 1900)


class Agent:
    def __init__(self, grouped_df, rows, stocks, indicators):
        self.iterations = 0 
        self.learning_rate = 0.01
        self.stocks = stocks
        self.grouped_df = grouped_df
        self.rows = rows
        self.indicators = indicators
        self.index = self.get_index()
        self.state = self.create_state(self.index)
        self.actor = Actor(self.get_state_size(), self.get_action_size(), self.learning_rate)

    def create_state(self,index):
        state = []
        for i in range(len(self.stocks)):
            stock_df = self.grouped_df.get_group(self.stocks[i])
            stock_df = stock_df.iloc[index]
            state.append(stock_df.to_numpy()[1:6])
        return np.array(state)

    def get_state_size(self):
        return len(self.state)*len(self.state[0])

    def get_action_size(self):
        return len(self.stocks)

    def get_index(self):
        return random.randint(100,self.rows-100)

    def train(self, iterations):
        x = []
        y = []
        logs = []
        for i in range(iterations):
            # generate actions using Actor
            actions = self.actor.select_action(self.state.tolist())
            # generate reward and advantage using Critic 
            critic = Critic(self.actor, self.state, self.generate_np_states(self.index), self.index, 1000)
            y.append(critic.generate_reward(self.state,actions))
            x.append(i)
            print(actions.tolist())
            print(x[i],y[i])
            logs.append(f"{x[i]},{y[i]}")
            states, actions, advantages = critic.generate_advantages()
            # Update policy
            self.actor.update_policy(states,actions,advantages)
            # update state 
            self.state = self.create_state(self.get_index())
            self.iterations +=1
        plt.plot(x,y)
        plt.show()
        f = open('resultlog.txt','w')
        f.writelines(logs)
        f.close()

    def generate_np_states(self, index):
        np_states = []
        for i in range(index,index+100):
            np_states.append(self.create_state(i))
        return np.array(np_states)
            
    def save_policy(self):
        pass

agent = Agent(grouped_df,1940,groups, 5)
agent.train(10000)

