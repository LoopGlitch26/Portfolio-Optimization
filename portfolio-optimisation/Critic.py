import math
class Critic:
    def __init__(self, actor, state, np_states,index, amount):
        self.actor = actor
        self.np_states = np_states
        self.index = index
        self.amount = amount

    def generate_actions(self):
        actions = []
        for i in range(30):
            actions.append(self.actor.select_action(self.np_states[i].tolist()))
        return actions
        
    def generate_advantages(self):
        states = self.np_states[:30].tolist()
        actions = self.generate_actions()
        rewards = []
        advantages = []
        for i in range(30):
            rewards.append(self.generate_reward(states[i],actions[i],i))

        avg_reward  = sum(rewards)/len(rewards)
        for i in range(30):
            advantages.append(rewards[i] - avg_reward)
        return (states,actions,advantages)

    def generate_reward(self,state,action,index=0):
        current_portfolio = 0
        for i in range(len(state)):
            current_portfolio += math.floor((action[i]*self.amount)/state[i][0]) * state[i][0]
        updated_portfolio = 0
        new_state = self.np_states[index+30]
        for i in range(len(state)):
            updated_portfolio += math.floor((action[i]*self.amount)/new_state[i][0]) * new_state[i][0]
        reward =  updated_portfolio - current_portfolio
       # print(f" Returns : ${reward} , Initial portfolio: ${current_portfolio}")
        return reward
