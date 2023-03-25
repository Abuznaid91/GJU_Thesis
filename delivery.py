# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
import imageio
from io import BytesIO
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


plt.style.use("seaborn-dark")


class DeliveryEnvironment(object):
    #print(" in the starting of class DeliveryEnvironment ")
    def __init__(self,n_stops = 10,max_box = 10,**kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")

        # Initialization
        self.n_stops = n_stops
        self.max_box = max_box
        self.stops = []

        # Generate stops
        self._generate_stops()
        self._generate_q_values()
        self.render(0)


        # Initialize first point
        self.reset()


    def _generate_stops(self):
        #print("in _generate_stops")
            # Generate geographical coordinates
# Abuznaid : generate n_stops of random numbers for both x and y
        xy = np.random.rand(self.n_stops, 2) * self.max_box
        self.x = xy[:,0]
        self.y = xy[:,1]


    def _generate_q_values(self):
        #print("in _generate_q_values")
        xy = np.column_stack([self.x, self.y]) # two columns
        self.q_stops = cdist(xy,xy) # distance matrix
        #in case you want adj matrix as input , use this
        #self.q_stops = np.array([[5 , 6, 7 , 9], [ 1 ,2 ,3 , 7], [ 1 ,2 ,3 , 2], [ 1 ,2 ,3 , 2]])
        #print("distance matrix", type(self.q_stops),"=", self.q_stops)


    def render(self, i, return_img=False):
        self.i=i
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops episode {}".format(self.i))

        # Show stops
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="blue", linewidth=1, linestyle="--")

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height)
            collection = PatchCollection([rect], facecolor="red", alpha=0.2)
            ax.add_collection(collection)

        plt.xticks([])
        plt.yticks([])

        if return_img:

            # create a memory buffer for the figure
            buffer = BytesIO()
            # save the figure to the buffer as a png image
            fig.savefig(buffer, format='png', dpi=300)

            # reset the buffer position to the start
            buffer.seek(0)

            # read the buffer as an imageio image object
            img_array = imageio.imread(buffer)


            plt.close()
            #return image
            return img_array
        else:
            plt.show()

    def reset(self):
        #print("in reset")

        # Stops placeholder
        self.stops = []

        # Abuznaid : start at zero every time , zero as first stop
        first_stop = 0
        self.stops.append(first_stop)

        return first_stop

    def step(self,destination):
        #print("in step")

        # Get current state
        state = self._get_state()
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state,new_state)

        # Append new_state to stops
        self.stops.append(destination)
        #print("self.stops in step = ", self.stops)
        done = len(self.stops) == self.n_stops

        return new_state,reward,done

    def _get_state(self):
        #print("in _get_state")
        return self.stops[-1]

    def _get_xy(self,initial = True):
        #print("in _get_xy")
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y

    def _get_reward(self,state,new_state):
        #print("in _get_reward")
        base_reward = self.q_stops[state,new_state]
        return base_reward


def run_episode(env,agent,verbose = 1):
    #print(" in run_episode",run_episode)

    s = env.reset()
    agent.reset_memory()

    max_step = env.n_stops
    
    episode_reward = 0
    
    i = 0
    while i < max_step:

        # Remember the states
        agent.remember_state(s)

        # Choose an action
        #print("choose  an action for the episode run")
        a = agent.act(s)

        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agent.train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward


class DeliveryQAgent():
    #print("in the start of class DeliveryQAgent")

    def __init__(self, states_size, actions_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9, gamma=0.7,
                 lr=0.8):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(states_size, actions_size)


    def build_model(self,states_size,actions_size):
        Q = np.zeros([states_size,actions_size])
        return Q

    def act(self,s):

        # Get Q Vector
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf
        #print("epsilon value =  ",self.epsilon)

        if np.random.rand() > self.epsilon:
            #print("choose the max action in Q")
            a = np.argmax(q)
        else:
            #print("choose action randomly")
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a

    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []

    def train(self,s,a,r,s_next):
        #print("inside q-Agent Train")
        self.Q[s,a] = self.Q[s,a] + self.lr * (r + self.gamma*np.max(self.Q[s_next,a]) - self.Q[s,a])
        #print(' Q value from training', self.Q[s,a])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_n_episodes(env,agent,name="training_abuzniad.gif",n_episodes=200,render_each=1,make_gif=False):
    #print("in run_n_episodes ")

    # Store the rewards
    rewards = []
    imgs = []
    episods =[]


    max_rewards =[]
    # plt.figure(figsize=(12, 3))
    # plt.title("Max Rewards over training")


    # Create the first figure and plot the rewards data
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))



    # Experience replay
    for i in tqdm(range(n_episodes)):

        episods.append(i)

        # Run the episode
        env,agent,episode_reward = run_episode(env,agent,verbose = 0)
        rewards.append(episode_reward)
        #print("reward =", episode_reward)

        if i==0:
            max_rewards.append(episode_reward)
        else:
            if episode_reward > max_rewards[-1]:
                max_rewards.append(episode_reward)
            else:
                max_rewards.append(max_rewards[-1])

        
        if i % render_each == 0:
            img = env.render(i=i,return_img = True)
            imgs.append(img)
    #print('rewards = ', rewards)

    # Show rewards
    # plt.figure(figsize = (15,3))
    # plt.title("Rewards over training")
    # plt.plot(rewards)
    # plt.show()

    #print("max_reward = ", max_rewards)

    # plt.plot(max_rewards)
    # plt.show()

    axs[0].plot(rewards)
    axs[0].set_title('Rewards')

    # Create the second figure and plot the max_rewards data
    axs[1].plot(episods,max_rewards)
    axs[1].set_title('Max Rewards')

    # Adjust the layout of the figures and show the plot
    plt.tight_layout()
    plt.show()




    if make_gif:
        imageio.mimsave(name,imgs,duration=0.1)

    return env,agent