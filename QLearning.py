from delivery import *


#define the number of stops to be shown in your environment
n_stops = 100


#create Environment where it models the places that the agent will travel between
#Environment is created randomly here, if you want to input a specific adj matrix uncomment the lines
env = DeliveryEnvironment(n_stops = n_stops)

# environment render showing the start position of the Agent
env.render(0)

#Agent Parameters:
#epsilon , epsilon_min and epsilon_decay : (ε) is a hyperparameter that is used to balance exploration and exploitation in the agent's decision-making process.
#low epsilon means greedy behaviour means that the agent will select the answer that his training ( q-tables ) tells
#High epsilon value means that the agent will explore new solutions randomly
#epsilon_decaying is a method used to allow exploration at the first learning epsiods where the q-table is not yet good enough to follow, and moving gradually to greedy behaviour while the number of training episods is higher
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.9

#The learning rate α determines how much weight the agent places on the new information obtained from the most recent experience
#high learning rate means that the agent quickly incorporates new information into its Q-value estimates
#low learning rate means that the agent updates its Q-values more slowly
learning_rate=0.8

#The discount factor is used to discount future rewards by a factor of γ^(t),
# where t is the number of time steps into the future. This means that rewards received in the future are worth less than rewards received immediately.
discount_factor = 0.7

#create agent class with his own parameters, the agent class is the responsible of learning and acting
agent = DeliveryQAgent(n_stops,n_stops, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, gamma=discount_factor,lr=learning_rate)


# start the process of learning, number of episods is how much trails will the agent do
number_of_episods =300
run_n_episodes(env,agent,name="training_32_stops_300_episods.gif",n_episodes=number_of_episods,render_each=1,make_gif=True)