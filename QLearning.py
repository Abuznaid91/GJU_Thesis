from delivery import *
env = DeliveryEnvironment(n_stops = 128,method = "distance")
#print(" main program , env =  then env.render :")
#env.render()
#print(" main program ,env.render , then agent = :")
agent = DeliveryQAgent(env.observation_space,env.action_space)
#agent = DeliveryQAgent()
#print(" main program , agent then run_n_episodes :")
run_n_episodes(env,agent,"training_abuznaid_stops.gif")