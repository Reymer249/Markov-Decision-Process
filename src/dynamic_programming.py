#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None  # will store a potential value solution table
        self.Q_sa = None  # will store a potential action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)
    
        ## IMPLEMENT YOUR VALUE ITERATION ALGORITHM HERE
        delta = theta
        while delta >= theta:
            delta = 0
            for state in env.states:
                x = V_s[state]

                max_trans_value = -np.inf
                for action in env.actions:  # The Bellman equation formula
                    s_prime, r = env.transition_function(state, action)
                    value = r + gamma * V_s[s_prime]
                    max_trans_value = value if value > max_trans_value else max_trans_value

                V_s[state] = max_trans_value
                delta = max(delta, abs(V_s[state] - x))
    
        self.V_s = V_s
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states, env.n_actions])

        ## IMPLEMENT YOUR Q-VALUE ITERATION ALGORITHM HERE
        delta = theta
        while delta >= theta:
            delta = 0
            for state in env.states:
                for action_n in range(len(env.actions)):
                    x = Q_sa[state][action_n]
                    s_prime, r = env.transition_function(state, env.actions[action_n])
                    Q_sa[state][action_n] = r + gamma * max(Q_sa[s_prime])  # The Bellman equation formula
                    delta = max(delta, abs(Q_sa[state][action_n] - x))

        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state()  # this is the current state of the environment, from which you will act
            available_actions = env.actions
            gamma = 1.0  # Must be the same, as in the call of self.value_iteration() function
            # Compute action values
            if table == 'V' and self.V_s is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.V_s HERE !!!
                greedy_action = None
                greedy_value = -np.inf
                for action in available_actions:
                    s_prime, r = env.transition_function(current_state, action)
                    considered_value = r + gamma*self.V_s[s_prime]  # We have to apply the Bellman equation again to
                    # find the best action to make. In other words, having the value function we still need to use
                    # Bellman equation for every state to get the policy.
                    if considered_value > greedy_value:
                        greedy_value = considered_value
                        greedy_action = action
                
            
            elif table == 'Q' and self.Q_sa is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.Q_sa here !!!
                # As you may see, here we just use the value we already have in the table. However, the table is
                # [n_actions] times bigger. There is an exchange between number of computations and the size of the table.
                greedy_action = available_actions[np.argmax(self.Q_sa[current_state])]

                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison2.txt')
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

