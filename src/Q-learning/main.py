import numpy as np


'''
Author: Rahul Sajnani
Date  : 26th December 2020

'''

def generateMap():
    '''
    Function to create map
    
    Input:
        None
    
    Returns:
        map - H x W - numpy ndarray 
            Map with 0 determining positions that contain obstacles and 1 is the path that the agent can take
        
        reward_map - H x W - numpy ndarray
            Reward map 
        
        source_position - [h, w]
        target_location - [h, w]
    '''


    level_map = np.array([[ 0, 0, 0, 0, 0, 0, 2, 0],
                          [ 0, 1, 1, 1, 1, 1, 1, 0],
                          [ 0, 1, 0, 1, 0, 1, 0, 0],
                          [ 0, 0, 1, 1, 1, 1, 1, 0],
                          [ 0, 1, 1, 1, 1, 1, 1, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0]])
    
    reward_map = np.ones(level_map.shape) * -1
    source_position = np.array([4, 1])
    target_position = np.array([0, 2])

    for i in range(level_map.shape[0]):
        for j in range(level_map.shape[1]):

            if level_map[i, j] == 0:
                reward_map[i, j] = -100
            elif level_map[i, j] == 2:
                reward_map[i, j] = 100

    level_dictionary = {'level_map': level_map,
                        'reward_map': reward_map,
                        'source_position': source_position,
                        'target_position': target_position}

    return level_dictionary


class Q_Learning:
    '''
    Q learning class
    '''

    def __init__(self):
        
        pass
        
    def getBestPath(self):
        '''
        Obtain best path from the Q table that is learned
        '''
        
        agent_position = self.level["source_position"]
        path = [(agent_position[0], agent_position[1])]
        q_table = self.q_table

        while self.level["reward_map"][agent_position[0], agent_position[1]] == -1:

            action_index = np.argmax(q_table[agent_position[0], agent_position[1], :])
            agent_position += self.actions[action_index]
            path.append((agent_position[0], agent_position[1]))
        
        return path


    def train(self, level_dictionary, learning_rate, decay_rate, training_iter, optimum_action_probability):
        '''
        Fill the Q table for the AI agent
        '''
        
        # Initialization
        self.level         = level_dictionary
        self.lr            = learning_rate
        self.decay         = decay_rate
        self.training_iter = training_iter
        self.q_table       = np.zeros((self.level["level_map"].shape[0], self.level["level_map"].shape[1], 4))
        self.actions       = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]) # right, left, down, up
        self.opt           = optimum_action_probability

        level_map = self.level["level_map"]
        reward_map = self.level["reward_map"].copy()

        # print(reward_map)
        for i in range(self.training_iter):
            
            agent_position = self.level["source_position"].copy()
            
            while reward_map[agent_position[0], agent_position[1]] == -1:
                
                
                # Till the agent does not crash/reach goal
                if np.random.rand() < self.opt:
                    action_index = np.argmax(self.q_table[agent_position[0], agent_position[1], :])
                else:
                    action_index = np.random.randint(4)

                agent_position_old = agent_position.copy()
                agent_position += self.actions[action_index]

                # Get reward
                reward_t = reward_map[agent_position_old[0], agent_position_old[1]]
                q_value_t = self.q_table[agent_position_old[0], agent_position_old[1], action_index].copy()
                
                # Temporal difference (Bellman equation)
                # TD_t = r_t + gamma * (max Q_t+1) - Q_t  
                temporal_difference_t = reward_t + self.decay * (np.max(self.q_table[agent_position[0], agent_position[1]])) - q_value_t

                # Updating Q values
                # Q_t = Q_t_old + learning_rate * TD_t
                q_value_new = q_value_t + self.lr * temporal_difference_t
                self.q_table[agent_position_old[0], agent_position_old[1], action_index] = q_value_new


            # For terminal locations
            self.q_table[agent_position[0], agent_position[1]] = reward_map[agent_position[0], agent_position[1]]
            
        
        # print(np.max(self.q_table, axis=2))
    
if __name__=="__main__":

    map_dictionary = generateMap()
    trainer = Q_Learning()
    trainer.train(map_dictionary, 0.9, 0.9, 100, 0.9)
    print("Shortest path indices:", trainer.getBestPath())
       