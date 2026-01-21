import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DQN_Agent

# make it so that its easy to swap between the presets
CONFIGS = {
    "five_options": {
        "num_actions": 5, 
        "reward_method": "exact", 
        "steps": 10000, 
        "alpha": 0.001
    },
    "degrees_360": {
        "num_actions": 360, 
        "reward_method": "exact", 
        "steps": 10000, 
        "alpha": 0.0005
    }
}

# alter this string to change the versions
current_run = CONFIGS["five_options"]

def get_stimulus_representation(angle_index, total_options):
    # make the sin and cosine pair
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target, method="exact"):
    # calculate reward; maybe we'll give points for being kinda close in the future
    if method == "exact":
        if prediction == target:
            return 1.0
        else:
            return 0.0
    # some some of partial credit here perhaps in the future
    return 0.0

def run_experiment():
    current_config = current_run
    sequence_length = 3
    
    # initialize agent according to config
    agent = DQN_Agent(
        input_dimensions = sequence_length * 2, 
        num_actions = current_config["num_actions"], 
        alpha = current_config["alpha"]
    )
    
    # arrays to save
    loss_history = []
    accuracy_history = []


    for i in range(current_config["steps"]):
        # get an angle out of the possible options (usually 5)
        target_index = random.randint(0, current_config["num_actions"] - 1)
        
        # give the 3x2 stimulus
        stim_seq = []
        for _ in range(sequence_length):
            stim_seq.extend(get_stimulus_representation(target_index, current_config["num_actions"]))
        
        # list to tensor , then go from [6] to [1,6] (need 2d input)
        state_tensor = torch.FloatTensor(stim_seq).unsqueeze(0)
        
        # choose an action and do it
        prediction = agent.choose_action(state_tensor)
        
        # calculate the reward
        reward = calculate_reward(prediction, target_index, method=current_config["reward_method"])
        
        # update the weights
        loss = agent.update(state_tensor, prediction, reward)
        
        # log the weights
        loss_history.append(loss)
        accuracy_history.append(reward) # 1 if right, 0 if wrong

    # plot
    plt.plot(loss_history, alpha=0.3)
    plt.title("loss")
    plt.show()

    

if __name__ == "__main__":
    run_experiment()


