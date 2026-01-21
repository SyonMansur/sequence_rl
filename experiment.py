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
    "various_degrees": {
        "num_actions": 10, 
        "reward_method": "exact", 
        "steps": 10000, 
        "alpha": 0.001
    }
}

# alter this string to change the versions
current_run = CONFIGS["various_degrees"]

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
    activation_history = []


    for i in range(current_config["steps"]):
        # get an angle out of the possible options (usually 5) 

        stimulus_angle = random.randint(0, current_config["num_actions"] - 1)

        target_index = stimulus_angle       
    
        # 6% flipped (90 degree rotation instead)

        if i > (current_config["steps"] / 2):
            if random.random() < 0.06:
                angles = np.linspace(0, 2 * np.pi, current_config["num_actions"], endpoint=False)
                original_theta = angles[target_index]
                target_theta = (original_theta - (np.pi / 2)) % (2 * np.pi)
                target_index = np.argmin(np.abs(angles - target_theta))

        # give the 3x2 stimulus
        stim_seq = []
        for _ in range(sequence_length):
            stim_seq.extend(get_stimulus_representation(stimulus_angle, current_config["num_actions"]))
        
        # list to tensor , then go from [6] to [1,6] (need 2d input)
        state_tensor = torch.FloatTensor(stim_seq).unsqueeze(0)
        
        # choose an action and do it
        prediction = agent.choose_action(state_tensor)
        
        # calculate the reward
        reward = calculate_reward(prediction, target_index, method=current_config["reward_method"])
        
        # update the weights
        loss = agent.update(state_tensor, prediction, reward)

        trial_activation = agent.model.activations['layer1'].cpu().numpy().flatten()
        activation_history.append(trial_activation)
        # log the weights
        loss_history.append(loss)
        accuracy_history.append(reward) # 1 if right, 0 if wrong



    # plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10,15))

    # loss
    ax1.plot(loss_history, color='red', alpha=0.5)
    ax1.set_title("rpe (loss)")

    # rolling reward - this is probability of success at any given moment
    # takes last 100 trials and plots what percent were correct
    rolling_r = []
    for j in range(len(accuracy_history)):
        start = max(0, j - 100)
        current_window = accuracy_history[start : j + 1]
        window_average = np.mean(current_window)
        rolling_r.append(window_average)
    ax2.plot(rolling_r, color='blue')
    ax2.set_title("learning curve")
    ax2.axvline(x = current_config["steps"] / 2, color = 'black', linestyle = '--')

    # heatmap
    im = ax3.imshow(activation_history, aspect='auto', cmap='viridis', interpolation='none')
    ax3.set_title("layer 1 activations")
    ax3.set_xlabel("neuron")
    ax3.set_ylabel("trial")
    fig.colorbar(im, ax=ax3, label="activation")

    rolling_loss = [np.mean(loss_history[max(0, j-100):j+1]) for j in range(len(loss_history))]
    ax4.plot(rolling_loss, color='darkred', linewidth=2, label='Mean Loss')
    ax4.set_title("mean loss over time")

    plt.tight_layout()
    plt.show()


    

if __name__ == "__main__":
    run_experiment()


