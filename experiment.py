import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DQN_Agent

# make it so that its easy to swap between the presets
CONFIGS = {
    "various_degrees": {
        "num_actions": 16, 
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

    # array to save what type of trial it is
    trial_type_array = []

    num_actions = current_config["num_actions"]
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False) # create array of values in radians 
    shift_lookup_table = {} # lookup table to find match
    for xx in range(num_actions):
        target_theta = (angles[xx] - (np.pi/2)) % (2*np.pi)
        shift_lookup_table[xx] = np.argmin(np.abs(angles - target_theta)) # find nearest neighbor to 90 degree shift


    for i in range(current_config["steps"]):
        # get an angle out of the possible options (usually 5) 

        stimulus_angle = random.randint(0, current_config["num_actions"] - 1)

        target_index = stimulus_angle       
    
        # 6% flipped (90 degree rotation instead)

        if i > (current_config["steps"] / 2):
            if random.random() < 0.06:
                # angles = np.linspace(0, 2 * np.pi, current_config["num_actions"], endpoint=False)
                # original_theta = angles[target_index]
                # target_theta = (original_theta - (np.pi / 2)) % (2 * np.pi)
                # target_index = np.argmin(np.abs(angles - target_theta))
                # stimulus_angle = random.randint(0, current_config["num_actions"] - 1)
                target_index = shift_lookup_table[stimulus_angle]

        # give the 3x2 stimulus
        stim_seq = []
        for _ in range(sequence_length):
            stim_seq.extend(get_stimulus_representation(stimulus_angle, current_config["num_actions"]))
        
        # list to tensor , then go from [6] to [1,6] (need 2d input)
        state_tensor = torch.FloatTensor(stim_seq).unsqueeze(0)

        # get the q values so i can look at the expected value
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        # choose an action and do it
        prediction = agent.choose_action(state_tensor)

        expected_value = q_values[0, prediction].item()

        
        # calculate the reward
        reward = calculate_reward(prediction, target_index, method=current_config["reward_method"])

        # 0 is expected reward, 1 is unexpected reward, 2 is unexpected lack of reward, 3 is expected lack of reward
        if expected_value >= 0.7 and reward == 1.0:
            trial_type = 0
        elif expected_value < 0.3 and reward == 1.0:
            trial_type = 1
        elif expected_value >= 0.7 and reward == 0.0:
            trial_type = 2
        elif expected_value < 0.3 and reward == 0.0:
            trial_type = 3
        else:
            trial_type = -1

        trial_type_array.append(trial_type)
        
        # update the weights
        loss = agent.update(state_tensor, prediction, reward)

        trial_activation = agent.model.activations['layer1'].cpu().numpy().flatten()
        activation_history.append(trial_activation)
        # log the weights
        loss_history.append(loss)
        accuracy_history.append(reward) # 1 if right, 0 if wrong
        
        
    # 0 is expected reward, 1 is unexpected reward, 2 is unexpected lack of reward, 3 is expected lack of reward
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
   
    # focus only on the second half of the experiment
    start_trial = current_config["steps"] // 2
    act_matrix = np.array(activation_history)
    types = np.array(trial_type_array)
    
    # get the mean activation per neuron, per the trial type (4 values per neuron)
    means_per_type = []
    for trial_category in range(4): # iterate through trial types
        # filter for trials matching the type within the second half
        matches = np.where((types == trial_category) & (np.arange(len(types)) >= start_trial))[0]
        
        if len(matches) > 0:
            means_per_type.append(np.mean(act_matrix[matches], axis=0)) # get mean activation
        else:
            # fill with zeros if the category never occurred (common for surprise early/late)
            means_per_type.append(np.zeros(act_matrix.shape[1]))


    # new figure for categorical analysis
    fig_cat, (tax1, tax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    neuron_indexes = np.arange(act_matrix.shape[1])
    bar_width = 0.2
    
    # plot mean activations across the four categories
    for category_position, mean_values in enumerate(means_per_type):
        tax1.bar(neuron_indexes + (category_position - 1.5) * bar_width, 
                    mean_values, bar_width, 
                    label=type_labels[category_position], 
                    color=colors[category_position])
    
    tax1.set_title("mean activation per reward category (second half)")
    tax1.set_ylabel("activation level")
    tax1.legend()


    # plot the omission signal (unexpected lack minus expected lack)
    # shows how neurons respond specifically to rule-breaking disappointment
    omission_delta = means_per_type[2] - means_per_type[3]
    tax2.bar(neuron_indexes, omission_delta, color='purple')
    tax2.set_title("omission signal (unexp lack - exp lack)")
    tax2.set_ylabel("delta activation")
    tax2.axhline(0, color='black', linewidth=1)

    # in both cases, here, the physical outcome is 0 reward
    # but because they're not 0, we know there's some internal state (violation of expectation)
    # positive bars mean that neurons fire more when a reward is missing (rule broken)

    # negative bars means they fire less when disappointed (imagine these are dopamine neurons)
    # 0 bars mean that they don't care about reward outcome

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


