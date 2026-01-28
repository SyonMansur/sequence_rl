import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DQN_Agent
from scipy.stats import pearsonr
from scipy.stats import zscore

# make it so that its easy to swap between the presets
CONFIGS = {
    "various_degrees": {
        "num_actions": 16, 
        "reward_method": "exact", 
        "steps": 20000, 
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

    # arrays to save components of the gradients
    l1_gradients_history = []
    l2_gradients_history = []

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
        loss, l1_fb, l2_fb = agent.update(state_tensor, prediction, reward)

        trial_activation = agent.model.activations['layer1'].cpu().numpy().flatten()
        activation_history.append(trial_activation)
        # log the weights
        loss_history.append(loss)
        accuracy_history.append(reward) # 1 if right, 0 if wrong

        l1_gradients_history.append(l1_fb)
        l2_gradients_history.append(l2_fb)
        

    
    # plotting labels 
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    start_trial = current_config["steps"] // 2 # only look at post habituation period
    neuron_indexes = np.arange(len(activation_history[0])) 
    width = 0.2 


    # convert to matrices for easier indexing
    act_matrix = np.array(activation_history)
    grad_matrix = np.array(l1_gradients_history) # just look at layer 1 for now
    types = np.array(trial_type_array)
    
    means_act = [] # activation means
    means_grad = [] # gradient means
    
    for t in range(4): # for each trial type
        # find matching trials in second half
        matches = np.where((types == t) & (np.arange(len(types)) >= start_trial))[0]
        
        if len(matches) > 0:
            means_act.append(np.mean(act_matrix[matches], axis=0))
            means_grad.append(np.mean(grad_matrix[matches], axis=0))
        else:
            means_act.append(np.zeros(act_matrix.shape[1]))
            means_grad.append(np.zeros(grad_matrix.shape[1]))


    # first figure will just be loss and accuracy over time
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # loss
    ax1.plot(loss_history, color='red', alpha=0.3, label='raw loss')
    rolling_loss = [np.mean(loss_history[max(0, j-100):j+1]) for j in range(len(loss_history))]
    ax1.plot(rolling_loss, color='darkred', linewidth=2, label='smoothed loss')
    ax1.set_title("rpe")
    ax1.legend()
    
    # accuracy
    rolling_acc = [np.mean(accuracy_history[max(0, j-100):j+1]) for j in range(len(accuracy_history))]
    ax2.plot(rolling_acc, color='blue')
    ax2.axvline(x=start_trial, color='black', linestyle='--', alpha=0.5, label='start of deviations')
    ax2.set_title("learning curve (accuracy)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('plots/1_performance_metrics.png')
    plt.close()


    # now let's look at the neural activations
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    
    # mean activations (subplot 1)
    for idx, mean_vals in enumerate(means_act):
        ax3.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, 
                label=type_labels[idx], color=colors[idx])
    ax3.set_title("mean activation per trial type")
    ax3.set_ylabel("activation (firing rate)")
    ax3.legend()
    
    # unexpected lack - expected lack (subplot 2)
    # if delta is zero, the neuron fires regardless of expectation
    # if delta is positive, its a disappointment neuron that fires because the prediction was violated
    # if delta is negative, its a reward expectation neuron that decreases firing when reward is omitted
    # delta negative is like a dopamine neuron (would fire at reward, but doesn't when reward not given)
    # this is because dopamine neurons encode RPE and RPE for unexpected lack is negative
    omission_delta = means_act[2] - means_act[3] # Unexp Lack - Exp Lack
    ax4.bar(neuron_indexes, omission_delta, color='purple')
    ax4.set_title("unexpected lack - expected lack")
    ax4.set_ylabel("delta activation")
    ax4.axhline(0, color='black', linewidth=1)

    # for what id imagine is a dopamine neuron, id want there to be a huge negative purple bar 
    # because we want dopamine neurons to encode RPE
    # for unexpected lack, RPE is 0 - 1 = -1
    # for expected lack, RPE is 0 - 0 = 0
    # so the difference should be negative 

    # we'd also expect the unexpected reward to be a lot bigger than the expected reward
    # because RPE for unexpected reward is 1 - 0 = +1
    # and for expected reward is 1 - 1 = 0
    
    plt.tight_layout()
    plt.savefig('plots/2_neural_activations.png')
    plt.close()


    # figure 3 for gradients (top-down feedback)
    fig3, ax5 = plt.subplots(figsize=(12, 6))
    
    # plot mean gradient
    for index, mean_grads in enumerate(means_grad):
        ax5.bar(neuron_indexes + (index-1.5)*width, mean_grads, width, 
                label=type_labels[index], color=colors[index])
        
    ax5.set_title("TD feedbacks (layer 1 gradients)")
    ax5.set_ylabel("mean gradient mag (|dLoss/dw|)")
    ax5.set_xlabel("neuron number")
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig('plots/3_top_down_feedback.png')
    plt.close()

    # for a dopamine neuron, id expect the green bar to be higher than the red bar because the surprise disappointment (red)
    # should cause less weight change because the activation is low

    # okay now we're gonna see if the above two are correlated across neurons
    # calculating the metrics
    # x axis: how does the neuron react to missing reward?

    # unexpected lack - expected lack (x axis)
    omission_response = means_act[2] - means_act[3] 
    # positive values mean the neuron increases firing when their prediction is violated (more firing when reward is missing than when they expect nothing)
    # negative values means that the neuron decreases firing when their prediction is violated (less firing when reward is missing than when they expect nothing)
    # the negative values is what i expect a dopamine neuron to look like

    # unexpected lack - unexpected reward (y axis)
    plasticity_bias = means_grad[2] - means_grad[1]
    # positive values here mean that the gradient is stronger during the rule flip -- they are responsible for learning the new rule
    # negative values are the ones that are supposed to learn from unexpected rewards to figure out this new rule

    # okay so in the plot, the top right quadrant are the neurons taht react when reward is missing (x) and get updated for it (y)
    # the bottom left are the neurons that shut down when reward is missing and mostly learn from rewards (dopamine-like)

    # calculate the separation score (correlation)
    correlation, pval = pearsonr(omission_response, plasticity_bias)
    
    fig4, ax6 = plt.subplots(figsize=(8, 8))
    ax6.scatter(omission_response, plasticity_bias, c='purple', alpha=0.6)
    
    # add a trendline so we can see the relationship
    m, b = np.polyfit(omission_response, plasticity_bias, 1)
    ax6.plot(omission_response, m*omission_response + b, color='black', linestyle='--')
    
    ax6.set_title(f"correlation {correlation:.3f}, p={pval:.3f})")
    ax6.set_xlabel("unexpected lack - expected lack (activations)")
    ax6.set_ylabel("unexpected lack - unexpected reward (gradients)")
    ax6.axhline(0, color='gray', alpha=0.3)
    ax6.axvline(0, color='gray', alpha=0.3)
    
    # save it
    plt.tight_layout()
    plt.savefig('plots/4_plasticity_separation.png')
    plt.close()


    # heatmap figure 5
    # stack the means into a matrix
    heatmap_data = np.stack(means_act, axis=1)
    
    # z-score across conditions for each neuron
    # axis=1 means we calculate mean/std across the 4 conditions for each row
    normalized_data = zscore(heatmap_data, axis=1)
    
    # sort indexes based on the raw omission response
    sort_indices = np.argsort(omission_response)
    sorted_data = normalized_data[sort_indices]
    
    fig5, ax7 = plt.subplots(figsize=(8, 10))
    
    # use 'coolwarm' centered at 0
    im = ax7.imshow(sorted_data, aspect='auto', cmap='coolwarm', interpolation='nearest', vmin=-2, vmax=2)
    
    ax7.set_title("z score neural population activity")
    ax7.set_xlabel("trial type")
    ax7.set_ylabel("neuron sorted by omission")
    
    ax7.set_xticks(np.arange(4))
    ax7.set_xticklabels(type_labels, rotation=45)
    
    plt.colorbar(im, ax=ax7, label='zcore activity')
    
    plt.tight_layout()
    plt.savefig('plots/5_population_heatmap.png')
    plt.close()
    print("heatmap generated.")


if __name__ == "__main__":
    run_experiment()


