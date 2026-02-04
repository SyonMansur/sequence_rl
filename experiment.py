import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DQN_Agent
from scipy.stats import pearsonr
from scipy.stats import zscore

CONFIGS = {
    "various_degrees": {
        "num_actions": 16, 
        "reward_method": "exact", 
        "steps": 20000, 
        "alpha": 0.001
    }
}

def get_stimulus_representation(angle_index, total_options):
    # make the sin and cosine pair
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target, method="exact"):
    if method == "exact":
        if prediction == target:
            return 1.0
        else:
            return 0.0
    return 0.0

def run_experiment(config_name):
    current_config = CONFIGS[config_name]
    sequence_length = 3
    
    # initialize agent according to config
    agent = DQN_Agent(
        input_dimensions=sequence_length * 2, 
        num_actions=current_config["num_actions"], 
        alpha=current_config["alpha"]
    )
    
    # arrays to save
    loss_history = []
    accuracy_history = []
    activation_history = [] # storing deltas now

    # array to save what type of trial it is
    trial_type_array = []

    # arrays to save components of the gradients
    l1_gradients_history = []
    l2_gradients_history = []

    num_actions = current_config["num_actions"]
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False) 
    
    shift_lookup_table = {} 
    for xx in range(num_actions):
        target_theta = (angles[xx] - (np.pi/2)) % (2*np.pi)
        shift_lookup_table[xx] = np.argmin(np.abs(angles - target_theta)) 


    for i in range(current_config["steps"]):
        # get an angle out of the possible options 
        stimulus_angle = random.randint(0, current_config["num_actions"] - 1)
        target_index = stimulus_angle        
    
        # 6% flipped (90 degree rotation instead)
        if i > (current_config["steps"] / 2):
            if random.random() < 0.06:
                target_index = shift_lookup_table[stimulus_angle]

        # give the 3x2 stimulus
        stim_seq = []
        stim_rep = get_stimulus_representation(stimulus_angle, current_config["num_actions"])
        for _ in range(sequence_length):
            stim_seq.extend(stim_rep)
        
        state_tensor = torch.tensor(stim_seq, dtype=torch.float32).unsqueeze(0)

        # get the q values so i can look at the expected value
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        # choose an action and do it
        prediction = agent.choose_action(state_tensor)
        expected_value = q_values[0, prediction].item()

        
        # calculate the reward
        reward = calculate_reward(prediction, target_index, method=current_config["reward_method"])

        # classify trial type
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
        # this runs the backward pass
        loss, l1_fb, l2_fb = agent.update(state_tensor, prediction, reward)

        # now capturing the change in activation
        # first, get the activation BEFORE the update (using old weights)
        # we can just grab it from the agent since update() ran a forward pass
        pre_activation = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        
        # second, run a forward pass with the NEW weights
        with torch.no_grad():
            agent.model(state_tensor)
        post_activation = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()

        # calculate delta
        activation_delta = post_activation - pre_activation
        activation_history.append(activation_delta)

        # log the weights
        loss_history.append(loss)
        accuracy_history.append(reward) 

        l1_gradients_history.append(l1_fb)
        l2_gradients_history.append(l2_fb)
        

    
    # plotting labels 
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    start_trial = current_config["steps"] // 2 
    neuron_indexes = np.arange(len(activation_history[0])) 
    width = 0.2 


    # convert to matrices 
    act_matrix = np.array(activation_history)
    grad_matrix = np.array(l1_gradients_history) 
    types = np.array(trial_type_array)
    
    means_act = [] 
    means_grad = [] 
    
    for t in range(4): 
        matches = np.where((types == t) & (np.arange(len(types)) >= start_trial))[0]
        
        if len(matches) > 0:
            means_act.append(np.mean(act_matrix[matches], axis=0))
            means_grad.append(np.mean(grad_matrix[matches], axis=0))
        else:
            means_act.append(np.zeros(act_matrix.shape[1]))
            means_grad.append(np.zeros(grad_matrix.shape[1]))


    # figure 1: performance
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


    # figure 2: activation changes
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    
    # mean activations 
    for idx, mean_vals in enumerate(means_act):
        ax3.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, 
                label=type_labels[idx], color=colors[idx])
    ax3.set_title("mean activation CHANGE per trial type")
    ax3.set_ylabel("delta activation (post - pre)")
    ax3.legend()
    
    # unexpected lack - expected lack 
    omission_delta = means_act[2] - means_act[3] 
    ax4.bar(neuron_indexes, omission_delta, color='purple')
    ax4.set_title("unexpected lack - expected lack (delta)")
    ax4.set_ylabel("difference in change")
    ax4.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig('plots/2_neural_activations.png')
    plt.close()


    # figure 3: top-down feedback
    fig3, ax5 = plt.subplots(figsize=(12, 6))
    
    for index, mean_grads in enumerate(means_grad):
        ax5.bar(neuron_indexes + (index-1.5)*width, mean_grads, width, 
                label=type_labels[index], color=colors[index])
        
    ax5.set_title("TD feedbacks (dL/dActivation)")
    ax5.set_ylabel("mean gradient mag")
    ax5.set_xlabel("neuron number")
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig('plots/3_top_down_feedback.png')
    plt.close()


    # figure 4: correlation
    # x axis: reaction to omission (change in firing)
    omission_response = means_act[2] - means_act[3] 

    # y axis: bias in error signal 
    # (do they get more error signal during omission or unexpected reward?)
    plasticity_bias = means_grad[2] - means_grad[1]

    correlation, pval = pearsonr(omission_response, plasticity_bias)
    
    fig4, ax6 = plt.subplots(figsize=(8, 8))
    ax6.scatter(omission_response, plasticity_bias, c='purple', alpha=0.6)
    
    m, b = np.polyfit(omission_response, plasticity_bias, 1)
    ax6.plot(omission_response, m*omission_response + b, color='black', linestyle='--')
    
    ax6.set_title(f"correlation {correlation:.3f}, p={pval:.3f})")
    ax6.set_xlabel("unexpected lack - expected lack (activation change)")
    ax6.set_ylabel("unexpected lack - unexpected reward (error signal)")
    ax6.axhline(0, color='gray', alpha=0.3)
    ax6.axvline(0, color='gray', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/4_plasticity_separation.png')
    plt.close()


    # figure 5: heatmap
    heatmap_data = np.stack(means_act, axis=1)
    
    # z-score normalization
    normalized_data = zscore(heatmap_data, axis=1)
    
    sort_indices = np.argsort(omission_response)
    sorted_data = normalized_data[sort_indices]
    
    fig5, ax7 = plt.subplots(figsize=(8, 10))
    
    im = ax7.imshow(sorted_data, aspect='auto', cmap='coolwarm', interpolation='nearest', vmin=-2, vmax=2)
    
    ax7.set_title("z score neural population activity CHANGE")
    ax7.set_xlabel("trial type")
    ax7.set_ylabel("neuron sorted by omission")
    
    ax7.set_xticks(np.arange(4))
    ax7.set_xticklabels(type_labels, rotation=45)
    
    plt.colorbar(im, ax=ax7, label='zcore activity change')
    
    plt.tight_layout()
    plt.savefig('plots/5_population_heatmap.png')
    plt.close()
    print("heatmap generated.")


if __name__ == "__main__":
    run_experiment("various_degrees")


