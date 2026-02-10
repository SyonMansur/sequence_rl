import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import DQN_Agent

CONFIG = {
    "num_actions": 16, 
    "steps": 20000, 
    "alpha": 0.001
}

def get_stimulus_representation(angle_index, total_options):
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target):
    return 1.0 if prediction == target else 0.0

def run_experiment():
    print(f"Running Experiment | generating all plots including q-values")
    
    sequence_length = 3
    agent = DQN_Agent(
        input_dimensions=sequence_length * 2, 
        num_actions=CONFIG["num_actions"], 
        alpha=CONFIG["alpha"]
    )
    
    # history arrays
    loss_history = []
    accuracy_history = []
    activation_history = [] 
    trial_type_array = []
    
    # new history: tracking the raw expectation (Q-value)
    q_value_history = []
    
    # feedback metrics
    l1_feedback_history = []
    l2_feedback_history = []
    raw_error_history = []

    num_actions = CONFIG["num_actions"]
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False) 
    
    shift_lookup_table = {} 
    for xx in range(num_actions):
        target_theta = (angles[xx] - (np.pi/2)) % (2*np.pi)
        shift_lookup_table[xx] = np.argmin(np.abs(angles - target_theta)) 

    # training loop
    for i in range(CONFIG["steps"]):
        stimulus_angle = random.randint(0, CONFIG["num_actions"] - 1)
        target_index = stimulus_angle        
    
        if i > (CONFIG["steps"] / 2):
            if random.random() < 0.06:
                target_index = shift_lookup_table[stimulus_angle]

        stim_seq = []
        stim_rep = get_stimulus_representation(stimulus_angle, CONFIG["num_actions"])
        for _ in range(sequence_length):
            stim_seq.extend(stim_rep)
        
        state_tensor = torch.tensor(stim_seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        prediction = agent.choose_action(state_tensor)
        expected_value = q_values[0, prediction].item()
        reward = calculate_reward(prediction, target_index)

        # classification
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
        
        # update step
        loss, l1_fb, l2_fb, raw_err = agent.update(state_tensor, prediction, reward)

        pre_activation = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        with torch.no_grad():
            agent.model(state_tensor) 
        post_activation = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        activation_delta = post_activation - pre_activation
        
        activation_history.append(activation_delta)
        loss_history.append(loss)
        accuracy_history.append(reward) 
        
        # save q value
        q_value_history.append(expected_value)
        
        l1_feedback_history.append(l1_fb)
        l2_feedback_history.append(l2_fb)
        raw_error_history.append(raw_err)
        
        if i % 1000 == 0:
            print(f"Step {i}, Loss: {loss:.4f}, Acc: {np.mean(accuracy_history[-100:]):.2f}")

    # --- PLOTTING ---
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    start_trial = CONFIG["steps"] // 2 
    neuron_indexes = np.arange(len(activation_history[0])) 
    width = 0.2 

    act_matrix = np.array(activation_history)
    l1_matrix = np.array(l1_feedback_history)
    l2_matrix = np.array(l2_feedback_history)
    raw_matrix = np.array([np.full(len(neuron_indexes), err) for err in raw_error_history])
    
    # q values are scalar, need to broadcast for bar chart consistency or just compute mean scalar
    # actually, for the q-value plot, we don't need neuron dimensions. 
    # we just want a single bar per trial type showing the average Q.
    q_matrix = np.array(q_value_history) 
    
    types = np.array(trial_type_array)
    
    means_act = [] 
    means_l1 = []
    means_l2 = []
    means_raw = []
    means_q = [] # store scalar means here
    
    for t in range(4): 
        matches = np.where((types == t) & (np.arange(len(types)) >= start_trial))[0]
        if len(matches) > 0:
            means_act.append(np.mean(act_matrix[matches], axis=0))
            means_l1.append(np.mean(l1_matrix[matches], axis=0))
            means_l2.append(np.mean(l2_matrix[matches], axis=0))
            means_raw.append(np.mean(raw_matrix[matches], axis=0))
            means_q.append(np.mean(q_matrix[matches])) # scalar mean
        else:
            zeros = np.zeros(act_matrix.shape[1])
            means_act.append(zeros)
            means_l1.append(zeros)
            means_l2.append(zeros)
            means_raw.append(zeros)
            means_q.append(0.0)

    # 1. AVERAGE LOSS
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(loss_history, color='red', alpha=0.3)
    rolling_loss = [np.mean(loss_history[max(0, j-100):j+1]) for j in range(len(loss_history))]
    ax1.plot(rolling_loss, color='darkred', linewidth=2)
    ax1.set_title(f"RPE (Loss)")
    
    rolling_acc = [np.mean(accuracy_history[max(0, j-100):j+1]) for j in range(len(accuracy_history))]
    ax2.plot(rolling_acc, color='blue')
    ax2.axvline(x=start_trial, color='black', linestyle='--')
    ax2.set_title("Learning Curve (Accuracy)")
    plt.tight_layout()
    plt.savefig('plots/1_average_loss.png')
    plt.close()

    # 2. NEURAL ACTIVATIONS
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    for idx, mean_vals in enumerate(means_act):
        ax3.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, 
                label=type_labels[idx], color=colors[idx])
    ax3.set_title("Mean Activation Delta (Plasticity)")
    ax3.legend()
    
    omission_delta = means_act[2] - means_act[3]
    ax4.bar(neuron_indexes, omission_delta, color='purple')
    ax4.set_title("Unexpected Lack - Expected Lack")
    ax4.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig('plots/2_neural_activations.png')
    plt.close()

    # 3. LAYER 1 TOPDOWN
    fig3, ax5 = plt.subplots(figsize=(12, 6))
    for index, mean_vals in enumerate(means_l1):
        ax5.bar(neuron_indexes + (index-1.5)*width, mean_vals, width, 
                label=type_labels[index], color=colors[index])
    ax5.set_title("Layer 1 Top-Down Signal (Filtered by Layer 2)")
    ax5.set_ylabel("Gradient Magnitude")
    ax5.legend()
    plt.tight_layout()
    plt.savefig('plots/3_layer_1_topdown.png')
    plt.close()

    # 4. LAYER 2 TOPDOWN
    fig4, ax6 = plt.subplots(figsize=(12, 6))
    for index, mean_vals in enumerate(means_l2):
        ax6.bar(neuron_indexes + (index-1.5)*width, mean_vals, width, 
                label=type_labels[index], color=colors[index])
    ax6.set_title("Layer 2 Top-Down Signal (Less Filtered)")
    ax6.set_ylabel("Gradient Magnitude")
    ax6.legend()
    plt.tight_layout()
    plt.savefig('plots/4_layer_2_topdown.png')
    plt.close()

    # 5. RAW SOURCE ERROR
    fig5, ax7 = plt.subplots(figsize=(12, 6))
    for index, mean_vals in enumerate(means_raw):
        ax7.bar(neuron_indexes + (index-1.5)*width, mean_vals, width, 
                label=type_labels[index], color=colors[index])
    ax7.set_title("Raw Output Error Source (Unfiltered)")
    ax7.set_ylabel("Error Magnitude |Prediction - Target|")
    ax7.legend()
    plt.tight_layout()
    plt.savefig('plots/5_raw_source_error.png')
    plt.close()
    
    # 6. RAW Q-VALUES (EXPECTATION ANALYSIS)
    # simple bar chart of the 4 scalar means
    fig6, ax8 = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(type_labels))
    ax8.bar(x_pos, means_q, color=colors, alpha=0.8)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(type_labels)
    ax8.set_title("Agent Confidence (Raw Q-Value Expectation)")
    ax8.set_ylabel("Predicted Value (0.0 - 1.0)")
    ax8.set_ylim(0, 1.1)
    
    # adding text labels on top of bars
    for i, v in enumerate(means_q):
        ax8.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('plots/6_agent_confidence.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()