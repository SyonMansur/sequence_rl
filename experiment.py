import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from model import DQN_Agent, DTPAgent

# config
CONFIG = {
    "num_actions": 16, 
    "steps": 90000, 
    "alpha": 0.001,
    "use_dtp": False
}

def get_stimulus_representation(angle_index, total_options):
    # angle to sin and cos
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target):
    # binary hit or miss
    return 1.0 if prediction == target else 0.0

def run_experiment():
    print("running")
    
    if not os.path.exists('plots'):
        os.makedirs('plots')

    sequence_length = 3
    if CONFIG["use_dtp"]:
        agent = DTPAgent(
            input_dimensions=sequence_length * 2, 
            num_actions=CONFIG["num_actions"], 
            alpha=CONFIG["alpha"]
        )
    else:
        agent = DQN_Agent(
            input_dimensions=sequence_length * 2, 
            num_actions=CONFIG["num_actions"], 
            alpha=CONFIG["alpha"]
        )
    
    # basic tracking
    loss_history = []
    accuracy_history = []
    trial_type_array = []
    
    # track acts and td errors
    raw_act_l1_history = []
    raw_act_l2_history = []
    l1_feedback_history = []
    l2_feedback_history = []
    
    # math parts
    delta_history = []
    w3_history = []
    a2_relu_history = []
    a2_gated_history = []

    num_actions = CONFIG["num_actions"]
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False) 
    
    # setup logic for the switch
    shift_lookup_table = {} 
    for xx in range(num_actions):
        target_theta = (angles[xx] - (np.pi/2)) % (2*np.pi)
        shift_lookup_table[xx] = np.argmin(np.abs(angles - target_theta)) 

    # main run
    for i in range(CONFIG["steps"]):
        stimulus_angle = random.randint(0, CONFIG["num_actions"] - 1)
        target_index = stimulus_angle        
    
        # hit them with the shift halfway thru
        if i > (CONFIG["steps"] / 2):
            if random.random() < 0.20:
                target_index = shift_lookup_table[stimulus_angle]

        # build out the input
        stim_seq = []
        stim_rep = get_stimulus_representation(stimulus_angle, CONFIG["num_actions"])
        for _ in range(sequence_length):
            stim_seq.extend(stim_rep)
        
        state_tensor = torch.tensor(stim_seq, dtype=torch.float32).unsqueeze(0)

        # network guess
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        prediction = agent.choose_action(state_tensor)
        expected_value = q_values[0, prediction].item()
        reward = calculate_reward(prediction, target_index)

        # tag the trial
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
        
        # grab pre acts
        pre_act_l1 = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        pre_act_l2 = agent.model.activation_tensors['layer2'].clone().detach().cpu().numpy().flatten()
        
        a_tensor = torch.tensor([[prediction]]) 
        r_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        # fire updates
        if CONFIG["use_dtp"]:
            agent.update_inverse(state_tensor)
            loss, l1_fb, l2_fb, raw_err, delta, w3_act, a2_relu, a2_gtd, w1_m, w2_m, w3_m = agent.update_forward(state_tensor, a_tensor, r_tensor)
        else:
            loss, l1_fb, l2_fb, raw_err, delta, w3_act, a2_relu, a2_gtd, w1_m, w2_m, w3_m = agent.update(state_tensor, a_tensor, r_tensor)

        # log everything
        raw_act_l1_history.append(pre_act_l1)
        raw_act_l2_history.append(pre_act_l2)
        loss_history.append(loss)
        accuracy_history.append(reward) 
        l1_feedback_history.append(l1_fb)
        l2_feedback_history.append(l2_fb)
        
        delta_history.append(delta)
        w3_history.append(w3_act)
        a2_relu_history.append(a2_relu)
        a2_gated_history.append(a2_gtd)
        
        if i % 10000 == 0:
            print(f"step {i} loss {loss:.4f} acc {np.mean(accuracy_history[-100:]):.2f}")

    # plot prep
    start_trial = CONFIG["steps"] // 2 
    types = np.array(trial_type_array)
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    
    l1_matrix = np.array(l1_feedback_history)
    l2_matrix = np.array(l2_feedback_history)
    act1_matrix = np.array(raw_act_l1_history)
    act2_matrix = np.array(raw_act_l2_history)
    
    neuron_indexes = np.arange(l1_matrix.shape[1]) 
    width = 0.2 

    # calc means
    means_l1 = []
    means_l2 = []
    means_act1 = []
    means_act2 = []
    
    for t in range(4): 
        matches = np.where((types == t) & (np.arange(len(types)) >= start_trial))[0]
        if len(matches) > 0:
            means_l1.append(np.mean(l1_matrix[matches], axis=0))
            means_l2.append(np.mean(l2_matrix[matches], axis=0))
            means_act1.append(np.mean(np.abs(act1_matrix[matches]), axis=0))
            means_act2.append(np.mean(np.abs(act2_matrix[matches]), axis=0))
        else:
            z1 = np.zeros(l1_matrix.shape[1])
            z2 = np.zeros(l2_matrix.shape[1])
            means_l1.append(z1)
            means_l2.append(z1)
            means_act1.append(z1)
            means_act2.append(z2)

    # plot one. loss and acc.
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    rolling_loss = [np.mean(loss_history[max(0, j-100):j+1]) for j in range(len(loss_history))]
    ax1.plot(rolling_loss, color='darkred', linewidth=2)
    ax1.set_title("loss")
    
    rolling_acc = [np.mean(accuracy_history[max(0, j-100):j+1]) for j in range(len(accuracy_history))]
    ax2.plot(rolling_acc, color='blue')
    ax2.axvline(x=start_trial, color='black', linestyle='--')
    ax2.set_title("accuracy")
    plt.tight_layout()
    plt.savefig('plots/1_loss_and_accuracy.png')
    plt.close()

    # plot two. layer one topdown.
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_l1):
        ax3.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax3.set_title("layer one td signal")
    ax3.legend()
    plt.tight_layout()
    plt.savefig('plots/2_layer_1_topdown.png')
    plt.close()

    # plot three. layer two topdown.
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_l2):
        ax4.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax4.set_title("layer two td signal")
    ax4.legend()
    plt.tight_layout()
    plt.savefig('plots/3_layer_2_topdown.png')
    plt.close()

    # plot four. layer one acts.
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_act1):
        ax5.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax5.set_title("layer one activations")
    ax5.legend()
    plt.tight_layout()
    plt.savefig('plots/4_layer_1_activations.png')
    plt.close()

    # plot five. layer two acts.
    fig5, ax6 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_act2):
        ax6.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax6.set_title("layer two activations")
    ax6.legend()
    plt.tight_layout()
    plt.savefig('plots/5_layer_2_activations.png')
    plt.close()
    
    # figure two b recreation
    # split second half into three blocks
    halfway = CONFIG["steps"] // 2
    sess_len = halfway // 3
    sessions = [
        (halfway, halfway + sess_len),
        (halfway + sess_len, halfway + 2 * sess_len),
        (halfway + 2 * sess_len, CONFIG["steps"])
    ]
    
    # arrays for plotting
    l1_dend_y, l1_dend_err = [], []
    l2_dend_y, l2_dend_err = [], []
    l1_soma_y, l1_soma_err = [], []
    l2_soma_y, l2_soma_err = [], []
    
    for start, end in sessions:
        # grab indices for this time block
        block_idx = (np.arange(len(types)) >= start) & (np.arange(len(types)) < end)
        viol_idx = np.where(block_idx & ((types == 1) | (types == 2)))[0]
        match_idx = np.where(block_idx & ((types == 0) | (types == 3)))[0]
        
        if len(viol_idx) == 0 or len(match_idx) == 0:
            continue
            
        # calc mean diff per neuron then avg across layer
        # l1 dendrite (td signal)
        l1_td_diff = np.mean(l1_matrix[viol_idx], axis=0) - np.mean(l1_matrix[match_idx], axis=0)
        l1_dend_y.append(np.mean(l1_td_diff))
        l1_dend_err.append(np.std(l1_td_diff) / np.sqrt(len(l1_td_diff)))
        
        # l2 dendrite (td signal)
        l2_td_diff = np.mean(l2_matrix[viol_idx], axis=0) - np.mean(l2_matrix[match_idx], axis=0)
        l2_dend_y.append(np.mean(l2_td_diff))
        l2_dend_err.append(np.std(l2_td_diff) / np.sqrt(len(l2_td_diff)))
        
        # l1 soma (forward act)
        l1_act_diff = np.mean(act1_matrix[viol_idx], axis=0) - np.mean(act1_matrix[match_idx], axis=0)
        l1_soma_y.append(np.mean(l1_act_diff))
        l1_soma_err.append(np.std(l1_act_diff) / np.sqrt(len(l1_act_diff)))
        
        # l2 soma (forward act)
        l2_act_diff = np.mean(act2_matrix[viol_idx], axis=0) - np.mean(act2_matrix[match_idx], axis=0)
        l2_soma_y.append(np.mean(l2_act_diff))
        l2_soma_err.append(np.std(l2_act_diff) / np.sqrt(len(l2_act_diff)))

    # draw the 2x2 grid
    fig_2b, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    x_vals = [1, 2, 3]
    
    axs[0, 0].errorbar(x_vals, l1_dend_y, yerr=l1_dend_err, fmt='-o', color='lightgreen', linewidth=2)
    axs[0, 0].set_title('l2/3-d (l1 dendrite (td feedback))')
    axs[0, 0].axhline(0, color='gray', linestyle='--')
    
    axs[0, 1].errorbar(x_vals, l2_dend_y, yerr=l2_dend_err, fmt='-o', color='forestgreen', linewidth=2)
    axs[0, 1].set_title('l5-d (l2 dendrite (td feedback))')
    axs[0, 1].axhline(0, color='gray', linestyle='--')
    
    axs[1, 0].errorbar(x_vals, l1_soma_y, yerr=l1_soma_err, fmt='-o', color='lightskyblue', linewidth=2)
    axs[1, 0].set_title('l2/3-s (l1 soma (activation))')
    axs[1, 0].axhline(0, color='gray', linestyle='--')
    
    axs[1, 1].errorbar(x_vals, l2_soma_y, yerr=l2_soma_err, fmt='-o', color='steelblue', linewidth=2)
    axs[1, 1].set_title('l5-s (l2 soma (activation))')
    axs[1, 1].axhline(0, color='gray', linestyle='--')
    
    for ax in axs.flat:
        ax.set_xticks([1, 2, 3])
        ax.set_ylabel('difference in signal (violation - match)')
        
    plt.tight_layout()
    plt.savefig('plots/6_fig2b_recreation.png')
    plt.close()

    # output the raw math at the end
    lime_idx = np.where((types == 1) & (np.arange(len(types)) >= start_trial))[0]
    red_idx = np.where((types == 2) & (np.arange(len(types)) >= start_trial))[0]
    
    if len(lime_idx) > 0 and len(red_idx) > 0:
        print("\n--- comparing red and lime math ---")
        
        d_lime = np.mean(np.array(delta_history)[lime_idx])
        d_red = np.mean(np.array(delta_history)[red_idx])
        print(f"avg err -> lime: {d_lime:.3f} | red: {d_red:.3f}")
        
        w3_lime = np.mean(np.abs(np.array(w3_history)[lime_idx]))
        w3_red = np.mean(np.abs(np.array(w3_history)[red_idx]))
        print(f"avg w3 -> lime: {w3_lime:.3f} | red: {w3_red:.3f}")
        
        l2_lime = np.mean(l2_matrix[lime_idx])
        l2_red = np.mean(l2_matrix[red_idx])
        print(f"avg l2 td -> lime: {l2_lime:.3f} | red: {l2_red:.3f}")
        
        relu_lime = np.mean(np.array(a2_relu_history)[lime_idx])
        relu_red = np.mean(np.array(a2_relu_history)[red_idx])
        print(f"avg l2 deriv -> lime: {relu_lime:.3f} | red: {relu_red:.3f}")
        
        gated_lime = np.mean(np.abs(np.array(a2_gated_history)[lime_idx]))
        gated_red = np.mean(np.abs(np.array(a2_gated_history)[red_idx]))
        print(f"avg gated td -> lime: {gated_lime:.4f} | red: {gated_red:.4f}")
        
        l1_lime = np.mean(l1_matrix[lime_idx])
        l1_red = np.mean(l1_matrix[red_idx])
        print(f"avg l1 td -> lime: {l1_lime:.4f} | red: {l1_red:.4f}")

if __name__ == "__main__":
    run_experiment()