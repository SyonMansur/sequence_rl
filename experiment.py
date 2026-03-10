import os
import random
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DQN_Agent, DTPAgent, PCAgent

# config
CONFIG = {
    "num_actions": 16, 
    "steps": 100000, 
    "alpha": 0.001,
    "use_dtp": False,
    "use_pc": False
}

def get_stimulus_representation(angle_index, total_options):
    # angle to sin and cos
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target):
    # binary hit or miss
    return 1.0 if prediction == target else 0.0

def get_agent(mode, input_dims, actions, hidden_dims, alpha):
    # init correct network based on mode
    if mode == 'dtp': return DTPAgent(input_dims, actions, hidden_dims, alpha)
    elif mode == 'pc': return PCAgent(input_dims, actions, hidden_dims, alpha)
    else: return DQN_Agent(input_dims, actions, hidden_dims, alpha)

# universal loop for all modes
def run_simulation(agent, mode, steps, num_actions, full_logging=True):
    sequence_length = 3
    angles = np.linspace(0, 2*np.pi, num_actions, endpoint=False) 
    
    # setup logic for the switch
    shift_lookup_table = {} 
    for xx in range(num_actions):
        target_theta = (angles[xx] - (np.pi/2)) % (2*np.pi)
        shift_lookup_table[xx] = np.argmin(np.abs(angles - target_theta)) 

    # basic tracking
    acc_hist = []
    log_dict = {
        "loss_history": [], "trial_type_array": [], "is_viol_history": [],
        "raw_act_l1_history": [], "raw_act_l2_history": [], "l1_feedback_history": [],
        "l2_feedback_history": [], "delta_history": [], "w3_history": [], 
        "w2_history": [], "a2_relu_history": [], "a2_gated_history": []
    }

    # wrap loop with progress bar
    for i in tqdm(range(steps), desc=f"run {mode}", leave=False):
        stimulus_angle = random.randint(0, num_actions - 1)
        target_index = stimulus_angle        
        is_viol = False
    
        # hit them with the shift halfway thru
        if i > (steps / 2):
            if random.random() < 0.06:
                target_index = shift_lookup_table[stimulus_angle]
                is_viol = True
                
        # build out the input
        stim_seq = []
        stim_rep = get_stimulus_representation(stimulus_angle, num_actions)
        for _ in range(sequence_length):
            stim_seq.extend(stim_rep)
        
        state_tensor = torch.tensor(stim_seq, dtype=torch.float32).unsqueeze(0)

        # network guess
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        prediction = agent.choose_action(state_tensor)
        expected_value = q_values[0, prediction].item()
        reward = calculate_reward(prediction, target_index)
        acc_hist.append(reward)

        # grab pre acts
        pre_act_l1 = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        pre_act_l2 = agent.model.activation_tensors['layer2'].clone().detach().cpu().numpy().flatten()
        
        a_tensor = torch.tensor([[prediction]]) 
        r_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        # fire updates
        if mode == 'dtp':
            agent.update_inverse(state_tensor)
            loss, l1_fb, l2_fb, raw_err, delta, w3_act, a2_relu, a2_gtd, w1_m, w2_m, w3_m = agent.update_forward(state_tensor, a_tensor, r_tensor)
        else:
            loss, l1_fb, l2_fb, raw_err, delta, w3_act, a2_relu, a2_gtd, w1_m, w2_m, w3_m = agent.update(state_tensor, a_tensor, r_tensor)

        # log everything if flag is true
        if full_logging:
            if expected_value >= 0.7 and reward == 1.0: trial_type = 0 
            elif expected_value < 0.3 and reward == 1.0: trial_type = 1 
            elif expected_value >= 0.7 and reward == 0.0: trial_type = 2 
            elif expected_value < 0.3 and reward == 0.0: trial_type = 3 
            else: trial_type = -1 
            
            log_dict["is_viol_history"].append(is_viol)
            log_dict["trial_type_array"].append(trial_type)
            log_dict["raw_act_l1_history"].append(pre_act_l1)
            log_dict["raw_act_l2_history"].append(pre_act_l2)
            log_dict["loss_history"].append(loss)
            log_dict["l1_feedback_history"].append(l1_fb)
            log_dict["l2_feedback_history"].append(l2_fb)
            log_dict["delta_history"].append(delta)
            log_dict["w3_history"].append(w3_m) 
            log_dict["w2_history"].append(w2_m)
            log_dict["a2_relu_history"].append(a2_relu)
            log_dict["a2_gated_history"].append(a2_gtd)

    return acc_hist, log_dict

def plot_main_experiment(acc_hist, log_dict, steps):
    print("plt raw exp data")
    start_trial = steps // 2 
    types = np.array(log_dict["trial_type_array"])
    is_viol_arr = np.array(log_dict["is_viol_history"])
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    
    l1_matrix = np.array(log_dict["l1_feedback_history"])
    l2_matrix = np.array(log_dict["l2_feedback_history"])
    act1_matrix = np.array(log_dict["raw_act_l1_history"])
    act2_matrix = np.array(log_dict["raw_act_l2_history"])
    
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

    # plot one loss and acc
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    rolling_loss = [np.mean(log_dict["loss_history"][max(0, j-100):j+1]) for j in range(len(log_dict["loss_history"]))]
    ax1.plot(rolling_loss, color='darkred', linewidth=2)
    ax1.set_title("loss")
    
    rolling_acc = [np.mean(acc_hist[max(0, j-100):j+1]) for j in range(len(acc_hist))]
    ax2.plot(rolling_acc, color='blue')
    ax2.axvline(x=start_trial, color='black', linestyle='--')
    ax2.set_title("accuracy")
    plt.tight_layout()
    plt.savefig('plots/1_loss_and_accuracy.png')
    plt.close()

    # plot two layer one topdown
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_l1):
        ax3.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax3.set_title("layer one td signal")
    ax3.legend()
    plt.tight_layout()
    plt.savefig('plots/2_layer_1_topdown.png')
    plt.close()

    # plot three layer two topdown
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_l2):
        ax4.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax4.set_title("layer two td signal")
    ax4.legend()
    plt.tight_layout()
    plt.savefig('plots/3_layer_2_topdown.png')
    plt.close()

    # plot four layer one acts
    fig4, ax5 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_act1):
        ax5.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax5.set_title("layer one activations")
    ax5.legend()
    plt.tight_layout()
    plt.savefig('plots/4_layer_1_activations.png')
    plt.close()

    # plot five layer two acts
    fig5, ax6 = plt.subplots(figsize=(12, 6))
    for idx, mean_vals in enumerate(means_act2):
        ax6.bar(neuron_indexes + (idx-1.5)*width, mean_vals, width, label=type_labels[idx], color=colors[idx])
    ax6.set_title("layer two activations")
    ax6.legend()
    plt.tight_layout()
    plt.savefig('plots/5_layer_2_activations.png')
    plt.close()
    
    # ---------------------------------------------------------
    # figure two b recreation with significance
    # ---------------------------------------------------------
    halfway = steps // 2
    sess_len = halfway // 3
    sessions = [
        (halfway, halfway + sess_len),
        (halfway + sess_len, halfway + 2 * sess_len),
        (halfway + 2 * sess_len, steps)
    ]
    
    # storage arrays
    storage = {
        "d1": {"y": [], "err": [], "p": [], "raw": []},
        "d2": {"y": [], "err": [], "p": [], "raw": []},
        "s1": {"y": [], "err": [], "p": [], "raw": []},
        "s2": {"y": [], "err": [], "p": [], "raw": []}
    }
    
    for start, end in sessions:
        window_idx = np.arange(start, end)
        viol_idx = window_idx[is_viol_arr[start:end] == True]
        match_idx = window_idx[is_viol_arr[start:end] == False]
        
        # skip if missing data
        if len(viol_idx) == 0 or len(match_idx) == 0:
            continue
            
        def process_layer(matrix, key):
            diff = np.mean(matrix[viol_idx], axis=0) - np.mean(matrix[match_idx], axis=0)
            storage[key]["y"].append(np.mean(diff))
            storage[key]["err"].append(stats.sem(diff))
            _, p = stats.ttest_1samp(diff, 0.0)
            storage[key]["p"].append(p if not np.isnan(p) else 1.0)
            storage[key]["raw"].append(diff)

        process_layer(l1_matrix, "d1")
        process_layer(l2_matrix, "d2")
        process_layer(act1_matrix, "s1")
        process_layer(act2_matrix, "s2")

    def get_sig_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        return ''

    def annotate_points(ax, x, y, err, p_vals, color):
        for i, pos in enumerate(x):
            sig = get_sig_stars(p_vals[i])
            if sig:
                offset = (max(y) - min(y)) * 0.15 if max(y) != min(y) else 0.01
                ax.text(pos, y[i] + err[i] + offset, sig, ha='center', color=color, fontweight='bold')

    def draw_brackets(ax, x, y, err, raw_data, color):
        # setup pairs for three sessions
        if len(raw_data) < 3: return
        y_max = max([val + e for val, e in zip(y, err)])
        pairs = [(0, 1), (1, 2), (0, 2)] 
        for i, (idx1, idx2) in enumerate(pairs):
            _, p = stats.ttest_rel(raw_data[idx1], raw_data[idx2])
            sig = get_sig_stars(p)
            if sig:
                h = y_max * 0.05
                level_y = y_max + (h * (i+1) * 2.5)
                ax.plot([x[idx1], x[idx1], x[idx2], x[idx2]], [level_y, level_y+h, level_y+h, level_y], lw=1.5, c=color)
                ax.text((x[idx1]+x[idx2])*0.5, level_y+h, sig, ha='center', va='bottom', color=color, fontweight='bold')

    fig_2b, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    x_vals = [1, 2, 3] 
    titles = ['l2/3-d (l1 dendrite proxy)', 'l5-d (l2 dendrite proxy)', 
              'l2/3-s (l1 soma proxy)', 'l5-s (l2 soma proxy)']
    keys = ["d1", "d2", "s1", "s2"]
    line_colors = ['lightgreen', 'forestgreen', 'lightskyblue', 'steelblue']

    for i, ax in enumerate(axs.flat):
        k = keys[i]
        c = line_colors[i]
        if not storage[k]["y"]: continue
        
        ax.errorbar(x_vals, storage[k]["y"], yerr=storage[k]["err"], fmt='-o', color=c, lw=2)
        ax.set_title(titles[i])
        ax.axhline(0, color='gray', ls='--')
        annotate_points(ax, x_vals, storage[k]["y"], storage[k]["err"], storage[k]["p"], c)
        draw_brackets(ax, x_vals, storage[k]["y"], storage[k]["err"], storage[k]["raw"], c)
        
        ax.set_xticks(x_vals)
        ax.set_xticklabels(['s1', 's2', 's3'])
        ax.set_ylabel('diff in signal (viol - match)')
        
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + (ylim[1]-ylim[0])*0.4)

    plt.tight_layout()
    plt.savefig('plots/6_fig2b_recreation.png')
    plt.close()
    
    # check weight norms against signal for both layers
    s_w2_norms = []
    s_w3_norms = []
    
    for start, end in sessions:
        window_idx = np.arange(start, end)
        # fallback for missing arrays
        if len(log_dict["w2_history"]) > window_idx[-1]:
            # calc true mag with abs
            s_w2_norms.append(np.mean(np.abs(np.array(log_dict["w2_history"])[window_idx])))
            s_w3_norms.append(np.mean(np.abs(np.array(log_dict["w3_history"])[window_idx])))
            
    if s_w3_norms and storage["d2"]["y"]:
        fig7, (ax_sig1, ax_sig2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # l one layer plot
        ax_w2 = ax_sig1.twinx()
        
        # plot err diff
        l1_a = ax_sig1.plot(x_vals, storage["d1"]["y"], color='steelblue', marker='o', lw=2, label='l1 td error')
        
        # plot wt growth
        l1_b = ax_w2.plot(x_vals, s_w2_norms, color='darkred', marker='x', ls='--', lw=2, label='w2 norm')
        
        ax_sig1.set_xticks(x_vals)
        ax_sig1.set_xticklabels(['s1', 's2', 's3'])
        ax_sig1.set_ylabel('td signal diff')
        ax_w2.set_ylabel('w2 norm')
        ax_sig1.set_title('bp w2 growth vs l1 error')
        
        lines1 = l1_a + l1_b
        labels1 = [l.get_label() for l in lines1]
        ax_sig1.legend(lines1, labels1, loc='upper left')

        # l two layer plot
        ax_w3 = ax_sig2.twinx()
        
        # plot err diff
        l2_a = ax_sig2.plot(x_vals, storage["d2"]["y"], color='steelblue', marker='o', lw=2, label='l2 td error')
        
        # plot wt growth
        l2_b = ax_w3.plot(x_vals, s_w3_norms, color='darkred', marker='x', ls='--', lw=2, label='w3 norm')
        
        ax_sig2.set_xticks(x_vals)
        ax_sig2.set_xticklabels(['s1', 's2', 's3'])
        ax_sig2.set_ylabel('td signal diff')
        ax_w3.set_ylabel('w3 norm')
        ax_sig2.set_title('bp w3 growth vs l2 error')
        
        lines2 = l2_a + l2_b
        labels2 = [l.get_label() for l in lines2]
        ax_sig2.legend(lines2, labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('plots/7_bp_weight_proof.png')
        plt.close()
        print("svd bp plt both")

    # check relu gating in both layers
    s_l1_zero_fraction = []
    s_l2_zero_fraction = []
    
    for start, end in sessions:
        window_idx = np.arange(start, end)
        viol_idx = window_idx[is_viol_arr[start:end] == True]
        if len(viol_idx) > 0:
            # act matrices shape (trials, neurons)
            l1_viol_acts = act1_matrix[viol_idx]
            l2_viol_acts = act2_matrix[viol_idx]
            s_l1_zero_fraction.append(np.mean(l1_viol_acts <= 1e-6))
            s_l2_zero_fraction.append(np.mean(l2_viol_acts <= 1e-6))
        else:
            s_l1_zero_fraction.append(0)
            s_l2_zero_fraction.append(0)
            
    if s_l1_zero_fraction and storage["d1"]["y"]:
        fig8, (ax_sig1, ax_sig2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # l1 plot
        ax_relu1 = ax_sig1.twinx()
        
        # plot err diff
        l1_a = ax_sig1.plot(x_vals, storage["d1"]["y"], color='steelblue', marker='o', lw=2, label='l1 td error')
        
        # plot relu near zero frac
        l1_b = ax_relu1.plot(x_vals, s_l1_zero_fraction, color='darkorange', marker='x', ls='--', lw=2, label='l1 relu near zero frac')
        
        ax_sig1.set_xticks(x_vals)
        ax_sig1.set_xticklabels(['s1', 's2', 's3'])
        ax_sig1.set_ylabel('td signal diff')
        ax_relu1.set_ylabel('frac relu near zero')
        ax_sig1.set_title('bp l1 relu gating vs error signal')
        
        lines1 = l1_a + l1_b
        labels1 = [l.get_label() for l in lines1]
        ax_sig1.legend(lines1, labels1, loc='center left')

        # l2 plot
        ax_relu2 = ax_sig2.twinx()
        
        # plot err diff
        l2_a = ax_sig2.plot(x_vals, storage["d2"]["y"], color='steelblue', marker='o', lw=2, label='l2 td error')
        
        # plot relu near zero frac
        l2_b = ax_relu2.plot(x_vals, s_l2_zero_fraction, color='darkorange', marker='x', ls='--', lw=2, label='l2 relu near zero frac')
        
        ax_sig2.set_xticks(x_vals)
        ax_sig2.set_xticklabels(['s1', 's2', 's3'])
        ax_sig2.set_ylabel('td signal diff')
        ax_relu2.set_ylabel('frac relu near zero')
        ax_sig2.set_title('bp l2 relu gating vs error signal')
        
        lines2 = l2_a + l2_b
        labels2 = [l.get_label() for l in lines2]
        ax_sig2.legend(lines2, labels2, loc='center left')
        
        plt.tight_layout()
        plt.savefig('plots/8_bp_relu_gating.png')
        plt.close()
        print("svd relu plt both")

    # output the raw math at the end
    lime_idx = np.where((types == 1) & (np.arange(len(types)) >= start_trial))[0]
    red_idx = np.where((types == 2) & (np.arange(len(types)) >= start_trial))[0]
    
    if len(lime_idx) > 0 and len(red_idx) > 0:
        print("cmp red lime math")
        
        d_lime = np.mean(np.array(log_dict["delta_history"])[lime_idx])
        d_red = np.mean(np.array(log_dict["delta_history"])[red_idx])
        print(f"err lime {d_lime:.3f} red {d_red:.3f}")
        
        w3_lime = np.mean(np.abs(np.array(log_dict["w3_history"])[lime_idx]))
        w3_red = np.mean(np.abs(np.array(log_dict["w3_history"])[red_idx]))
        print(f"w three lime {w3_lime:.3f} red {w3_red:.3f}")
        
        l2_lime = np.mean(l2_matrix[lime_idx])
        l2_red = np.mean(l2_matrix[red_idx])
        print(f"l two td lime {l2_lime:.3f} red {l2_red:.3f}")
        
        relu_lime = np.mean(np.array(log_dict["a2_relu_history"])[lime_idx])
        relu_red = np.mean(np.array(log_dict["a2_relu_history"])[red_idx])
        print(f"l two deriv lime {relu_lime:.3f} red {relu_red:.3f}")
        
        gated_lime = np.mean(np.abs(np.array(log_dict["a2_gated_history"])[lime_idx]))
        gated_red = np.mean(np.abs(np.array(log_dict["a2_gated_history"])[red_idx]))
        print(f"gtd td lime {gated_lime:.4f} red {gated_red:.4f}")
        
        l1_lime = np.mean(l1_matrix[lime_idx])
        l1_red = np.mean(l1_matrix[red_idx])
        print(f"l one td lime {l1_lime:.4f} red {l1_red:.4f}")

def run_experiment():
    print("run mn exp")
    if not os.path.exists('plots'): os.makedirs('plots')
    
    mode = 'bp'
    if CONFIG["use_dtp"]: mode = 'dtp'
    elif CONFIG["use_pc"]: mode = 'pc'
        
    agent = get_agent(mode, 6, CONFIG["num_actions"], 64, CONFIG["alpha"])
    acc_hist, log_dict = run_simulation(agent, mode, CONFIG["steps"], CONFIG["num_actions"], full_logging=True)
    plot_main_experiment(acc_hist, log_dict, CONFIG["steps"])

# standalone benchmark 
def run_benchmarks():
    print("strt bench")
    if not os.path.exists('plots'): os.makedirs('plots')
    
    trials_test = [5000, 25000, 50000, 100000]
    hidden_test = [16, 32, 64]
    actions_test = [8, 16, 32]
    modes = ['bp', 'dtp', 'pc']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # plot one total trials
    for mode in modes:
        res = []
        for t in trials_test:
            agent = get_agent(mode, 6, 16, 64, 0.001)
            acc, _ = run_simulation(agent, mode, t, 16, full_logging=False)
            res.append(np.mean(acc[-1000:]))
        axes[0].plot(trials_test, res, marker='o', label=mode)
    axes[0].set_title('acc vs total trials')
    axes[0].set_xlabel('trials')
    axes[0].set_ylabel('final acc')
    axes[0].legend()
    
    # plot two hidden nodes
    for mode in modes:
        res = []
        for h in hidden_test:
            agent = get_agent(mode, 6, 16, h, 0.001)
            acc, _ = run_simulation(agent, mode, 100000, 16, full_logging=False)
            res.append(np.mean(acc[-1000:]))
        axes[1].plot(hidden_test, res, marker='o', label=mode)
    axes[1].set_title('acc vs hidden nodes')
    axes[1].set_xlabel('hidden dims')
    
    # plot three actions
    for mode in modes:
        res = []
        for a in actions_test:
            agent = get_agent(mode, 6, a, 64, 0.001)
            acc, _ = run_simulation(agent, mode, 100000, a, full_logging=False)
            res.append(np.mean(acc[-1000:]))
        axes[2].plot(actions_test, res, marker='o', label=mode)
    axes[2].set_title('acc vs num actions')
    axes[2].set_xlabel('actions')
    
    plt.tight_layout()
    plt.savefig('plots/9_benchmarks.png')
    plt.close()
    print("svd bench plts")

if __name__ == "__main__":
    run_experiment()
    run_benchmarks()