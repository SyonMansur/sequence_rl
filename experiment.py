import os
import random
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DQN_Agent, DTPAgent, PCAgent

# config for main parameters
CONFIG = {
    "number_of_actions": 16, 
    "steps": 100000, 
    "alpha": 0.001,
    "use_dtp": False,
    "use_pc": False
}

def get_stimulus_representation(angle_index, total_options):
    # convert the discrete angle index into a continuous sine and cosine representation
    # gives the network a structured continuous input
    angles = np.linspace(0, 2 * np.pi, total_options, endpoint=False)
    theta = angles[angle_index]
    return [np.sin(theta), np.cos(theta)]

def calculate_reward(prediction, target):
    # simple binary reward system for hits and misses
    return 1.0 if prediction == target else 0.0

def get_agent(mode, input_dimensions, actions, hidden_dimensions, alpha):
    # get correct network architecture based on the selected mode
    if mode == 'dtp': return DTPAgent(input_dimensions, actions, hidden_dimensions, alpha)
    elif mode == 'pc': return PCAgent(input_dimensions, actions, hidden_dimensions, alpha)
    else: return DQN_Agent(input_dimensions, actions, hidden_dimensions, alpha)

def run_simulation(agent, mode, steps, number_of_actions, full_logging=True):
    # set up sequence length and angle distribution for trials
    sequence_length = 3
    angles = np.linspace(0, 2 * np.pi, number_of_actions, endpoint=False) 
    
    # build a lookup table for the rule shift
    # maps the original angle to the shifted target angle
    shift_lookup_table = {} 
    for index in range(number_of_actions):
        target_theta = (angles[index] - (np.pi / 2)) % (2 * np.pi)
        shift_lookup_table[index] = np.argmin(np.abs(angles - target_theta)) 

    # track basic metrics and initialize the big logging dictionary
    # tracks all network internals over time for plotting
    accuracy_history = []
    logging_dictionary = {
        "loss_history": [], "trial_type_array": [], "is_violation_history": [],
        "raw_activation_layer_one_history": [], "raw_activation_layer_two_history": [], 
        "layer_one_feedback_history": [], "layer_two_feedback_history": [], 
        "delta_history": [], "weight_three_history": [], "weight_two_history": [], 
        "activation_two_relu_history": [], "activation_two_gated_history": []
    }

    # progress bar :D
    for step_index in tqdm(range(steps), desc=f"run {mode}", leave=False):
        stimulus_angle = random.randint(0, number_of_actions - 1)
        target_index = stimulus_angle        
        is_violation = False
    
        # target shift midway through the training run
        if step_index > (steps / 2):
            if random.random() < 0.06:
                target_index = shift_lookup_table[stimulus_angle]
                is_violation = True
                
        # construct the stimulus sequence by repeating the representation
        stimulus_sequence = []
        stimulus_representation = get_stimulus_representation(stimulus_angle, number_of_actions)
        for _ in range(sequence_length):
            stimulus_sequence.extend(stimulus_representation)
        
        state_tensor = torch.tensor(stimulus_sequence, dtype=torch.float32).unsqueeze(0)

        # grab the network prediction without tracking gradients yet
        with torch.no_grad():
            q_values = agent.model(state_tensor)
        
        # choose the action and calculate the resulting reward
        prediction = agent.choose_action(state_tensor)
        expected_value = q_values[0, prediction].item()
        reward = calculate_reward(prediction, target_index)
        accuracy_history.append(reward)

        # extract the pre activation tensors from the hidden layers
        pre_activation_layer_one = agent.model.activation_tensors['layer1'].clone().detach().cpu().numpy().flatten()
        pre_activation_layer_two = agent.model.activation_tensors['layer2'].clone().detach().cpu().numpy().flatten()
        
        action_tensor = torch.tensor([[prediction]]) 
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        # apply the learning updates based on the specific network mode
        if mode == 'dtp':
            agent.update_inverse(state_tensor)
            loss, layer_one_feedback, layer_two_feedback, raw_error, delta, weight_three_activation, activation_two_relu, activation_two_gated, weight_one_magnitude, weight_two_magnitude, weight_three_magnitude = agent.update_forward(state_tensor, action_tensor, reward_tensor)
        else:
            loss, layer_one_feedback, layer_two_feedback, raw_error, delta, weight_three_activation, activation_two_relu, activation_two_gated, weight_one_magnitude, weight_two_magnitude, weight_three_magnitude = agent.update(state_tensor, action_tensor, reward_tensor)

        # append everything to the logging dictionary if the flag is active
        # classify trial types based on expectation vs reality
        if full_logging:
            if expected_value >= 0.7 and reward == 1.0: trial_type = 0 
            elif expected_value < 0.3 and reward == 1.0: trial_type = 1 
            elif expected_value >= 0.7 and reward == 0.0: trial_type = 2 
            elif expected_value < 0.3 and reward == 0.0: trial_type = 3 
            else: trial_type = -1 
            
            logging_dictionary["is_violation_history"].append(is_violation)
            logging_dictionary["trial_type_array"].append(trial_type)
            logging_dictionary["raw_activation_layer_one_history"].append(pre_activation_layer_one)
            logging_dictionary["raw_activation_layer_two_history"].append(pre_activation_layer_two)
            logging_dictionary["loss_history"].append(loss)
            logging_dictionary["layer_one_feedback_history"].append(layer_one_feedback)
            logging_dictionary["layer_two_feedback_history"].append(layer_two_feedback)
            logging_dictionary["delta_history"].append(delta)
            logging_dictionary["weight_three_history"].append(weight_three_magnitude) 
            logging_dictionary["weight_two_history"].append(weight_two_magnitude)
            logging_dictionary["activation_two_relu_history"].append(activation_two_relu)
            logging_dictionary["activation_two_gated_history"].append(activation_two_gated)

    return accuracy_history, logging_dictionary

def plot_main_experiment(accuracy_history, logging_dictionary, steps):
    print("plt raw exp")
    start_trial = steps // 2 
    types_array = np.array(logging_dictionary["trial_type_array"])
    is_violation_array = np.array(logging_dictionary["is_violation_history"])
    
    # define labels and colors
    type_labels = ['exp reward', 'unexp reward', 'unexp lack', 'exp lack']
    colors = ['forestgreen', 'lime', 'red', 'gray']
    
    # pull tracking matrices and convert to numpy
    layer_one_matrix = np.array(logging_dictionary["layer_one_feedback_history"])
    layer_two_matrix = np.array(logging_dictionary["layer_two_feedback_history"])
    activation_one_matrix = np.array(logging_dictionary["raw_activation_layer_one_history"])
    activation_two_matrix = np.array(logging_dictionary["raw_activation_layer_two_history"])
    
    neuron_indexes = np.arange(layer_one_matrix.shape[1]) 
    bar_width = 0.2 

    # lists to hold calculated
    means_layer_one = []
    means_layer_two = []
    means_activation_one = []
    means_activation_two = []
    
    # get column wise means
    # filter out any trials from before the rule shift
    for trial_type_index in range(4): 
        matching_indices = np.where((types_array == trial_type_index) & (np.arange(len(types_array)) >= start_trial))[0]
        if len(matching_indices) > 0:
            means_layer_one.append(np.mean(layer_one_matrix[matching_indices], axis=0))
            means_layer_two.append(np.mean(layer_two_matrix[matching_indices], axis=0))
            means_activation_one.append(np.mean(np.abs(activation_one_matrix[matching_indices]), axis=0))
            means_activation_two.append(np.mean(np.abs(activation_two_matrix[matching_indices]), axis=0))
        else:
            zero_array_one = np.zeros(layer_one_matrix.shape[1])
            zero_array_two = np.zeros(layer_two_matrix.shape[1])
            means_layer_one.append(zero_array_one)
            means_layer_two.append(zero_array_one)
            means_activation_one.append(zero_array_one)
            means_activation_two.append(zero_array_two)

    # plot one: rolling loss and overall accuracy
    figure_one, (axis_one, axis_two) = plt.subplots(2, 1, figsize=(10, 8))
    rolling_loss = [np.mean(logging_dictionary["loss_history"][max(0, index-100):index+1]) for index in range(len(logging_dictionary["loss_history"]))]
    axis_one.plot(rolling_loss, color='darkred', linewidth=2)
    axis_one.set_title("loss")
    
    rolling_accuracy = [np.mean(accuracy_history[max(0, index-100):index+1]) for index in range(len(accuracy_history))]
    axis_two.plot(rolling_accuracy, color='blue')
    axis_two.axvline(x=start_trial, color='black', linestyle='--')
    axis_two.set_title("accuracy")
    plt.tight_layout()
    plt.savefig('plots/1_loss_and_accuracy.png')
    plt.close()

    # plot two: the top down signal hitting the first layer
    figure_two, axis_three = plt.subplots(figsize=(12, 6))
    for list_index, mean_values in enumerate(means_layer_one):
        axis_three.bar(neuron_indexes + (list_index - 1.5) * bar_width, mean_values, bar_width, label=type_labels[list_index], color=colors[list_index])
    axis_three.set_title("layer one td signal")
    axis_three.legend()
    plt.tight_layout()
    plt.savefig('plots/2_layer_1_topdown.png')
    plt.close()

    # plot three: top down signal hitting the second layer
    figure_three, axis_four = plt.subplots(figsize=(12, 6))
    for list_index, mean_values in enumerate(means_layer_two):
        axis_four.bar(neuron_indexes + (list_index - 1.5) * bar_width, mean_values, bar_width, label=type_labels[list_index], color=colors[list_index])
    axis_four.set_title("layer two td signal")
    axis_four.legend()
    plt.tight_layout()
    plt.savefig('plots/3_layer_2_topdown.png')
    plt.close()

    # plot four: raw activation values for layer one
    figure_four, axis_five = plt.subplots(figsize=(12, 6))
    for list_index, mean_values in enumerate(means_activation_one):
        axis_five.bar(neuron_indexes + (list_index - 1.5) * bar_width, mean_values, bar_width, label=type_labels[list_index], color=colors[list_index])
    axis_five.set_title("layer one activations")
    axis_five.legend()
    plt.tight_layout()
    plt.savefig('plots/4_layer_1_activations.png')
    plt.close()

    # plot five: raw activation values for layer two
    figure_five, axis_six = plt.subplots(figsize=(12, 6))
    for list_index, mean_values in enumerate(means_activation_two):
        axis_six.bar(neuron_indexes + (list_index - 1.5) * bar_width, mean_values, bar_width, label=type_labels[list_index], color=colors[list_index])
    axis_six.set_title("layer two activations")
    axis_six.legend()
    plt.tight_layout()
    plt.savefig('plots/5_layer_2_activations.png')
    plt.close()
    
    # ---------------------------------------------------------
    # figure 2b from gillan et al. paper
    # ---------------------------------------------------------
    
    
    halfway_point = steps // 2
    session_length = halfway_point // 3
    
    # chop the second half of training into three discrete sessions
    sessions = [
        (halfway_point, halfway_point + session_length),
        (halfway_point + session_length, halfway_point + 2 * session_length),
        (halfway_point + 2 * session_length, steps)
    ]
    
    # nested dictionary to hold the significance testing data
    storage = {
        "d1": {"y": [], "error": [], "p_value": [], "raw": []},
        "d2": {"y": [], "error": [], "p_value": [], "raw": []},
        "s1": {"y": [], "error": [], "p_value": [], "raw": []},
        "s2": {"y": [], "error": [], "p_value": [], "raw": []}
    }
    
    # iterate through each session window and run stats
    for start_index, end_index in sessions:
        window_indices = np.arange(start_index, end_index)
        violation_indices = window_indices[is_violation_array[start_index:end_index] == True]
        match_indices = window_indices[is_violation_array[start_index:end_index] == False]
        
        # safely skip if there wasn't enough data in this specific window
        if len(violation_indices) == 0 or len(match_indices) == 0:
            continue
            
        def process_layer(matrix, key):
            # compute the difference in means between violations and standard matches
            difference = np.mean(matrix[violation_indices], axis=0) - np.mean(matrix[match_indices], axis=0)
            storage[key]["y"].append(np.mean(difference))
            storage[key]["error"].append(stats.sem(difference))
            _, p_value = stats.ttest_1samp(difference, 0.0)
            storage[key]["p_value"].append(p_value if not np.isnan(p_value) else 1.0)
            storage[key]["raw"].append(difference)

        process_layer(layer_one_matrix, "d1")
        process_layer(layer_two_matrix, "d2")
        process_layer(activation_one_matrix, "s1")
        process_layer(activation_two_matrix, "s2")

    def get_significance_stars(p_value):
        # helper to format p values
        if p_value < 0.001: return '***'
        elif p_value < 0.01: return '**'
        elif p_value < 0.05: return '*'
        return ''

    def annotate_points(axis, x_coordinates, y_coordinates, errors, p_values, color):
        # apply the significance stars directly to the plotted points
        for index, position in enumerate(x_coordinates):
            sig_string = get_significance_stars(p_values[index])
            if sig_string:
                offset = (max(y_coordinates) - min(y_coordinates)) * 0.15 if max(y_coordinates) != min(y_coordinates) else 0.01
                axis.text(position, y_coordinates[index] + errors[index] + offset, sig_string, ha='center', color=color, fontweight='bold')

    def draw_brackets(axis, x_coordinates, y_coordinates, errors, raw_data, color):
        # connecting brackets
        if len(raw_data) < 3: return
        max_y = max([value + error for value, error in zip(y_coordinates, errors)])
        pairs = [(0, 1), (1, 2), (0, 2)] 
        for index, (first_index, second_index) in enumerate(pairs):
            _, p_value = stats.ttest_rel(raw_data[first_index], raw_data[second_index])
            sig_string = get_significance_stars(p_value)
            if sig_string:
                height = max_y * 0.05
                level_y = max_y + (height * (index + 1) * 2.5)
                axis.plot([x_coordinates[first_index], x_coordinates[first_index], x_coordinates[second_index], x_coordinates[second_index]], [level_y, level_y + height, level_y + height, level_y], lw=1.5, c=color)
                axis.text((x_coordinates[first_index] + x_coordinates[second_index]) * 0.5, level_y + height, sig_string, ha='center', va='bottom', color=color, fontweight='bold')

    # 2x2 subplot
    figure_six, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    x_values = [1, 2, 3] 
    titles = ['l2/3-d (l1 dendrite proxy)', 'l5-d (l2 dendrite proxy)', 
              'l2/3-s (l1 soma proxy)', 'l5-s (l2 soma proxy)']
    keys = ["d1", "d2", "s1", "s2"]
    line_colors = ['lightgreen', 'forestgreen', 'lightskyblue', 'steelblue']

    # plot all four
    for index, axis in enumerate(axes.flat):
        current_key = keys[index]
        current_color = line_colors[index]
        if not storage[current_key]["y"]: continue
        
        axis.errorbar(x_values, storage[current_key]["y"], yerr=storage[current_key]["error"], fmt='-o', color=current_color, lw=2)
        axis.set_title(titles[index])
        axis.axhline(0, color='gray', ls='--')
        annotate_points(axis, x_values, storage[current_key]["y"], storage[current_key]["error"], storage[current_key]["p_value"], current_color)
        draw_brackets(axis, x_values, storage[current_key]["y"], storage[current_key]["error"], storage[current_key]["raw"], current_color)
        
        axis.set_xticks(x_values)
        axis.set_xticklabels(['s1', 's2', 's3'])
        axis.set_ylabel('diff in signal (viol - match)')
        
        ylim = axis.get_ylim()
        axis.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.4)

    plt.tight_layout()
    plt.savefig('plots/6_fig2b_recreation.png')
    plt.close()
    
    # track weight magnitude growth against the top down error signal
    session_weight_two_norms = []
    session_weight_three_norms = []
    
    # split and process by session window
    for start_index, end_index in sessions:
        window_indices = np.arange(start_index, end_index)
        if len(logging_dictionary["weight_two_history"]) > window_indices[-1]:
            # calculate magnitude using absolute values
            session_weight_two_norms.append(np.mean(np.abs(np.array(logging_dictionary["weight_two_history"])[window_indices])))
            session_weight_three_norms.append(np.mean(np.abs(np.array(logging_dictionary["weight_three_history"])[window_indices])))
            
    # plot comparisons
    if session_weight_three_norms and storage["d2"]["y"]:
        figure_seven, (axis_signal_one, axis_signal_two) = plt.subplots(1, 2, figsize=(14, 6))
        
        # layer one weight logic
        axis_weight_two = axis_signal_one.twinx()
        line_one_a = axis_signal_one.plot(x_values, storage["d1"]["y"], color='steelblue', marker='o', lw=2, label='l1 td error')
        line_one_b = axis_weight_two.plot(x_values, session_weight_two_norms, color='darkred', marker='x', ls='--', lw=2, label='w2 norm')
        
        axis_signal_one.set_xticks(x_values)
        axis_signal_one.set_xticklabels(['s1', 's2', 's3'])
        axis_signal_one.set_ylabel('td signal diff')
        axis_weight_two.set_ylabel('w2 norm')
        axis_signal_one.set_title('bp w2 growth vs l1 error')
        
        lines_one = line_one_a + line_one_b
        labels_one = [line.get_label() for line in lines_one]
        axis_signal_one.legend(lines_one, labels_one, loc='upper left')

        # layer two weight logic
        axis_weight_three = axis_signal_two.twinx()
        line_two_a = axis_signal_two.plot(x_values, storage["d2"]["y"], color='steelblue', marker='o', lw=2, label='l2 td error')
        line_two_b = axis_weight_three.plot(x_values, session_weight_three_norms, color='darkred', marker='x', ls='--', lw=2, label='w3 norm')
        
        axis_signal_two.set_xticks(x_values)
        axis_signal_two.set_xticklabels(['s1', 's2', 's3'])
        axis_signal_two.set_ylabel('td signal diff')
        axis_weight_three.set_ylabel('w3 norm')
        axis_signal_two.set_title('bp w3 growth vs l2 error')
        
        lines_two = line_two_a + line_two_b
        labels_two = [line.get_label() for line in lines_two]
        axis_signal_two.legend(lines_two, labels_two, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('plots/7_bp_weight_proof.png')
        plt.close()
        print("svd bp plts")

    # check relu gating
    session_layer_one_zero_fraction = []
    session_layer_two_zero_fraction = []
    
    # dead neuron checks
    for start_index, end_index in sessions:
        window_indices = np.arange(start_index, end_index)
        violation_indices = window_indices[is_violation_array[start_index:end_index] == True]
        if len(violation_indices) > 0:
            layer_one_violation_activations = activation_one_matrix[violation_indices]
            layer_two_violation_activations = activation_two_matrix[violation_indices]
            session_layer_one_zero_fraction.append(np.mean(layer_one_violation_activations <= 1e-6))
            session_layer_two_zero_fraction.append(np.mean(layer_two_violation_activations <= 1e-6))
        else:
            session_layer_one_zero_fraction.append(0)
            session_layer_two_zero_fraction.append(0)
            
    if session_layer_one_zero_fraction and storage["d1"]["y"]:
        figure_eight, (axis_signal_one, axis_signal_two) = plt.subplots(1, 2, figsize=(14, 6))
        
        # map layer one relu mechanics
        axis_relu_one = axis_signal_one.twinx()
        line_one_a = axis_signal_one.plot(x_values, storage["d1"]["y"], color='steelblue', marker='o', lw=2, label='l1 td error')
        line_one_b = axis_relu_one.plot(x_values, session_layer_one_zero_fraction, color='darkorange', marker='x', ls='--', lw=2, label='l1 relu near zero frac')
        
        axis_signal_one.set_xticks(x_values)
        axis_signal_one.set_xticklabels(['s1', 's2', 's3'])
        axis_signal_one.set_ylabel('td signal diff')
        axis_relu_one.set_ylabel('frac relu near zero')
        axis_signal_one.set_title('bp l1 relu gating vs error signal')
        
        lines_one = line_one_a + line_one_b
        labels_one = [line.get_label() for line in lines_one]
        axis_signal_one.legend(lines_one, labels_one, loc='center left')

        # map layer two relu mechanics
        axis_relu_two = axis_signal_two.twinx()
        line_two_a = axis_signal_two.plot(x_values, storage["d2"]["y"], color='steelblue', marker='o', lw=2, label='l2 td error')
        line_two_b = axis_relu_two.plot(x_values, session_layer_two_zero_fraction, color='darkorange', marker='x', ls='--', lw=2, label='l2 relu near zero frac')
        
        axis_signal_two.set_xticks(x_values)
        axis_signal_two.set_xticklabels(['s1', 's2', 's3'])
        axis_signal_two.set_ylabel('td signal diff')
        axis_relu_two.set_ylabel('frac relu near zero')
        axis_signal_two.set_title('bp l2 relu gating vs error signal')
        
        lines_two = line_two_a + line_two_b
        labels_two = [line.get_label() for line in lines_two]
        axis_signal_two.legend(lines_two, labels_two, loc='center left')
        
        plt.tight_layout()
        plt.savefig('plots/8_bp_relu_gating.png')
        plt.close()
        print("svd relu plts")

    # print out final math
    lime_indices = np.where((types_array == 1) & (np.arange(len(types_array)) >= start_trial))[0]
    red_indices = np.where((types_array == 2) & (np.arange(len(types_array)) >= start_trial))[0]
    
    if len(lime_indices) > 0 and len(red_indices) > 0:
        print("cmp red lime")
        
        delta_lime = np.mean(np.array(logging_dictionary["delta_history"])[lime_indices])
        delta_red = np.mean(np.array(logging_dictionary["delta_history"])[red_indices])
        print(f"err lime {delta_lime:.3f} red {delta_red:.3f}")
        
        weight_three_lime = np.mean(np.abs(np.array(logging_dictionary["weight_three_history"])[lime_indices]))
        weight_three_red = np.mean(np.abs(np.array(logging_dictionary["weight_three_history"])[red_indices]))
        print(f"w three lime {weight_three_lime:.3f} red {weight_three_red:.3f}")
        
        layer_two_lime = np.mean(layer_two_matrix[lime_indices])
        layer_two_red = np.mean(layer_two_matrix[red_indices])
        print(f"l two td lime {layer_two_lime:.3f} red {layer_two_red:.3f}")
        
        relu_lime = np.mean(np.array(logging_dictionary["activation_two_relu_history"])[lime_indices])
        relu_red = np.mean(np.array(logging_dictionary["activation_two_relu_history"])[red_indices])
        print(f"l two deriv lime {relu_lime:.3f} red {relu_red:.3f}")
        
        gated_lime = np.mean(np.abs(np.array(logging_dictionary["activation_two_gated_history"])[lime_indices]))
        gated_red = np.mean(np.abs(np.array(logging_dictionary["activation_two_gated_history"])[red_indices]))
        print(f"gtd td lime {gated_lime:.4f} red {gated_red:.4f}")
        
        layer_one_lime = np.mean(layer_one_matrix[lime_indices])
        layer_one_red = np.mean(layer_one_matrix[red_indices])
        print(f"l one td lime {layer_one_lime:.4f} red {layer_one_red:.4f}")

def run_experiment():
    print("run exp")
    
    # default bp
    mode = 'bp'
    if CONFIG["use_dtp"]: mode = 'dtp'
    elif CONFIG["use_pc"]: mode = 'pc'
        
    agent = get_agent(mode, 6, CONFIG["number_of_actions"], 64, CONFIG["alpha"])
    accuracy_history, logging_dictionary = run_simulation(agent, mode, CONFIG["steps"], CONFIG["number_of_actions"], full_logging=True)
    plot_main_experiment(accuracy_history, logging_dictionary, CONFIG["steps"])

def run_benchmarks():
    print("strt bench")
    if not os.path.exists('plots'): os.makedirs('plots')
    
    # testing parameters
    trials_test = [5000, 25000, 50000, 100000]
    hidden_test = [16, 32, 64]
    actions_test = [8, 16, 32]
    modes = ['bp', 'dtp', 'pc']
    
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # accuracy plotting
    for mode in modes:
        results = []
        for trials in trials_test:
            agent = get_agent(mode, 6, 16, 64, 0.001)
            accuracy, _ = run_simulation(agent, mode, trials, 16, full_logging=False)
            results.append(np.mean(accuracy[-1000:]))
        axes[0].plot(trials_test, results, marker='o', label=mode)
    axes[0].set_title('acc vs total trials')
    axes[0].set_xlabel('trials')
    axes[0].set_ylabel('final acc')
    axes[0].legend()
    
    # node count plotting
    for mode in modes:
        results = []
        for hidden in hidden_test:
            agent = get_agent(mode, 6, 16, hidden, 0.001)
            accuracy, _ = run_simulation(agent, mode, 100000, 16, full_logging=False)
            results.append(np.mean(accuracy[-1000:]))
        axes[1].plot(hidden_test, results, marker='o', label=mode)
    axes[1].set_title('acc vs hidden nodes')
    axes[1].set_xlabel('hidden dims')
    
    # output action space plotting
    for mode in modes:
        results = []
        for actions in actions_test:
            agent = get_agent(mode, 6, actions, 64, 0.001)
            accuracy, _ = run_simulation(agent, mode, 100000, actions, full_logging=False)
            results.append(np.mean(accuracy[-1000:]))
        axes[2].plot(actions_test, results, marker='o', label=mode)
    axes[2].set_title('acc vs num actions')
    axes[2].set_xlabel('actions')
    
    plt.tight_layout()
    plt.savefig('plots/9_benchmarks.png')
    plt.close()
    print("svd bench plts")

if __name__ == "__main__":
    run_experiment()
    run_benchmarks()