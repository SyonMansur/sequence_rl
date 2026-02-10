import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5):
        super(QNetwork, self).__init__() 
        
        # standard linear layers. nothing fancy here.
        # removed the feedback alignment stuff so its just normal backprop now.
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        
        self.relu = nn.ReLU()
        
        # dict to save activations for plotting later
        self.activation_tensors = {}

    def forward(self, x):
        # layer 1 pass
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        
        # we need to save a1 to analyze the feedback signal arriving here.
        # using retain_grad because pytorch usually tosses intermediate grads to save memory.
        # we need them to see what the upper layers are "telling" layer 1 to do.
        if a1.requires_grad: a1.retain_grad()
        self.activation_tensors['layer1'] = a1
        
        # layer 2 pass
        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        
        # saving layer 2 as well so we can compare the feedback signal at different depths.
        # helps us see where the signal gets blocked if the neurons are silent.
        if a2.requires_grad: a2.retain_grad()
        self.activation_tensors['layer2'] = a2
        
        # output layer
        return self.fc3(a2)

class DQN_Agent:
    def __init__(self, input_dimensions=6, num_actions=5, alpha=0.01, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        # hidden dim is scaled up a bit. just following the config.
        self.model = QNetwork(input_dimensions, num_actions*4, num_actions)
        
        # standard adam optimizer. mse loss for regression since we predict q-values.
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) 
        self.criterion = nn.MSELoss() 

    def choose_action(self, state_tensor):
        # epsilon greedy policy.
        # sometimes explore random actions.
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        
        # otherwise exploit best known action.
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 
        
    def update(self, state_tensor, action, reward):
        self.model.train() 
        
        # get q values for all actions
        all_q_values = self.model(state_tensor) 
        
        # pick the value for the action we actually took
        current_q = all_q_values[0, action] 
        target_q = torch.tensor(reward, dtype=torch.float32) 
        
        # calc standard loss. this drives the weight updates.
        loss = self.criterion(current_q, target_q) 
        
        # NEW METRIC: raw error magnitude.
        # this is the "source" error |prediction - target|.
        # calculating this before gradients to see the raw signal magnitude.
        # detached so it doesn't mess with the graph.
        raw_error = torch.abs(current_q - target_q).detach().item()
        
        # run backprop
        self.optimizer.zero_grad() 
        loss.backward() 
        
        # extract gradients for analysis.
        # layer 1 grads show the instruction arriving at the bottom of the network.
        l1_grads = self.model.activation_tensors['layer1'].grad.detach().cpu().numpy().flatten()
        l1_feedback = np.abs(l1_grads)

        # layer 2 grads show the instruction arriving at the middle.
        l2_grads = self.model.activation_tensors['layer2'].grad.detach().cpu().numpy().flatten()
        l2_feedback = np.abs(l2_grads)

        # actually update the weights
        self.optimizer.step() 
        
        # returning all three metrics so main.py can choose what to plot.
        # 1. loss (training progress)
        # 2. l1 feedback (filtered instruction)
        # 3. l2 feedback (less filtered instruction)
        # 4. raw error (pure source signal)
        return loss.item(), l1_feedback, l2_feedback, raw_error