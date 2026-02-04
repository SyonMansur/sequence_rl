import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5):
        super(QNetwork, self).__init__() 
        # input to hidden
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        # hidden to hidden
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        # hidden to output layer
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        
        self.relu = nn.ReLU()

        # storage for live tensors so we can grab gradients
        self.activation_tensors = {}

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        
        # tell pytorch to keep the grad for this tensor
        # this is dLoss/dActivation (the signal from above)
        if a1.requires_grad:
            a1.retain_grad()
        self.activation_tensors['layer1'] = a1

        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        if a2.requires_grad:
            a2.retain_grad()
        self.activation_tensors['layer2'] = a2

        return self.fc3(a2) 
    

class DQN_Agent:
    def __init__(self, input_dimensions=6, num_actions=5, alpha=0.01, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.model = QNetwork(input_dimensions=input_dimensions, output_dimensions=num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) 
        self.criterion = nn.MSELoss() 

    def choose_action(self, state_tensor):
        # epsilon greedy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 
        
    def update(self, state_tensor, action, reward):
        self.model.train() 
        
        all_q_values = self.model(state_tensor) 
        current_q = all_q_values[0, action] 

        # target is just reward here
        target_q = torch.tensor(reward, dtype=torch.float32) 

        loss = self.criterion(current_q, target_q) 

        self.optimizer.zero_grad() 
        loss.backward() 

        # grab the gradient of the activation itself
        # taking abs to see magnitude of the error signal
        l1_grads = self.model.activation_tensors['layer1'].grad.detach().cpu().numpy().flatten()
        l1_feedback = np.abs(l1_grads)

        l2_grads = self.model.activation_tensors['layer2'].grad.detach().cpu().numpy().flatten()
        l2_feedback = np.abs(l2_grads)

        self.optimizer.step() 

        return loss.item(), l1_feedback, l2_feedback

