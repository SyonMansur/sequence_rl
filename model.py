import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dimensions = 6, hidden_dimensions = 64, output_dimensions = 5):
        super(QNetwork, self).__init__() # standard for inheritance of the parent class
        # input to hidden
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        # hidden to hidden
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        # hidden to output layer
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        # 3 layers allows for nonlinear relationships
        self.relu = nn.ReLU()
        # relu is my activation function that is the all or nothing like a neuron

        # in order to record the activations, make a dictionary
        # activation are like the firing rate of the individual nodes
        self.activations = {}

        # so for each possible option (base 5), we get an activation value
        # which is the sum of all the activations of the individual nodes (times their unique weight)
        # highest sum gets chosen

    def forward(self, x):
        x1 = self.relu(self.fc1(x)) # multiply by weights of the first layer
        self.activations['layer1'] = x1.detach() # save firing rates without calculating gradients (efficient)

        x2 = self.relu(self.fc2(x1)) # do it again
        self.activations['layer2'] = x2.detach()

        return self.fc3(x2) # output layer; these are the Q values
    

class DQN_Agent:
    def __init__(self, input_dimensions = 6, num_actions = 5, alpha = 0.01, epsilon = 0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.model = QNetwork(input_dimensions = input_dimensions, output_dimensions = num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr = alpha) # adam as a standard way to adjust the weights
        # i would usually use SGD, but seems like ADAM is standard and useful because it  scales updates and doesn't need
        # as much hyperparameter tuning (more out of the box)
        self.criterion = nn.MSELoss() # mean squared error loss

    def choose_action(self, state_tensor):
        # epsilon greedy, like the classic

        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() # efficiency because i dont need to do backprop here (update instead)
        
    def update(self, state_tensor, action, reward):
        self.model.train() # set to training mode
        all_q_values = self.model(state_tensor) # get the activations of the output layer (1x5 matrix)
        current_q = all_q_values[0, action] # get the value for the action i actually took

        target_q = torch.tensor(float(reward)) # what was the actual reward?

        loss = self.criterion(current_q, target_q) # get RPE using MSE


        self.optimizer.zero_grad() # clears previous errors
        loss.backward() # backprop



        l1_gradients = self.model.fc1.weight.grad.clone().cpu().numpy()
        l1_feedback = np.mean(np.abs(l1_gradients), axis = 1)

        l2_gradients = self.model.fc2.weight.grad.clone().cpu().numpy()
        l2_feedback = np.mean(np.abs(l2_gradients), axis = 1)

        self.optimizer.step() # ADAM to adjust weights

        return loss.item(), l1_feedback, l2_feedback

        
