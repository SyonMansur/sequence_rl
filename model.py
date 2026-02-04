import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# feedback alignment classes
class FeedbackAlignmentFunction(torch.autograd.Function): # inherit from autograd; we manually write calculus rules
    @staticmethod # belong to the class, not the instance
    def forward(ctx, input, weight, bias, feedback_matrix): # below, using a context box from pytorch
        ctx.save_for_backward(input, weight, bias, feedback_matrix) # save all of the inputs/weights/etc. to calculate gradients later
        output = input.mm(weight.t()) # transpose the weight matrix for proper multiplication
        if bias is not None: # if bias, reshape from [64] to [1, 64]
            output += bias.unsqueeze(0).expand_as(output)
        return output
     # now let's do the backward pass
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, feedback_matrix = ctx.saved_tensors # get the tensors we saved earlier
        # now, we're actually just going to use the feedback matrix instead of the weight matrix
        grad_input = grad_output.mm(feedback_matrix) # ok now its normal calculation
        grad_weight = grad_output.t().mm(input) # error times input, etc.
        grad_bias = grad_output.sum(0) if bias is not None else None # sum errors across batch
        return grad_input, grad_weight, grad_bias, None # return None for feedback_matrix, get everything else

# custom linear layer using feedback alignment (container)
class LinearFA(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearFA, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) # create learnable parameters
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # fixed random feedback weights
        self.feedback_matrix = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5) # optimized way to pick initial weights
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # randomize feedback
        self.feedback_matrix.data.normal_(0, 1)

    def forward(self, input):
        return FeedbackAlignmentFunction.apply(input, self.weight, self.bias, self.feedback_matrix)


class QNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5, learning_rule="backprop"):
        super(QNetwork, self).__init__() 
        
        # layer selection
        if learning_rule == "feedback_alignment":
            LayerType = LinearFA
        else:
            LayerType = nn.Linear

        # use LayerType() instead of nn.Linear()
        self.fc1 = LayerType(input_dimensions, hidden_dimensions)
        self.fc2 = LayerType(hidden_dimensions, hidden_dimensions)
        
        # usually we want the output layer to also be FA so it sends 
        # random feedback to layer 2
        self.fc3 = LayerType(hidden_dimensions, output_dimensions)
        
        self.relu = nn.ReLU()
        self.activation_tensors = {}

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        if a1.requires_grad: a1.retain_grad()
        self.activation_tensors['layer1'] = a1

        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        if a2.requires_grad: a2.retain_grad()
        self.activation_tensors['layer2'] = a2

        return self.fc3(a2)

class DQN_Agent:
    def __init__(self, input_dimensions=6, num_actions=5, alpha=0.01, epsilon=0.1, learning_rule="backprop"):
        self.num_actions = num_actions
        self.epsilon = epsilon
        # pass the rule to the network
        self.model = QNetwork(input_dimensions, num_actions*4, num_actions, learning_rule=learning_rule)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) 
        self.criterion = nn.MSELoss() 

    # choose_action and update methods remain exactly the same ...
    def choose_action(self, state_tensor):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 
        
    def update(self, state_tensor, action, reward):
        self.model.train() 
        all_q_values = self.model(state_tensor) 
        current_q = all_q_values[0, action] 
        target_q = torch.tensor(reward, dtype=torch.float32) 
        loss = self.criterion(current_q, target_q) 
        self.optimizer.zero_grad() 
        loss.backward() 
        l1_grads = self.model.activation_tensors['layer1'].grad.detach().cpu().numpy().flatten()
        l1_feedback = np.abs(l1_grads)
        l2_grads = self.model.activation_tensors['layer2'].grad.detach().cpu().numpy().flatten()
        l2_feedback = np.abs(l2_grads)
        self.optimizer.step() 
        return loss.item(), l1_feedback, l2_feedback


