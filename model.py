import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# standard backprop on a simple feedforward network
class QNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5):
        super(QNetwork, self).__init__() 
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        
        # relu for backprop. dtp uses tanh.
        self.activation = nn.ReLU() 
        self.activation_tensors = {}

    def forward(self, x):
        # layer one
        z1 = self.fc1(x)
        a1 = self.activation(z1)
        
        # hold onto grads for plotting later
        if a1.requires_grad: a1.retain_grad()
        self.activation_tensors['layer1'] = a1
        
        # layer two
        z2 = self.fc2(a1)
        a2 = self.activation(z2)
        if a2.requires_grad: a2.retain_grad()
        self.activation_tensors['layer2'] = a2
        
        # final output
        return self.fc3(a2)

# agent controller
class DQN_Agent:
    def __init__(self, input_dimensions=6, num_actions=5, hidden_dims=64, alpha=0.01, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.model = QNetwork(input_dimensions, hidden_dims, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) 
        self.criterion = nn.MSELoss() 

    def choose_action(self, state_tensor):
        # random action if under epsilon threshold
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 
        
    def update(self, state_tensor, action_tensor, reward_tensor):
        self.model.train() 
        
        # forward pass
        all_q_values = self.model(state_tensor) 
        current_q = all_q_values.gather(1, action_tensor)
        target_q = reward_tensor
        
        # loss
        loss = self.criterion(current_q, target_q) 
        raw_error = torch.mean(torch.abs(current_q - target_q)).detach().item()
        
        # standard backprop
        self.optimizer.zero_grad() 
        loss.backward() 
        
        # clip grads so it doesnt crash
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # get flat arrays for plotting
        l1_grads = self.model.activation_tensors['layer1'].grad.detach().cpu().numpy().flatten()
        l1_feedback = np.abs(l1_grads)
        l2_grads = self.model.activation_tensors['layer2'].grad.detach().cpu().numpy().flatten()
        l2_feedback = np.abs(l2_grads)

        # grab math parts for chain rule proof
        delta = (current_q - target_q).detach().item()
        w3_act = self.model.fc3.weight[action_tensor.item()].detach().cpu().numpy()
        a2_val = self.model.activation_tensors['layer2'].detach().cpu().numpy().flatten()
        a2_relu = (a2_val > 0).astype(float)
        a2_gtd = l2_grads * a2_relu
        
        w1_m = torch.mean(torch.abs(self.model.fc1.weight)).item()
        w2_m = torch.mean(torch.abs(self.model.fc2.weight)).item()
        w3_m = torch.mean(torch.abs(self.model.fc3.weight)).item()

        self.optimizer.step() 
        
        return loss.item(), l1_feedback, l2_feedback, raw_error, delta, w3_act, a2_relu, a2_gtd, w1_m, w2_m, w3_m

# dtp
#  lee et al 2015
class DTPNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5):
        super(DTPNetwork, self).__init__() 
        
        # fwd layers
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        
        # inv layers. only need layer two.
        self.inv2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.inv3 = nn.Linear(output_dimensions, hidden_dimensions)
        
        # gotta use tanh here
        self.activation = nn.Tanh()
        self.activation_tensors = {}
        
        # orthogonal init from the paper. zero out biases.
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.activation(z1)
        if a1.requires_grad: a1.retain_grad()
        self.activation_tensors['layer1'] = a1
        
        z2 = self.fc2(a1)
        a2 = self.activation(z2)
        if a2.requires_grad: a2.retain_grad()
        self.activation_tensors['layer2'] = a2
        
        return self.fc3(a2)

class DTPAgent:
    def __init__(self, input_dimensions=6, num_actions=5, hidden_dims=64, alpha=0.01, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        self.model = DTPNetwork(input_dimensions, hidden_dims, num_actions)
        self.criterion = nn.MSELoss() 
        
        # split optimizers for fwd and inv
        self.optimizer_fwd = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'inv' not in n], 
            lr=alpha
        )
        self.optimizer_inv = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'inv' in n], 
            lr=alpha * 2.0
        )

    def choose_action(self, state_tensor):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 
        
    def update_inverse(self, state_tensor):
        self.model.train() 
        
        # fwd pass to get activations
        with torch.no_grad():
            _ = self.model(state_tensor)
            
        a1 = self.model.activation_tensors['layer1']
        a2 = self.model.activation_tensors['layer2']
        noise_level = 0.1 
        
        # rapid mini updates
        for _ in range(3): 
            # add noise
            noise_a1 = a1.detach() + torch.randn_like(a1) * noise_level
            
            # forward thru f2
            clean_z2 = self.model.fc2(noise_a1)
            clean_a2 = self.model.activation(clean_z2)
            
            # backward thru inv2
            recon_a1 = self.model.activation(self.model.inv2(clean_a2))
            
            # try to fix first layer
            inv_loss_1 = self.criterion(recon_a1, noise_a1)
            
            # add noise
            noise_a2 = a2.detach() + torch.randn_like(a2) * noise_level
            
            # forward thru f3
            clean_z3 = self.model.fc3(noise_a2)
            
            # backward thru inv3
            recon_a2 = self.model.activation(self.model.inv3(clean_z3))
            
            # try to fix sec layer
            inv_loss_2 = self.criterion(recon_a2, noise_a2)
            
            total_inv_loss = inv_loss_1 + inv_loss_2
            
            self.optimizer_inv.zero_grad()
            total_inv_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer_inv.step()

    def update_forward(self, state_tensor, action_tensor, reward_tensor):
        self.model.train() 
        
        all_q_values = self.model(state_tensor) 
        current_q = all_q_values.gather(1, action_tensor)
        target_q = reward_tensor
        
        # global loss; keep graph to diff later
        fwd_loss_main = self.criterion(current_q, target_q)
        raw_error = torch.mean(torch.abs(current_q - target_q)).detach().item()
        
        a1 = self.model.activation_tensors['layer1']
        a2 = self.model.activation_tensors['layer2']
        
        # pure dtp mapping for top layer
        target_a3 = all_q_values.clone().detach()
        target_a3[0, action_tensor] = target_q
        
        with torch.no_grad():
            inv_pred_target_a2 = self.model.activation(self.model.inv3(target_a3))
            inv_pred_current_a2 = self.model.activation(self.model.inv3(all_q_values.detach()))
            
            correction_a2 = inv_pred_target_a2 - inv_pred_current_a2
            correction_a2 = torch.clamp(correction_a2, -0.5, 0.5)
            target_a2 = a2.detach() + correction_a2
            
            inv_pred_target = self.model.activation(self.model.inv2(target_a2))
            inv_pred_current = self.model.activation(self.model.inv2(a2))
            
            correction = inv_pred_target - inv_pred_current
            correction = torch.clamp(correction, -0.5, 0.5)
            target_a1 = a1.detach() + correction
            
            l2_grad_proxy = torch.abs(target_a2 - a2).cpu().numpy().flatten()
            l1_grad_proxy = torch.abs(target_a1 - a1).cpu().numpy().flatten()
            
        l1_feedback = l1_grad_proxy
        l2_feedback = l2_grad_proxy

        # treat each layer as local supervised problem
        loss_3 = fwd_loss_main
        loss_2 = self.criterion(a2, target_a2.detach())
        loss_1 = self.criterion(a1, target_a1.detach())
        total_fwd_loss = loss_3 + loss_2 + loss_1

        self.optimizer_fwd.zero_grad() 
        total_fwd_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer_fwd.step() 
        
        # mock chain rule parts for dtp so experiment.py doesnt crash
        delta = raw_error
        w3_act = self.model.fc3.weight[action_tensor.item()].detach().cpu().numpy()
        a2_val = a2.detach().cpu().numpy().flatten()
        a2_deriv = (1.0 - a2_val**2) # tanh deriv
        a2_gtd = l2_grad_proxy * a2_deriv
        w1_m = torch.mean(torch.abs(self.model.fc1.weight)).item()
        w2_m = torch.mean(torch.abs(self.model.fc2.weight)).item()
        w3_m = torch.mean(torch.abs(self.model.fc3.weight)).item()
        
        return total_fwd_loss.item(), l1_feedback, l2_feedback, raw_error, delta, w3_act, a2_deriv, a2_gtd, w1_m, w2_m, w3_m

# predictive coding
class PCNetwork(nn.Module):
    def __init__(self, input_dimensions=6, hidden_dimensions=64, output_dimensions=5):
        super(PCNetwork, self).__init__() 
        
        # fwd layers
        self.fc1 = nn.Linear(input_dimensions, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.fc3 = nn.Linear(hidden_dimensions, output_dimensions)
        
        # top down generative layers for pred error
        self.td2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.td3 = nn.Linear(output_dimensions, hidden_dimensions)
        
        self.activation = nn.Tanh()
        self.activation_tensors = {}
        
        # orthogonal
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.activation(z1)
        self.activation_tensors['layer1'] = a1
        
        z2 = self.fc2(a1)
        a2 = self.activation(z2)
        self.activation_tensors['layer2'] = a2
        
        return self.fc3(a2)

class PCAgent:
    def __init__(self, input_dimensions=6, num_actions=5, hidden_dims=64, alpha=0.01, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        self.model = PCNetwork(input_dimensions, hidden_dims, num_actions)
        self.criterion = nn.MSELoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        
        # pc specific hyperparams
        self.inf_lr = 0.05 
        self.inf_steps = 10 

    def choose_action(self, state_tensor):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 

    def update(self, state_tensor, action_tensor, reward_tensor):
        self.model.train() 
        
        # phase one inference
        # get standard fwd pass to seed the starting activations
        with torch.no_grad():
            init_q = self.model(state_tensor)
            init_a1 = self.model.activation_tensors['layer1']
            init_a2 = self.model.activation_tensors['layer2']
            
        # set up free floating params for the neural activities
        a1 = init_a1.clone().detach().requires_grad_(True)
        a2 = init_a2.clone().detach().requires_grad_(True)
        q = init_q.clone().detach().requires_grad_(True)
        
        act_optimizer = optim.SGD([a1, a2, q], lr=self.inf_lr)
        
        # run inference loop to let network settle
        for _ in range(self.inf_steps):
            act_optimizer.zero_grad()
            
            pred_a1 = self.model.activation(self.model.fc1(state_tensor))
            pred_a2 = self.model.activation(self.model.fc2(a1))
            pred_q = self.model.fc3(a2)
            
            # sum of squared local prediction errors
            e1 = self.criterion(a1, pred_a1)
            e2 = self.criterion(a2, pred_a2)
            e3 = self.criterion(q, pred_q)
            
            # anchor the top layer to the task reward
            target_q = q.clone().detach()
            target_q[0, action_tensor] = reward_tensor
            e_task = self.criterion(q, target_q)
            
            total_energy = e1 + e2 + e3 + e_task
            total_energy.backward()
            act_optimizer.step()
            
        # phase two learning
        # update weights based on the settled activations
        self.optimizer.zero_grad()
        
        pred_a1 = self.model.activation(self.model.fc1(state_tensor))
        pred_a2 = self.model.activation(self.model.fc2(a1.detach()))
        pred_q = self.model.fc3(a2.detach())
        
        loss_1 = self.criterion(a1.detach(), pred_a1)
        loss_2 = self.criterion(a2.detach(), pred_a2)
        loss_3 = self.criterion(q.detach(), pred_q)
        
        total_loss = loss_1 + loss_2 + loss_3
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # proxy td feedback is literal gradient of the settled energy
        l1_feedback = np.abs(a1.grad.detach().cpu().numpy().flatten()) if a1.grad is not None else np.zeros_like(a1.detach().cpu().numpy().flatten())
        l2_feedback = np.abs(a2.grad.detach().cpu().numpy().flatten()) if a2.grad is not None else np.zeros_like(a2.detach().cpu().numpy().flatten())
        
        # no crash
        with torch.no_grad():
            current_q = q.gather(1, action_tensor)
            raw_error = torch.mean(torch.abs(current_q - reward_tensor)).item()
            delta = raw_error
            w3_act = self.model.fc3.weight[action_tensor.item()].cpu().numpy()
            a2_val = a2.detach().cpu().numpy().flatten()
            a2_deriv = (1.0 - a2_val**2) 
            a2_gtd = l2_feedback * a2_deriv
            w1_m = torch.mean(torch.abs(self.model.fc1.weight)).item()
            w2_m = torch.mean(torch.abs(self.model.fc2.weight)).item()
            w3_m = torch.mean(torch.abs(self.model.fc3.weight)).item()
        
        return total_loss.item(), l1_feedback, l2_feedback, raw_error, delta, w3_act, a2_deriv, a2_gtd, w1_m, w2_m, w3_m