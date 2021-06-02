import numpy as np
import torch

from tqdm import tqdm
import abc

import torch.nn as nn


def inv_sherman_morrison_iter(a, A_inv):
    """Inverse of a matrix for combinatorial case.
    """
    temp = A_inv.float()    
    for u in a:
        u = u.float()                     
        Au = torch.matmul(temp, u)
        temp = temp - torch.ger(Au, Au)/(1+torch.matmul(u.T, Au))    
    return temp       

class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self, 
                 input_size=1, 
                 hidden_layer_width=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                ):
        super(Model, self).__init__()
        
        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]                        
        else:
            size  = [input_size] + [hidden_layer_width,] * (self.n_layers-1) + [1]
            ##
            self.layers = [nn.Linear(size[i], size[i+1], bias=False) \
                           for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)
        
        # dropout layer
        self.dropout = nn.Dropout(p=p)
        
        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
                    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x   

class UCBorTS(abc.ABC):
    """Base class for UCB and TS methods.
    """
    def __init__(self,
                 ucb_or_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 reg_factor=1.0,
                 gamma=1, ## for UCB, gamma
                 nu=1, ## for TS, nu
                 delta=0.1,
                 training_period=1,
                 throttle=int(1e2),
                 device=torch.device('cpu')
                ):
        ## select whether UCB or TS
        self.ucb_or_ts = ucb_or_ts
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta

        # multiplier for the confidence bound            
        self.gamma = gamma

        # exploration variance for TS
        self.nu = nu
        
        # train approximator only few rounds
        self.training_period = training_period
        
        # throttle tqdm updates
        self.throttle = throttle
        
        # device
        self.device = device
        
        self.reset()
    
    ## for UCB
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        if self.ucb_or_ts == "UCB":
            self.exploration_bonus = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.mu_hat = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.upper_confidence_bounds = torch.ones((self.bandit.T, self.bandit.n_arms)).to(self.device)
        else:
            pass

    ## for TS
    def reset_sample_rewards(self):
        """Initialize sample rewards and related quantities.
        """
        if self.ucb_or_ts == "TS":
            self.sigma_square = torch.ones((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.mu_hat = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.sample_rewards = torch.zeros((self.bandit.T, self.bandit.n_arms, self.bandit.n_samples)).to(self.device)
            self.optimistic_sample_rewards = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
        else:
            pass
    
    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = torch.zeros(self.bandit.T).to(self.device)

    def reset_actions(self):
        """Initialize cache of actions (actions: played set of arms of each round).
        """
        self.actions = torch.zeros((self.bandit.T, self.bandit.n_assortment)).to(self.device)
    
    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = (torch.eye(self.approximator_dim).to(self.device)/self.reg_factor).float()        
    
    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = torch.zeros((self.bandit.n_arms, self.approximator_dim)).to(self.device)
        
    def sample_action(self):        
        """Return the action (set of arms) to play based on current estimates
        """
        ## for UCB
        if self.ucb_or_ts == "UCB":
            a = self.upper_confidence_bounds[self.iteration].to('cpu').numpy()
        ## for TS
        if self.ucb_or_ts == "TS":
            a = self.optimistic_sample_rewards[self.iteration].to('cpu').numpy()

        ind = np.argpartition(a, -1*self.bandit.n_assortment)[-1*self.bandit.n_assortment:]
        s_ind = ind[np.argsort(a[ind])][::-1]
        return torch.Tensor(s_ind.copy()).to(self.device)               

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass
    
    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def update_confidence_bounds(self):
        """Update the confidence bounds for all arms at time t.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass
    
    ## for UCB
    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all set of arms.
        """
        # update self.grad_approx
        self.update_output_gradient()
        
        # UCB exploration bonus
        self.exploration_bonus[self.iteration] = torch.Tensor(
            [
                self.confidence_multiplier * torch.sqrt(torch.dot(self.grad_approx[a].float(), torch.matmul(self.A_inv.float(), self.grad_approx[a].T.float()))) for a in self.bandit.arms
            ]
        )        
        
        # update reward prediction mu_hat
        self.predict()
        
        # estimated combined bound for reward
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[self.iteration]

    ## for TS
    def update_sample_rewards(self):
        """Update sample rewards and related quantities for all set of arms.
        """        
        # update self.grad_approx
        self.update_output_gradient() 
        
        # update sigma_square        
        self.sigma_square[self.iteration] = torch.Tensor([self.reg_factor * \
                                             torch.dot(self.grad_approx[a].float(), torch.matmul(self.A_inv.float(), self.grad_approx[a].T.float())) \
                                             for a in self.bandit.arms]).to(self.device)
                
        # update reward prediction mu_hat
        self.predict()
        
        # update sample reward
        self.sample_multi_rewards()
        
        # update optimistic sample reward for each arm
        for a in self.bandit.arms:
            self.optimistic_sample_rewards[self.iteration][a] = torch.max(self.sample_rewards[self.iteration][a])
    
    def sample_multi_rewards(self):
        self.sample_rewards.to('cpu')
        for a in self.bandit.arms:
            for j in range(self.bandit.n_samples):
                self.sample_rewards[self.iteration][a][j] = np.random.normal(loc = self.mu_hat[self.iteration, a].to('cpu'),
                                                                             scale = (self.nu**2) * self.sigma_square[self.iteration, a].to('cpu')
                                                                            )                                                                                                                                           
        self.sample_rewards.to(self.device)
    
    def update_A_inv(self):
        """Update A_inv by using an iteration of Sherman_Morrison formula
        """
        self.A_inv = inv_sherman_morrison_iter(
            self.grad_approx[self.action.to('cpu').numpy()],
            self.A_inv
        )               
        
    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'total regret': 0.0,
            '% optimal set of arms': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):                
                ## for UCB
                if self.ucb_or_ts == "UCB":
                    # update confidence of all set of arms based on observed features at time t
                    self.update_confidence_bounds()
                ## for TS
                if self.ucb_or_ts == "TS":
                    ## update sample rewards of all set of arms based on observed features at time t
                    self.update_sample_rewards()                
                
                # pick action (set of arm) with the highest boosted estimated reward
                self.action = self.sample_action()
                self.actions[t] = self.action
                # update approximator                          
                self.train() ### lin and neural training are different
                # update exploration indicator A_inv
                self.update_A_inv()
                
                ## compute regret                
                self.regrets[t] = self.bandit.best_round_reward[t] - self.bandit.round_reward_function(self.bandit.rewards[t, self.action.to('cpu').numpy()])                 
                
                # increment counter
                self.iteration += 1
                
                # log
                postfix['total regret'] += self.regrets[t].to('cpu').numpy()
                n_optimal_arm = np.sum(
                    np.prod(
                        (self.actions[:self.iteration].to('cpu').numpy()==self.bandit.best_super_arm[:self.iteration].to('cpu').numpy())*1, 
                        axis=1)                                                          
                )
                postfix['% optimal set of arms'] = '{:.2%}'.format(n_optimal_arm / self.iteration)
                
                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)

class Neural(UCBorTS):
    """CN-UCB or CN-TS.
    """
    def __init__(self,
                 ucb_or_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 hidden_layer_width=20,
                 n_layers=2,
                 reg_factor=1.0,
                 delta=0.01,
                 gamma=1, ## for UCB
                 nu=1, ## for TS
                 training_window=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 training_period=1,
                 throttle=1,
                 device=torch.device('cpu'),
                ):

        # hidden size of the NN layers
        self.hidden_layer_width = hidden_layer_width
        # number of layers
        self.n_layers = n_layers
        
        # number of rewards in the training buffer
        self.training_window = training_window
        
        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.device = device
            
        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.n_features, 
                           hidden_layer_width=self.hidden_layer_width,
                           n_layers=self.n_layers,
                           p=self.p
                          ).to(self.device)        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        super().__init__(ucb_or_ts,
                         bandit, 
                         reg_factor=reg_factor,
                         gamma=gamma, ## for UCB
                         nu=nu, ## for TS
                         delta=delta,
                         throttle=throttle,
                         training_period=training_period,
                         device=self.device
                        )

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)
    
    @property
    def confidence_multiplier(self):
        """Constant equal to gamma
        """
        return self.gamma
    
    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        for a in self.bandit.arms:
            
            x = self.bandit.features[self.iteration, a].reshape(1,-1).float()                
            
            self.model.zero_grad()
            y = self.model(x)
            y.backward()
                        
            self.grad_approx[a] = torch.cat([
                w.grad.detach().flatten() / np.sqrt(self.hidden_layer_width)
                for w in self.model.parameters() if w.requires_grad]
            ).to(self.device)
            
            
    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_upper_confidence_bounds() ## for UCB
        self.reset_sample_rewards() ## for TS
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    ## inital parameters
    def set_init_param(self, parameters):
        self.init_param = self.param_to_tensor(parameters)

    ## torch Parameter object to Tensor object
    def param_to_tensor(self, parameters):
        a = torch.empty(1).to(self.device)
        for p in parameters:
            a = torch.cat((a, p.data.flatten()))
        return a[1:].to(self.device)    
        
    def train(self):
        """Train neural approximator.        
        """
        ### train only when training_period occurs
        if self.iteration % self.training_period == 0:                        
            iterations_so_far = range(np.max([0, self.iteration-self.training_window]), self.iteration+1)
            actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1].to('cpu').numpy() # this is a matrix            

            temp = torch.cat([self.bandit.features[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            x_train = torch.reshape(temp, (1,-1,self.bandit.n_features)).squeeze().float().to(self.device)

            temp = torch.cat([self.bandit.rewards[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            y_train = torch.reshape(temp, (1,-1)).squeeze().float().to(self.device)

            # train mode
            self.model.train()
            for _ in range(self.epochs):
                ## computing the regularization parameter
                tmp = (self.param_to_tensor(self.model.parameters()) - self.init_param).to(torch.device('cpu')).numpy()
                param_diff = np.linalg.norm(tmp)
                regularization = (self.reg_factor*self.hidden_layer_width*param_diff**2)/2

                ## update weight
                y_pred = self.model.forward(x_train).squeeze()
                ### loss = nn.MSELoss()(y_train, y_pred)
                loss = nn.MSELoss(reduction='sum')(y_train, y_pred)/2 + regularization            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            pass
                                        
    def predict(self):
        """Predict reward.
        """
        # eval mode
        self.model.eval()        
        self.mu_hat[self.iteration] = self.model.forward(self.bandit.features[self.iteration].float()).detach().squeeze()

class Lin(UCBorTS):
    """LinUCB or LinTS.
    """
    def __init__(self,
                 ucb_or_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 bound_theta=1.0,
                 gamma=1, ## for UCB                 
                 nu=1, ## for TS
                 throttle=int(1e2),
                 device=torch.device('cpu')
                ):

        # range of the linear predictors
        self.bound_theta = bound_theta
        
        super().__init__(ucb_or_ts, 
                         bandit, 
                         reg_factor=reg_factor,
                         gamma=gamma, ## for UCB
                         nu=nu, ## for TS
                         delta=delta,
                         throttle=throttle,
                        )
        self.device = device

    @property
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        return self.bandit.n_features
    
    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        """
        self.grad_approx = self.bandit.features[self.iteration]
    
    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds() ## for UCB
        self.reset_sample_rewards() ## for TS
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

        # randomly initialize linear predictors within their bounds        
        self.theta = torch.from_numpy(np.random.uniform(-1, 1, self.bandit.n_features) * self.bound_theta).to(self.device)

        # initialize reward-weighted features sum at zero
        self.b = torch.zeros(self.bandit.n_features).to(self.device).float()

    @property
    def confidence_multiplier(self):
        """Use exploration variance (nu) instead of confidence scaling factor (gamma)
        """
        return self.gamma
    

    def train(self):
        """Update linear predictor theta.
        """        
        self.theta = torch.matmul(self.A_inv.float(), self.b.float())                      
        tmp = torch.from_numpy(np.sum(np.array([ self.bandit.rewards[self.iteration][i].to('cpu').numpy()*
                      self.bandit.features[self.iteration, self.actions[self.iteration].to('cpu').numpy()][i].to('cpu').numpy()
                                   for i in range(0, self.bandit.n_assortment) ]
                             ), axis = 0)).to(self.device)
        self.b = self.b.double() + tmp.double()
            
    def predict(self):
        """Predict reward.
        """
        self.mu_hat[self.iteration] = torch.Tensor(
            [
                torch.dot(self.bandit.features[self.iteration, a], self.theta.double()) for a in self.bandit.arms
            ]
        )

