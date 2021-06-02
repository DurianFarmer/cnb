import numpy as np
import torch
import itertools

SEED_FEATURE = 2000
SEED_NOISE = 3000

class Bandit():
    def __init__(self,
                 T,
                 n_arms,                 
                 n_features,                                  
                 h,                                                   
                 noise_coef=1.0,                 
                 n_assortment=1,
                 n_samples=1,
                 round_reward_function=sum,
                 device=torch.device('cpu'),
                 n_sim=20,
                 unif=False
                ):
        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features        
        
        # average reward function
        # h : R^d -> R
        self.h = h

        # standard deviation of Gaussian reward noise
        self.noise_coef = noise_coef
        
        # number of assortment (top-K)
        self.n_assortment = n_assortment                
        
        # (TS) number of samples for each round and arm
        self.n_samples = n_samples
        
        # round reward function
        self.round_reward_function = round_reward_function                
        
        # device
        self.device = device
        
        # number of simulation
        self.n_sim = n_sim
        
        # generate feature vectors and noise
        self.X = self.generate_features(unif)
        self.xi = self.generate_noise()

    @property                
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)
    
    def generate_features(self, unif):
        """Generate feature vectors for every round and simulation
        """
        X = np.zeros((self.n_sim, self.T, self.n_arms, self.n_features))
        
        # set random seed for feature vector
        np.random.seed(SEED_FEATURE)
        for i in range(self.n_sim):
            if unif:
                x = np.random.uniform(low=-1.0, high=1.0, size=(self.T, self.n_arms, self.n_features))
            else:
                x = np.random.randn(self.T, self.n_arms, self.n_features)
            x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
            X[i] = x
        return torch.from_numpy(X).to(self.device) 
    
    def generate_noise(self):
        """Generate noise for every round and simulation
        """
        xi = np.zeros((self.n_sim, self.T, self.n_arms))
        
        # set random seed for noise
        np.random.seed(SEED_NOISE)
        for i in range(self.n_sim):   
            x = np.random.randn(self.T, self.n_arms)        
            xi[i] = x
        return torch.from_numpy(xi).to(self.device)
                                                                            
    def reset(self, i):
        """Generate new features and new rewards.
        """
        self.reset_features(i)
        self.reset_rewards(i)
    
    def reset_features(self, i):
        """Generate normalized random N(0,1) features.
        """        
        self.features = self.X[i]    

    def reset_rewards(self, i):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """            
        self.rewards = torch.Tensor(
            [
                self.h( self.features[t, k] ) + self.noise_coef*self.xi[i][t][k] 
                for t,k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        ## to be used only to compute regret, NOT by the algorithm itself        
        a = self.rewards.to('cpu').numpy()
        ind = np.argpartition(a, -1*self.n_assortment, axis=1)[:,-1*self.n_assortment:]        
        s_ind = np.array([list(ind[i][np.argsort(a[i][ind[i]])][::-1]) for i in range(0, np.shape(a)[0])])
        
        self.best_super_arm = torch.from_numpy(s_ind).to(self.device)
        self.best_rewards = torch.Tensor([a[i][s_ind[i]] for i in range(0,np.shape(a)[0])]).to(self.device)
        self.best_round_reward = self.round_reward_function(self.best_rewards).to(self.device)