import numpy as np
import torch
import os
import argparse
from combinatorial_bandit import Bandit
from options import Neural, Lin

if not os.path.exists('regrets'):
    os.mkdir('regrets')

SEED_HIDDEN = 1000
PI = 3.14

def experiment(
    # related to algorithm
    neural_or_lin, 
    ucb_or_ts, 
    
    # related to score function
    h_str,
    noise_coef,

    # related to feature vector
    unif,
    n_arms,
    n_features,

    # related to combinatorial choices
    n_assortment,
    n_samples,

    # related to number of rounds per simulation
    total_rounds,

    # related to number of simulations per experiment
    n_sim,
    
    # related to coefficients
    reg_factor,
    delta,
    nu,
    gamma,

    # related to neural network
    hidden_layer_width,     
    epochs,
    dropout,
    learning_rate,
    training_period,
    training_window,

    # filename to save the result
    save           
    ):

    """ kind explanation
    """
    T = total_rounds    
    p = dropout

    ## device to compute the algorithm
     # for nueral, use gpu if availabe
     # for linear, use cpu
    if neural_or_lin == 'neural' and torch.cuda.is_available():
        device = torch.device('cuda')        
    else:
        device = torch.device('cpu')        

    ## score function
    np.random.seed(SEED_HIDDEN) # random seed        
    tmp = np.random.randn(n_features)
    a = torch.from_numpy(tmp / np.linalg.norm(tmp, ord=2)).to(device)    
    if h_str == "h1":        
        def h(x):
            return torch.dot(x, a).to(device)    
    elif h_str == "h2":        
        def h(x):
            return (torch.dot(x, a)**2).to(device)    
    elif h_str == "h3":        
        def h(x):
            return torch.cos(PI*torch.dot(x, a)).to(device)
    elif h_str == "h4":        
        def h(x):
            return torch.sin(PI*torch.dot(x, a)).to(device)

    ## reward function
    def F(x): # round_reward_function
        if x.dim == 1: # if x is a vector
            return torch.sum(x)
        else: # if x is a matrix
            return torch.sum(x, dim=-1)                

    ## Bandit
    bandit = Bandit(T,
                    n_arms,
                    n_features, 
                    h,
                    noise_coef=noise_coef,
                    n_assortment=n_assortment,
                    n_samples=n_samples,
                    round_reward_function=F,
                    device=device,
                    n_sim=n_sim,
                    unif=unif
                   )
       
    ## Learning algorithm and regret
    regrets = np.empty((n_sim, T))
    
    ## Repeat simulation n_sim times
    for i in range(n_sim):
        bandit.reset(i) # use the pre-generated feature vectors and noises for i-th simulation
        ## Neural
        if neural_or_lin == 'neural':
            model = Neural(ucb_or_ts,
                           bandit,
                           hidden_layer_width,
                           reg_factor=reg_factor,
                           delta=delta,
                           gamma=gamma,
                           nu=nu,
                           p=p,
                           training_window=training_window,
                           learning_rate=learning_rate,
                           epochs=epochs,
                           training_period=training_period,
                           device=device
                          )
            
            model.set_init_param(model.model.parameters()) # keep initial parameters for regularization
        
        ## Linear     
        elif neural_or_lin == 'lin':
            model = Lin(ucb_or_ts,
                        bandit,
                        reg_factor=reg_factor,
                        delta=delta,
                        gamma=gamma,
                        nu=nu,
                        device=device
                        )

        model.run()
        regrets[i] = np.cumsum(model.regrets.to('cpu').numpy())
        np.cumsum(model.regrets.to('cpu').numpy())
    if save: # save regrets
        np.save('regrets/' + save, regrets)
    return regrets    
      
#-------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # related to algorithm
    parser.add_argument('--neural_or_lin', type=str, default='neural') # 'neural' for neural bandit, 'lin' for linear bandit
    parser.add_argument('--ucb_or_ts', type=str, default='UCB') # 'UCB' for UCB, 'TS' for TS

    # related to score function
    parser.add_argument('--score_ftn', type=str, default='h2') # score functions. h1, h2, h3. 'h1': linear, 'h2': quadratic, 'h3': cosine
    parser.add_argument('--noise_coef', type=float, default=0.01) # float. coefficient of the noise of scores: noise = noise_coef*N(0,1), score = h_n + noise

    # related to arm or feature vector
    parser.add_argument('--unif', type=str, default='False') # If True, sample feature vectors from uniform dist. Else, sample feature vectors from normal dist.
    parser.add_argument('--n_arms', type=int, default=20) # N
    parser.add_argument('--n_features', type=int, default=80) # d
    
    # related to combinatorial selection or multiple sampling
    parser.add_argument('--n_assortment', type=int, default=4) # K
    parser.add_argument('--n_samples', type=int, default=1) # M
    
    # related to number of rounds per simulation         
    parser.add_argument('--total_rounds', type=int, default=2000) # T

    # related to number of simulations per experiment
    parser.add_argument('--n_sim', type=int, default=20) # number of simulations for one experiment

    # related to coefficients
    parser.add_argument('--reg_factor', type=float, default=1.0) # lambda
    parser.add_argument('--delta', type=float, default=0.1) # delta
    parser.add_argument('--nu', type=float, default=1.0) # nu
    parser.add_argument('--gamma', type=float, default=1.0) # gamma

    # related to neural network
    parser.add_argument('--hidden_layer_width', type=int, default=100) # m
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--training_period', type=int, default=10) # update the network only when "round % training_period == 0"
    parser.add_argument('--training_window', type=int, default=100) # use the recent "training_window" rewards for updating the network

    # filename to save the result
    parser.add_argument('--save', type=str, default='')
    
    args = parser.parse_args()    
    unif = True if (args.unif == 'True') else False
    
    # run the experiment
    experiment(
        # related to algorithm
        args.neural_or_lin,
        args.ucb_or_ts,
        
        # related to score function
        args.score_ftn,
        args.noise_coef,
        
        # related to feature vector
        unif,
        args.n_arms,
        args.n_features, 

        # related to combinatorial choices
        args.n_assortment,
        args.n_samples,
        
        # related to number of rounds per simulation
        args.total_rounds,

        # related to number of simulations per experiment
        args.n_sim,

        # related to coefficients
        args.reg_factor,
        args.delta,
        args.nu,
        args.gamma,

        # related to neural network
        args.hidden_layer_width,
        args.epochs,
        args.dropout,
        args.learning_rate,
        args.training_period,
        args.training_window,

        # filename to save the result
        args.save
        )
