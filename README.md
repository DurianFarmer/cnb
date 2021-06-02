# Combinatorial Neural Bandints

# Code Description
- experiment.py
- combinatorial_bandit.py
- options.py
- plot_results.py

# How to Use the Code

### Required Modules

### How to Use Argparse
    # related to algorithm
    parser.add_argument('--neural_or_lin', type=str, default='neural') # 'neural' for neural bandit, 'lin' for linear bandit
    parser.add_argument('--ucb_or_ts', type=str, default='UCB') # 'UCB' for UCB, 'TS' for TS

    # related to score function
    parser.add_argument('--score_ftn', type=str, default='h2') # score functions. h1, h2, h3. 'h1': linear, 'h2': quadratic, 'h3': cosine
    parser.add_argument('--noise_coef', type=float, default=0.01) # float. coefficient of the noise of scores: noise = noise_coef*N(0,1), score = h_n + noise

    # related to feature vector
    parser.add_argument('--unif', type=str, default='False') # If True, sample feature vectors from uniform dist. Else, sample feature vectors from normal dist.
    parser.add_argument('--n_arms', type=int, default=20) # N
    parser.add_argument('--n_features', type=int, default=80) # d
    
    # related to combinatorial choices
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

# How to Run the Experiments

### Hidden functions
- <img src="https://latex.codecogs.com/gif.latex?-- " /> 

- Linear: <img src="https://latex.codecogs.com/gif.latex?h_{1}(\mathbf{x}_{t,i}) = \mathbf{x}_{t,i}^{\top}\mathbf{a} " />
- Quadratic: $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$
- Non-linear: $h_{3}(\mathbf{x}_{t,i}) = \cos(\pi \mathbf{x}_{t,i}^{\top}\mathbf{a})$
- where $\mathbf{a} is sampled from N(0,1)$ and then normalized

### Experiment 1

##### For $d=80$, for each hidden function, compare the following algorithms
- CN-UCB
- CN-TS: optimistic sampling, sample size = 10
- CN-TS(1): single sampling
- CombLinUCB
- CombLinTS

##### $h_{1}(\mathbf{x}_{t,i}) = \mathbf{x}_{t,i}^{\top}\mathbf{a}$, $d=80$
- CNUCB
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CNUCB
```

- CNTS
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h1 --n_features 80 --n_samples 10 \
--total_rounds 2000 --save exp1_h1_CNTS
```

- CNTS(M=1)
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h1 --n_features 80 --n_samples 1 \
--total_rounds 2000 --save exp1_h1_CNTS_M1
```

- CombLinUCB
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CombLinUCB
```

- CombLinTS
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CombLinTS
```

##### $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$, $d=80$
- CNUCB
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CNUCB
```

- CNTS
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 80 --n_samples 10 \
--total_rounds 2000 --save exp1_h2_CNTS
```

- CNTS(M=1)
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 80 --n_samples 1 \
--total_rounds 2000 --save exp1_h2_CNTS_M1
```

- CombLinUCB
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CombLinUCB
```

- CombLinTS
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CombLinTS
```

##### $h_{3}(\mathbf{x}_{t,i}) = \cos(\pi \mathbf{x}_{t,i}^{\top}\mathbf{a})$, $d=80$
- CNUCB
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CNUCB
```

- CNTS
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h3 --n_features 80 --n_samples 10 \
--total_rounds 4000 --save exp1_h3_CNTS
```

- CNTS(M=1)
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h3 --n_features 80 --n_samples 1 \
--total_rounds 4000 --save exp1_h3_CNTS_M1
```

- CombLinUCB
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CombLinUCB
```

- CombLinTS
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CombLinTS
```

### Experiment 2

##### For $h_{2}$, for $d=\{40, 80, 120\}$, compare the following algorithms
- CN-UCB
- CN-TS: optimistic sampling, sample size = 10
- CN-TS(1): single sampling

##### $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$, $d=40$
- CNUCB
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 40 \
--total_rounds 2000 --save exp2_40_CNUCB
```

- CNTS
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 40 --n_samples 10 \
--total_rounds 2000 --save exp2_40_CNTS
```

- CNTS(M=1)
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 40 --n_samples 1 \
--total_rounds 2000 --save exp2_40_CNTS_M1
```

##### $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$, $d=80$
- use the results of Experiment 1, $h_{2}$.

##### $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$, $d=120$
- CNUCB
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 120 \
--total_rounds 4000 --save exp2_120_CNUCB
```

- CNTS
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 120 --n_samples 10 \
--total_rounds 4000 --save exp2_120_CNTS
```

- CNTS(M=1)
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 120 --n_samples 1 \
--total_rounds 4000 --save exp2_120_CNTS_M1
```

# How to Plot the Results
