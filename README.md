<<<<<<< Updated upstream
# Code for Combinatorial Neural Bandits

This repository is the official implementation of [Combinatorial Neural Bandits](https://arxiv.org/abs/--). 

## Requirements

- The code uses `PyTorch 1.7.1`.
- We used an Anaconda environment for the experiment. Use the following to setup the modules & dependencies:
```setup
conda env create -f environment.yml
```
- Or you may use pip requirements:
```setup
 pip install -r requirements.txt
```

## Code Description
- `experiment.py`: Runs the experiments. Please refer to [Details](#details).
- `combinatorial_bandit.py`: Defines the combinatorial bandit.
- `options.py`: Defines the agent's strategies: Neural or Linear / UCB or TS (single sampling or optimistic sampling).
- `plot_results.py`: Plots the experiment results.

[Details](#details) is long. If you are interested in how we used the code for the experiments, please go to [How to Run the Code](#how-to-run-the-code).

## Details
### experiment.py
`experiment.py` has 22 argument parsers. You may choose different values for testing the algorithm.

- Related to the Algorithm

| Argparse Command | Default | Note |
|---|---|---|
| --neural_or_lin | nueral | 'neural' for neural bandit, 'lin' for linear bandit |
| --ucb_or_ts | UCB | 'UCB' for UCB, 'TS' for TS |

<br>

- Related to the Score Function

| Argparse Command | Default | Note |
|---|---|---|
| --score_ftn | h2 | 'h1', 'h2' or 'h3'. 'h1': linear, 'h2': quadratic, 'h3': cosine |
| --noise_coef | 0.01 | the intesity of the noise of the scores, *cf. score = h(x) + noise* |

<br>

- Related to Arm or Feature Vector

| Argparse Command | Default | Note |
|---|---|---|
| --unif | False | If True, sample feature vectors from uniform dist. Else, sample feature vectors from normal dist. |
| --n_arms | 20 | *N* |
| --n_features | 20 | *d* |

<br>

- Related to Combinatorial Selection or Multiple Sampling

| Argparse Command | Default | Note |
|---|---|---|
| --n_assortment | False | *K* |
| --n_samples | 1 | *M*. Sampling number for ***CN-TS*** |

<br>

- Related to Rounds per Simulation / Simulations per Experiment

| Argparse Command | Default | Note |
|---|---|---|
| --total_rounds | 2000 | *T* |
| --n_sim | 20 | Number of simulations for one experiment. Average the simulation results for final experiment result. |

<br>

- Related to Coefficients

| Argparse Command | Default | Note |
|---|---|---|
| --reg_factor | 1.0 | *lambda* |
| --delta | 0.1 | *delta* |
| --nu | 1.0 | *nu* |
| --gamma | 1.0 | *gamma* |

<br>

- Related to the Neural Network

| Argparse Command | Default | Note |
|---|---|---|
| --hidden_layer_width | 100 | *m* |
| --epochs | 100 |  |
| --dropout | 0.0 |  |
| --learning_rate | 0.01 |  |
| --training_period | 10 | update the network only when "round % training_period == 0" |
| --training_window | 100 | use the recent "training_window" rewards for updating the network |

<br>

- Filename to Save the Result

| Argparse Command | Default | Note |
|---|---|---|
| --save | '' |  |

<br>

### plot_results.py
`plot_results.py` has 1 argument parser.

| Argparse Command | Default | Note |
|---|---|---|
| --exp | 1 | Experiment number. '1' for plotting the results of Experiment 1, '2' for plotting the results of Experiment 2 |


## How to Run the Code
=======
# Combinatorial Neural Bandints

## Code Description
- experiment.py
- combinatorial_bandit.py
- options.py
- plot_results.py

## How to Use the Code

### Required Modules

### How to Use Argparse
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

# How to Run the Experiments

### Hidden functions
- Linear: $h_{1}(\mathbf{x}_{t,i}) = \mathbf{x}_{t,i}^{\top}\mathbf{a}$
- Quadratic: $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$
- Non-linear: $h_{3}(\mathbf{x}_{t,i}) = \cos(\pi \mathbf{x}_{t,i}^{\top}\mathbf{a})$
- where $\mathbf{a} is sampled from N(0,1)$ and then normalized
>>>>>>> Stashed changes

### Experiment 1
Experiment 1 compares ***CN-UCB***, ***CN-TS*** (optimistic sampling) and ***CN-TS(M=1)*** (single sampling) with prior linear combinatorial algorithms, ***CombLinUCB*** and ***CombLinTS***.  The score functions are ***{h1, h2, h3}*** and the feature dimension is ***d=80***.

<<<<<<< Updated upstream
To run Experiment 1 in the paper, run the following commands:
=======
For $d=80$, for each hidden function, compare the following algorithms
- CN-UCB
- CN-TS: optimistic sampling, sample size = 10
- CN-TS(1): single sampling
- CombLinUCB
- CombLinTS
>>>>>>> Stashed changes

***h1 (linear)***
- ***CN-UCB***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CNUCB
```

- ***CN-TS***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h1 --n_features 80 --n_samples 10 \
--total_rounds 2000 --save exp1_h1_CNTS
```

- ***CN-TS(M=1)***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h1 --n_features 80 --n_samples 1 \
--total_rounds 2000 --save exp1_h1_CNTS_M1
```

- ***CombLinUCB***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CombLinUCB
```

- ***CombLinTS***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h1 --n_features 80 \
--total_rounds 2000 --save exp1_h1_CombLinTS
```

***h2 (linear)***
- ***CN-UCB***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CNUCB
```

- ***CN-TS***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 80 --n_samples 10 \
--total_rounds 2000 --save exp1_h2_CNTS
```

- ***CN-TS(M=1)***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 80 --n_samples 1 \
--total_rounds 2000 --save exp1_h2_CNTS_M1
```

- ***CombLinUCB***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CombLinUCB
```

- ***CombLinTS***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h2 --n_features 80 \
--total_rounds 2000 --save exp1_h2_CombLinTS
```

***h3 (cosine)***
- ***CN-UCB***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CNUCB
```

- ***CN-TS***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h3 --n_features 80 --n_samples 10 \
--total_rounds 4000 --save exp1_h3_CNTS
```

- ***CN-TS(M=1)***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h3 --n_features 80 --n_samples 1 \
--total_rounds 4000 --save exp1_h3_CNTS_M1
```

- ***CombLinUCB***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts UCB \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CombLinUCB
```

- ***CombLinTS***
```bash
python3 experiment.py \
--neural_or_lin lin --ucb_or_ts TS \
--score_ftn h3 --n_features 80 \
--total_rounds 4000 --save exp1_h3_CombLinTS
```

### Experiment 2
Experiment 2 compares ***CN-UCB***, ***CN-TS*** (optimistic sampling) and ***CN-TS(M=1)*** (single sampling) as the feature dimension increases ***d={40, 80, 120}***.  The score function is quadratic ***h2***.

To run Experiment 2 in the paper, run the following commands:

***d=40***
- ***CN-UCB***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 40 \
--total_rounds 2000 --save exp2_40_CNUCB
```

- ***CN-TS***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 40 --n_samples 10 \
--total_rounds 2000 --save exp2_40_CNTS
```

- ***CN-TS(M=1)***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 40 --n_samples 1 \
--total_rounds 2000 --save exp2_40_CNTS_M1
```

***d=80***

Use the results of Experiment 1, ***h2***.

***d=120***
- ***CN-UCB***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts UCB \
--score_ftn h2 --n_features 120 \
--total_rounds 4000 --save exp2_120_CNUCB
```

- ***CN-TS***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 120 --n_samples 10 \
--total_rounds 4000 --save exp2_120_CNTS
```

- ***CN-TS(M=1)***
```bash
python3 experiment.py \
--neural_or_lin neural --ucb_or_ts TS \
--score_ftn h2 --n_features 120 --n_samples 1 \
--total_rounds 4000 --save exp2_120_CNTS_M1
```

### Plot the Results
Plot the Results
After getting all of the results, use the following code to plot the results of Experiment 1 & 2:
- Experiment 1
```bash
python3 plot_results.py --exp 1
```

- Experiment 2
```bash
python3 plot_results.py --exp 2
```

## Results

Our algortihms achieves the following performance:

### Experiment 1 (score function *{h1, h2, h3}*, feature dimension *d=80*)
![exp1](./plots/Figure1.png)

Compared to linear baselines, the cumulative regrets for ***CN-UCB*** and ***CN-TS*** show a sub-linear tendency even when the score function is quadratic or non-linear.
This suggests that our algorithms can be more applicable to various and complex reward functions.

### Experiment 2 (quadratic score function *h2*, feature dimension *d={40,80,120}*)
![exp2](./plots/Figure2.png)

The empirical results suggest that ***CN-UCB*** and ***CN-TS*** have a scalability in feature dimension ***d*** no higher than linear.
When ***d*** is large, ***CN-TS*** shows a lower cumulative regret compared to ***CN-TS(M=1)***. 
This observation supports our claim that ***CN-TS*** secures a constant probability of optimism by drawing multiple ***M*** samples.

## Contact Info.
Please contact [Taehyun Hwang](th.hwang@snu.ac.kr) or [Kyuwook Chai](kwchai@snu.ac.kr) for anything.
