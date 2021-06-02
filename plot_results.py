import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

if not os.path.exists('plots'):
    os.mkdir('plots')

######## Define funtions and properties    
    
regrets_dict = {'CN-UCB': ('',''), 'CN-TS': ('',''), 'CN-TS(M=1)': ('',''), 'CombLinUCB': ('',''), 'CombLinTS': ('','')}

def set_regrets_dict(*r_str_tuple):
    # reset regrets_dict
    for k in regrets_dict:
        regrets_dict[k] = ('','')
    
    for r_str in r_str_tuple:
        if '_CNUCB.npy' in r_str:
            regrets_dict['CN-UCB'] = (r_str, '.')        
        elif '_CNTS.npy' in r_str:
            regrets_dict['CN-TS'] = (r_str, '+')
        elif '_CNTS_M1.npy' in r_str:
            regrets_dict['CN-TS(M=1)'] = (r_str, 'x')
        elif '_CombLinUCB.npy' in r_str:
            regrets_dict['CombLinUCB'] = (r_str, 's')
        elif '_CombLinTS.npy' in r_str:
            regrets_dict['CombLinTS'] = (r_str, 'd')
            
def plot(ax, h, d, regrets_dict):
    
    h1 = r'$h_{1}(\mathbf{x}) = \mathbf{x}^{\top}\mathbf{a}$'
    h2 = r'$h_{2}(\mathbf{x}) = (\mathbf{x}^{\top}\mathbf{a})^{2}$'
    h3 = r'$h_{3}(\mathbf{x}) = \cos(\pi \mathbf{x}^{\top}\mathbf{a})$'
    h4 = r'$h_{4}(\mathbf{x}) = \sin(\pi \mathbf{x}^{\top}\mathbf{a})$'
        
    if h == "h1":
        hidden = h1
    elif h == "h2":
        hidden = h2
    elif h == "h3":
        hidden = h3
    elif h == "h4":
        hidden = h4            
    
    for label, value in regrets_dict.items():        
        result_path = value[0]
        marker = value[1]
        if result_path:
            # total_reg is a numpy array (L x T) where L is the number of repeated experiments
            total_reg = np.load('regrets/' + result_path)

            T = total_reg.shape[-1]            
            steps=np.arange(1,T+1)
            freq = int(T/10)

            avg_reg = total_reg.mean(axis=0)
            sd_req = total_reg.std(axis=0)
            
            ax.errorbar(steps, avg_reg, sd_req, errorevery=freq, marker=marker, markevery=freq, label=label, markersize=6, linewidth=2, elinewidth=1, capsize=3)
    
    ax.grid(color='0.85', )
    ax.set_xlabel('Round ($t$)', size = 14)
    ax.set_ylabel('Cumulative Regret', size = 14)
    ax.set_title(r'{}, $d$={}'.format(hidden, d), size = 14)

    ax.legend(loc='upper left', prop={'size': 12})
    
    if T == 2000:
        ax.set_xticks([0,250,500,750,1000,1250,1500,1750,2000])    
    elif T == 4000:
        ax.set_xticks([0,500,1000,1500,2000,2500,3000,3500,4000])    
    ax.tick_params(labelsize=12)    


######### Set Figures & Plot
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='1') # experiment number. 1 or 2      
    args = parser.parse_args()
    exp_no = args.exp    
    
    plt.style.use('default')
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.4*3,4.8)) # 4:1
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24,6))
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(32,8))
    
    if exp_no == '1':
        d = 80

        for idx, h in enumerate(['h1', 'h2', 'h3']):        
            try:
                set_regrets_dict(f'exp{exp_no}_{h}_CNUCB.npy',                  
                            f'exp{exp_no}_{h}_CNTS.npy',
                            f'exp{exp_no}_{h}_CNTS_M1.npy',
                            f'exp{exp_no}_{h}_CombLinUCB.npy',
                            f'exp{exp_no}_{h}_CombLinTS.npy'
                            )
                plot(axs[idx], h, d, regrets_dict)          
            except:
                pass
    
        plt.savefig(f'plots/exp{exp_no}')

    if exp_no == '2':
        h = 'h2'

        for idx, d in enumerate([40, 80, 120]):        
            try:
                set_regrets_dict(f'exp{exp_no}_{d}_CNUCB.npy',                  
                                f'exp{exp_no}_{d}_CNTS.npy',
                                f'exp{exp_no}_{d}_CNTS_M1.npy'                            
                                )
                plot(axs[idx], h, d, regrets_dict)
            except:
                try:
                    set_regrets_dict(f'exp1_{h}_CNUCB.npy',                  
                                    f'exp1_{h}_CNTS.npy',
                                    f'exp1_{h}_CNTS_M1.npy'                            
                                    )
                    plot(axs[idx], h, d, regrets_dict)
                except:
                    pass
                pass

        plt.savefig(f'plots/exp{exp_no}')        