import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from game_types import GenericGame

class MonteCarloSimulator:
    
    def __init__(self, game:GenericGame) -> None:
        self._game = game 
        self._prob_space , self._retn_space = self.initialize_game()
        self.simulated_paths = None
        self.final_returns = None

    def initialize_game(self):
        return self._game.game_setup 

    def compute_std_return(self) -> float:
        return np.std(self.final_returns)

    def compute_avg_return(self) -> float:
        return np.mean(self.final_returns)

    def compute_median_return(self) -> float:
        return np.median(self.final_returns)

    def compute_percentile_return(self, pctile) -> float:
        return np.percentile(self.final_returns,pctile)

    def compute_mode_return(self) -> float:
        return_freqs , edges = np.histogram(self.final_returns, bins='auto')
        bin_centroids = 0.5*(edges[1:] + edges[:-1])
        return_space = bin_centroids[(return_freqs > 0.)]
        prob_space = (return_freqs[(return_freqs > 0.)]).astype(float)
        mode_idx = np.argmax(prob_space)
        return np.mean(return_space[mode_idx]) 

    def __call__(self, frac:float, hold_reward_func:Callable[[float],float], 
                 n_paths:int=1000, n_periods:int=100, rng_seed:int=42) -> None:
        sequences = np.zeros((n_paths, n_periods))
        sequences[:,0] = 1.0

        def modify_reward(frac:float, g:Callable[[float],float], reward:float) -> float:
            return (frac*reward) + ((1.0-frac) * g(reward))

        modified_returns = np.array([ modify_reward(frac, hold_reward_func, x) for x in self._retn_space ])
        
        np.random.seed(rng_seed)
        sequences[:,1:] = np.random.choice(modified_returns, size=(n_paths,n_periods-1), p=self._prob_space)
        cumulative_returns = np.cumprod(sequences, axis=1)
        final_returns = np.prod(sequences, axis=1)

        self.simulated_paths , self.final_returns = cumulative_returns , final_returns 
        return None        

    def plot_paths(self, nbins:int=30, size:tuple[int]=(15,5)) -> None:
        n_paths = len(self.final_returns);
        n_periods = len(self.simulated_paths[0,:]) 

        mean = self.compute_avg_return()
        median = self.compute_median_return()
        mode = self.compute_mode_return()    
        btm_pct = np.percentile(self.final_returns,5.0)

        plt.style.use('ggplot');
        fig , ax = plt.subplots(1,2, figsize=size);
        ax[0].set_title(f'{n_paths} Simulated Paths');
        ax[0].semilogy(self.simulated_paths.T, alpha=0.6);
        ax[0].set_xlim(0,n_periods);
        ax[0].set_xlabel('# periods');
        ax[0].set_ylabel('Evolution of 1 unit');

        ax[1].set_title(f'Distribution of Returns after {n_periods} periods');
        ax[1].hist(self.final_returns, bins=nbins, label=f'Mean={mean:.1f}, Median={median:.1f},Mode={mode:.1f}, Bottom 5%ile = {btm_pct:.1f}'); 
        ax[1].set_xlabel('Return');
        ax[1].set_ylabel('# of paths');
        ax[1].legend();

        
        fig.tight_layout();
        return None  
        