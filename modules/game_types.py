import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Callable
from abc import ABC, abstractmethod

class PriceData:
    def __init__(self, instrument :str, start_date:str, end_date:str, re_sample:str="W-THU") -> None:
        self._data = self.fetch_and_clean_data(instrument, start_date, end_date, re_sample)
        self._close = self._data['Close'].copy()
        self._changes = self._data['changes'].copy()

    def fetch_and_clean_data(self, ins:str, start:str, end:str, re_sample:str) -> pd.DataFrame:
        df = yf.download(ins, start=start, end=end) ; 
        df = df.drop(["Open", "High", "Low", "Adj Close", "Volume"],axis=1)
        df = df.resample(re_sample).last()
        df['changes'] = df['Close'] / df['Close'].shift(1)
        clean_data = df.dropna()
        return clean_data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def price_changes(self) -> pd.Series:
        return self._changes

    # add hurst coefficient property, other autocorrelation properties to evaluate distance from random walk

    def plot(self, width:float=12 , height:float=5) -> None:
        plt.style.use('ggplot');
        fig , ax = plt.subplots(2,1,figsize=(width,height), sharex=True);
        ax[0].plot(self._close.index, self._close.values)
        ax[0].set_ylabel('Closing Price History')
        ax[1].plot(self._changes.index, self._changes.values)
        ax[1].set_ylabel('Price Return History')
        fig.tight_layout();  
        return None


class GenericGame(ABC):
    def __init__(self) -> None:
        self._prob_space = 0.0
        self._retn_space = 0.0
    
    @abstractmethod
    def compute_probs_retns(self):
        pass

    @property
    def game_setup(self):
        return self._prob_space , self._retn_space 

    def compute_EV(self) -> float:
        return np.sum(self._prob_space * self._retn_space)

    def compute_BEV(self) -> float:
        return np.exp(np.sum(self._prob_space * np.log(self._retn_space)))


class CoinTossGame(GenericGame):
    def __init__(self, head_prob:float, rewd:np.ndarray) -> None:
        super().__init__()
        self.param_1 = head_prob
        self.param_2 = rewd
        self._prob_space , self._retn_space = self.compute_probs_retns()
        
    def compute_probs_retns(self) -> tuple[np.ndarray]:
        return np.array([self.param_1, (1.0-self.param_1)]) , self.param_2


class FinancialTimeSeries(GenericGame):
    def __init__(self, time_series:PriceData, nbins:int) -> None:
        super().__init__()
        self._nbins = nbins
        self._ts = time_series.price_changes 
        self._prob_space , self._retn_space = self.compute_probs_retns()

    def compute_probs_retns(self) -> tuple[np.ndarray]:
        return_freqs , edges = np.histogram(self._ts.values, bins=self._nbins)
        bin_centroids = 0.5*(edges[1:] + edges[:-1])
        return_space = bin_centroids[(return_freqs > 0.)]
        prob_space = (return_freqs[(return_freqs > 0.)]).astype(float)
        prob_space /= np.sum(prob_space)
        return prob_space , return_space