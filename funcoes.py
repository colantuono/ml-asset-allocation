import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import yfinance as yf
import scipy.stats
from scipy.optimize import minimize
import math

                        
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())
                         
def annualize_rets(r, periods_per_year=12):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year=12):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate=0.03, periods_per_year=12):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    vol = (weights.T @ covmat @ weights)**0.5
    return vol 

def sortino_ratio(returns, riskfree_rate, target_return=0, periods_per_year=12):
    average_return = np.mean(returns) * periods_per_year 
    downside_returns = returns[returns < target_return]   
    downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)

    sortino_ratio = (average_return - riskfree_rate) / downside_deviation
    return sortino_ratio

def summary_stats(r, riskfree_rate=0.03, precision=5):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    ann_so = r.aggregate(sortino_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    max_dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    avg_dd = r.aggregate(lambda r: drawdown(r).Drawdown.mean())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    return pd.DataFrame({
        "Annualized Return": round(ann_r*100,2),
        "Annualized Vol": round(ann_vol*100,2),
        "Sharpe Ratio": round(ann_sr,precision),
        "Sortino Ratio": round(ann_so,precision),
        'Average Drawdown':round(avg_dd*100,2),
        "Max Drawdown": round(max_dd*100,2),
        "Skewness": round(skew,precision),
        "Kurtosis": round(kurt,precision),
        "Cornish-Fisher VaR (5%)": round(cf_var5*100,2),
        "Historic CVaR (5%)": round(hist_cvar5*100,2),
    })
    
def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def msr(riskfree_rate, er, cov, max_allocation=None):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    if max_allocation == None:
        max_allocation = 1/n*10
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, max_allocation),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))
                         
def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[1]]
    return w/w.sum()

def backtest_ws(r, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert List of weights to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)

def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample

def risk_contribution(w,cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib

def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)

def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)


"""
MINHAS FUNÇÕES
"""  
def weight_msr(returns, max_allocation=None, cov_estimator=sample_cov):
     # Risk-free rate

    mean_returns = annualize_rets(returns)
    cov_matrix = cov_estimator(returns)

    num_assets = len(mean_returns)
    if max_allocation == None:
        max_allocation = 1/num_assets*10
    initial_weights = np.ones(num_assets) / num_assets  # Equal weights to start with

    def negative_sharpe(weights):
        port_return = np.sum(mean_returns * weights)
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return ) / port_stddev
        return -sharpe_ratio

    constraints = (
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}  # Sum of weights = 1
    )

    bounds = tuple((0, max_allocation) for asset in range(num_assets))  # Each weight between 0 and 0.20 (20%)

    optimal_weights = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    ).x

    return optimal_weights

from sklearn.decomposition import PCA
def weight_pca(df, cov_estimator=sample_cov, **kwargs):
    rets = (df - df.mean()) / df.std()
    cov_matrix = cov_estimator(rets)
        
    pca = PCA()
    pca_fitted = pca.fit(cov_matrix)
    
    pca_comp = pca_fitted.components_;
    pc_w = pca_comp/ pca_comp.sum()
    
    stats = summary_stats(pd.DataFrame(pc_w),0).sort_values('Sharpe Ratio', ascending=False)
    max_port = stats.index[stats['Sharpe Ratio'] == stats['Sharpe Ratio'].max()][0]
    res = pc_w[:,max_port] / np.sum(pc_w[:,max_port]) # Normalize to sum to 1
    return res

def weight_eigen(df, cov_estimator=sample_cov, **kwargs):
    rets = (df - df.mean()) / df.std()
    cov_matrix = cov_estimator(rets)
    
    D, S = np.linalg.eigh(cov_matrix)
    
    stats = summary_stats(pd.DataFrame(S),0).sort_values('Sharpe Ratio', ascending=False)
    max_port = stats.index[stats['Sharpe Ratio'] == stats['Sharpe Ratio'].max()][0]
    res = S[:,max_port] / np.sum(S[:,max_port]) # Normalize to sum to 1
    return res

def pipeline(df: pd.DataFrame, training_period: int, oos_period: int, algo: str, show_pesos=False, **kwargs):
    retornos = pd.DataFrame()
    for i, month in enumerate(df.index):
        df = df.dropna(axis='columns')
        if df[(i+training_period+1) : (i+training_period+oos_period+2)].shape[0] > oos_period:
            train_df = df[(i) : (i+training_period+1)].dropna(axis='columns')
            train_df = train_df.pct_change().dropna(axis='rows')
            ## DANGER OF DATA LEAKAGE
            oos_df = df[(i+training_period+1) : (i+training_period+1+oos_period+1)][train_df.columns]
            oos_df = oos_df.pct_change((oos_period)).dropna() 
            # DANGER OF DATA LEAKAGE
            
            pesos_algo = algo(train_df, **kwargs)
            pesos_df = pd.DataFrame(data={'pesos':pesos_algo}, index=train_df.columns).sort_values(by='pesos', ascending=False).T
    
            stock_rets = []  
            retorno_mes = []
            for n, date in enumerate(oos_df.index):
                for stock in oos_df.columns:
                    stock_rets.append(oos_df[stock][n] * pesos_df[stock][n])

                retorno_mes.append(sum(stock_rets))
                retornos[date] = retorno_mes
                
    retornos = retornos.T 
    retornos.rename(columns={0:'rets'}, inplace=True)
    return retornos 



now = dt.datetime.now() 
start = now - dt.timedelta(days=365*10)

def tickers(tickers):
    if isinstance(tickers, str):
        tickers = list(pd.read_csv(tickers.columns))
    stocks = [stock+'.SA' for stock in tickers]    
    return list(stocks)

def download_portfolio(tickers: list, cols="Adj Close", interval='1mo', start=start, end=now, fill = False, drop=False):
    """
    Reads tickers from file, downloads market data using yfinance and removes new tickers 
    """
    portfolio = yf.download(tickers, start=start, end=end, interval=interval)
    if drop:
        portfolio = portfolio.dropna(axis='columns')
    if fill:
        portfolio.fillna(axis=0, method='bfill', inplace=True)
        
    portfolio.drop('Unnamed: 0', axis='columns')
        
    return portfolio[cols]

def returns(prices, window=1):
    return prices.pct_change(window)[1:]
    
def plot_dd(returns):
    drawdown(returns)[['Wealth', 'Peaks']].plot(figsize=(12,5))
    
def underwater_plot(returns, figsize=(12,5)):
    drawdown(returns)['Drawdown'].plot(figsize=figsize)

from bcb import sgs
def get_selic(start_date='2000-01-01', end_date='2022-12-31'):
    # Busca a série da SELIC no SGS
    selic = sgs.get({'selic':4390}, start = start_date, end=end_date) 
    selic = selic.to_period('M')
    return selic / 100

def plot_rets_distribuition(r, benchmark=None, density_size=10, figsize=(20,8)):
    plt.figure(figsize=figsize)
    plt.vlines(r.mean(), 0, density_size, color='r', linestyle='dashed', linewidth=2,label='strat mean')
    plt.vlines(r.median(), 0, density_size, color='orange', linestyle='dashed', linewidth=2, label='strat median')
    if benchmark is not None:
        sns.distplot(benchmark, hist=True, bins=int(len(benchmark)/2), color='g', label='Benchmark')
    sns.distplot(r, hist=True, bins=int(len(r)/2), color='b', label='Strategy')
    plt.title("Returns Distribution")
    plt.legend()
    plt.show;
