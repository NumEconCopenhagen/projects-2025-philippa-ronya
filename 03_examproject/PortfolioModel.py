""" a portfolio with a risky and a safe asset (Problem 3)

Starting point for the exam. The methods raising NotImplementedError are the ones
you should write yourself.

"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'
        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    # the share of wealth in the risky asset after trading, and the amount traded
    def trade(self,theta):
        """ apply the no-trade-band rule

        Args:

            theta (ndarray): share of wealth in the risky asset before trading

        Returns:

            theta_post (ndarray): share of wealth in the risky asset after trading
            traded (ndarray): boolean, True where the portfolio is traded

        """

        par = self.par

        # a. is the portfolio outside the no-trade band?
        traded = np.abs(theta-par.theta_star) > par.Delta

        # b. if traded, go all the way back to the target, otherwise stay put
        theta_post = np.where(traded,par.theta_star,theta)

        return theta_post,traded

    # simulate all N portfolios forward T periods
    def simulate(self,R=None):
        """ simulate all N portfolios forward T periods

        Args:

            R (ndarray,optional): gross returns on the risky asset with shape (N,T).
                If None, they are drawn using draw_returns().

        Stores in self.sim:

            W (ndarray): wealth, shape (N,T+1)
            theta (ndarray): share in the risky asset before trading, shape (N,T+1)
            traded (ndarray): boolean, True where the portfolio is traded, shape (N,T)

        """

        par = self.par
        sim = self.sim

        # a. draw returns if not given
        if R is None:
            R = self.draw_returns()

        Rf = np.exp(par.r)

        # b. allocate containers
        W = np.empty((par.N,par.T+1))
        theta = np.empty((par.N,par.T+1))
        traded = np.empty((par.N,par.T),dtype=bool)

        # c. initial values: everyone starts exactly at the target
        W[:,0] = par.W0
        theta[:,0] = par.theta_star

        # d. loop over time, vectorized over portfolios
        for t in range(par.T):

            # i. trade towards the target if outside the band
            theta_post,traded_t = self.trade(theta[:,t])
            traded[:,t] = traded_t

            # ii. pay the transaction cost on the amount traded
            amount_traded = np.abs(theta_post-theta[:,t])
            W_post = W[:,t]*(1-par.tau*amount_traded)

            # iii. realize returns
            W[:,t+1] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf

            # iv. share in the risky asset going into next period
            theta[:,t+1] = theta_post*W_post*R[:,t]/W[:,t+1]

        # e. store results
        sim.R = R
        sim.W = W
        sim.theta = theta
        sim.traded = traded

    # the numbers to report for a rule, including expected utility
    def summary(self):
        """ compute the six summary numbers for the current simulation

        Returns:

            out (SimpleNamespace) with fields:

                n_trades (float): average number of times a portfolio is traded
                avg_dist (float): average |theta_t - theta_star| before trading
                mean_WT (float): mean of terminal wealth
                median_WT (float): median of terminal wealth
                p10_WT (float): 10th percentile of terminal wealth
                EU (float): expected utility, mean of u(WT)

        """

        par = self.par
        sim = self.sim

        out = SimpleNamespace()

        # a. average number of trades per portfolio
        out.n_trades = np.mean(np.sum(sim.traded,axis=1))

        # b. average distance to target, before trading, over all periods t=0,...,T-1
        out.avg_dist = np.mean(np.abs(sim.theta[:,:-1]-par.theta_star))

        # c. terminal wealth statistics
        WT = sim.W[:,-1]
        out.mean_WT = np.mean(WT)
        out.median_WT = np.median(WT)
        out.p10_WT = np.percentile(WT,10)

        # d. expected utility
        out.EU = np.mean(self.u(WT))

        return out
