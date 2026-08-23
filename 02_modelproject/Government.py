from types import SimpleNamespace

import numpy as np
from scipy import optimize

from Consumer import ConsumerClass

# the five product taxes used in section 4, written as the vector (tau_1,tau_2,tau_3)
# that a single rate tau is multiplied by
TAX_DIRECTIONS = {
    'food only'      : np.array([1.0,0.0,0.0]),
    'bus only'       : np.array([0.0,1.0,0.0]),
    'train only'     : np.array([0.0,0.0,1.0]),
    'bus and train'  : np.array([0.0,1.0,1.0]),
    'all three goods': np.array([1.0,1.0,1.0]),
}

class GovernmentClass:
    """ a government that taxes the consumer of ConsumerClass

    The government has two instruments:

        T      : a lump-sum tax, which lowers income from I to I-T
        tau    : three product taxes, which raise the price of good j
                 from p_j to (1+tau_j)*p_j

    Total revenue is

        R = T + sum_j tau_j * p_j * x_j                                  (eq. 5)

    where p_j is the *pre-tax* price and x_j is what the consumer buys
    *given the taxes*. The revenue leaves the model: the consumer gets
    nothing back.

    All the government does is translate (T,tau) into the prices and the income
    that the consumer faces (see .set_taxes()). Everything written in sections
    1-3 therefore keeps working unchanged: the consumer object is exactly the
    same object, only with different numbers in par.p1, par.p2, par.p3 and par.I.

    """

    def __init__(self,consumer=None,par=None):

        # a. the consumer being taxed (default: the complements calibration)
        self.con = ConsumerClass() if consumer is None else consumer

        # b. setup
        self.setup()

        # c. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ store the pre-tax prices and income, and start with no taxes """

        par = self.par = SimpleNamespace()
        con = self.con

        # a. the pre-tax ("producer") prices and the pre-tax income.
        #    These never change again -- they are what the taxes are applied to.
        par.p0 = np.array([con.par.p1,con.par.p2,con.par.p3])
        par.I0 = con.par.I

        # b. the current taxes
        par.T = 0.0
        par.tau = np.zeros(3)

        # c. start from a clean slate
        self.set_taxes(T=0.0,tau=np.zeros(3))

    def __str__(self):

        par = self.par

        lines = ['GovernmentClass']
        lines.append(f'  pre-tax prices = {par.p0}, pre-tax income = {par.I0:.4f}')
        lines.append(f'  T = {par.T:.4f}, tau = {par.tau}')
        lines.append('  consumer faces:')
        lines.append('    '+str(self.con).replace('\n','\n    '))

        return '\n'.join(lines)

    ##################
    # 1. the taxes   #
    ##################

    def set_taxes(self,T=0.0,tau=None):
        """ translate the taxes into the prices and the income the consumer faces

        Args:

            T (float): lump-sum tax
            tau (ndarray): the three product taxes, defaults to no product taxes

        Returns:

            None, but self.con.par is updated

        """

        par = self.par
        con = self.con

        if tau is None: tau = np.zeros(3)
        tau = np.asarray(tau,dtype=float)

        assert np.all(tau >= 0), 'the product taxes must be non-negative'
        assert T < par.I0, 'the lump-sum tax cannot take more than the whole income'

        # a. remember the taxes
        par.T = T
        par.tau = tau

        # b. consumer prices: p_j -> (1+tau_j)*p_j
        con.par.p1 = (1+tau[0])*par.p0[0]
        con.par.p2 = (1+tau[1])*par.p0[1]
        con.par.p3 = (1+tau[2])*par.p0[2]

        # c. consumer income: I -> I-T
        con.par.I = par.I0-T

    def rate_to_taxes(self,tau_rate,direction):
        """ one rate applied in one of the five directions in TAX_DIRECTIONS """

        if isinstance(direction,str): direction = TAX_DIRECTIONS[direction]

        return tau_rate*np.asarray(direction,dtype=float)

    ####################
    # 2. tax revenue   #
    ####################

    def tax_revenue(self,T=0.0,tau=None,s0=None,**kwargs):
        """ the revenue raised by (T,tau), eq. 5

        The consumer is re-solved at the taxed prices and the taxed income,
        because x* in eq. 5 is what the consumer buys *given the taxes*.

        Args:

            T (float): lump-sum tax
            tau (ndarray): the three product taxes
            s0 (ndarray): starting guess for the consumer problem

        Returns:

            (SimpleNamespace): revenue R, utility u, quantities and budget shares

        """

        par = self.par
        con = self.con
        res = SimpleNamespace()

        # a. let the consumer face the taxes
        self.set_taxes(T=T,tau=tau)

        # b. solve the consumer problem at those prices and that income
        opt = con.solve(s0=s0,**kwargs)

        # c. the quantities bought given the taxes
        x = np.array(con.quantities(opt.s1,opt.w))

        # d. revenue: the lump-sum tax plus tau_j*p_j*x_j, with p_j the pre-tax price
        R = T + np.sum(par.tau*par.p0*x)

        # e. results
        res.R = R
        res.u = opt.u
        res.x = x
        res.s = np.array([opt.s1,opt.s2,opt.s3])
        res.s1 = opt.s1
        res.w = opt.w
        res.T = T
        res.tau = par.tau.copy()

        return res

    def revenue_curve(self,tau_rates,direction):
        """ revenue and utility along a grid of rates in one tax direction

        Args:

            tau_rates (ndarray): grid of tax rates
            direction (str or ndarray): one of the keys of TAX_DIRECTIONS

        Returns:

            (tuple): revenue and utility, both arrays of the same length as tau_rates

        """

        R = np.empty(tau_rates.size)
        u = np.empty(tau_rates.size)

        for i,tau_rate in enumerate(tau_rates):
            res = self.tax_revenue(T=0.0,tau=self.rate_to_taxes(tau_rate,direction))
            R[i] = res.R
            u[i] = res.u

        # leave the consumer untaxed again
        self.set_taxes(T=0.0,tau=np.zeros(3))

        return R,u

    def lump_sum_curve(self,T_values):
        """ revenue and utility along a grid of lump-sum taxes """

        R = np.empty(T_values.size)
        u = np.empty(T_values.size)

        for i,T in enumerate(T_values):
            res = self.tax_revenue(T=T,tau=np.zeros(3))
            R[i] = res.R
            u[i] = res.u

        self.set_taxes(T=0.0,tau=np.zeros(3))

        return R,u

    def max_revenue(self,direction,tau_max=20.0,N=2000):
        """ the top of the Laffer curve in one tax direction, by a grid search

        Exactly as in section 2.1: lay a grid over the tax rate, evaluate the
        revenue in every point and keep the best one.

        Args:

            direction (str or ndarray): one of the keys of TAX_DIRECTIONS
            tau_max (float): the largest rate on the grid
            N (int): number of grid points

        Returns:

            (SimpleNamespace): the best rate, the revenue there, and whether the
            maximum is interior (a real top) or at the end of the grid

        """

        opt = SimpleNamespace()

        tau_rates = np.linspace(0.0,tau_max,N)
        R,u = self.revenue_curve(tau_rates,direction)

        i = np.argmax(R)

        opt.tau_rates = tau_rates
        opt.R_grid = R
        opt.tau_star = tau_rates[i]
        opt.R_max = R[i]
        opt.interior = (i > 0) and (i < N-1)

        return opt

    #####################
    # 3. root-finding   #
    #####################

    def find_rate(self,R_target,direction,bracket=(0.0,5.0)):
        """ the tax rate that raises exactly R_target in one tax direction

        Args:

            R_target (float): the revenue the government needs
            direction (str or ndarray): one of the keys of TAX_DIRECTIONS
            bracket (tuple): interval to look in, revenue-minus-target must
                change sign between the two end points

        Returns:

            (SimpleNamespace): the rate, the revenue, the utility and the
            quantities at that rate

        """

        # a. revenue minus the target, as a function of the rate
        def f(tau_rate):
            res = self.tax_revenue(T=0.0,tau=self.rate_to_taxes(tau_rate,direction))
            return res.R-R_target

        # b. find the root
        sol = optimize.root_scalar(f,bracket=list(bracket),method='brentq')

        # c. evaluate everything at the solution
        out = self.tax_revenue(T=0.0,tau=self.rate_to_taxes(sol.root,direction))
        out.rate = sol.root
        out.converged = sol.converged
        out.instrument = direction if isinstance(direction,str) else str(direction)

        self.set_taxes(T=0.0,tau=np.zeros(3))

        return out

    def find_lump_sum(self,R_target):
        """ the lump-sum tax that raises exactly R_target

        It is of course T = R_target, but we find it with the same root-finder
        so that all six instruments are treated in the same way.

        """

        def f(T):
            return self.tax_revenue(T=T,tau=np.zeros(3)).R-R_target

        sol = optimize.root_scalar(f,bracket=[0.0,self.par.I0*0.99],method='brentq')

        out = self.tax_revenue(T=sol.root,tau=np.zeros(3))
        out.rate = sol.root
        out.converged = sol.converged
        out.instrument = 'lump-sum'

        self.set_taxes(T=0.0,tau=np.zeros(3))

        return out
