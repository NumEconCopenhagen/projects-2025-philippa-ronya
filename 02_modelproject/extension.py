from types import SimpleNamespace

import numpy as np
from scipy import optimize

from Consumer import ConsumerClass
from Government import GovernmentClass

class GreenGovernmentClass(GovernmentClass):
    """ a government that taxes travel because travel pollutes

    Extension of section 4 in two directions:

    1. Every unit of a good emits CO2: e = (e_1,e_2,e_3) per unit, so total
       emissions are E = sum_j e_j x_j. Bus trips are dirty (diesel), train
       trips are much cleaner (electricity), and food is left out of the
       transport externality (e_1 = 0).

    2. The revenue does *not* leave the model any more: it is handed straight
       back to the consumer as a lump-sum rebate, so the tax is revenue neutral
       and only the *relative prices* do any work. Because the consumer's income
       then depends on the revenue and the revenue depends on the consumer's
       income, this is a fixed point, and we find it with a root-finder.

    Social welfare is

        W = u - delta*E

    where delta > 0 is the damage from one unit of emissions, measured in the
    same units as utility.

    """

    def setup(self):

        # a. everything the ordinary government has
        super().setup()

        par = self.par

        # b. emission intensities per unit of each good
        par.e = np.array([0.00,0.10,0.02]) # food, bus, train

        # c. the damage from one unit of emissions, in utility units
        par.delta = 1.0

    ####################
    # 1. emissions     #
    ####################

    def emissions(self,x):
        """ total emissions from a bundle of quantities """

        return np.sum(self.par.e*np.asarray(x))

    ##########################################
    # 2. the equilibrium with a rebate       #
    ##########################################

    def _product_revenue(self,rebate,tau):
        """ the product-tax revenue when the consumer gets `rebate` back """

        par = self.par

        # a. income is I0 + rebate, i.e. a *negative* lump-sum tax
        self.set_taxes(T=-rebate,tau=tau)

        # b. the consumer's choice at those prices and that income.
        #    The tolerances are tighter than the default because this utility
        #    is itself the objective of an *outer* optimizer in
        #    .solve_optimal_tax(): if the inner solve is only accurate to 1e-8,
        #    the outer finite-difference gradient is pure noise.
        opt = self.con.solve(options={'ftol':1e-16,'gtol':1e-14})
        x = np.array(self.con.quantities(opt.s1,opt.w))

        # c. tau_j*p_j*x_j with p_j the pre-tax price
        return np.sum(par.tau*par.p0*x),opt,x

    def solve_with_rebate(self,tau):
        """ the equilibrium when all the revenue is rebated lump-sum

        The rebate has to equal the revenue it itself generates, so we solve

            product_revenue(rebate) - rebate = 0

        with brentq.

        Args:

            tau (ndarray): the three product taxes

        Returns:

            (SimpleNamespace): rebate, utility, emissions, welfare, quantities
            and budget shares in the equilibrium

        """

        par = self.par
        res = SimpleNamespace()

        tau = np.asarray(tau,dtype=float)

        # a. the fixed point in the rebate
        f = lambda rebate: self._product_revenue(rebate,tau)[0]-rebate

        sol = optimize.root_scalar(f,bracket=[0.0,par.I0*10],method='brentq')

        # b. evaluate everything at the fixed point
        rebate,opt,x = self._product_revenue(sol.root,tau)

        res.rebate = sol.root
        res.tau = tau.copy()
        res.u = opt.u
        res.x = x
        res.s = np.array([opt.s1,opt.s2,opt.s3])
        res.E = self.emissions(x)
        res.W = res.u-par.delta*res.E

        # c. leave the consumer untaxed again
        self.set_taxes(T=0.0,tau=np.zeros(3))

        return res

    def rebate_closed_form(self,tau):
        """ the same fixed point in closed form, as a check

        Preferences are homothetic, so the budget shares do not depend on income
        and the revenue is proportional to income:

            R = I_net * sum_j tau_j*s_j/(1+tau_j),   I_net = I0 + R

        """

        tau = np.asarray(tau,dtype=float)

        # a. the shares in the equilibrium (they are income-independent)
        eq = self.solve_with_rebate(tau)

        # b. the implied rebate
        phi = np.sum(tau*eq.s/(1+tau))

        return self.par.I0*phi/(1-phi)

    ##########################################
    # 3. hitting an emission target          #
    ##########################################

    def find_rate_for_emissions(self,cut,direction,bracket=(0.0,20.0)):
        """ the tax rate that cuts emissions by `cut` percent

        Args:

            cut (float): the required reduction, e.g. 0.20 for 20 percent
            direction (ndarray): the direction the single rate is applied in
            bracket (tuple): interval for the root-finder

        Returns:

            (SimpleNamespace): the equilibrium at the required rate

        """

        # a. emissions with no tax at all
        E0 = self.solve_with_rebate(np.zeros(3)).E

        # b. the target
        E_target = (1-cut)*E0

        # c. emissions minus the target, as a function of the rate
        f = lambda rate: self.solve_with_rebate(rate*np.asarray(direction)).E-E_target

        sol = optimize.root_scalar(f,bracket=list(bracket),method='brentq')

        # d. evaluate everything there
        out = self.solve_with_rebate(sol.root*np.asarray(direction))
        out.rate = sol.root
        out.E0 = E0
        out.E_target = E_target

        return out

    ##########################################
    # 4. the welfare-maximizing green tax    #
    ##########################################

    def welfare(self,tau23):
        """ social welfare W = u - delta*E for travel taxes (tau_2,tau_3)

        Food is left untaxed as a normalization: with the revenue rebated, a
        tax on *all three* goods at the same rate changes nothing at all, so
        only the taxes on bus and train relative to food are pinned down.

        """

        tau = np.array([0.0,tau23[0],tau23[1]])

        return self.solve_with_rebate(tau).W

    def solve_optimal_tax(self,tau_max=5.0,s0=None):
        """ the (tau_2,tau_3) that maximizes W, with L-BFGS-B """

        # start from the textbook Pigouvian rate
        if s0 is None: s0 = self.pigou_benchmark()[1:]

        obj = lambda tau23: -self.welfare(tau23)

        # the outer objective is itself the result of two numerical solves, so
        # it is only smooth down to a point: eps must be well above that noise
        res = optimize.minimize(obj,s0,method='L-BFGS-B',
            bounds=((0,tau_max),(0,tau_max)),
            options={'eps':1e-5,'ftol':1e-14,'gtol':1e-12})

        out = self.solve_with_rebate(np.array([0.0,res.x[0],res.x[1]]))
        out.tau2 = res.x[0]
        out.tau3 = res.x[1]
        out.res = res

        return out

    def pigou_benchmark(self):
        """ the textbook Pigouvian rate, delta*e_j/(lambda*p_j)

        lambda is the marginal utility of income, which for these homothetic
        preferences is just u/I evaluated with no taxes.

        """

        par = self.par

        eq0 = self.solve_with_rebate(np.zeros(3))
        lam = eq0.u/par.I0

        return par.delta*par.e/(lam*par.p0)
