import numpy as np
#from numpy.random import RandomState
from utils import flux, mas_to_rad, rad_to_mas, ed_to_uv



# TODO: First, use the same ``loga``, ``beta_e`` (geometry) & ``logs`` for all
# sources, the the same geometry but distribution of ``logs`` and then
# distributions for all.
class Simulation(object):
    def __init__(self, loga, logs, beta_e, bsls, alpha_e=2.):
        """
        Class that implements simulation of RA survey.

        :param loga:
            2 values for min & max of ``loga``/``mu_loga`` prior range
            [log(mas)].
        :param logs:
            2 values for min & max of ``logs``/``mu_logs`` prior range
            [log(Jy)].
        :param beta_e:
            2 values for min & max of beta-parameter prior range for beta
            function.
        :param bsls:
            Array-like of baselines [ED].
        :param alpha_e (optional):
            Alpha-parameter of beta function.
        """
        self.loga = loga
        self.logs = logs
        self.beta_e = beta_e
        self.bsls = np.asarray(bsls)
        self.alpha_e = alpha_e
        self._p = []

    def run1(self, n_acc, fr_list, tol_list, bsls_borders=None, s_thr=0.05):
        """
        Run simulation with same values of ``loga``, ``beta_e`` & ``logs`` till
        ``n_acc`` parameters are accepted.
        :param n_acc:
            Accepted number of parameters to stop simulation.
        :param fr_list
            Array-like of observed fractions.
        :param tol_list:
            Array-like of tolerances. The same length as ``fr_list``.
        :param bsls_borders:
            Array-like of borders for baseline ranges. Fractions will be
            compared in intervals [bsls_borders[0], bsls_borders[1]],
            [bsls_borders[1], bsls_borders[2]], ..., [bsls_borders[n-1],
            bsls_borders[n]]. Length must be ``len(fr_list) + 1``.
        :param s_thr:
            Flux detection threshold [Jy].

        :notes:
            Size of samples used to count acceptance fractions is determined by
            number of baselines used (``self.bsls``)

        :return:
            List of n lists of parameters [``loga``, ``beta_e``, ``logs``] that
            give detection fractions equal to observed within tolerance.
        """
        np.random.seed(123)
        # Assertions
        assert(len(fr_list) == len(tol_list))
        assert(len(fr_list) == len(bsls_borders) - 1)
        # Initialize counting variable
        n = 0

        # Partition baselines in ranges
        bsls_partitioned = list()
        for i in range(len(bsls_borders) - 1):
            bsls_partitioned.append(self.bsls[np.where((self.bsls > bsls_borders[i]) &
                                             (self.bsls < bsls_borders[i + 1]))[0]])
        print bsls_partitioned
        print [len(part) for part in bsls_partitioned]
        while n <= n_acc:
            print "Accepted up to now : " + str(self.p)
            # Simulate ``n_bsls`` sources
            # First simulate one value
            loga = np.random.uniform(self.loga[0], self.loga[1])
            logs = np.random.uniform(self.logs[0], self.logs[1])
            beta_e = np.random.uniform(self.beta_e[0], self.beta_e[1])
            e = np.random.beta(self.alpha_e, beta_e)
            # Save in list
            params = [loga, beta_e, logs]
            print "Trying parameters " + str(params)
            # For each range of baselines check summary statistics
            for i, baselines in enumerate(bsls_partitioned):
                n_ = len(baselines)
                print "n_ is " + str(n_)
                # Calculate detection fraction in this baseline range
                # Repeat to make ``n_`` sources with the same parameters
                loga_ = np.asarray(loga).repeat(n_)
                logs_ = np.asarray(logs).repeat(n_)
                e_ = np.asarray(e).repeat(n_)
                # Simulate ``n_`` random positional angles for baselines
                pa = np.random.uniform(0., np.pi, size=n_)
                print "Got pa " + str(pa)
                baselines = ed_to_uv(baselines)
                print "Calculating flux for source with :"
                print "S = " + str(np.exp(logs_[0])) + " Jy"
                print "a = " + str(np.exp(loga_[0])) + " mas"
                print "e = " + str(e_)
                print "On baselines " + str(baselines)
                print "lengths : "
                print len(baselines), len(pa), len(logs_), len(loga_), len(e_)
                fluxes = flux(baselines, pa, np.exp(logs_),
                              mas_to_rad * np.exp(loga_), e_)
                print "Got fluxes " + str(fluxes)
                n_det = len(np.where(fluxes > s_thr)[0])
                det_fr = float(n_det) / n_
                print "Got detection fraction " + str(det_fr)
                # If fail to get right fraction in this range then go to next
                # loop of while
                if abs(det_fr - fr_list[i]) > tol_list[i]:
                    # If we got stuck here - then reject current parameters and
                    # got to next ``while``-loop
                    print "Rejecting parameter!"
                    break
            # This ``else`` is part of ``for``-loop
            else:
                # If ```for``-list is exhausted, then keep current parameters
                # because the fractions in all baseline ranges are within
                # tolerance of the observed.
                print "This parameter is accepted!"
                n += 1
                self._p.append(params)

#    def run(self, n_acc, fr_list, tol_list, s_thr=0.05, size=10 ** 4.):
#        """
#        Run simulation till ``n_acc`` accepted.
#        """
#        # Initialize counting variable
#        fr_array = np.asarray(fr_list)
#        tol_array = np.asarray(tol_list)
#        n = 0
#        while n < n_acc:
#            # Sample from priors
#            mu_loga = np.random.unif(self.loga[0], self.loga[1])
#            std_loga = np.random.gamma(0.1, 0.1)
#            mu_logs = np.random.unif(self.logs[0], self.logs[1])
#            std_logs = np.random.gamma(0.1, 0.1)
#            beta_e = np.random.unif(self.beta_e[0], self.beta_e[1])
#            fractions = list()
#            for fr in fr_list:
#                fractions.append([detection_fraction(self.loga, self.std_loga,
#                                                self.beta_e, self.logs,
#                                                self.std_logs, bsls,
#                                                self.alpha_e, s_thr=s_thr,
#                                                size=size) for bsls in
#                                  self.bsls_list])
#                fractions = np.asarray(fractions)
#            boolean = [abs(frac - fr)]
#            if abs(frac - fr) <= tol:
#                p = [mu_loga, std_loga, mu_logs, std_logs, beta_e]
#                print "Accepted parameters: " + str(p)
#                self.p.append(p)
#                n += 1
#            else:
#                print "Rejected parameters: " + str(p)

    def reset(self):
        self._p = list()

    @property
    def p(self):
        return np.atleast_2d(self._p)

    @p.setter
    def p(self, value):
        self._p = value


if __name__ == '__main__':

    simulation = Simulation([-3., -0.], [-2., 0.], [1., 5.],
                            np.arange(0.1, 30, 0.001))
    bsls_borders = [10., 20., 25.]
    fr_list = [0.2, 0.05]
    tol_list = [0.05, 0.02]
    simulation.run1(100, fr_list, tol_list, bsls_borders=bsls_borders,
                    s_thr=0.02)
    pass


#    lines = [line.strip() for line in open('exp_bsl_st.txt')]
#    data = list()
#    for line in lines:
#       data.extend([line.split()])
#    exp_name, base_ed, status = zip(*data)
#    exp_names_u, indxs = np.unique(exp_name, return_index=True)
#    baselines = []
#    for ind, exp in zip(indxs, exp_names_u):
#        baselines.append(data[ind][1])
#    status = []
#    for ind, exp in zip(indxs, exp_names_u):
#        status.append(data[ind][2])
#    baselines = [float(bsl) for bsl in baselines]
#    # Resample to get "more data"
#    big = np.random.choice(baselines, size=10000)
#    bsl_kde = gaussian_kde(big)
