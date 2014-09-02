import numpy as np
#from numpy.random import RandomState
from scipy.stats import halfcauchy
from utils import flux, mas_to_rad, rad_to_mas, ed_to_uv, get_ratio_hdi
from load_data import get_baselines_exper_averaged,\
    get_detection_fraction_in_baseline_range


class Simulation(object):
    def __init__(self, mu_logs, mu_loga, beta_e, bsls, std_loga=None,
                 std_logs=None, hc_scale=0.2, alpha_e=5.0):
        """
        Class that implements simulation of RA survey.

        :param mu_loga:
            2 values for min & max of uniform prior range on mean of log(major
            axis) [log(mas)].
        :param mu_logs:
            2 values for min & max of uniform prior range on mean of log(full
            flux) [log(Jy)].
        :param beta_e:
            2 values for min & max of ``beta_e`` prior range of beta-parameter
            of axis ratio beta distribution.
        :param bsls:
            Array-like of baselines [ED].
        :param std_logs (optional):
            2 values for min & max of uniform prior range on std of log(full
            flux) [log(Jy)]. If ``None`` then use half-cauchy prior.
        :param std_loga (optional):
            2 values for min & max of uniform prior range on std of log(major
            axis) [log(mas)]. If ``None`` then use half-cauchy prior.
        :param hc_scale (optional):
            Scale parameter for half-cauchy prior.
        :param alpha_e (optional):
            Value of alpha-parameter of axis ratio beta distribution.
        """
        self.mu_loga = mu_loga
        self.std_loga = std_loga
        self.mu_logs = mu_logs
        self.std_logs = std_logs
        self.beta_e = beta_e
        self.alpha_e = alpha_e
        self.hc_scale = hc_scale
        self.bsls = np.asarray(bsls)
        self._p = []

    def run(self, n_acc, fr_list, bsls_borders=None, s_thr=0.05):
        """
        Run simulation till ``n_acc`` parameters are accepted.
        :param n_acc:
            Accepted number of parameters to stop simulation.
        :param fr_list
            List of lists of observed fractions. Each list contains low and high
            point of corresponding HDI interval.
        :param bsls_borders:
            Array-like of borders for baseline ranges. Fractions will be
            compared in intervals [bsls_borders[0], bsls_borders[1]],
            [bsls_borders[1], bsls_borders[2]], ..., [bsls_borders[n-1],
            bsls_borders[n]]. Length must be ``len(fr_list) + 1``.
        :param s_thr:
            Flux detection threshold [Jy].

        :notes:
            Size of samples used to count acceptance fractions is determined by
            number of baselines in each baseline interval.

        :return:
            List of n lists of parameters [``mu_loga``, ``std_loga``,
            ``mu_logs``, ``std_logs``, ``beta_e``] that generate samples of
            sources that give detection fractions equal to observed ones within
            specified tolerance.
        """
        np.random.seed(123)
        # Assertions
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
            params = self.draw_parameters()
            print "Trying parameters " + str(params)
            # For each range of baselines check summary statistics
            for i, baselines in enumerate(bsls_partitioned):

                n_ = len(baselines)
                # Simulate ``n_`` random positional angles for baselines
                pa = np.random.uniform(0., np.pi, size=n_)
                baselines = ed_to_uv(baselines)
                sample = self.create_sample(params, size=n_)
                det_fr = self.observe_sample(sample, baselines, pa, s_thr)
                print "Got detection fraction " + str(det_fr)
                # If fail to get right fraction in this range then go to next
                # loop of while
                if (det_fr < fr_list[i][0]) or (det_fr > fr_list[i][1]):
                    # If we got stuck here - then reject current parameters and
                    # got to next ``while``-loop
                    print str(det_fr) + " not in HDI : " + str(fr_list[i])
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

    def draw_parameters(self):
        """
        Draw parameters from priors specified in constructor.
        """
        mu_loga = np.random.uniform(self.mu_loga[0], self.mu_loga[1])
        if self.std_loga is None:
            std_loga = float(halfcauchy.rvs(scale=self.hc_scale, size=1))
        else:
            std_loga = np.random.uniform(self.std_loga[0], self.std_loga[1])
        mu_logs = np.random.uniform(self.mu_logs[0], self.mu_logs[1])
        if self.std_logs is None:
            std_logs = float(halfcauchy.rvs(scale=self.hc_scale, size=1))
        else:
            std_logs = np.random.uniform(self.std_logs[0], self.std_logs[1])
        beta_e = np.random.uniform(self.beta_e[0], self.beta_e[1])
        return mu_logs, std_logs, mu_loga, std_loga, beta_e

    def create_sample(self, parameters, size):
        """
        Create sample os sources (sizes, full fluxes & axis ratios) with size =
        ``size`` using parameters of distributions, specified in ``parameters``.
        :param parameters:
            Array-like of ``mu_loga``, ``std_loga``, ``mu_logs``, ``std_logs``,
            ``beta_e``.
        :param size:
            Size of sample to generate.
        :return:
            Numpy arrays of size, full flux & axis ratios each of size =
            ``size``.
        """
        mu_logs, std_logs, mu_loga, std_loga, beta_e = parameters
        # Draw sample of size ``size`` from distributions with parameters
        loga = np.random.normal(mu_loga, std_loga, size=size)
        logs = np.random.normal(mu_logs, std_logs, size=size)
        e = np.random.beta(self.alpha_e, beta_e, size=size)
        return np.exp(logs), np.exp(loga), e

    def observe_sample(self, sample, baselines, pa, s_thr):
        """
        Test ``sample`` of sources for detection fraction on ``baselines`` with
        positional angles ``pa``.
        :param sample:
            Array-like of (total flux, size, axis ratio) numpy arrays.
        :param baselines:
            Numpy array of baseline length [lambda].
        :param pa:
            Numpy array of baseline PA [rad].
        :param s_thr:
            Threshold detection flux on each baseline [Jy].
        :return:
            Detection fraction.
        """
        s, a, e = sample
        print "a before " + str(a[::30])
        a *= mas_to_rad
        print "a after" + str(a[::30])
        print "s " + str(s[::30])
        n = len(baselines)
        fluxes = flux(baselines, pa, s, a, e)
        print "Got fluxes " + str(fluxes[::30])
        print "On baselines " + str(baselines[::30])
        n_det = len(np.where(fluxes > s_thr)[0])
        return float(n_det) / n

    def reset(self):
        self._p = list()

    @property
    def p(self):
        return np.atleast_2d(self._p)

    @p.setter
    def p(self, value):
        self._p = value


if __name__ == '__main__':

    fname = '/home/ilya/Dropbox/survey/exp_bsl_st.txt'
    bsls = get_baselines_exper_averaged(fname)
    bsls_borders = [5., 10., 20., 30.]
    fractions = get_detection_fraction_in_baseline_range(fname, bsls_borders)
    frac_list = list()
    for frac in fractions:
        hdi0, hdi1 = get_ratio_hdi(frac[0], frac[1])
        frac_list.append([hdi0, hdi1])
    print "Using fractions :"
    print frac_list

    simulation = Simulation([-2., -0.], [-3., 0.], [1., 25.], bsls)
    simulation.run(100, frac_list, bsls_borders=bsls_borders, s_thr=0.03)

