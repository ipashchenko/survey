import sys
import numpy as np
from scipy.stats import halfcauchy
from utils import flux, mas_to_rad, rad_to_mas, ed_to_uv, get_ratio_hdi,\
    partition_baselines_
from load_data import get_baselines_s_threshold, \
    get_detection_fractions_in_baseline_ranges


class Simulation(object):
    def __init__(self, mu_logs, mu_loga, beta_e, bsls_s_thrs_statuses,
                 std_loga=None, std_logs=None, hc_scale=0.2, alpha_e=5.0):
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
        :param bsls_s_thrs_statuses:
            Array-like of (baseline, threshold flux, status) [ED, Jy, y/n].
        :param std_logs: (optional)
            2 values for min & max of uniform prior range on std of log(full
            flux) [log(Jy)]. If ``None`` then use half-cauchy prior.
        :param std_loga: (optional)
            2 values for min & max of uniform prior range on std of log(major
            axis) [log(mas)]. If ``None`` then use half-cauchy prior.
        :param hc_scale: (optional)
            Scale parameter for half-cauchy prior.
        :param alpha_e: (optional)
            Value of alpha-parameter of axis ratio beta distribution.
        """
        self.mu_loga = mu_loga
        self.std_loga = std_loga
        self.mu_logs = mu_logs
        self.std_logs = std_logs
        self.beta_e = beta_e
        self.alpha_e = alpha_e
        self.hc_scale = hc_scale
        self.bsls_s_thrs_statuses = bsls_s_thrs_statuses
        self._p = []
        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.
        """
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        """
        Set the state of the random number generator.
        """
        try:
           self._random.set_state(state)
        except:
           pass

    # TODO: don't need to supply fr_list
    def run(self, n_acc, bsls_borders=None, rstate0=None):
        """
        Run simulation till ``n_acc`` parameters are accepted.
        :param n_acc:
            Accepted number of parameters to stop simulation.
        :param bsls_borders:
            Array-like of borders for baseline ranges. Fractions will be
            compared in intervals [bsls_borders[0], bsls_borders[1]],
            [bsls_borders[1], bsls_borders[2]], ..., [bsls_borders[n-1],
            bsls_borders[n]]. Length must be ``len(fr_list) + 1``.
        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

        :notes:
            Size of samples used to count acceptance fractions is determined by
            number of baselines in each baseline interval.
            List of n lists of parameters [``mu_loga``, ``std_loga``,
            ``mu_logs``, ``std_logs``, ``beta_e``] that generate samples of
            sources that give detection fractions equal to observed ones within
            specified tolerance.
        """
        # Assertions
        fr_list = get_detection_fractions_in_baseline_ranges(self.bsls_s_thrs_statuses,
                                                             bsls_borders)
        assert(len(fr_list) == len(bsls_borders) - 1)
        # Initialize counting variable
        n = 0

        if rstate0 is None:
            rstate0 = self.random_state

        # Partition baselines in ranges
        bsls_s_thrs_partitioned = partition_baselines_(self.bsls_s_thrs_statuses[['bl', 's_thr']],
                                                       bsls_borders)
        while n <= n_acc:
            # Use custom generator for pa generation
            self.random_state = rstate0
            print "Accepted up to now : " + str(self.p)
            params = self.draw_parameters()
            print "Trying parameters " + str(params)
            # For each range of baselines check summary statistics
            for i, bsls_s_thrs in enumerate(bsls_s_thrs_partitioned):

                n_ = len(bsls_s_thrs)
                # Simulate ``n_`` random positional angles for baselines
                # Use the same seed to get the same pa at each iteration
                pa = self._random.uniform(0., np.pi, size=n_)
                baselines = ed_to_uv(bsls_s_thrs['bl'])
                s_thrs = bsls_s_thrs['s_thr']
                sample = self.create_sample(params, size=n_)
                det_fr = self.observe_sample(sample, baselines, pa, s_thrs)
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

    def observe_sample(self, sample, baselines, pa, s_thrs):
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
            Numpy array of threshold detection flux on each baseline [Jy].
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
        n_det = len(np.where(fluxes > 5. * s_thrs)[0])
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

    #if sys.argv[1] == 'c':
    #    band = 'c'
    #elif sys.argv[1] == 'l':
    #    band = 'l'
    #else:
#        sys.exit('USE c OR l!')
    band = 'c'
    print "Using " + band + "-band"
    bsls_s_thrs_statuses = get_baselines_s_threshold(band)

    bsls_borders = [2., 5., 10., 17., 30.]

    print "Using " + str(bsls_borders)
    simulation = Simulation([-4., -0.], [-6., 0.], [1., 35.],
                            bsls_s_thrs_statuses)
    simulation.run(100, bsls_borders=bsls_borders)
