import sys
import math
import numpy as np
import cPickle
from scipy.stats import halfcauchy, halfnorm
from utils import flux, mas_to_rad, rad_to_mas, ed_to_uv, get_ratio_hdi, \
    partition_baselines_, flux_
from load_data import get_baselines_s_threshold, \
    get_detection_fractions_in_baseline_ranges_


class Simulation(object):
    def __init__(self, mu_logs, mu_logtb, bsls_s_thrs_statuses,
                 std_logs=None, std_logtb=None):
        """
        Class that implements simulation of RA survey.

        :param mu_logs:
            2 values for min & max of uniform prior range on mean of log(full
            flux) [log(Jy)].
        :param mu_logtb:
            2 values for min & max of uniform prior range on mean of log(Tb)
            [log(K)].
        :param bsls_s_thrs_statuses:
            Array-like of (baseline, threshold flux, status) [ED, Jy, y/n].
        :param std_logs: (optional)
            2 values for min & max of uniform prior range on std of log(full
            flux) [log(Jy)].
        :param std_logtb: (optional)
            2 values for min & max of uniform prior range on std of log(Tb)
            [log(K)].
        """
        self.mu_logs = mu_logs
        self.std_logs = std_logs
        self.mu_logtb = mu_logtb
        self.std_logtb = std_logtb
        self.bsls_s_thrs_statuses = bsls_s_thrs_statuses
        self._p = []
        # List for keeping fractions for accepted parameters
        self._summary = []
        # List for keeping distance to data summary statistics for accepted
        # parameters
        self._d = list()
        self._data_sum = None
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
    def run(self, n_acc, eps, bsls_borders=None, rstate0=None):
        """
        Run simulation till ``n_acc`` parameters are accepted.
        :param n_acc:
            Accepted number of parameters to stop simulation.
        :param eps:
            Tolerance for acceptance.
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
            number of baselines in each baseline interval. List of n lists of
            parameters [``mu_logs``, ``std_logs``, ``mu_logtb``, ``std_logtb``]
            that generate samples of sources that give detection fractions equal
            to observed ones within specified tolerance.
        """
        # Calculate detection fractions in given baseline ranges
        fr_list = get_detection_fractions_in_baseline_ranges_(self.bsls_s_thrs_statuses,
                                                              bsls_borders)
        # Save detection fractions as data summary statistics
        self._data_sum = np.array(fr_list)
        print "Data summary statistics : ", self._data_sum
        # Assertion on consistency
        assert(len(fr_list) == len(bsls_borders) - 1)
        # Initialize counting variable
        n = 0

        if rstate0 is None:
            rstate0 = self.random_state

        # Partition baselines in ranges
        bsls_s_thrs_partitioned = partition_baselines_(self.bsls_s_thrs_statuses[['bl', 's_thr']],
                                                       bsls_borders)
        while n <= n_acc:
            # Use custom generator for pa generation to get the same random pa's
            self.random_state = rstate0
            params = self.draw_parameters()
            #print "Trying parameters " + str(params)
            # Create list to collect summary statistics
            summary_statistics = []
            # For each range of baselines check summary statistics
            for i, bsls_s_thrs in enumerate(bsls_s_thrs_partitioned):
                #print "bsls_s_thrs", bsls_s_thrs

                n_ = len(bsls_s_thrs)
                baselines = bsls_s_thrs['bl']
                s_thrs = bsls_s_thrs['s_thr']
                sample = self.create_sample(params, size=n_)
                #print "Using sample :", sample
                det_fr = self.observe_sample(sample, baselines, s_thrs)
                #print "detection fr.", det_fr
                summary_statistics.append(det_fr)

            summary_statistics = np.array(summary_statistics)
            #print "Parameter's data summary statistics : ", summary_statistics
            d = math.sqrt(((summary_statistics - self._data_sum) ** 2 /
                           self._data_sum ** 2).sum())
            #print "Distance is__________________________>"
            #print d
            if d < eps:
                print "This parameter is accepted!"
                n += 1
                self._p.append(params)
                print "Accepted up to now : " + str(self.p)
                self._summary.append(list(summary_statistics))
                self._d.append(d)

    def draw_parameters(self):
        """
        Draw parameters from priors specified in constructor.
        """
        std_logtb = np.random.uniform(self.std_logtb[0], self.std_logtb[1])
        mu_logtb = np.random.uniform(self.mu_logtb[0], self.mu_logtb[1])
        # Based on VSOP N(0.21, 0.9) C-band data
        # actually logS ~ N(-0.43, 0.94)
        mu_logs = np.random.uniform(-3., 2.)
        std_logs = np.random.uniform(self.std_logs[0], self.std_logs[1])
        return mu_logs, std_logs, mu_logtb, std_logtb

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
        mu_logs, std_logs, mu_logtb, std_logtb = parameters
        # Draw sample of size ``size`` from distributions with parameters
        logs = np.random.normal(mu_logs, std_logs, size=size)
        logtb = np.random.normal(mu_logtb, std_logtb, size=size)
        return np.exp(logs), np.exp(logtb)

    def observe_sample(self, sample, baselines, s_thrs):
        """
        Test ``sample`` of sources for detection fraction on ``baselines``.
        :param sample:
            Array-like of (total flux, brigtness temperature) numpy arrays.
        :param baselines:
            Numpy array of baseline length [ED].
        :param s_thr:
            Numpy array of threshold detection flux on each baseline [Jy].
        :return:
            Detection fraction.
        """
        v0, tb = sample
        #print "v0"
        #print v0
        #print "tb"
        #print tb
        #print "baselines"
        #print baselines
        n = len(baselines)
        fluxes = flux_(baselines, v0, tb)
        #print "Fluxes :", fluxes[::10]
        n_det = len(np.where(fluxes > 5. * s_thrs)[0])
        return float(n_det) / n

    def reset(self):
        self._p = list()

    @property
    def p(self):
        return np.atleast_2d(self._p)

    @property
    def summary(self):
        return np.atleast_2d(self._summary)


if __name__ == '__main__':

    if sys.argv[1] == 'c':
        band = 'c'
    elif sys.argv[1] == 'l':
        band = 'l'
    else:
        sys.exit('USE c OR l!')
    print "Using " + band + "-band"
    file_ = open('bsls_s_thrs_statuses.dat', 'r')
    bsls_s_thrs_statuses = cPickle.load(file_)
    file_.close()
    #bsls_s_thrs_statuses, names = get_baselines_s_threshold(band)
    #file_ = open('bsls_s_thrs_statuses.dat', 'w')
    #cPickle.dump(bsls_s_thrs_statuses, file_)
    #file_.close()

    # [ 0.66911765  0.41517857  0.16058394  0.04693141] - fractions
    bsls_borders = [2., 5., 10., 17., 30.]

    print "Using " + str(bsls_borders)
    sim = Simulation(None, [26., 32.], bsls_s_thrs_statuses,
                     std_logs=[0.0, 8.5], std_logtb=[0., 7.5])
    sim.run(300, 0.15, bsls_borders=bsls_borders)
