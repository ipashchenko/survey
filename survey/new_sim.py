import math
import cPickle
import numpy as np
from load_data import get_detection_fractions_in_baseline_ranges_, \
    get_data_for_simulation
from utils import flux_


class Simulation(object):
    def __init__(self, data, mu_logv0, mu_logtb, std_logv0=None,
                 std_logtb=None):
        """
        Class that implements simulation of RA survey.

        :param data:
            Numpy structured array with fields (source, baseline, threshold
            flux, status, parameter, fluxes).
        :param mu_logv0:
            2 values for min & max of uniform prior range on mean of log(full
            flux) [log(Jy)].
        :param mu_logtb:
            2 values for min & max of uniform prior range on mean of log(Tb)
            [log(K)].
        :param std_logv0: (optional)
            2 values for min & max of uniform prior range on std of log(full
            flux) [log(Jy)].
        :param std_logtb: (optional)
            2 values for min & max of uniform prior range on std of log(Tb)
            [log(K)].
        """
        self.mu_logv0 = mu_logv0
        self.std_logv0 = std_logv0
        self.mu_logtb = mu_logtb
        self.std_logtb = std_logtb
        self.data = data
        self.sources = np.unique(data['source'])
        # List for keeping accepted parameters
        self._p = []
        # List for keeping fractions for accepted parameters
        self._summary_statistics = list()
        # List for keeping distance to data summary statistics for accepted
        # parameters
        self._distance = list()
        self._data_summary_statistic = None
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

    def run(self, n_acc, eps, baseline_bins=None, rstate0=None):
        """
        Run simulation till ``n_acc`` parameters are accepted.

        :param n_acc:
            Accepted number of parameters to stop simulation.
        :param eps:
            Tolerance for acceptance.
        :param baseline_bins:
            Array-like of bins for baseline ranges. Fractions will be
            compared in intervals [baseline_bins[0], baseline_bins[1]],
            [baseline_bins[1], baseline_bins[2]], ..., [baseline_bins[n-1],
            baseline_bins[n]].
        :param rstate0: (optional)
            The state of the random number generator.
            See the :attr:`Sampler.random_state` property for details.

        :notes:
            Size of samples used to count acceptance fractions is determined by
            number of baselines in each baseline interval.
        """
        # Calculate detection fractions in given baseline bins
        det_fracs = get_detection_fractions_in_baseline_ranges_(self.data,
                                                                baseline_bins)
        # Save observed detection fractions as data summary statistics
        self._data_summary_statistic = np.array(det_fracs)
        print "Data summary statistics : ", self._data_summary_statistic

        # Assertion on consistency
        assert(len(det_fracs) == len(baseline_bins) - 1)

        # Initialize counting variable
        n = 0

        if rstate0 is None:
            rstate0 = self.random_state

        # Main circle
        while n <= n_acc:

            # Copy data for modifying parameters and fluxes
            data = self.data.copy()
            # Draw parameters of distributions from priors
            params = self.draw_parameters()
            # Draw parameters of #sources sources from distributions
            sample = self.create_sample(params, size=len(self.sources))
            # Fill parameters field
            for i, source in enumerate(self.sources):
                data['sample'][np.where(data['source'] == source)] = sample[i]
            # Fill flux field (calculate fluxes)
            data['flux'] = self.observe_sample(data['sample'], data['bl'],
                                               data['s_thr'])
            # Calculate detection fractions in each bin of baselines
            summary_statistic = list()
            # Loop over baseline bins
            for i in range(len(baseline_bins) - 1):
                indxs = np.where(np.logical_and(data['bl'] > baseline_bins[i],
                                                data['bl'] <
                                                baseline_bins[i + 1]))
                # Count only data in current baseline bin
                subdata = data[indxs]
                # Count number of detections
                n_det = len(np.where(subdata['flux'] >
                                     5. * subdata['s_thr'])[0])
                summary_statistic.append(float(n_det) / len(subdata))

            summary_statistic = np.array(summary_statistic)

            # Calculate distance from data summary statistics
            d = math.sqrt(((summary_statistic -
                            self._data_summary_statistic) ** 2 /
                           self._data_summary_statistic ** 2).sum())
            #print "Distance is --->", d
            if d < eps:
                print "This parameter is accepted!"
                n += 1
                self._p.append(params)
                print "Accepted up to now : " + str(self.p)
                self._summary_statistics.append(list(summary_statistic))
                self._distance.append(d)

    def draw_parameters(self):
        """
        Draw parameters from uniform priors (ranges are specified in
        constructor).
        """
        mu_logtb = np.random.uniform(self.mu_logtb[0], self.mu_logtb[1])
        std_logtb = np.random.uniform(self.std_logtb[0], self.std_logtb[1])
        # Actually VSOP gives logS ~ N(-0.43, 0.94)
        mu_logv0 = np.random.uniform(self.mu_logv0[0], self.mu_logv0[1])
        std_logv0 = np.random.uniform(self.std_logv0[0], self.std_logv0[1])
        return mu_logv0, std_logv0, mu_logtb, std_logtb

    def create_sample(self, parameters, size):
        """
        Create sample os sources (full fluxes, T_b & axis ratios) with size =
        ``size`` using parameters of distributions, specified in ``parameters``.
        :param parameters:
            Array-like of ``mu_logv0``, ``std_logv0``, ``mu_logtb``,
            ``std_logtb``.
        :param size:
            Size of sample to generate.
        :return:
            Numpy arrays of full flux & Tb each of size = ``size``.
        """
        mu_logv0, std_logv0, mu_logtb, std_logtb = parameters
        # Draw sample of size ``size`` from distributions with parameters
        logv0 = np.random.normal(mu_logv0, std_logv0, size=size)
        logtb = np.random.normal(mu_logtb, std_logtb, size=size)
        return np.squeeze(np.dstack((np.exp(logv0), np.exp(logtb),)))

    def observe_sample(self, sample, baselines, threshold_fluxes):
        """
        Test ``sample`` of sources for detection fraction on ``baselines``.
        :param sample:
            Array-like of (total flux, brigtness temperature) numpy arrays.
        :param baselines:
            Numpy array of baseline length [ED].
        :param threshold_fluxes:
            Numpy array of threshold detection flux on each baseline [Jy].
        :return:
            Detection fraction.
        """
        v0 = sample[:, 0]
        tb = sample[:, 1]
        return flux_(baselines, v0, tb)

    def reset(self):
        self._p = list()

    @property
    def p(self):
        return np.atleast_2d(self._p)

    @property
    def summary(self):
        return np.atleast_2d(self._summary)

if __name__ == '__main__':

    #if sys.argv[1] == 'c':
    #    band = 'c'
    #elif sys.argv[1] == 'l':
    #    band = 'l'
    #else:
    #    sys.exit('USE c OR l!')
    band = 'c'
    print "Using " + band + "-band"
    file_ = open('data.dat', 'r')
    data = cPickle.load(file_)
    file_.close()
    #data, names = get_data_for_simulation(band)
    #file_ = open('data.dat', 'w')
    #cPickle.dump(data, file_)
    #file_.close()

    # [ 0.66911765  0.41517857  0.16058394  0.04693141] - fractions
    baseline_bins = [2., 5., 10., 17., 30.]

    print "Using " + str(baseline_bins)
    sim = Simulation(data, [-3., 1.], [26., 32.], std_logtb=[0., 7.5],
                     std_logv0=[0., 5.])
    sim.run(100, 0.25, baseline_bins=baseline_bins)
