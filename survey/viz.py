import cPickle
import numpy as np
import pylab as pl
from utils import partition_baselines_, ed_to_uv, flux_


def plot_fractions(p, bsls_s_thrs_statuses, bsls_borders):
    """
    Plot detections fraction for given parameters
    :param p:
    :param bsls_s_thrs_statuses:
    :return:
    """
    mu_logv0, std_logv0, mu_logtb, std_logtb = p
    # Partition baselines in ranges
    bsls_s_thrs_partitioned =\
        partition_baselines_(bsls_s_thrs_statuses[['bl', 's_thr']],
                             bsls_borders)
    fractions = list()
    baseline_means = list()
    for i, bsls_s_thrs in enumerate(bsls_s_thrs_partitioned):
        print bsls_s_thrs

        n = len(bsls_s_thrs)
    # Create sample of sources
        logv0 = np.random.normal(mu_logv0, std_logv0, size=n)
        logtb = np.random.normal(mu_logtb, std_logtb, size=n)
        v0 = np.exp(logv0)
        tb = np.exp(logtb)
        sample = (v0, tb,)
        print "Sample :", sample

        baselines = bsls_s_thrs['bl']
        baseline_means.append(np.mean(baselines))
        s_thrs = bsls_s_thrs['s_thr']
        det_fr = observe_sample(sample, baselines, s_thrs)
        fractions.append(det_fr)

    pl.plot(baseline_means, fractions, '.k')
    pl.show()


def observe_sample(sample, baselines, s_thrs):
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
    n = len(baselines)
    fluxes = flux_(baselines, v0, tb)
    print fluxes
    n_det = len(np.where(fluxes > 5. * s_thrs)[0])
    return float(n_det) / n


if __name__ == '__main__':
    file_ = open('bsls_s_thrs_statuses.dat', 'r')
    bsls_s_thrs_statuses = cPickle.load(file_)
    file_.close()
    plot_fractions([0.21, 0.9, 27.6, 0.3], bsls_s_thrs_statuses,
                   bsls_borders=[2., 5., 10., 17., 30.])
