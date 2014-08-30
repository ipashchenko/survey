import numpy as np
#from numpy.random import RandomState
from scipy.stats import gaussian_kde


mas_to_rad = 4.85 * 10 ** (-9)
rad_to_mas = 206.3 * 10 ** 6


def flux(r, pa, amp, std_x, e):
    """
    Return flux of elliptical gaussian source with major axis (along x-axis,
    which is along u-axis) FWHM = 2*sqrt(2*log(2))*s_x at uv-radius ``r`` and
    positional angle ``pa``.
    :param r:
        Uv-radius [lambda^(-1)]
    :param pa:
        Positional angle of baseline (from u to v) [rad].
    :param amp:
        Amplitude of component [Jy].
    :param std_x:
        Major axis std in image plane [mas]. Corresponds to u-coordinate.
    :param e:
        Ratio of minor to major axis (``std_y``/``std_x``) of elliptical
        component.
    :return:
        Value of correlated flux at uv-point ()
    """

    std_u = 1. / (std_x * mas_to_rad)

    return amp * np.exp(-(r ** 2. * (1. + e ** 2. * np.tan(pa) ** 2.)) /
                        (2. * std_u ** 2. * (1. + np.tan(pa) ** 2.)))


def get_samples_from_beta(alpha, beta, *args, **kwargs):
    return np.random.beta(alpha, beta, *args, **kwargs)


def get_samples_from_normal(mean, tau, size=10 ** 4):
    return np.random.normal(loc=mean, scale=1. / np.sqrt(tau), size=size)


def detection_fraction(mu_loga, std_loga, beta_e, mu_logs, std_logs, bsls,
                       alpha_e=2., s_thr=0.05, size=10 ** 5):
    """
    Returns detection fraction for distribution of source sizes with log(major
    axis) ~ N(loga, std_loga), minor-to-major axis distribution ~
    Beta(alpha_e, beta_e), log(flux) at zero (u,v)-point distribution ~
    N(mu_logS, tau_logS) and baseline length pdf ``bsl_pdf`` (e.g. from kernel
    estimate).

    :param mu_loga:
        Mean of normal distribution for loga, where a [mas].
    :param std_loga:
        Std of normal distribution for loga, where a [mas].
    :param beta_e:
    :param mu_logs:
        Mean of normal distribution for logs, where s [Jy].
    :param std_logs:
        Std of normal distribution for logs, where s [Jy].
    :param bsls:
        Baselines. Array-like with size=``size``. If it is not, then resample
        to ``size``.
    :param alpha_e (optional):
        Parameter of Beta-distribution for axis ratios. Default = 2.
    :param s_thr:
        Threshold flux for detection [Jy].
    :param size:
        Size of samples to draw.g
    :return:
    """
    #prng = RandomState(123)
    np.random.seed(123)
    loga = np.random.normal(loc=mu_loga, scale=std_loga, size=size)
    a = np.exp(loga)
    e = np.random.beta(alpha_e, beta_e, size=size)
    logs = np.random.normal(loc=mu_logs, scale=std_logs, size=size)
    s = np.exp(logs)
    if not len(bsls) == size:
        bsls = np.random.choice(bsls, size=size)
    pa = np.random.unif(0., np.pi, size=size)
    fluxes = flux(bsls, pa, s, a, e)
    n_det = len(np.where(fluxes > s_thr)[0])
    return float(n_det) / size


# TODO: First, use the same ``loga``, ``beta_e`` (geometry) & ``logs`` for all
# sources, the the same geometry but distribution of ``logs`` and then
# distributions for all.
class Simulation(object):
    def __init__(self, loga, logs, beta_e, bsls, alpha_e=2.):
        """
        loga - 2 values for min & max of ``loga``/``mu_loga`` prior range.
        logs - 2 values for min & max of ``logs``/``mu_logs`` prior range.
        beta - 2 values for min & max of beta prior range.
        bsls - Array-like of baselines.
        """
        self.loga = loga
        self.logs = logs
        self.beta_e = beta_e
        self.bsls = np.asarray(bsls)
        self.alpha_e = alpha_e
        self.p = list()

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
            Flux detection threshold.

        :notes:
            Size of samples used to count acceptance fractions is determined by
            number of baselines used (``self.bsls``)

        :return:
            List of n lists of parameters [``loga``, ``beta_e``, ``logs``] that
            give detection fractions equal to observed within tolerance.
        """
        # Assertions
        assert(len(fr_list) == len(tol_list))
        assert(len(fr_list) == len(bsls_borders) - 1)
        # Initialize counting variable
        n = 0
        # Initialize accepted parameters container
        parameters = list()

        n_bsls = len(self.bsls)

        fr_array = np.asarray(fr_list)
        tol_array = np.asarray(tol_list)

        # Partition baselines in ranges
        bsls_partitioned = list()
        for i in range(len(bsls_borders) - 1):
            bsls_partitioned.append(np.where(self.bsls > bsls_borders[i] &
                                             self.bsls < bsls_borders[i + 1]))
        while n <= n_acc:
            # Simulate ``n_bsls`` sources
            # First simulate one value
            loga = np.random.unif(self.loga[0], self.loga[1])
            logs = np.random.unif(self.logs[0], self.logs[1])
            beta_e = np.random.unif(self.beta_e[0], self.beta_e[1])
            e = np.random.beta(self.alpha_e, beta_e)
            # Save in list
            params = [loga, beta_e, logs]
            # Repeat to make ``n_bsls`` sources with the same parameters
            loga = loga.repeat(n_bsls)
            logs = logs.repeat(n_bsls)
            beta_e = beta_e.repeat(n_bsls)
            # Simulate ``n_bsls`` random positional angles for baselines
            pa = np.random.unif(0., np.pi, size=n_bsls)
            # For each range of baselines check summary statistics
            for i, baselines in enumerate(bsls_partitioned):
                n_ = len(baselines)
                # Calculate detection fraction in this baseline range
                fluxes = flux(baselines, pa[:n_], np.exp(logs[:n_]),
                              np.exp(loga[:n_]), e[:n_])
                n_det = len(np.where(fluxes > s_thr)[0])
                det_fr = float(n_det) / n_
                # If fail to get right fraction in this range then go to next
                # loop of while
                if abs(det_fr - fr_list[i]) > tol_list[i]:
                    break
            # If we got stuck here, then fractions in all baseline ranges are
            # within tolerance of the observed.
            n += 1
            parameters.append(params)

    def run(self, n_acc, fr_list, tol_list, s_thr=0.05, size=10 ** 4.):
        """
        Run simulation till ``n_acc`` accepted.
        """
        # Initialize counting variable
        fr_array = np.asarray(fr_list)
        tol_array = np.asarray(tol_list)
        n = 0
        while n < n_acc:
            # Sample from priors
            mu_loga = np.random.unif(self.loga[0], self.loga[1])
            std_loga = np.random.gamma(0.1, 0.1)
            mu_logs = np.random.unif(self.logs[0], self.logs[1])
            std_logs = np.random.gamma(0.1, 0.1)
            beta_e = np.random.unif(self.beta_e[0], self.beta_e[1])
            fractions = list()
            for fr in fr_list:
                fractions.append([detection_fraction(self.loga, self.std_loga,
                                                self.beta_e, self.logs,
                                                self.std_logs, bsls,
                                                self.alpha_e, s_thr=s_thr,
                                                size=size) for bsls in
                                  self.bsls_list])
                fractions = np.asarray(fractions)
            boolean = [abs(frac - fr)]
            if abs(frac - fr) <= tol:
                p = [mu_loga, std_loga, mu_logs, std_logs, beta_e]
                print "Accepted parameters: " + str(p)
                self.p.append(p)
                n += 1
            else:
                print "Rejected parameters: " + str(p)

    def reset(self):
        self.p = list()

    @property
    def p(self):
        return np.atleast_2d(self.p)


if __name__ == '__main__':
    lines = [line.strip() for line in open('exp_bsl_st.txt')]
    data = list()
    for line in lines:
       data.extend([line.split()])
    exp_name, base_ed, status = zip(*data)
    exp_names_u, indxs = np.unique(exp_name, return_index=True)
    baselines = []
    for ind, exp in zip(indxs, exp_names_u):
        baselines.append(data[ind][1])
    status = []
    for ind, exp in zip(indxs, exp_names_u):
        status.append(data[ind][2])
    baselines = [float(bsl) for bsl in baselines]
    # Resample to get "more data"
    big = np.random.choice(baselines, size=10000)
    bsl_kde = gaussian_kde(big)







