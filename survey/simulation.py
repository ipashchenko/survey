import numpy as np
from numpy.random import RandomState


def flux(r, pa, amp, std_x, e):
    """
    Return flux of elliptical gaussian source with major axis (along x-axis,
    which is along u-axis) FWHM = 2*sqrt(2*log(2))*s_x at uv-radius ``r`` and
    positional angle ``pa``.
    :param r:
        Uv-radius [lambda^(-1)]
    :param pa:
        Positional angle of baseline (from u to v).
    :param std_x:
        Major axis std in image plane [mas]. Corresponds to u-coordinate.
    :param e:
        Ratio of minor to major axis (``std_y``/``std_x``) of elliptical
        component.
    :return:
        Value of correlated flux at uv-point ()
    """

    std_u = 1. / std_x

    return amp * np.exp(-(r ** 2. * (1. + e ** 2. * np.tan(pa) ** 2.)) /
                        (2. * std_u ** 2. * (1. + np.tan(pa) ** 2.)))


def get_samples_from_beta(alpha, beta, *args, **kwargs):
    return np.random.beta(alpha, beta, *args, **kwargs)


def get_samples_from_normal(mean, tau, size=10 ** 4):
    return np.random.normal(loc=mean, scale=1. / np.sqrt(tau), size=size)


def detection_fraction(mu_loga, std_loga, beta_e, mu_logS, std_logS, bsl_pdf,
                       alpha_e=2., S_thr=0.05, size=10 ** 5):
    """
    Returns detection fraction for distribution of source sizes with log(major
    axis) ~ N(mu_loga, tau_loga), minor-to-major axis distribution ~
    Beta(alpha_e, beta_e), log(flux) at zero (u,v)-point distribution ~
    N(mu_logS, tau_logS) and baseline length pdf ``bsl_pdf`` (e.g. from kernel
    estimate).

    :param mu_loga:
    :param std_loga:
    :param beta_e:
    :param mu_logS:
    :param std_logS:
    :param bsl_pdf:
    :param alpha_e (optional):
        Parameter of Beta-distribution for axis ratios. Default = 2.
    :param S_thr:
        Threshold flux for detection [Jy].
    :param size:
    :return:
    """
    prng = RandomState(123)
    np.random.seed(123)
    loga = np.random.normal(loc=mu_loga, scale=std_loga, size=size)
    a = np.exp(loga)
    e = np.random.beta(alpha_e, beta_e, size=size)
    logS = np.random.normal(loc=mu_logS, scale=std_logS, size=size)
    S = np.exp(logS)
    bsl = bsl_pdf.resample(size=size)
    pa = np.random.unif(0., np.pi, size=size)
    fluxes = flux(bsl, pa, S, a, e)
    n_det = len(np.where(fluxes > S_thr)[0])
    return float(n_det) / size


class Simulation(object):
    def __init__(self, mu_loga, mu_logs, beta_e, bsl_kde, alpha_e=2.):
        """
        mu_loga - 2 values for min & max of mu_loga prior range.
        mu_logs - 2 values for min & max of mu_logs prior range.
        beta - 2 values for min & max of beta prior range.
        """
        self.mu_loga = mu_loga
        self.mu_logs = mu_logs
        self.beta_e = beta_e
        self.bsl_kde = bsl_kde
        self.alpha_e = alpha_e
        self.p = list()

    def run(self, n_acc, fr, tol, s_thr=0.05, size=10 ** 4.):
        """
        Run simulation till ``n_acc`` accepted.
        """
        # Initialize counting variable
        n = 0
        while n < n_acc:
            # Sample from priors
            mu_loga = np.random.unif(self.mu_loga[0], self.mu_loga[1])
            std_loga = np.random.gamma(0.1, 0.1)
            mu_logs = np.random.unif(self.mu_logs[0], self.mu_logs[1])
            std_logs = np.random.gamma(0.1, 0.1)
            beta_e = np.random.unif(self.beta_e[0], self.beta_e[1])
            frac = detection_fraction(self.mu_loga, self.std_loga, self.beta_e,
                                      self.mu_logs, self.std_logs, self.bsl_pdf,
                                      self.alpha_e, s_thr=s_thr, size=size):
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
