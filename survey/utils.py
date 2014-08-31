import numpy as np


mas_to_rad = 4.85 * 10 ** (-9)
rad_to_mas = 206.3 * 10 ** 6


def ed_to_uv(r, lambda_cm=18.):
    return r * 12742. * 100000. / lambda_cm


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
        Major axis std in image plane [rad]. Corresponds to u-coordinate.
    :param e:
        Ratio of minor to major axis (``std_y``/``std_x``) of elliptical
        component.
    :return:
        Value of correlated flux at uv-point ()
    """

    std_u = 1. / (std_x)

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