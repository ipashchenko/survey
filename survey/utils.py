import numpy as np
from scipy.optimize import fmin
from scipy.stats import beta as sbeta


mas_to_rad = 4.85 * 10 ** (-9)
rad_to_mas = 206.3 * 10 ** 6


def ed_to_uv(r, lambda_cm=18.):
    return r * 12742. * 100000. / lambda_cm

def partition_baselines(bsls, s_thrs, borders):
    """
    Function that partition array ``bsls`` on ``len(borders) - 1`` parts and
    returns list of tuples (bsls_partitioned, s_thrs_partitioned,)
    :param bsls:
    :param s_thrs:
    :param borders:
    :return:
    """
    bsls_partitioned = list()
    s_thrs_partitioned = list()
    for i in range(len(borders) - 1):
        indxs = np.where(np.logical_and(bsls > borders[i],
                                        bsls < borders[i + 1]))[0]
        bsls_partitioned.append(bsls[indxs])
        s_thrs_partitioned.append(s_thrs[indxs])
    return zip(bsls_partitioned, s_thrs_partitioned)


def partition_baselines_(bsls_s_thrs, borders):
    """
    Function that partition 2d numpy array ``bsls_s_thrs`` based on ``borders``
    baseline ranges.
    :param bsls_s_thrs:
    :param borders:
    :return:
    """
    bsls_s_thrs_partitioned = list()
    for i in range(len(borders) - 1):
        indxs = np.where(np.logical_and(bsls_s_thrs[:, 0] > borders[i],
                                        bsls_s_thrs[:, 0] < borders[i + 1]))[0]
        bsls_s_thrs_partitioned.append(bsls_s_thrs[indxs])
    return bsls_s_thrs_partitioned


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

    std_u = 1. / (2. * np.pi * std_x)

    return amp * np.exp(-(r ** 2. * (1. + e ** 2. * np.tan(pa) ** 2.)) /
                        (2. * std_u ** 2. * (1. + np.tan(pa) ** 2.)))


def size(flux, r, flux0, r0=1.):
    """
    Get size estimate from measured flux ``flux`` at baseline ``r`` if at
    baseline``r0`` measured flux is ``flux0``. Using gaussian model.
    :param flux:
    :param r:
    :param r0:
    :return:
    """
    pass


def hdi_of_icdf(name, args, cred_mass=0.95, tol=1e-08):
    """
    :param name:
        Distribution with percent point function (inverse of CDF) defined as
        ``name.ppf(q)``, where ``q`` - lower tail probability.
    :param cred_mass (optional):
        Desired mass of HDI region.
    :param tol (optioanl):
        Passed to ``scipy.optimize.fmin`` with ``xtol`` keyword argument.
    :param args:
        Positional arguments to distribution ``name``.
    :param kwargs:
        Keyword arguments to distribution ``name``.
    """
    name = name(*args)

    def interval_width(low_tail_prob, name, cred_mass):
        return name.ppf(cred_mass + low_tail_prob) - name.ppf(low_tail_prob)

    hdi_low_tail_prob = fmin(interval_width, 0, (name, cred_mass,), xtol=tol)

    return name.ppf(hdi_low_tail_prob), name.ppf(hdi_low_tail_prob + cred_mass)


def get_ratio_hdi(m, n, cred_mass=0.95):
    """
    Get hdi for ratio ``m/n`` for binominal model.
    :param m:
        ``heads``.
    :param n:
        ``all``
    :return:
    """
    assert(n >= m)
    hdi = hdi_of_icdf(sbeta, [m + 1., n - m + 1.], cred_mass=cred_mass)
    return float(hdi[0]), float(hdi[1])
