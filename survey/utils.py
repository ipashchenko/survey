import numpy as np
from scipy.optimize import fmin


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


def hdi_of_icdf(name, cred_mass=0.95, tol=1e-08, *args, **kwargs):
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
    name = name(*args, **kwargs)

    def interval_width(low_tail_prob, name, cred_mass):
        return name.ppf(cred_mass + low_tail_prob) - name.ppf(low_tail_prob)

    hdi_low_tail_prob = fmin(interval_width, 0, (name, cred_mass,), xtol=tol)

    return name.ppf(hdi_low_tail_prob), name.ppf(hdi_low_tail_prob + cred_mass)
