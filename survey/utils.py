import re
import math
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
        indxs = np.where(np.logical_and(bsls_s_thrs['bl'] > borders[i],
                                        bsls_s_thrs['bl'] < borders[i + 1]))[0]
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


def flux_(b, v0, tb):
    """
    Flux of circular gaussian source with full flux ``v0`` and brightness
    temperature ``tb`` at baseline ``b``.
    :param b:
        Baseline [baseline, ED]
    :param v0:
        Amplitude of component [Jy].
    :param tb:
        Brightness temperature of source [K].
    :return:
        Value of correlated flux.
    """
    b *= 12742. * 10. ** 3
    k = 1.38 * 10 ** (-23)
    return v0 * np.exp(-math.pi * b ** 2. * v0 * 10 ** (-26) / (2. * k * tb))


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


def j2000_from_racat(fname):

    dict_ = dict()

    file_ = open(fname, 'U')
    lines = file_.readlines()
    for line in lines:
        if line.strip().startswith('!'):
            continue
        try:

            name = re.findall(r".+source='(\S+)'", line)[0]
            print name
            j2000 = line.strip().split()[1].strip(",'")
            print j2000
            dict_.update({str(name): str(j2000)})
        except IndexError:
            pass

    return dict_


def cross_correlate_ra_with_vsop(names, vsop_cores_table, racat):
    """
    Get fluxes of VSOP cores for sources observed with RA.

    :param names:
        Iterable of RA DB names (B1950).
    :param vsop_cores_table:
        Text file with data from VSOP.
    :param racat:
        Radioastron catalogue.
    :return:
        Fluxes of cores (VSOP 5GHz data) for RA DB sources that were observed
        with VSOP.
    """
    names = set(names)
    file_ = open(vsop_cores_table, 'U')
    lines = file_.readlines()
    fluxes = list()
    for name in names:
        print "Searching for " + name
        j2000 = find_j2000_from_racat(name, racat)
        if j2000 is not None:
            for line in lines:
                if j2000 in line:
                    flux = line.strip().split()[1]
                    print "Source core " + name + " got flux " + str(flux)
                    fluxes.append(float(flux))
    return fluxes


def find_j2000_from_racat(name, racat):
    """
    Function that returns J2000 name from RA catalogue given B1950 or
    alternative name.

    :param name:
        B1950 or alternative (like 3C279) name.
    :return:
        J2000 name.
    """
    file_ = open(racat, 'U')
    lines = file_.readlines()
    j2000 = None
    for line in lines:
        if name in line:
            j2000 = line.strip().split()[1].strip(",'")

    try:
        return j2000
    except UnboundLocalError:
        print "Can't find J2000 for " + name
        return None
