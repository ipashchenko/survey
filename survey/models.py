import numpy as np
import math


def flux_from_1_asymmetric(r, pa, amp, std_x, e):
    """
    Return flux of elliptical gaussian source with major axis (along x-axis,
    which corresponds to u-axis) FWHM = 2*sqrt(2*log(2))*s_x at uv-radius ``r``
    and positional angle ``pa``.
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


def flux_from_1_asymmetric_full(u, v, s_full, bmaj, e, bpa=0, x0=0, y0=0):
    """
    Return complex flux of elliptical gaussian source with major axis std
    ``bmaj`` (which corresponds to FWHM = 2 * sqrt(2 * log(2)) * ``std``) and
     positional angle of major axis equals to ``bpa`` (from x to y of image
     plane) at uv-point (``u``,``v``) of uv-plane.

    :param u:
        u-coordinate [lambda^(-1)].
    :param v:
        v-coordinate [lambda^(-1)].
    :param s_full:
        Full flux of component [Jy].
    :param bmaj:
        Major axis std in image plane [rad].
    :param bpa:
        Positional angle of major axis in image plane (from x to y) [rad].
    :param e:
        Ratio of minor to major axis of elliptical component.
    :param x0:
        Center of component in image plane [rad].
    :param y0:
        Center of component in image plane [rad].
    :return:
        Complex value of correlated flux at given uv-point.
    """
    # Rotate the uv-plane on angle ``-bpa``
    u_ = u * math.cos(bpa) + v * math.sin(bpa)
    v_ = -u * math.sin(bpa) + v * math.cos(bpa)

    bmin = bmaj * e
    ft = s_full * math.pi * bmaj * bmin * np.exp(-math.pi ** 2 * 1j *
                                              (bmaj ** 2 * u_ ** 2 +
                                               bmin ** 2 * v_ ** 2))
    # Multiply on phases of ``x0``, ``y0`` in rotated system
    x0_ = x0 * math.cos(bpa) + y0 * math.sin(bpa)
    y0_= -x0 * math.sin(bpa) + y0 * math.cos(bpa)
    ft *= math.exp(2 * math.pi * 1j * (x0_ * u_ + y0_ * v_))

    return ft


def ft_of_2d_gaussian(u, v, s_full, bmaj, e, bpa=0, x0=0, y0=0):
    """
    Returns value of the Fourier transform of 2d elliptical gaussian with major
    axis ``bmaj``, full flux ``s_full``, axis ratio ``e``, positional angle of
    major axis ``bpa`` (counted from x-axis of image plane counter clockwise)
    and center ``x0``, ``y0`` in image plane in point ``u``,
    ``v`` of uv-plane.
    :param u:
    :param v:
    :param s_full:
    :param bmaj:
    :param e:
    :param bpa:
    :param x0:
    :param y0:
    :return:
    :notes:
        The value of the Fourier transform of gaussian function (Wiki):

            g(x, y) = A * exp[-(a*(x-x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0)**2)]  (1)

            where:

                a = cos(\theta)**2/(2*std_x**2) + sin(\theta)**2/(2*std_y**2)
                b = sin(2*\theta)/(2*std_x**2) - sin(2*\theta)/(2*std_y**2)
                (corresponds to rotation counter clockwise)
                c = sin(\theta)**2/(2*std_x**2) + cos(\theta)**2/(2*std_y**2)

            for x0=0, y0=0 in point u,v of uv-plane is (Briggs Thesis):

            2*pi*A*(4*a*c-b**2)**(-1/2)*exp[(4*pi**2/(4*a*c-b**2))*(-c*u**2+b*u*v-a*v**2)] (2)

            As we parametrize the flux as ``s_full`` - full flux of gaussian
            (that is at zero (u,v)-spacing), then change coefficient in front of
            exponent to ``s_full``.

            shift of (x0, y0) in image plane corresponds to phase shift in
            uv-plane:

            ft(x0,y0) = ft(x0=0,y0=0) * exp(-2*pi*(u*x0 + v*y0))
    """
    # Construct parameter of gaussian function (1)
    std_x = bmaj
    std_y = e * bmaj
    theta = bpa
    a = math.cos(theta) ** 2. / (2. * std_x ** 2.) +\
        math.sin(theta) ** 2. / (2. * std_y ** 2.)
    b = math.sin(2. * theta) / (2. * std_x ** 2.) -\
        math.sin(2. * theta) / (2. * std_y ** 2.)
    c = math.sin(theta) ** 2. / (2. * std_x ** 2.) + \
        math.cos(theta) ** 2. / (2. * std_y ** 2.)
    # Calculate the value of FT in point (u,v) for x0=0,y0=0 case using (2)
    k = (4. * a * c - b ** 2.)
    #ft = s_full * 2. * math.pi * k ** (-0.5) * math.exp((4. * math.pi ** 2. / k)
    #                                                    * (-c * u ** 2. +
    #                                                       b * u * v -
    #                                                       a * v ** 2.))
    ft = s_full * math.exp((4. * math.pi ** 2. / k) * (-c * u ** 2. +
                                                       b * u * v -
                                                       a * v ** 2.))
    ft = complex(ft)
    # If x0=!0 or y0=!0 then shift phase
    if x0 or y0:
        ft *= math.exp(-2. * math.pi * 1j * (u * x0 + v * y0))

    return ft


def flux_from_2_asymmetric(u, v, s_full, e_s, b_maj_small, e_b_small,
                           b_maj_big, e_b_big, d_pa):
    """
    Return flux of source that consists of 2 elliptical gaussian components with
    major axis of small component (along x-axis, which corresponds to u-axis)
    FWHM = 2 * sqrt(2 * log(2)) * ``b_maj_small`` at uv-radius ``r`` and
    positional angle ``pa``. Second component has major axis ``b_maj_big`` and
    difference between positional angles of 2 gaussians (small gaussian pa - big
    gaussian pa) equals ``d_pa``.

    :param r:
    :param pa:
    :param s_full:
    :param e_s:
    :param b_maj_small:
    :param e_b_small:
    :param b_maj_big:
    :param e_b_big:
    :param d_pa:
    :return:
    """
    pass


def ft_2dgaussian(uvs, amp, x0, y0, bmaj, bmin, bpa):
    """
    Return the Fourie Transform of 2D gaussian defined in image plane by
    it's amplitude ``amp``, center ``x0`` & ``y0``, major and minor axes
    ``bmaj`` & ``bmin`` and positional angle of major axis ``bpa``.

    :param uvs:
        Iterable of uv-points for which calculate FT.

    :param amp:
        Amplitude of gaussian [Jy].

    :param x0:
        X-coordinate of gaussian center [rad].

    :param y0:
        Y-coordinate of gaussian center [rad].

    :param bmaj:
        Size of major axis [rad].

    :param bmin:
        Size of min axis [rad].

    :param bpa:
        Positional angle of major axis [rad].

    :return:
        Numpy array of complex visibilities for specified points ``uvs``.
        Length of resulting array = len(uvs).
    """
    # Rotate the uv-plane on angle -bpa
    uvs_ = uvs.copy()
    uvs_[:, 0] = uvs[:, 0] * math.cos(bpa) + uvs[:, 1] * math.sin(bpa)
    uvs_[:, 1] = -uvs[:, 0] * math.sin(bpa) + uvs[:, 1] * math.cos(bpa)
    # Sequence of FT of gaussian(amp, x0=0, y0=0, bmaj, bmin) with len(ft) =
    # len(uvs)
    ft = amp * math.pi * bmaj * bmin * np.exp(-math.pi ** 2 *
                                              (bmaj ** 2 * uvs_[:, 0] ** 2 +
                                               bmin ** 2 * uvs_[:, 1] ** 2))
    # Multiply on phases of x0, y0 in rotated system
    x0_ = x0 * math.cos(bpa) + y0 * math.sin(bpa)
    y0_= -x0 * math.sin(bpa) + y0 * math.cos(bpa)
    ft *= np.exp(2 * math.pi * 1j * (x0_ * uvs_[:, 0] + y0_ * uvs_[:, 1]))

    return ft
