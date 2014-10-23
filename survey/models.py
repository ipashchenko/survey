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


def ft_of_delta(uv, s_full, x0=0., y0=0.):
    """
    Returns value of the Fourier transform of delta function with full flux
    ``s_full`` and position ``x0``, ``y0`` in image plane at points ``u``, ``v``
    of uv-plane.
    :param uv:
        2D numpy array of uv-coordinates [lambda^(-1)].
    :param s_full:
        Full flux of component [Jy].
    :param x0: (optional)
        Position of delta-function in image plane [rad].
    :param y0: (optional)
        Position of delta-function in image plane [rad].
    :return:
        Numpy array of FT values at given points of uv-plane.
    """
    ft = np.empty(np.shape(uv)[-1], dtype=complex)
    ft[:] = s_full
    # If x0=!0 or y0=!0 then shift phase
    if x0 or y0:
        ft *= np.exp(-2. * math.pi * 1j * (uv[0] * x0 + uv[1] * y0))
    return ft


def ft_of_2d_gaussian(uv, s_full, bmaj, e, bpa=0., x0=0., y0=0.):
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
    # In our parametrization we need only functional dependence as flux at zero
    # uv-spacings is given by ``s_full``.
    #ft = s_full * 2. * math.pi * k ** (-0.5) * math.exp((4. * math.pi ** 2. / k)
    #                                                    * (-c * u ** 2. +
    #                                                       b * u * v -
    #                                                       a * v ** 2.))
    ft = s_full * math.exp((4. * math.pi ** 2. / k) * (-c * uv[0] ** 2. +
                                                       b * uv[0] * uv[1] -
                                                       a * uv[1] ** 2.))
    ft = complex(ft)
    # If x0=!0 or y0=!0 then shift phase
    if x0 or y0:
        ft *= np.exp(-2. * math.pi * 1j * (uv[0] * x0 + uv[1] * y0))
    return ft
