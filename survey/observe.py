import numpy as np
from utils import flux, ed_to_uv, mas_to_rad
from load_data import load_data, get_baselines_exper_averaged


class Survey(object):
    def __init__(self, baselines, pa=None):
        """
        Class that implements brightness temperature survey.
        :param baselines:
            Array-like of baselines [ED].
        :param pa:
            Array-like of positional angle of baselines relative to major axis
            of sources.
        """
        # Baselines from ED to uv:
        self.baselines = ed_to_uv(np.asarray(baselines))
        self.pa = pa

    def observe(self, population, pa=None):
        """
        Observe population ``population`` of sources with baselines and
        positional angle of baselines relative to major axis of sources given
        in constructor.
        :param population:
            Population of sources. Instance of ``Population``.
        :param pa (optional):
            Array-like of positional angle of baselines relative to major axis
            of sources.
        :return:
           Numpy array of fluxes [Jy].
        """
        # Use pa from argument if any
        if pa is None:
            pa = self.pa
        # Generate population of sources
        population.generate(len(self.baselines))
        amps, std_x, e = population.population

        return flux(self.baselines, pa, amps, mas_to_rad * std_x, e)


class Population(object):
    def __init__(self, mu_logs, std_logs, mu_loga, std_loga, alpha_e, beta_e):
        """
        :param mu_loga:
            Mean of log(major axis) [log(mas)].
        :param std_loga:
            Std of log(major axis) [log(mas)].
        :param mu_logs:
            Mean of log(full flux) [log(Jy)]
        :param std_logs:
            Std of log(full flux) [log(Jy)]
        :param beta_e:
            Beta-parameter of axis ratio beta distribution.
        :param alpha_e:
            Alpha-parameter of axis ratio beta distribution.
        """
        self.mu_loga = mu_loga
        self.std_loga = std_loga
        self.mu_logs = mu_logs
        self.std_logs = std_logs
        self.alpha_e = alpha_e
        self.beta_e = beta_e
        # Major axis [mas]
        self.a = None
        # Full fluxes [Jy]
        self.s = None
        # Minor-to-major axis ratios
        self.e = None

    def generate(self, size):
        """
        Generate population with size ``size``.
        :param size:
            Size of population to generate.
        """
        loga = np.random.normal(self.mu_loga, self.std_loga, size=size)
        logs = np.random.normal(self.mu_logs, self.std_logs, size=size)
        e = np.random.beta(self.alpha_e, self.beta_e, size=size)
        self.a = np.exp(loga)
        self.s = np.exp(logs)
        self.e = e

    @property
    def population(self):
        return self.s, self.a, self.e

    @population.deleter
    def population(self):
        """
        Clear population of sources.
        """
        self.a = None
        self.s = None
        self.e = None

    def projected_sizes(self, pa):
        """
        Get real projected sizes of population of sources along given positional
        angles ``pa``.
        :param pa:
            Array-like positional angles to calculate projections [rad].
        :return:
            Numpy array with real projected sizes.
        """
        return self.a * np.sqrt((1. + np.tan(pa) ** 2.) / (1. +
                                                           self.e ** (-2.) *
                                                           np.tan(pa) ** 2.))


if __name__ == '__main__':
    # Create 3 populations
    population1 = Population(-0.7, 0.2, -0.5, 0.4, 5., 5.)
    # As 1 but more compact and less full flux
    population2 = Population(-1.0, 0.2, -1.0, 0.4, 5., 5.)
    # As 1 but more alongated
    population3 = Population(-0.7, 0.2, -0.5, 0.4, 5., 15.)

    # Get baselines from real RA survey
    fname = '/home/ilya/Dropbox/survey/exp_bsl_st.txt'
    data = load_data(fname)
    baselines = get_baselines_exper_averaged(fname)

    # Generate random pa
    pa1 = np.random.uniform(0., np.pi, size=len(baselines))
    pa2 = np.random.uniform(0., np.pi, size=len(baselines))
    pa3 = np.random.uniform(0., np.pi, size=len(baselines))

    # Generate survey
    survey = Survey(baselines)
    # Observe 3 populations in this survey
    # Get correlated fluxes
    flux1 = survey.observe(population1, pa=pa1)
    flux2 = survey.observe(population2, pa=pa2)
    flux3 = survey.observe(population3, pa=pa3)
    # Get real projected sizes
    pr_size1 = population1.projected_sizes(pa1)
    pr_size2 = population1.projected_sizes(pa2)
    pr_size3 = population1.projected_sizes(pa3)
