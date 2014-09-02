import numpy as np
from utils import flux, ed_to_uv
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
        self.baselines = ed_to_uv(baselines)
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
        pa = pa or self.pa
        # Generate population of sources
        amps, std_x, e = population.generate(len(self.baselines))

        return flux(self.baselines, pa, amps, std_x, e)


class Population(object):
    def __init__(self, mu_loga, std_loga, mu_logs, std_logs, alpha_e, beta_e):
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

    def generate(self, size):
        loga = np.random.normal(self.mu_loga, self.std_loga, size=size)
        logs = np.random.normal(self.mu_logs, self.std_logs, size=size)
        e = np.random.beta(self.alpha_e, self.beta_e, size=size)
        return (np.exp(logs), np.exp(loga), e)


if __name__ == '__main__':
    # Create 3 populations
    population1 = Population(-0.5, 0.2, -0.5, 0.4, 5., 5.)
    # As 1 but more compact and less full flux
    population2 = Population(-1.0, 0.2, -1.0, 0.4, 5., 5.)
    # As 1 but more alongated
    population3 = Population(-0.5, 0.2, -0.5, 0.4, 5., 15.)

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
    flux1 = survey.observe(population1, pa=pa1)
    flux2 = survey.observe(population2, pa=pa2)
    flux3 = survey.observe(population3, pa=pa3)
