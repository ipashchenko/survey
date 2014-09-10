import numpy as np
from utils import get_ratio_hdi


vfloat = np.vectorize(float)
SEFD_dict = {'RADIO-AS': {{'K': {'L': 46700., 'R': 36800}},
                          {'C': {'L': 11600., 'R': None}},
                          {'L': {'L': 2760., 'R': 2930.}}},
             'GBT-VLBA': {{'K': {'L': 23., 'R': 23.}},
                          {'C': {'L': 8., 'R': 8.}},
                          {'L': {'L': 10., 'R': 10.}}},
             'EFLSBERG': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'YEBES40M': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'ZELENCHK': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'EVPTRIYA': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'SVETLOE': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}},
             'BADARY': {{'C': {'L': None, 'R': None}},
                        {'L': {'L': None, 'R': None}}},
             'TORUN': {{'C': {'L': None, 'R': None}},
                       {'L': {'L': None, 'R': None}}},
             'ARECIBO': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}},
             'WSTRB-07': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'VLA-N8': {{'C': {'L': None, 'R': None}},
                        {'L': {'L': None, 'R': None}}},
             'KALYAZIN': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'MEDICINA': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'NOTO': {{'C': {'L': None, 'R': None}},
                      {'L': {'L': None, 'R': None}}},
             'HARTRAO': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}},
             'HOBART26': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'MOPRA': {{'C': {'L': None, 'R': None}},
                       {'L': {'L': None, 'R': None}}},
             'WARK12M': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}},
             'TIDBIN64': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'DSS63': {{'C': {'L': None, 'R': None}},
                       {'L': {'L': None, 'R': None}}},
             'PARKES': {{'C': {'L': None, 'R': None}},
                        {'L': {'L': None, 'R': None}}},
             'USUDA64': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}},
             'JODRELL2': {{'C': {'L': None, 'R': None}},
                          {'L': {'L': None, 'R': None}}},
             'ATCA104': {{'C': {'L': None, 'R': None}},
                         {'L': {'L': None, 'R': None}}}}


def load_data(fname):
    """
    Load data from file
    :param fname:
    :return:
    """
    lines = [line.strip() for line in open(fname)]
    data = list()
    for line in lines:
        data.extend([line.split()])
    return data


def get_baselines_exper_averaged(fname):
    """
    Return list of baselines where each baseline is mean of baselines for each
    unique experiment.
    :param fname:
    :return:
    """
    data = load_data(fname)
    exp_name, base_ed, status = zip(*data)
    exp_names_u, indxs = np.unique(exp_name, return_index=True)
    adata = np.atleast_2d(data)
    mean_baselines = list()
    for exp_name_u in exp_names_u:
        exp_baselines = adata[np.where(adata[:, 0] == exp_name_u)][:, 1]
        mean_baselines.append(np.mean(vfloat(exp_baselines)))
    return np.asarray(mean_baselines)


def get_unique_experiments_dict(data):
    exp_name, base_ed, status = zip(*data)
    exp_names_u, indxs = np.unique(exp_name, return_index=True)
    adata = np.atleast_2d(data)
    summary = dict()
    for exp_name_u in exp_names_u:
        exp_statuses = adata[np.where(adata[:, 0] == exp_name_u)][:, 2]
        exp_baselines = vfloat(adata[np.where(adata[:, 0] == exp_name_u)][:, 1])
        if 'y' in exp_statuses:
            summary.update({exp_name_u: ('y', np.mean(exp_baselines))})
        else:
            summary.update({exp_name_u: ('n', np.mean(exp_baselines))})
    return summary


def get_detection_fraction(data):
    summary = get_unique_experiments_dict(data)
    temp = np.atleast_2d(summary.values())
    n_det = list(temp[:, 0]).count('y')
    n_all = len(summary)
    return n_det, n_all, float(n_det) / n_all


def get_detection_fraction_in_baseline_range(fname, bsls_borders):
    data = load_data(fname)
    adata = np.atleast_2d(data)
    fractions = list()
    for i in range(len(bsls_borders) - 1):
        adata_ = adata[np.where((vfloat(adata[:, 1]) > bsls_borders[i]) &
                       (vfloat(adata[:, 1]) < bsls_borders[i + 1]))[0]]
        n_det, n_all, frac = get_detection_fraction(adata_)
        fractions.append([n_det, n_all, frac])
    return fractions


if __name__ == '__main__':

    fname_c = '/home/ilya/Dropbox/survey/exp_bsl_st_c.txt'
    fname_l = '/home/ilya/Dropbox/survey/exp_bsl_st_l.txt'
    data_c = load_data(fname_c)
    data_l = load_data(fname_l)
    summary_c = get_unique_experiments_dict(data_c)
    summary_l = get_unique_experiments_dict(data_l)
    mean_baselines_c = get_baselines_exper_averaged(fname_c)
    mean_baselines_l = get_baselines_exper_averaged(fname_l)
    det_fr_c = get_detection_fraction(data_c)
    det_fr_l = get_detection_fraction(data_l)
    fractions_c = get_detection_fraction_in_baseline_range(fname_c, [5, 10.,
                                                                     17., 30.])
    fractions_l = get_detection_fraction_in_baseline_range(fname_l, [5, 10.,
                                                                     17., 30.])
