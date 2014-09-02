import numpy as np


vfloat = np.vectorize(float)


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
    return mean_baselines


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

    fname = '/home/ilya/Dropbox/survey/exp_bsl_st.txt'
    data = load_data(fname)
    summary = get_unique_experiments_dict(data)
    mean_baselines = get_baselines_exper_averaged(fname)
    det_fr = get_detection_fraction(data)
    fractions = get_detection_fraction_in_baseline_range(fname, [0, 10., 20.,
                                                                 30.])
