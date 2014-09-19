import numpy as np
import math
import psycopg2
from utils import get_ratio_hdi


dtype_converter_dict = {'integer': 'int', 'smallint': 'int', 'character': '|S',
                        'character varying': '|S', 'real': '<f8',
                        'timestamp without time zone': np.object}

vfloat = np.vectorize(float)
n_q = 0.637
SEFD_dict = {'RADIO-AS': {'K': {'L': 46700., 'R': 36800},
                          'C': {'L': 11600., 'R': None},
                          'L': {'L': 2760., 'R': 2930.}},
             'GBT-VLBA': {'K': {'L': 23., 'R': 23.},
                          'C': {'L': 8., 'R': 8.},
                          'L': {'L': 10., 'R': 10.}},
             'EFLSBERG': {'C': {'L': 20., 'R': 20.},
                          'L': {'L': 19., 'R': 19.}},
             'YEBES40M': {'C': {'L': 160., 'R': 160.},
                          'L': {'L': None, 'R': None}},
             'ZELENCHK': {'C': {'L': 400., 'R': 400.},
                          'L': {'L': 300., 'R': 300.}},
             'EVPTRIYA': {'C': {'L': 44., 'R': 44.},
                          'L': {'L': 44., 'R': 44.}},
             'SVETLOE': {'C': {'L': 250., 'R': 250.},
                         'L': {'L': 360., 'R': 360.}},
             'BADARY': {'C': {'L': 200., 'R': 200.},
                        'L': {'L': 330., 'R': 330.}},
             'TORUN': {'C': {'L': 220., 'R': 220.},
                       'L': {'L': 300., 'R': 300.}},
             'ARECIBO': {'C': {'L': 5., 'R': 5.},
                         'L': {'L': 3., 'R': 3.}},
             'WSTRB-07': {'C': {'L': 120., 'R': 120.},
                          'L': {'L': 40., 'R': 40.}},
             'VLA-N8': {'C': {'L': None, 'R': None},
                        'L': {'L': None, 'R': None}},
             # Default values for KL
            'KALYAZIN': {'C': {'L': 150., 'R': 150.},
                         'L': {'L': 140., 'R': 140.}},
             'MEDICINA': {'C': {'L': 170., 'R': 170.},
                          'L': {'L': 700., 'R': 700.}},
             'NOTO': {'C': {'L': 260., 'R': 260.},
                      'L': {'L': 784., 'R': 784.}},
             'HARTRAO': {'C': {'L': 650., 'R': 650.},
                         'L': {'L': 430., 'R': 430.}},
             'HOBART26': {'C': {'L': 640., 'R': 640.},
                          'L': {'L': 470., 'R': 470.}},
             'MOPRA': {'C': {'L': 350., 'R': 350.},
                       'L': {'L': 340., 'R': 340.},
                       'K': {'L': 900., 'R': 900.}},
             'WARK12M': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'TIDBIN64': {'C': {'L': None, 'R': None},
                          'L': {'L': None, 'R': None}},
             'DSS63': {'C': {'L': 24., 'R': 24.},
                       'L': {'L': 24., 'R': 24.}},
             'PARKES': {'C': {'L': 110., 'R': 110.},
                        'L': {'L': 40., 'R': 40.},
                        'K': {'L': 810., 'R': 810.}},
             'USUDA64': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'JODRELL2': {'C': {'L': 320., 'R': 320.},
                          'L': {'L': 320., 'R': 320.}},
             'ATCA104': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}}}


def dtype_converter(data_type, char_length):
    """
    Converts psycopg2 data types to python data types.
    :param data_type:
        Psycopg2 data type.
    :param char_length:
        If not ``None``, then shows char length.
    :return:
    """
    result = dtype_converter_dict[data_type]
    if char_length:
        result += str(char_length)

    return result


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


def get_array_from_dbtable(host='odin.asc.rssi.ru', port='5432',
                           db='ra_results', user='guest', password='WyxRep0Dav',
                           table_name='pima_observations'):
    """
    Function that returns numpy structured array from user-specified db table.
    :param host:
    :param port:
    :param db:
    :param user:
    :param password:
    :param table_name:
    :return:
    """
    connection = psycopg2.connect(host=host, port=port, dbname=db,
                                  password=password, user=user)
    cursor = connection.cursor()
    # First know column names
    cursor.execute("select column_name, data_type, character_maximum_length from\
                   information_schema.columns where table_schema = \'public\'\
                   and table_name=\'" + table_name + "\'")
    result = cursor.fetchall()
    dtype = list()
    #column_names, data_types, char_lengths = zip(*result):
    for column_name, data_type, char_length in result:
        dtype.append((column_name, dtype_converter(data_type, char_length)))

    # Convert to numpy data types

    # Now read the table and put to structured array
    cursor.execute("select * from " + table_name)
    table_data = cursor.fetchall()
    struct_array = np.zeros(len(table_data), dtype=dtype)
    for i, (column_name, data_type, char_length,) in enumerate(result):
        struct_array[column_name] = zip(*table_data)[i]

    return struct_array


def s_thr_from_obs_row(row, raise_ra=True, n_q=0.637, dnu=16. * 10 ** 6, n=2):
    """
    Function that calculates sigma of detection from structured array row.
    :param row:
        Row of 2D structured array. Actually, an object with __getitem__ method
        and corresponding keys.
    :return:
        Sigma for detection using upper and lower bands.
    """
    rt1 = row['st1']
    rt2 = row['st2']
    polar = row['polar']
    band = row['band'].upper()
    try:
        SEFD_rt1 = SEFD_dict[rt1][band.upper()][polar[0]]
    except KeyError:
        #raise Exception("There's no entry for " + rt1 + " in SEFD dictionary!")
        return None
    except TypeError:
        raise Exception("There's no SEFD data for " + rt1 + " !")
    try:
        SEFD_rt2 = SEFD_dict[rt2][band.upper()][polar[1]]
    except KeyError:
        #raise Exception("There's no entry for " + rt2 + " in SEFD dictionary!")
        return None
    except TypeError:
        raise Exception("There's no SEFD data for " + rt2 + " !")

    try:
        result = (1. / n_q) * math.sqrt((SEFD_rt1 * SEFD_rt2) / (n * dnu *
                                                                 row['solint']))
    except TypeError:
        return None

    return result


# TODO: Execute query with asc.db.DB class before fetching results.
def get_baselines_s_threshold(band, struct_array=None):
    """
    Returns numpy 2D-arrays of (baseline, s_thr, stutus), where s_thr is
    determined by integration time and ground station.
    :param fname:
    :return:

    :notes:
        Each single experiment contribute one baseline. If no detection in
        experiment then use most sensitive baseline to set ``s_thr``. If there
        are any number of detections then use the least sensitive baseline among
        baselines with detection to set ``s_thr``. Thus, we use the most
        information from detections/nondetections.
    """
    if band not in ('k', 'c', 'l', 'p',):
        raise Exception("band must be k, c, l or p!")
    if struct_array is None:
        struct_array = get_array_from_dbtable()
    # Choose only data with parallel hands
    struct_array = struct_array[np.where(np.logical_or(struct_array['polar'] ==
                                                       'LL',
                                                       struct_array['polar'] ==
                                                       'RR'))]
    # Choose only data with band
    struct_array = struct_array[np.where(struct_array['band'] == band)]
    # Choose only data with RA
    struct_array = struct_array[np.where(np.logical_or(struct_array['st1'] ==
                                                       'RADIO-AS',
                                                       struct_array['st2'] ==
                                                       'RADIO-AS'))]

    # Get unique experiments
    exp_names_u, indxs = np.unique(struct_array['exper_name'],
                                   return_index=True)
    # List for collecting results
    results = list()
    # Loop over unique experiments to find threshold flux
    for exp in exp_names_u:
        exp_array = struct_array[np.where(struct_array['exper_name'] == exp)]
        # Get observations with detections
        obs_detections = exp_array[np.where(exp_array['status'] == 'y')]
        obs_nondetections = exp_array[np.where(np.logical_or(exp_array['status']
                                                             == 'n',
                                                             exp_array['status']
                                                             == 'u'))]
        # If experiment has detections
        if obs_detections.size:
            # List with threshold fluxes for each observation with detection
            s_thrs = list()
            for obs_ in obs_detections:
                s_thr = s_thr_from_obs_row(obs_, band)
                s_thrs.append(s_thr)
            print "S_thr for experiment ", s_thrs
            # Choose baseline with the highest threshold flux among detections
            index = s_thrs.index(max(s_thrs))
            # If all s_thr are None then nothing will go to result (see append)
            observation = obs_detections[index]
            status = 'y'

        # If experiment doesn't have detections
        else:
            # List with threshold fluxes for each observation with nondetections
            s_thrs = list()
            for obs_ in obs_nondetections:
                s_thr = s_thr_from_obs_row(obs_, band)
                s_thrs.append(s_thr)
            print "S_thr for experiment ", s_thrs
            # Choose observation with the lowest threshold flux among
            # nondetections
            try:
                index = s_thrs.index(min(s_thr for s_thr in s_thrs if s_thr is
                                         not None))
            # if all s_thr are None then use any (first)
            except ValueError:
                index = 0
            observation = obs_nondetections[index]
            status = 'n'

        print observation['st1'], observation['st2'], observation['base_ed'],\
            s_thr_from_obs_row(observation), status
        s_thr_result = s_thr_from_obs_row((observation))
        if s_thr_result is not None:
            results.append([observation['base_ed'], s_thr_result, status])

    dtype = [('bl', '>f4'), ('s_thr', '>f4'), ('status', '|S1')]
    output = np.zeros(len(results), dtype=dtype)
    output['bl'], output['s_thr'], output['status'] = zip(*results)

    return output


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
        fractions.append([get_ratio_hdi(n_det, n_all), frac])
    return fractions


def get_detection_fractions_in_baseline_ranges(bsls_s_thrs_status,
                                               bsls_borders):
    array_ = bsls_s_thrs_status
    fractions = list()
    for i in range(len(bsls_borders) - 1):
        array__ = array_[np.where(np.logical_and(vfloat(array_['bl']) >
                                                 bsls_borders[i],
                                                 vfloat(array_['bl']) <
                                                 bsls_borders[i + 1]))]
        n_det = list(array__['status']).count('y')
        n_all = len(array__)
        fractions.append(get_ratio_hdi(n_det, n_all,cred_mass=0.5))
    return fractions


def get_experiments_for_each_source(data):
    adata = np.atleast_2d(data)
    exp_names, sources, base_ed, status = zip(*data)
    sources_u, indxs = np.unique(sources, return_index=True)
    source_dict = dict()
    for source in sources_u:
        source_data = adata[np.where(adata[:, 1] == source)]
        source_experiments = np.unique(source_data[:, 0])
        source_dict.update({source: [len(source_experiments),
                            source_experiments]})
    return source_dict


def get_data_for_each_source(data):
    """
    fname should contain:
        exper_name, source, st2, snr, solint, u, v - for polar='RR' or 'LL', for
        st1='RADIO-AS' and band='x'
    data = load_data(fname)
    # We need:  st2, snr & solint (for approx. amp. cal.), u&v.

    :return:
        dictionary with keys = source names and values - numpy array (u, v, snr,
        ampl, solint, st2, status)
    """
    adata = np.atleast_2d(data)
    exp_names, sources, st2s, polars, snrs, ampls, solints, us, vs, bsls_ed,\
        statuses = zip(*data)
    sources_u, indxs = np.unique(sources, return_index=True)
    print "Unique sources: " + str(sources_u)
    source_dict = dict()
    for source in sources_u:
        # Constructing structured array:
        source_data = adata[np.where(adata[:, 1] == source)]
        source_us = source_data[:, 7]
        source_vs = source_data[:, 8]
        source_snrs = source_data[:, 4]
        source_ampls = source_data[:, 5]
        source_polars = source_data[:, 3]
        source_st2s = source_data[:, 2]
        source_statuses = source_data[:, -1]
        source_array = np.dstack((source_us, source_vs, source_snrs,
                                  source_ampls, source_polars, source_st2s,
                                  source_statuses,))

        source_dict.update({source: source_array[0]})

    return source_dict



if __name__ == '__main__':

    fname_c = '/home/ilya/Dropbox/survey/exp_bsl_st_c.txt'
    fname_l = '/home/ilya/Dropbox/survey/exp_bsl_st_l.txt'
    fname_k = '/home/ilya/Dropbox/survey/exp_bsl_st_k.txt'
    data_c = load_data(fname_c)
    data_l = load_data(fname_l)
    data_k = load_data(fname_k)
    summary_c = get_unique_experiments_dict(data_c)
    summary_l = get_unique_experiments_dict(data_l)
    summary_k = get_unique_experiments_dict(data_k)
    mean_baselines_c = get_baselines_exper_averaged(fname_c)
    mean_baselines_l = get_baselines_exper_averaged(fname_l)
    mean_baselines_k = get_baselines_exper_averaged(fname_k)
    det_fr_c = get_detection_fraction(data_c)
    det_fr_l = get_detection_fraction(data_l)
    det_fr_k = get_detection_fraction(data_k)
    fractions_c = get_detection_fraction_in_baseline_range(fname_c, [0., 5, 10.,
                                                                     17., 30.])
    fractions_l = get_detection_fraction_in_baseline_range(fname_l, [0., 5, 10.,
                                                                     17., 30.])
    fractions_k = get_detection_fraction_in_baseline_range(fname_k, [0., 5, 10.,
                                                                     17., 30.])



