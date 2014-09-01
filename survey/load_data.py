import numpy as np

if __name__ == '__main__':

    lines = [line.strip() for line in open('/home/ilya/Dropbox/survey/exp_bsl_st.txt')]
    data = list()
    for line in lines:
        data.extend([line.split()])
    exp_name, base_ed, status = zip(*data)
    exp_names_u, indxs = np.unique(exp_name, return_index=True)
    baselines = []
    for ind, exp in zip(indxs, exp_names_u):
        baselines.append(data[ind][1])
    status = []
    for ind, exp in zip(indxs, exp_names_u):
        status.append(data[ind][2])
    baselines = [float(bsl) for bsl in baselines]
