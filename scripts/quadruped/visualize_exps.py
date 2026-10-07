import pickle
import matplotlib.pyplot as plt 
import numpy as np

N=20
scenario_list_bams = np.zeros((N,3))
error_list_bams = np.zeros((N,1))

import numpy as np

from scipy.spatial.distance import cdist

import numpy as np

fail_list = ['failure','success']
idx_list = [1,2,3]
for fail in fail_list:
    for idx in idx_list:
        with open(_REPO / "experiments_quadruped" / "sample_trials" / fail / f"{fail}_{idx}_trial.pkl", "rb") as f:
            data = pickle.load(f)
        print("________________________________________")
        print(f"{fail}_{idx}")
        print(data['command']['vx'], data['command']['vy'], data['command']['wz'])
        print(data['error_summary']['mean_abs_err_sum'])
        print(data['records'][-1]['vx'],data['records'][-1]['vy'],data['records'][-1]['wz'])
        print("________________________________________")

