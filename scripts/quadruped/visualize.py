import pickle
import matplotlib.pyplot as plt 
import numpy as np

N=20
scenario_list_bams = np.zeros((N,3))
error_list_bams = np.zeros((N,1))

import numpy as np

from scipy.spatial.distance import cdist

import numpy as np
rng = np.random.default_rng(seed=42)
X = np.column_stack([
    rng.uniform(-0.4,1.0, size=20),
    rng.uniform(-0.8,0.8, size=20),
    rng.uniform(-0.8, 0.8, size=20),
])

print(X) 


for i in range(N):
    if i!=17:
        # if i<=8:
        #     with open(f"results/go2_hardware/bams/fixed_vx0{10+1}.pkl",'rb') as f:
        #             data = pickle.load(f)
        # else:
        with open(f"results/go2_hardware/bams_2/fixed_vx{10+i}.pkl",'rb') as f:
            data = pickle.load(f)
    scenario_list_bams[i,0] = data['command']['vx']
    scenario_list_bams[i,1] = data['command']['vy']
    scenario_list_bams[i,2] = data['command']['wz']

    error_list_bams[i] = data['error_summary']['mean_abs_err_sum']

scenario_list_scout = np.zeros((N,3))
error_list_scout = np.zeros((N,1))
for i in range(N):
    # if i<=8:
    #     with open(f"results/go2_hardware/scout/fixed_vx0{i+1}.pkl",'rb') as f:
    #             data = pickle.load(f)
    # else :
    with open(f"results/go2_hardware/scout_2/fixed_vx{i+10}.pkl",'rb') as f:
        data = pickle.load(f)
    scenario_list_scout[i,0] = data['command']['vx']
    scenario_list_scout[i,1] = data['command']['vy']
    scenario_list_scout[i,2] = data['command']['wz']

    error_list_scout[i] = data['error_summary']['mean_abs_err_sum']

scenario_list_random = np.zeros((N,3))
error_list_random = np.zeros((N,1))
for i in range(N):
    if i<=8:
        with open(f"results/go2_hardware/random/fixed_vx0{i+1}.pkl",'rb') as f:
                data = pickle.load(f)
    else :
        with open(f"results/go2_hardware/random/fixed_vx{i+1}.pkl",'rb') as f:
            data = pickle.load(f)
    scenario_list_random[i,0] = data['command']['vx']
    scenario_list_random[i,1] = data['command']['vy']
    scenario_list_random[i,2] = data['command']['wz']

    error_list_random[i] = data['error_summary']['mean_abs_err_sum']

scenario_select_scout = np.array([scenario_list_scout[i] for i in range(N) if error_list_scout[i]>=0.7])
scenario_select_random = np.array([scenario_list_random[i] for i in range(N) if error_list_random[i]>=0.7])
scenario_select_bams = np.array([scenario_list_bams[i] for i in range(N) if error_list_bams[i]>=0.7])

A = cdist(scenario_select_scout,scenario_select_scout)
upper_vals = A[np.triu_indices(A.shape[0], k=1)]
upper_sum = upper_vals.sum()
print("SCOUT",upper_sum, len(scenario_select_scout))


B = cdist(scenario_select_bams,scenario_select_bams)
upper_vals = B[np.triu_indices(B.shape[0], k=1)]
upper_sum = upper_vals.sum()
print("BAMS",upper_sum, len(scenario_select_bams))


C= cdist(scenario_select_random,scenario_select_random)
upper_vals = C[np.triu_indices(C.shape[0], k=1)]
upper_sum = upper_vals.sum()
print("Random",upper_sum, len(scenario_select_random))

plt.figure()
plt.scatter(scenario_select_scout[:,0],scenario_select_scout[:,1],label='Scout')
plt.scatter(scenario_select_random[:,0],scenario_select_random[:,1],label='Random')
plt.scatter(scenario_select_bams[:,0],scenario_select_bams[:,1],label='BAMS')
plt.legend()

plt.figure()
plt.scatter(scenario_select_scout[:,0],scenario_select_scout[:,2],label='Scout')
plt.scatter(scenario_select_random[:,0],scenario_select_random[:,2],label='Random')
plt.scatter(scenario_select_bams[:,0],scenario_select_bams[:,2],label='BAMS')
plt.legend()

plt.figure()
plt.scatter(scenario_select_scout[:,1],scenario_select_scout[:,2],label='Scout')
plt.scatter(scenario_select_random[:,1],scenario_select_random[:,2],label='Random')
plt.scatter(scenario_select_bams[:,1],scenario_select_bams[:,2],label='BAMS')
plt.legend()




# N=14
# print(scenario_list[10:N,:],error_list[0:N])
# plt.figure()
# plt.scatter(scenario_list[10:N,0],scenario_list[10:N,1],c=error_list[10:N])
# plt.title('Vx and Vy')
# plt.colorbar()

# plt.figure()
# plt.scatter(scenario_list[10:N,0],scenario_list[10:N,2],c=error_list[10:N])
# plt.title('Vx and wz')
# plt.colorbar()

# plt.figure()
# plt.scatter(scenario_list[10:N,1],scenario_list[10:N,2],c=error_list[10:N])
# plt.title('Vy and wz')
# plt.colorbar()
    

