import numpy as np
import os

lamdashu = 10
WEISHU = 152 # Adjust this value based on the number of PSSM files you have
PSSM_DIR = 'pssm'  
PSSM_EXTENSION = '.pssm'
SAVE_DIR = 'your save dir'
pssms = []


for i in range(1, WEISHU + 1):
    filename = f"{i}{PSSM_EXTENSION}"
    filepath = os.path.join(PSSM_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

            all_pssm_scores = []
            for line in lines:
                pssm_scores = [float(value) for value in line.strip().split()]
                if len(pssm_scores) == 20:
                    all_pssm_scores.extend(pssm_scores)
                else:
                    print(f"Warning: Line in file {filepath} does not contain exactly 20 PSSM scores.")
            pssm_data = np.array(all_pssm_scores).reshape(-1, 20)
            pssms.append(pssm_data)
    else:
        print(f"Warning: File {filepath} does not exist.")
normalized_pssms = [1 / (1 + np.exp(-pssm)) for pssm in pssms]
pse_pssms = []

for lamda in range(1, lamdashu + 1):
    pse_pssm = np.zeros((WEISHU, 20))
    for i, pssm in enumerate(normalized_pssms):
        M, N = pssm.shape
        for j in range(N):
            for k in range(M - lamda):
                pse_pssm[i, j] += (pssm[k, j] - pssm[k + lamda, j]) ** 2
            pse_pssm[i, j] /= max(1, M - lamda) if M - lamda > 0 else 0
    pse_pssms.append(pse_pssm)

pse_pssm_combined = np.concatenate(pse_pssms, axis=1)

aac_features = np.array([pssm.mean(axis=0) for pssm in normalized_pssms])

psepssm = np.concatenate((aac_features, pse_pssm_combined), axis=1)

if psepssm.shape[1] == 1:
    psepssm = psepssm.squeeze(1)


save_path = os.path.join(SAVE_DIR, f'psepssm.npy')
np.save(save_path, psepssm)
