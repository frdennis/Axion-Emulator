import time
start = time.time()

#from Generate import Generate
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib

from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)
import sys
import pandas as pd
import h5py
import csv
from tqdm import tqdm

from _predict import Predictor

class cm_emulator_class:
    def __init__(self, version=0):
        self.predictor = Predictor.from_path(f"./emulators/lightning_logs/version_{version}")

    def __call__(
        self,
        params,
        return_tensors: bool = False,
    ):
        inputs = np.array(params)
        return self.predictor(inputs).reshape(-1)


_emulator = cm_emulator_class(version=4,)


Omega_b = 0.049
Omega_cdm = 0.2637
A_s = 2.1e-9
n_s = 0.966
OmegaMnu = 0.0
h = 0.6766
redshift = 0.0
k = np.logspace(-3, 0.78, 256)

# ['OmegaCDM', 'logA_s', 'f_ax', 'logm_ax', 'z', 'k',]
#params1 = []
#params1.append([Omega_cdm, np.log10(A_s), 0.25, np.log10(1e-23), redshift, k[j]])

fig, ax = plt.subplots(figsize=(6,6), nrows=2, sharex=True)

[[axi.grid(), axi.set_xscale("log")] for axi in ax]

N = 15
cmap = plt.cm.get_cmap("viridis", N)
f_ax = np.linspace(0.01, 1.0, N)
m_ax = np.logspace(-28, -22, N)

# omm = np.linspace(0.3, 0.399, N)
for i, f in enumerate(f_ax):
    params = [[Omega_cdm, np.log10(A_s), f, np.log10(1e-26), redshift, ki] for ki in k]
    emu = _emulator(params)
    ax[0].plot(k, emu, c=cmap(i))

for i, m in enumerate(m_ax):
    params = [[Omega_cdm, np.log10(A_s), 0.5, np.log10(m), redshift, ki] for ki in k]    
    emu = _emulator(params)
    ax[1].plot(k, emu, c=cmap(i))


sm1 = plt.cm.ScalarMappable(cmap=cmap)
sm1.set_array(f_ax)
cbar1 = plt.colorbar(sm1, ax=ax[0])

sm2 = plt.cm.ScalarMappable(cmap=cmap)
sm2.set_array(np.log10(m_ax))
cbar2 = plt.colorbar(sm2, ax=ax[1])

cbar1.set_label(r"$f_{\rm ax}$")
cbar2.set_label(r"log10($m_{\rm ax}$)")
#ax[0].set_xlabel("k [h/Mpc]")
ax[1].set_xlabel("k [h/Mpc]")
ax[0].set_ylabel(r"r = $P_{\rm ax}$(k)/$P_{\Lambda \rm CDM}$(k)")
ax[1].set_ylabel(r"r = $P_{\rm ax}$(k)/$P_{\Lambda \rm CDM}$(k)")

plt.tight_layout()
plt.savefig("test_axion_dependency.pdf")

end = time.time()
print("Time : ", end - start, "s")

plt.show()

