from pyscf import tools, lo, scf, fci
import numpy as np
import matplotlib.pyplot as plt
from time import time

from openms.qmc import afqmc



if __name__ == "__main__":
    from pyscf import gto, scf, fci
    bond = 1.6 # Bohr
    natoms = 2
    atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
    mol = gto.M(atom=atoms, basis='321g', unit='Bohr', verbose=3)

    # HF
    T0 = time()
    print("Doing HF calculation.")
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    T_HF = time() - T0

    # FCI
    T0 = time()
    print("Doing FCI calculation.")
    fcisolver = fci.FCI(mf)
    fci_energy = fcisolver.kernel()[0]
    T_FCI = time() - T0

    # AFQMC
    T0 = time()
    print("Doing AFQMC calculation.")
    num_walkers     = 10000
    dt              = 0.05
    total_time      = 50.0
    afqmc_obj       = afqmc.AFQMC(mol, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, energies = afqmc_obj.kernel()
    T_AFQMC         = time() - T0

    times     = np.array(times)
    energies  = np.array(energies)
    AFQMC_AVE = np.average(energies[len(times)//4:].real)
    AFQMC_STD = np.std(energies[len(times)//4:].real)
    plt.plot(times, times*0 + AFQMC_AVE, '-', c='black', lw=6, alpha=0.25, label="<AFQMC> (%1.0f s, E = %1.3f $\pm$ %1.3f)" % (T_AFQMC,AFQMC_AVE,AFQMC_STD))
    plt.plot(times, energies.real, '-', c='black', lw=2, label="AFQMC")
    plt.plot(times, times*0 + hf_energy,  '--', c='blue', lw=2, label="HF (%1.0f s, E = %1.3f)" % (T_FCI,hf_energy))
    plt.plot(times, times*0 + fci_energy, '--', c='red', lw=2, label="FCI (%1.0f s, E = %1.3f)" % (T_FCI,fci_energy))
    plt.ylabel("Ground State Energy (a.u.)", fontsize=15)
    plt.xlabel("Projection Time, $\\tau$ (a.u.)", fontsize=15)
    plt.title("$d\\tau$ = %1.3f a.u.,  $N_\mathrm{w}$ = %1.0f" % (dt, num_walkers), fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig("01-AFQMC_H2.jpg", dpi=300)

