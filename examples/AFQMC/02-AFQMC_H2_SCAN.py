from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
from time import time
import subprocess as sp

from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "02-AFQMC_H2_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)


    bmin = 1.0
    bmax = 6.0
    bond_list_coarse = np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    bond_list_fine   = np.arange( bmin,bmax+0.05,0.05 ) # Bohr

    time_list = []
    E_AFQMC   = []
    E_HF      = np.zeros( len(bond_list_fine) )
    E_FCI     = np.zeros( len(bond_list_fine) )

    print("\n\tDoing HF and FCI calculations on a fine grid.")
    for bi,b in enumerate(bond_list_fine):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)

        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis='sto3g', unit='Bohr', verbose=3)

        # HF
        mf = scf.RHF(mol)
        E_HF[bi] = mf.kernel()

        # FCI
        fcisolver = fci.FCI(mf)
        E_FCI[bi] = fcisolver.kernel()[0]


    print("\n\tDoing AFQMC calculations on a coarse grid.")
    for bi,b in enumerate(bond_list_coarse):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)
        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis='sto3g', unit='Bohr', verbose=3)
        # AFQMC
        num_walkers     = 10
        dt              = 0.01
        total_time      = 10.0
        afqmc_obj       = afqmc.AFQMC(mol, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        time_list.append( np.array(times) )
        E_AFQMC.append( np.array(energies) )

    
    E_AFQMC = np.array(E_AFQMC).real
    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_AFQMC  = E_AFQMC[:,EQ_TIME:]
    AFQMC_AVE = np.average(E_AFQMC, axis=-1)
    AFQMC_STD = np.std(E_AFQMC, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    

    ### Plot all trajectories ###
    EREF = AFQMC_AVE
    plt.imshow( E_AFQMC[:,:] - EREF[:,None], origin='lower', cmap="bwr", extent=[0,total_time,bmin,bmax], aspect='auto')
    plt.colorbar(pad=0.01, label="$E(\\tau) - \langle E(\\tau \\rightarrow \infty) \\rangle$" )
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Projection Time, $\\tau$ (a.u.)", fontsize=15)
    #plt.title("$E(\\tau) - \langle E(\\tau \\rightarrow \infty) \\rangle$", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/02-AFQMC_H2_SCAN_TRAJ_T_%1.0f_dt_%1.4f_Nw_%1.0f.jpg" % (total_time,dt,num_walkers),dpi=300)
    plt.clf()

    

    plt.errorbar(bond_list_coarse, AFQMC_AVE, yerr=AFQMC_STD, fmt='-o', elinewidth=1, ecolor='black', capsize=2, c='black', mfc='black', label="AFQMC")
    plt.plot(bond_list_fine, E_HF , '--', c='blue', lw=2, label="HF")
    plt.plot(bond_list_fine, E_FCI, '--', c='red', lw=2, label="FCI")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f.jpg" % (total_time,dt,num_walkers), dpi=300)
    plt.clf()

