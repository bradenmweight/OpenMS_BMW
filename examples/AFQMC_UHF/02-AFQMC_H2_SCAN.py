from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
from time import time
import subprocess as sp

from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "02-AFQMC_H2_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis = '321g' # "cc-pvdz" # "sto3g"
    bmin = 1.0
    bmax = 6.0
    bond_list_coarse = np.arange( bmin,bmax+0.25,0.25 ) # Bohr
    bond_list_fine   = np.arange( bmin,bmax+0.05,0.05 ) # Bohr

    time_list  = []
    E_AFQMC    = []
    E_uAFQMC   = []
    E_RHF      = np.zeros( len(bond_list_fine) )
    E_UHF      = np.zeros( len(bond_list_fine) )
    E_FCI      = np.zeros( len(bond_list_fine) )

    print("\n\tDoing HF and FCI calculations on a fine grid.")
    for bi,b in enumerate(bond_list_fine):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)

        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

        # RHF
        mf = scf.RHF(mol)
        E_RHF[bi] = mf.kernel()

        # UHF -- Need to break symmetry of initial guess to get right solution
        mf = scf.UHF(mol)
        dm_alpha, dm_beta = mf.get_init_guess()
        dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
        dm = (dm_alpha,dm_beta)
        E_UHF[bi] = mf.kernel(dm)

        # FCI
        fcisolver = fci.FCI(mf)
        E_FCI[bi] = fcisolver.kernel()[0]


    print("\n\tDoing AFQMC calculations on a coarse grid.")
    for bi,b in enumerate(bond_list_coarse):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)
        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)
        # AFQMC
        num_walkers     = 5000
        dt              = 0.01 # 0.1
        total_time      = 10.0
        afqmc_obj       = afqmc.AFQMC(mol, numdets=1, trial="RHF", dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        E_AFQMC.append( np.array(energies) )
        
        afqmc_obj       = afqmc.AFQMC(mol, numdets=1, trial="UHF", dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        E_uAFQMC.append( np.array(energies) )
        if ( bi == 0 ):
            time_list = np.array(times)
            #break

        print( "rAFQMC Energy: %1.3f" % (np.average(E_AFQMC[-1][len(time_list)//4:], axis=-1) ) )
        print( "uAFQMC Energy: %1.3f" % (np.average(E_uAFQMC[-1][len(time_list)//4:], axis=-1) ) )

    
    E_AFQMC = np.array(E_AFQMC).real
    E_uAFQMC = np.array(E_uAFQMC).real
    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_AFQMC  = E_AFQMC[:,EQ_TIME:]
    E_uAFQMC  = E_uAFQMC[:,EQ_TIME:]
    AFQMC_AVE = np.average(E_AFQMC, axis=-1)
    uAFQMC_AVE = np.average(E_uAFQMC, axis=-1)
    AFQMC_STD = np.std(E_AFQMC, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    uAFQMC_STD = np.std(E_uAFQMC, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO

    plt.errorbar(bond_list_coarse, AFQMC_AVE, yerr=AFQMC_STD, fmt='o', elinewidth=1, ecolor='black', capsize=2, c='black', mfc='black', label="AFQMC@RHF")
    plt.errorbar(bond_list_coarse, uAFQMC_AVE, yerr=AFQMC_STD, fmt='o', elinewidth=1, ecolor='red', capsize=2, c='red', mfc='red', label="AFQMC@UHF")
    plt.plot(bond_list_fine, E_FCI,  '-', c='green', lw=6, alpha=0.4, label="FCI")
    plt.plot(bond_list_fine, E_RHF , '-', c='blue', lw=6, alpha=0.4, label="RHF")
    plt.plot(bond_list_fine, E_UHF , '-', c='blue', lw=2, label="UHF")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis), dpi=300)
    plt.clf()

