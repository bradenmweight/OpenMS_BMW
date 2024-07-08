from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
from time import time
import subprocess as sp

from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "01-AFQMC_QED_H2_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis = "sto3g"
    bmin = 1.0
    bmax = 10.0
    #bond_list_coarse = np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    bond_list_coarse = np.arange( bmin,bmax+0.5,0.5 ) # Bohr
    #bond_list_coarse = [10.] # np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    bond_list_fine   = np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    #bond_list_fine   = np.arange( bmin,bmax+0.05,0.05 ) # Bohr

    time_list   = []
    E_AFQMC     = []
    E_AFQMC_QED = []
    E_HF        = np.zeros( len(bond_list_fine) )
    E_FCI       = np.zeros( len(bond_list_fine) )

    print("\n\tDoing HF and FCI calculations on a fine grid.")
    for bi,b in enumerate(bond_list_fine):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)

        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

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
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)
        # AFQMC
        num_walkers     = 1000 # 5000
        dt              = 0.1
        total_time      = 10.0
        afqmc_obj       = afqmc.AFQMC(mol, numdets=1, trial="RHF", dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        E_AFQMC.append( np.array(np.real(energies)) )
        afqmc_obj       = afqmc.QEDAFQMC(mol, cavity_freq=np.array([0.1]), cavity_coupling=np.array([1e-4]), cavity_vec=np.array([np.array([1,1,1])]), numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        E_AFQMC_QED.append( np.array(np.real(energies)) )
        print("\n\tI did QED-AFQMC.\n")
        
        print( "AFQMC Energy: %1.6f" % np.average(E_AFQMC[-1][len(times)//4:], axis=-1) )
        print( "QED-AFQMC Energy: %1.6f" % np.average(E_AFQMC_QED[-1][len(times)//4:], axis=-1) )
    
        if ( bi == 0 ):
            time_list = np.array(times)






    E_AFQMC = np.array(E_AFQMC).real
    E_AFQMC_QED = np.array(E_AFQMC_QED).real
    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_AFQMC   = E_AFQMC[:,EQ_TIME:]
    E_AFQMC_QED   = E_AFQMC_QED[:,EQ_TIME:]
    AFQMC_AVE = np.average(E_AFQMC, axis=-1)
    AFQMC_AVE_QED = np.average(E_AFQMC_QED, axis=-1)
    AFQMC_STD = np.std(E_AFQMC, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    AFQMC_STD_QED = np.std(E_AFQMC_QED, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    

    ### Plot all trajectories ###
    # EREF = AFQMC_AVE
    # plt.imshow( E_AFQMC[:,:] - EREF[:,None], origin='lower', cmap="bwr", extent=[0,total_time,bmin,bmax], aspect='auto')
    # plt.colorbar(pad=0.01, label="$E(\\tau) - \\langle E(\\tau \\rightarrow \\infty) \\rangle$" )
    # plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    # plt.ylabel("Projection Time, $\\tau$ (a.u.)", fontsize=15)
    # #plt.title("$E(\\tau) - \langle E(\\tau \\rightarrow \infty) \\rangle$", fontsize=15)
    # plt.tight_layout()
    # plt.savefig("%s/02-AFQMC_H2_SCAN_TRAJ_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis),dpi=300)
    # plt.clf()

    

    plt.errorbar(bond_list_coarse, AFQMC_AVE, yerr=AFQMC_STD, fmt='-o', elinewidth=1, ecolor='black', capsize=2, c='black', mfc='black', label="AFQMC")
    plt.errorbar(bond_list_coarse, AFQMC_AVE_QED, yerr=AFQMC_STD, fmt='--s', elinewidth=1, ecolor='red', capsize=2, c='red', mfc='red', label="QED-AFQMC")
    plt.plot(bond_list_fine, E_HF , '--', c='blue', lw=2, label="HF")
    plt.plot(bond_list_fine, E_FCI, '--', c='green', lw=2, label="FCI")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis), dpi=300)
    plt.clf()

