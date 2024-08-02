from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
from time import time
import subprocess as sp

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "02-AFQMC_QED_coupling_scan"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis        = "ccpVTZ"
    photon_basis = 'fock'
    NFock        = 5
    bond_length = 2.8 # Bohr
    A0_list = np.arange( 0.0, 0.51, 0.01 ) # np.arange( 0.0, 0.225, 0.025 )


    time_list   = []
    E_AFQMC_QED = []
    E_QEDHF     = np.zeros( len(A0_list) )
    E_FCI       = np.zeros( len(A0_list) )


    print("\n\tDoing QED calculations on a coarse grid.")
    for A0i,A0 in enumerate(A0_list):
        print("\n\n\tDoing calculations for A0 = %1.3f a.u." % A0)
        atoms = [("H", -bond_length/2, 0, 0), ("H", bond_length/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

        # Cavity parameters
        cavity_freq     = np.array([20/27.2114])
        lam             = np.sqrt(2*cavity_freq) * A0 # Convert from A0 to lambda coupling
        cavity_coupling = lam
        cavity_vec      = np.array([np.array([1,0,0])])
        cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec)

        # QED-HF
        #mf       = scf.RHF(mol)
        #mf.kernel()
        #dm       = mf.make_rdm1()
        qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        qedmf.max_cycle = 5000
        #E_QEDHF[A0i] = qedmf.kernel(dm0=dm) # Supply initial guess for QED-HF from HF
        E_QEDHF[A0i] = qedmf.kernel()

        # QMC params
        num_walkers     = 5000 # 5000 is converged for this system
        dt              = 0.1
        total_time      = 10.0

        # QED-AFQMC
        afqmc_obj       = afqmc.QEDAFQMC(mol, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies = afqmc_obj.kernel()
        E_AFQMC_QED.append( np.array(np.real(energies)) )
        
        print( "QED-AFQMC Energy: %1.6f" % np.average(E_AFQMC_QED[-1][len(times)//4:], axis=-1) )
    
        if ( A0i == 0 ):
            time_list = np.array(times)





    E_AFQMC_QED = np.array(E_AFQMC_QED).real
    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_AFQMC_QED   = E_AFQMC_QED[:,EQ_TIME:]
    AFQMC_AVE_QED = np.average(E_AFQMC_QED, axis=-1)
    AFQMC_STD_QED = np.std(E_AFQMC_QED, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    

    plt.errorbar(A0_list, AFQMC_AVE_QED - AFQMC_AVE_QED[0], yerr=AFQMC_STD_QED, fmt='o-', elinewidth=1, ecolor='black', capsize=2, c='black', mfc='black', label="QED-AFQMC")
    plt.plot(A0_list, E_QEDHF - E_QEDHF[0], 's--', c='red', lw=2, label="QED-HF")
    plt.xlabel("Coupling Strength, $A_0$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(0,A0_list[-1])
    plt.ylim(0,0.5)
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis), dpi=300)
    plt.clf()

