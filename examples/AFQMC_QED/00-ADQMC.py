from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import sys
from time import time

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "01-AFQMC_QED_H2_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis        = "sto3g"
    photon_basis = 'fock'
    NFock        = 5

    # Cavity parameters
    cavity_freq     = np.array([2/27.2114])
    cavity_coupling = np.array([0.05]) # np.sqrt(2*cavity_freq) * 0.1 # Convert from A0 to lambda coupling
    cavity_vec      = np.array([np.array([1,0,0])])
    cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)



    atoms = [("H", 0.0, 0, 0), ("H", 10.0, 0, 0)]
    mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

    # HF
    mf       = scf.RHF(mol)
    E_HF     = mf.kernel()

    # FCI
    E_FCI     = fci.FCI(mf).kernel()[0]


    try:
        qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        qedmf.max_cycle = 5000
        qedmf.kernel()
        if ( qedmf.conv_check == False ):
            print("   Warning! QED-HF did not converge. Setting energy to NaN.")
            E_QEDHF = float('nan')
        else:
            E_QEDHF = qedmf.e_tot
            print( "Energy QED-HF:", E_QEDHF )
    except np.linalg.LinAlgError:
        print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
        E_QEDHF = float('nan')



    # QMC params
    num_walkers     = 5000 # 5000 is converged for this system
    dt              = 0.01 # 0.1 is converged for this system
    total_time      = 10.0
    
    # AFQMC
    T0 = time()
    afqmc_obj                 = afqmc.AFQMC(mol, compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_AFQMC, WFN_AFQMC = afqmc_obj.kernel()
    #print( "AFQMC Final Wavefunction:\n", WFN_AFQMC[-1] )
    #print( "NORM  Final Wavefunction:\n", np.linalg.det(np.einsum("Faj,Fak->jk", WFN_AFQMC[-1].conj(), WFN_AFQMC[-1] ) ) )
    print( "AFQMC Time: %1.3f seconds" % (time() - T0) )

    # QED-AFQMC
    T0 = time()
    afqmc_obj = afqmc.QEDAFQMC(mol, compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_AFQMC_QED, WFN_AFQMC_QED = afqmc_obj.kernel()
    #print( "QED-AFQMC Final Wavefunction:\n", WFN_AFQMC_QED[-1] )
    #print( "NORM      Final Wavefunction:\n", np.linalg.det(np.einsum("Faj,Fak->jk", WFN_AFQMC_QED[-1].conj(), WFN_AFQMC_QED[-1] ) ) )
    print( "QED-AFQMC Time: %1.3f seconds" % (time() - T0) )
    



    times         = np.array(times)
    WFN_AFQMC     = np.array(WFN_AFQMC)
    WFN_AFQMC_QED = np.array(WFN_AFQMC_QED)
    E_AFQMC       = np.array(E_AFQMC).real
    E_AFQMC_QED   = np.array(E_AFQMC_QED).real

    E_AFQMC_AVE     = np.average(E_AFQMC[len(times)//4:]).real
    E_AFQMC_QED_AVE = np.average(E_AFQMC_QED[len(times)//4:]).real
    print( "AFQMC Energy: %1.6f" % E_AFQMC_AVE)
    print( "QED-AFQMC Energy: %1.6f" % E_AFQMC_QED_AVE)




    rho_ph  = np.einsum( "TFaj,TGak->TFGjk", WFN_AFQMC_QED.conj(), WFN_AFQMC_QED ) # (Time,Fock,MO,MO)
    rho_ph  = np.linalg.det( rho_ph ).real # (Time,Fock)
    PHOT_AFQMC_QED = np.einsum( "F,TFF->T", np.arange(NFock), rho_ph ) # Measure photon number
    PHOT_AFQMC_QED = np.average( PHOT_AFQMC_QED[len(times)//4:] )

    plt.plot( times, E_AFQMC, "-", c="blue" )
    plt.plot( times, E_AFQMC_QED, "--", c='blue' )
    plt.plot( times, times*0 + E_AFQMC_AVE, "-", lw=5, alpha=0.5,c="blue", label="AFQMC" )
    plt.plot( times, times*0 + E_AFQMC_QED_AVE, "--", lw=5, alpha=0.5, c='blue', label="QED-AFQMC" )
    plt.plot( times, times*0 + E_HF, "-", c="red", label="HF" )
    plt.plot( times, times*0 + E_QEDHF, "--", c="red", label="QED-HF" )
    plt.plot( times, times*0 + E_FCI, "-", c='black', label="FCI" )
    plt.plot( times, times*0 + E_FCI+cavity_freq[0]/2, "--", c='black', label="FCI+$\\frac{1}{2}\\hbar \\omega_\\mathrm{c}$" )
    plt.legend()
    plt.xlim(times[0], times[-1])
    plt.ylim(-1.2, -0.4)
    plt.xlabel("Projection Time, $\\tau$ (a.u.)", fontsize=15)
    plt.ylabel("Ground State Energy, $E_0$ (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u.  $\\omega_\\mathrm{c}$ = %1.2f a.u. $\\langle \\hat{a}^\\dag \\hat{a} \\rangle$ = %1.2f" % (cavity_coupling[0], cavity_freq[0], PHOT_AFQMC_QED), fontsize=15)
    plt.tight_layout()
    plt.savefig("time_traj.png", dpi=300)


