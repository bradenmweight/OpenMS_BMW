from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import sys
from time import time

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "00"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis        = "321g"
    photon_basis = 'fock'
    NFock        = 3

    # Cavity parameters
    cavity_freq     = np.array([2/27.2114])
    cavity_coupling = np.array([0.0]) # np.sqrt(2*cavity_freq) * 0.1 # Convert from A0 to lambda coupling
    cavity_vec      = np.array([np.array([1,0,0])])
    cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)



    atoms = [("H", 0.0, 0, 0), ("H", 2.0, 0, 0)]
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


    # UHF -- Need to break symmetry of initial guess to get right solution
    mf = scf.UHF(mol)
    dm_alpha, dm_beta = mf.get_init_guess()
    dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    dm = (dm_alpha,dm_beta)
    E_UHF = mf.kernel(dm)


    # QMC params
    num_walkers     = 5000 # 5000 is converged for this system
    dt              = 0.1 # 0.1 is converged for this system
    total_time      = 10.0 # 10.0
    
    # rAFQMC
    T0 = time()
    afqmc_obj                 = afqmc.AFQMC(mol, trial="RHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_rAFQMC, _        = afqmc_obj.kernel()
    print( "AFQMC Time: %1.3f seconds" % (time() - T0) )

    # uAFQMC
    T0 = time()
    afqmc_obj                 = afqmc.AFQMC(mol, trial="UHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_uAFQMC, _        = afqmc_obj.kernel()
    print( "AFQMC Time: %1.3f seconds" % (time() - T0) )


    # QED-rAFQMC
    T0 = time()
    afqmc_obj                = afqmc.QEDAFQMC(mol, trial="RHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_rAFQMC_QED, _   = afqmc_obj.kernel()
    print( "QED-AFQMC Time: %1.3f seconds" % (time() - T0) )
    
    # QED-uAFQMC
    T0 = time()
    afqmc_obj                = afqmc.QEDAFQMC(mol, trial="UHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_uAFQMC_QED, _   = afqmc_obj.kernel()
    print( "QED-AFQMC Time: %1.3f seconds" % (time() - T0) )

    


    times          = np.array(times)
    E_rAFQMC       = np.array(E_rAFQMC).real
    E_uAFQMC       = np.array(E_uAFQMC).real
    E_rAFQMC_QED   = np.array(E_rAFQMC_QED).real
    E_uAFQMC_QED   = np.array(E_uAFQMC_QED).real

    E_rAFQMC_AVE     = np.average(E_rAFQMC[len(times)//4:]).real
    E_uAFQMC_AVE     = np.average(E_uAFQMC[len(times)//4:]).real
    E_rAFQMC_QED_AVE = np.average(E_rAFQMC_QED[len(times)//4:]).real
    E_uAFQMC_QED_AVE = np.average(E_uAFQMC_QED[len(times)//4:]).real
    print( "rAFQMC Energy: %1.6f" % E_rAFQMC_AVE)
    print( "uAFQMC Energy: %1.6f" % E_uAFQMC_AVE)
    print( "rQED-AFQMC Energy: %1.6f" % E_rAFQMC_QED_AVE)
    print( "uQED-AFQMC Energy: %1.6f" % E_uAFQMC_QED_AVE)




    plt.plot( times, E_rAFQMC, "-", c="blue" )
    plt.plot( times, E_uAFQMC, "--", c="blue" )
    plt.plot( times, E_rAFQMC_QED, "-", c='green' )
    plt.plot( times, E_uAFQMC_QED, "--", c='green' )
    plt.plot( times, times*0 + E_rAFQMC_AVE, "-", lw=5, alpha=0.5,c="blue", label="rAFQMC" )
    plt.plot( times, times*0 + E_uAFQMC_AVE, "--", lw=5, alpha=0.5,c="blue", label="uAFQMC" )
    plt.plot( times, times*0 + E_rAFQMC_QED_AVE, "-", lw=3, alpha=0.5, c='green', label="QED-rAFQMC" )
    plt.plot( times, times*0 + E_uAFQMC_QED_AVE, "--", lw=3, alpha=0.5, c='green', label="QED-uAFQMC" )
    plt.plot( times, times*0 + E_HF, "-", c="red", label="RHF" )
    plt.plot( times, times*0 + E_UHF, "-.", c="red", label="UHF" )
    plt.plot( times, times*0 + E_QEDHF, "--", c="red", label="QED-HF" )
    plt.plot( times, times*0 + E_FCI, "-", c='black', label="FCI" )
    plt.plot( times, times*0 + E_FCI+cavity_freq[0]/2, "--", c='black', label="FCI+$\\frac{1}{2}\\hbar \\omega_\\mathrm{c}$" )
    plt.legend()
    plt.xlim(times[0], times[-1])
    #plt.ylim(-1.2, -0.4)
    plt.xlabel("Projection Time, $\\tau$ (a.u.)", fontsize=15)
    plt.ylabel("Ground State Energy, $E_0$ (a.u.)", fontsize=15)
    plt.title("$\\lambda$ = %1.2f a.u.  $\\omega_\\mathrm{c}$ = %1.2f a.u." % (cavity_coupling[0], cavity_freq[0]), fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/time_traj.png", dpi=300)


