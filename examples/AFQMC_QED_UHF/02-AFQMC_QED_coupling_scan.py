from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import sys
from time import time

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "02-AFQMC_QED_coupling_scan"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis            = "321g"
    photon_basis     = 'fock'
    NFock            = 3
    bmin             = 0.0
    bmax             = 0.5
    coupling_list_coarse = np.arange( bmin,bmax+0.05,0.05 ) # Bohr

    time_list      = []
    E_rAFQMC_QED   = []
    E_uAFQMC_QED   = []
    E_rHF          = 0
    E_uHF          = 0
    E_FCI          = 0
    E_rHF_QED      = np.zeros( len(coupling_list_coarse) )


    # Cavity parameters
    cavity_freq     = np.array([0.1]) # a.u.
    cavity_vec      = np.array([np.array([1,0,0])])


    atoms = [("H", -2.8/2, 0, 0), ("H", 2.8/2, 0, 0)]
    mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

    # HF
    mf       = scf.RHF(mol)
    E_rHF    = mf.kernel()

    # UHF -- Need to break symmetry of initial guess to get right solution
    mf                = scf.UHF(mol)
    dm_alpha, dm_beta = mf.get_init_guess()
    dm_beta[:2,:2]    = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    dm                = (dm_alpha,dm_beta)
    E_uHF             = mf.kernel(dm)

    # FCI
    fcisolver = fci.FCI(mf)
    E_FCI = fcisolver.kernel()[0]



    # QMC params
    num_walkers     = 25000 # 5000 is converged for this system
    dt              = 0.01 # 0.1 is converged for this system
    total_time      = 10.0

    # rAFQMC
    afqmc_obj          = afqmc.AFQMC(mol, trial="RHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_rAFQMC, _ = afqmc_obj.kernel()

    # uAFQMC
    afqmc_obj          = afqmc.AFQMC(mol, trial="UHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    times, E_uAFQMC, _ = afqmc_obj.kernel()


    print("\n\tDoing AFQMC calculations.")
    for bi,b in enumerate(coupling_list_coarse):
        print("\n\n\tDoing calculations for R(H-H) = %1.3f Bohr." % b)

        cavity_coupling = np.array([b]) # np.sqrt(2*cavity_freq) * 0.1 # Convert from A0 to lambda coupling
        cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)

        try:
            # QED-HF
            #mf       = scf.RHF(mol)
            #mf.kernel()
            #dm       = mf.make_rdm1()
            #E_QEDHF[bi] = qedmf.kernel(dm0=dm) # Supply initial guess for QED-HF from HF
            qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
            qedmf.max_cycle = 5000
            qedmf.kernel()
            if ( qedmf.conv_check == False ):
                print("   Warning! QED-HF did not converge. Setting energy to NaN.")
                E_rHF_QED[bi] = float('nan')
            else:
                E_rHF_QED[bi] = qedmf.e_tot
                print( "Energy QED-HF:", E_rHF_QED[bi] )
        except np.linalg.LinAlgError:
            print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
            E_rHF_QED[bi] = float('nan')

        # QED-rAFQMC
        afqmc_obj = afqmc.QEDAFQMC(mol, trial="RHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, e_rAFQMC_QED, _ = afqmc_obj.kernel()
        E_rAFQMC_QED.append(e_rAFQMC_QED)
        
        # QED-uAFQMC
        afqmc_obj = afqmc.QEDAFQMC(mol, trial="UHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, e_uAFQMC_QED, _ = afqmc_obj.kernel()
        E_uAFQMC_QED.append(e_uAFQMC_QED)
    

        if ( bi == 0 ):
            time_list = np.array(times)


    E_rAFQMC       = np.array(E_rAFQMC).real
    E_uAFQMC       = np.array(E_uAFQMC).real
    E_rAFQMC_QED   = np.array(E_rAFQMC_QED).real
    E_uAFQMC_QED   = np.array(E_uAFQMC_QED).real

    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_rAFQMC       = E_rAFQMC[EQ_TIME:]
    E_uAFQMC       = E_uAFQMC[EQ_TIME:]
    E_rAFQMC_QED   = E_rAFQMC_QED[:,EQ_TIME:]
    E_uAFQMC_QED   = E_uAFQMC_QED[:,EQ_TIME:]
    E_rAFQMC_AVE   = np.average(E_rAFQMC)
    E_uAFQMC_AVE   = np.average(E_uAFQMC)
    E_rAFQMC_AVE_QED = np.average(E_rAFQMC_QED, axis=1)
    E_uAFQMC_AVE_QED = np.average(E_uAFQMC_QED, axis=1)
    E_rAFQMC_STD     = np.std(E_rAFQMC) # Since this is a biased walk, we also need to add correlated error TODO
    E_uAFQMC_STD     = np.std(E_uAFQMC) # Since this is a biased walk, we also need to add correlated error TODO
    E_rAFQMC_STD_QED = np.std(E_rAFQMC_QED, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    E_uAFQMC_STD_QED = np.std(E_uAFQMC_QED, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    

    plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_rHF , '-', c='red', lw=2, label="HF")
    plt.plot(coupling_list_coarse,   E_rHF_QED , '--', c='red', lw=2, label="QED-HF")
    plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_FCI, '-', c='blue', lw=2, label="FCI")
    plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_FCI + 0.5 * np.sum(cavity_freq), '--', c='blue', lw=2, label="FCI + $\\frac{1}{2}\\omega_\\mathrm{c}$")
    plt.errorbar(coupling_list_coarse, coupling_list_coarse*0 + E_rAFQMC_AVE, yerr=E_rAFQMC_STD, fmt='o', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC@RHF")
    plt.errorbar(coupling_list_coarse, coupling_list_coarse*0 + E_uAFQMC_AVE, yerr=E_uAFQMC_STD, fmt='s', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC@UHF")
    plt.errorbar(coupling_list_coarse, E_rAFQMC_AVE_QED, yerr=E_rAFQMC_STD_QED, fmt='o', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@RHF")
    plt.errorbar(coupling_list_coarse, E_uAFQMC_AVE_QED, yerr=E_uAFQMC_STD_QED, fmt='s', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@UHF")
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    # plt.ylim(np.min(E_FCI)*1.1,0.0)
    plt.tight_layout()
    plt.savefig("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), dpi=300)
    plt.clf()
    np.savetxt("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), np.c_[coupling_list_coarse, E_rAFQMC_AVE, E_uAFQMC_AVE, E_rAFQMC_AVE_QED, E_uAFQMC_AVE_QED, E_rHF_QED], fmt="%1.6f", header="R_HH  AFQMC@RHF AFQMC@UHR  QED-AFQMC@RHF  QED-AFQMC@UHF  QED-RHF")

