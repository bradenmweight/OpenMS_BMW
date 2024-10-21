from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import sys
from time import time

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "02-AFQMC_QED_COUPLING_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis            = "sto3g"
    photon_basis     = 'fock'
    NFock            = 1
    LAMmin           = 0.0
    LAMmax           = 0.1
    dLAM             = 0.025
    coupling_list_coarse = np.arange( LAMmin,LAMmax+dLAM,dLAM )

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


    atoms = [("Li", -3.5/2, 0, 0), ("H", 3.5/2, 0, 0)] # ~3.5 a.u. is GS minimum
    mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

    # QMC params
    num_walkers     = 5000 # 5000 is converged for this system
    dt              = 0.25 # 0.1 is converged for this system
    total_time      = 4.0 # 10.0 is converged for this system


    # # HF
    # mf       = scf.RHF(mol)
    # E_rHF    = mf.kernel()

    # # UHF -- Need to break symmetry of initial guess to get right solution
    # mf                = scf.UHF(mol)
    # dm_alpha, dm_beta = mf.get_init_guess()
    # dm_beta[:2,:2]    = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
    # dm                = (dm_alpha,dm_beta)
    # E_uHF             = mf.kernel(dm)

    # # FCI
    # E_FCI = fci.FCI(mf).kernel()[0]



    # # rAFQMC
    # afqmc_obj          = afqmc.AFQMC(mol, trial="RHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    # times, E_rAFQMC, _ = afqmc_obj.kernel()

    # # uAFQMC
    # afqmc_obj          = afqmc.AFQMC(mol, trial="UHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
    # times, E_uAFQMC, _ = afqmc_obj.kernel()


    def get_ave_N( wfn ):
        wfn    = np.array( wfn )[len(times)//4:]
        NORM   = np.einsum( "tFSaj,tFSak->tSjk", wfn.conj(), wfn )
        NORM   = np.linalg.det( NORM )
        NORM   = np.prod( NORM, axis=-1 )
        print("WFN NORM", NORM)
        ave_N  = np.einsum( "tFSaj,F,tFSak,t->tSjk", wfn, np.arange(NFock), wfn, 1/NORM )
        ave_N  = np.linalg.det( ave_N )
        ave_N  = np.prod( ave_N, axis=-1 )
        ave_N  = np.average( ave_N )
        return ave_N

    print("\n\tDoing AFQMC calculations.")
    for LAMi,LAM in enumerate(coupling_list_coarse):
        print("\n\n\tDoing calculations for lambda_c = %1.3f a.u." % LAM)

        cavity_coupling = np.array([LAM]) # = np.sqrt(2*cavity_freq) * A0 # Convert from A0 to lambda coupling
        cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)

        try:
            # QED-HF
            #mf       = scf.RHF(mol)
            #mf.kernel()
            #dm       = mf.make_rdm1()
            #E_QEDHF[LAMi] = qedmf.kernel(dm0=dm) # Supply initial guess for QED-HF from HF
            qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
            qedmf.max_cycle = 500
            qedmf.kernel()
            if ( qedmf.conv_check == False ):
                print("   Warning! QED-HF did not converge. Setting energy to NaN.")
                E_rHF_QED[LAMi] = float('nan')
            else:
                E_rHF_QED[LAMi] = qedmf.e_tot
                print( "Energy QED-HF:", E_rHF_QED[LAMi] )
        except np.linalg.LinAlgError:
            print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
            E_rHF_QED[LAMi] = float('nan')

        # QED-rAFQMC
        afqmc_obj = afqmc.QEDAFQMC(mol, trial="RHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, e_rAFQMC_QED, wfn_rAFQMC_QED = afqmc_obj.kernel()
        E_rAFQMC_QED.append(e_rAFQMC_QED)
        ave_N = get_ave_N( wfn_rAFQMC_QED )
        print("QED-AFQMC Energy:", np.average(np.array(e_rAFQMC_QED)[len(times)//4:]) )
        print("Average Photon Number:", ave_N )
        
        # # QED-uAFQMC
        # afqmc_obj = afqmc.QEDAFQMC(mol, trial="UHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        # times, e_uAFQMC_QED, _ = afqmc_obj.kernel()
        # E_uAFQMC_QED.append(e_uAFQMC_QED)
        # get_ave_N( wfn_rAFQMC_QED )
        # print("Average Photon Number:", ave_N )
    

        if ( LAMi == 0 ):
            time_list = np.array(times)


    # E_rAFQMC       = np.array(E_rAFQMC).real
    # E_uAFQMC       = np.array(E_uAFQMC).real
    E_rAFQMC_QED   = np.array(E_rAFQMC_QED).real
    E_uAFQMC_QED   = np.array(E_uAFQMC_QED).real

    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    # E_rAFQMC       = E_rAFQMC[EQ_TIME:]
    # E_uAFQMC       = E_uAFQMC[EQ_TIME:]
    E_rAFQMC_QED   = E_rAFQMC_QED[:,EQ_TIME:]
    # E_uAFQMC_QED   = E_uAFQMC_QED[:,EQ_TIME:]
    # E_rAFQMC_AVE   = np.average(E_rAFQMC)
    # E_uAFQMC_AVE   = np.average(E_uAFQMC)
    E_rAFQMC_AVE_QED = np.average(E_rAFQMC_QED, axis=1)
    # E_uAFQMC_AVE_QED = np.average(E_uAFQMC_QED, axis=1)
    # E_rAFQMC_STD     = np.std(E_rAFQMC)
    # E_uAFQMC_STD     = np.std(E_uAFQMC)
    E_rAFQMC_STD_QED = np.std(E_rAFQMC_QED, axis=1) 
    # E_uAFQMC_STD_QED = np.std(E_uAFQMC_QED, axis=1) 
    

    plt.plot(coupling_list_coarse,   E_rHF_QED - E_rHF_QED[0], '--', c='red', lw=2, label="QED-HF")
    plt.errorbar(coupling_list_coarse, E_rAFQMC_AVE_QED - E_rAFQMC_AVE_QED[0], yerr=E_rAFQMC_STD_QED, fmt='o', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@RHF")
    # plt.errorbar(coupling_list_coarse, E_uAFQMC_AVE_QED - E_uAFQMC_AVE_QED[0], yerr=E_uAFQMC_STD_QED, fmt='s', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@UHF")

    #plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_rHF , '-', c='red', lw=2, label="HF")
    # plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_FCI, '-', c='blue', lw=2, label="FCI")
    # plt.plot(coupling_list_coarse,   coupling_list_coarse*0 + E_FCI + 0.5 * np.sum(cavity_freq), '--', c='blue', lw=2, label="FCI + $\\frac{1}{2}\\omega_\\mathrm{c}$")
    # plt.errorbar(coupling_list_coarse, coupling_list_coarse*0 + E_rAFQMC_AVE, yerr=E_rAFQMC_STD, fmt='o', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC@RHF")
    # plt.errorbar(coupling_list_coarse, coupling_list_coarse*0 + E_uAFQMC_AVE, yerr=E_uAFQMC_STD, fmt='s', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC@UHF")
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(LAMmin,LAMmax)
    #plt.ylim(np.min(E_FCI)*1.1,0.0)
    #plt.ylim(0.0, 0.5)
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFOCK_%d_LAM_SCAN.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), dpi=300)
    plt.clf()

    np.savetxt("%s/02-AFQMC_H2_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFOCK_%d_LAM_SCAN.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), np.c_[coupling_list_coarse, E_rHF_QED, E_rAFQMC_AVE_QED, E_rAFQMC_STD_QED ], fmt="%1.6f", header="R_HH  QED-RHF  QED-AFQMC@RHF  STD(QED-AFQMC@RHF)")

