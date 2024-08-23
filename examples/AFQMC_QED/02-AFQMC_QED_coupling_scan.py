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

    basis        = "sto3g"
    photon_basis = 'fock'
    NFock        = 3
    bond_length  = 1.5 # 2.8 # Bohr
    lam_list     = np.arange( 0.0, 0.5+0.1, 0.1 )



    atoms = [("H", -bond_length/2, 0, 0), ("H", bond_length/2, 0, 0)]
    mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

    # HF
    mf       = scf.RHF(mol)
    mf.kernel()

    # FCI
    E_FCI = fci.FCI(mf).kernel()[0]



    time_list   = []
    E_AFQMC_QED = []
    E_QEDHF     = np.zeros( len(lam_list) )

    WFN_AFQMC_QED   = []

    print("\n\tDoing QED calculations on a coarse grid.")
    for lami,lam in enumerate(lam_list):
        print("\n\n\tDoing calculations for lamda = %1.3f a.u." % lam)
        atoms = [("H", -bond_length/2, 0, 0), ("H", bond_length/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

        # Cavity parameters
        cavity_freq     = np.array([2/27.2114])
        cavity_coupling = np.array([lam])
        cavity_vec      = np.array([np.array([1,0,0])])
        cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec)

        try:
            # QED-HF
            #mf       = scf.RHF(mol)
            #mf.kernel()
            #dm       = mf.make_rdm1()
            #E_QEDHF[bi] = qedmf.kernel(dm0=dm) # Supply initial guess for QED-HF from HF
            qedmf = mqed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
            qedmf.max_cycle = 500
            qedmf.kernel()
            if ( qedmf.conv_check == False ):
                print("   Warning! QED-HF did not converge. Setting energy to NaN.")
                E_QEDHF[lami] = float('nan')
            else:
                E_QEDHF[lami] = qedmf.e_tot
                print( "Energy QED-HF:", E_QEDHF[lami] )
        except np.linalg.LinAlgError:
            print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
            E_QEDHF[lami] = float('nan')
        
        # QMC params
        num_walkers     = 5000
        dt              = 0.01
        total_time      = 10.0

        # QED-AFQMC
        T0 = time()
        afqmc_obj       = afqmc.QEDAFQMC(mol, compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies, wfn = afqmc_obj.kernel()
        E_AFQMC_QED.append( np.array(np.real(energies)) )
        WFN_AFQMC_QED.append( wfn )
        print( "AFQMC Final Wavefunction:\n", wfn[-1] )
        print( "NORM      Final Wavefunction:\n", np.linalg.det(np.einsum("Faj,Fak->jk", WFN_AFQMC_QED[-1][-1].conj(), WFN_AFQMC_QED[-1][-1] ) ) )

        print( "QED-AFQMC Energy: %1.6f" % np.average(E_AFQMC_QED[-1][len(times)//4:], axis=-1) )
        print( "AFQMC Time      : %1.2f seconds" % (time()-T0) )

        if ( lami == 0 ):
            time_list = np.array(times)





    E_AFQMC_QED = np.array(E_AFQMC_QED).real
    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_AFQMC_QED   = E_AFQMC_QED[:,EQ_TIME:]
    AFQMC_AVE_QED = np.average(E_AFQMC_QED, axis=-1)
    AFQMC_STD_QED = np.std(E_AFQMC_QED, axis=-1) # Since this is a biased walk, we also need to add correlated error TODO
    

    WFN_AFQMC_QED = np.array(WFN_AFQMC_QED)
    WFN_AFQMC_QED = WFN_AFQMC_QED[:,EQ_TIME:]
    WFN_AFQMC_QED = np.average(WFN_AFQMC_QED, axis=1)



    plt.errorbar(lam_list, AFQMC_AVE_QED, yerr=AFQMC_STD_QED, fmt='o--', elinewidth=1, ecolor='black', capsize=2, c='black', mfc='black', label="QED-AFQMC")
    plt.plot(lam_list, E_QEDHF, 's--', c='red', lw=2, label="QED-HF")
    #plt.plot(A0_list, A0_list*0 + E_FCI, '-', c='blue', lw=2, label="FCI")
    plt.plot(lam_list, lam_list*0 + E_FCI + cavity_freq[0]/2, '--', c='blue', lw=2, label="FCI+$\\frac{1}{2}\\hbar \\omega_\\mathrm{c}$")
    plt.xlabel("Coupling Strength, $\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(lam_list[0],lam_list[-1])
    #plt.ylim(0,0.5)
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFock_%d.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), dpi=300)
    plt.clf()
    np.savetxt("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFock_%d.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), np.c_[lam_list, AFQMC_AVE_QED - AFQMC_AVE_QED[0], E_QEDHF - E_QEDHF[0]], fmt="%1.3f", header="lambda   AFQMC(lambda)-AFQMC(0)   HF(lambda)-HF(0)" )


    rho_ph  = np.einsum( "AFaj,AGak->AFGjk", WFN_AFQMC_QED.conj(), WFN_AFQMC_QED ) # (Time,Fock,MO,MO)
    rho_ph  = np.linalg.det( rho_ph ) # (Time,Fock)
    PHOT_AFQMC_QED = np.einsum( "F,AFF->A", np.arange(NFock), rho_ph ) # Measure photon number
    
    plt.plot(lam_list, PHOT_AFQMC_QED.real, '-', c='black', lw=2, label="$\\langle \\hat{a}^\\dag \\hat{a} \\rangle$")
    plt.xlabel("Coupling Strength, $\\\lambda_\\mathrm{c}$ (a.u.)", fontsize=15)
    plt.ylabel("Average Photon Number, $\\langle \\hat{a}^\\dag \\hat{a} \\rangle$", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(lam_list[0],lam_list[-1])
    plt.tight_layout()
    plt.savefig("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFock_%d_PHOTON_NUMBER.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), dpi=300)
    plt.clf()
    np.savetxt("%s/02-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_NFock_%d_PHOTON_NUMBER.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,NFock), np.c_[lam_list, PHOT_AFQMC_QED], fmt="%1.6f", header="Lambda  PHOT_AFQMC_QED")

