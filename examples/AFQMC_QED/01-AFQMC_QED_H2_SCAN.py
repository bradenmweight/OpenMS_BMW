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

    basis        = "ccpvdz"
    photon_basis = 'fock'
    NFock        = 3
    bmin = 1.0
    bmax = 6.0
    bond_list_coarse = np.arange( bmin,bmax+0.5,0.5 ) # Bohr
    #bond_list_coarse = [10.] # np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    bond_list_fine   = np.arange( bmin,bmax+0.05,0.05 ) # Bohr
    #bond_list_fine   = np.arange( bmin,bmax+0.025,0.025 ) # Bohr

    time_list      = []
    E_AFQMC        = []
    WFN_AFQMC      = []
    WFN_AFQMC_QED  = []
    PHOT_AFQMC_QED = []
    E_AFQMC_QED    = []
    E_HF           = np.zeros( len(bond_list_fine) )
    E_QEDHF        = np.zeros( len(bond_list_coarse) )
    E_FCI          = np.zeros( len(bond_list_fine) )


    # Cavity parameters
    cavity_freq     = np.array([20/27.2114])
    if ( len(sys.argv) == 2 ):
        lam = float(sys.argv[1])
        print("Running lambda = %1.3f" % lam)
        cavity_coupling = np.array([lam]) # np.sqrt(2*cavity_freq) * 0.1 # Convert from A0 to lambda coupling
    else:
        cavity_coupling = np.array([0.0]) # np.sqrt(2*cavity_freq) * 0.1 # Convert from A0 to lambda coupling
    
    cavity_vec      = np.array([np.array([1,0,0])])
    cavity_mode     = np.einsum("m,md->md", cavity_coupling, cavity_vec ) / np.linalg.norm(cavity_vec,axis=-1)



    print("\n\tDoing HF and FCI calculations on a fine grid.")
    for bi,b in enumerate(bond_list_fine):
        print("Doing calculations for R(H-H) = %1.3f Bohr." % b)

        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

        # HF
        mf       = scf.RHF(mol)
        E_HF[bi] = mf.kernel()

        # FCI
        fcisolver = fci.FCI(mf)
        E_FCI[bi] = fcisolver.kernel()[0]

    np.savetxt("%s/01-AFQMC_H2_SCAN_FCI_HF_basis_%s.dat" % (DATA_DIR,basis), np.c_[bond_list_fine, E_HF, E_FCI], fmt="%1.6f", header="R_HH  E_HF  E_FCI")



    print("\n\tDoing AFQMC calculations on a coarse grid.")
    for bi,b in enumerate(bond_list_coarse):
        print("\n\n\tDoing calculations for R(H-H) = %1.3f Bohr." % b)
        atoms = [("H", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

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
                E_QEDHF[bi] = float('nan')
            else:
                E_QEDHF[bi] = qedmf.e_tot
                print( "Energy QED-HF:", E_QEDHF[bi] )
        except np.linalg.LinAlgError:
            print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
            E_QEDHF[bi] = float('nan')

        # QMC params
        num_walkers     = 5000 # 5000 is converged for this system
        dt              = 0.1 # 0.1 is converged for this system
        total_time      = 10.0
        
        # AFQMC
        T0 = time()
        afqmc_obj            = afqmc.AFQMC(mol, compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies, wfn = afqmc_obj.kernel()
        E_AFQMC.append( np.array(np.real(energies)) )
        WFN_AFQMC.append( wfn )
        print( "AFQMC Final Wavefunction:\n", WFN_AFQMC[-1][-1] )
        print( "NORM  Final Wavefunction:\n", np.linalg.det(np.einsum("Faj,Fak->jk", WFN_AFQMC[-1][-1].conj(), WFN_AFQMC[-1][-1] ) ) )
        print( "AFQMC Time: %1.3f seconds" % (time() - T0) )

        # QED-AFQMC
        T0 = time()
        afqmc_obj            = afqmc.QEDAFQMC(mol, compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, energies, wfn = afqmc_obj.kernel()
        E_AFQMC_QED.append( np.array(np.real(energies)) )
        WFN_AFQMC_QED.append( wfn )
        print( np.shape(WFN_AFQMC_QED[-1][-1]) )
        print( "QED-AFQMC Final Wavefunction:\n", WFN_AFQMC_QED[-1][-1] )
        print( "NORM      Final Wavefunction:\n", np.linalg.det(np.einsum("Faj,Fak->jk", WFN_AFQMC_QED[-1][-1].conj(), WFN_AFQMC_QED[-1][-1] ) ) )
        print( "QED-AFQMC Time: %1.3f seconds" % (time() - T0) )
        
        print( "AFQMC Energy: %1.6f" % np.average(E_AFQMC[-1][len(times)//4:], axis=-1) )
        print( "QED-AFQMC Energy: %1.6f" % np.average(E_AFQMC_QED[-1][len(times)//4:], axis=-1) )
    
        if ( bi == 0 ):
            time_list = np.array(times)


    WFN_AFQMC     = np.array(WFN_AFQMC)
    WFN_AFQMC_QED = np.array(WFN_AFQMC_QED)
    E_AFQMC       = np.array(E_AFQMC).real
    E_AFQMC_QED   = np.array(E_AFQMC_QED).real

    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    WFN_AFQMC     = WFN_AFQMC[:,EQ_TIME:]
    WFN_AFQMC_QED = WFN_AFQMC_QED[:,EQ_TIME:]
    E_AFQMC       = E_AFQMC[:,EQ_TIME:]
    E_AFQMC_QED   = E_AFQMC_QED[:,EQ_TIME:]
    AFQMC_AVE     = np.average(E_AFQMC, axis=1)
    WFN_AFQMC     = np.average(WFN_AFQMC, axis=1)
    WFN_AFQMC_QED = np.average(WFN_AFQMC_QED, axis=1)
    AFQMC_AVE_QED = np.average(E_AFQMC_QED, axis=1)
    AFQMC_STD     = np.std(E_AFQMC, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    AFQMC_STD_QED = np.std(E_AFQMC_QED, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    

    plt.plot(bond_list_fine, E_HF , '-', c='red', lw=2, label="HF")
    plt.plot(bond_list_coarse, E_QEDHF , '--', c='red', lw=2, label="QED-HF")
    plt.plot(bond_list_fine, E_FCI, '-', c='blue', lw=2, label="FCI")
    plt.plot(bond_list_fine, E_FCI + 0.5 * np.sum(cavity_freq), '--', c='blue', lw=2, label="FCI + $\\frac{1}{2}\\omega_\\mathrm{c}$")
    plt.errorbar(bond_list_coarse, AFQMC_AVE, yerr=AFQMC_STD, fmt='o', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC")
    plt.errorbar(bond_list_coarse, AFQMC_AVE_QED, yerr=AFQMC_STD_QED, fmt='o', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    plt.ylim(np.min(E_FCI)*1.1,0.0)
    plt.tight_layout()
    plt.savefig("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), dpi=300)
    plt.clf()
    np.savetxt("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), np.c_[bond_list_coarse, AFQMC_AVE, AFQMC_AVE_QED, E_QEDHF], fmt="%1.6f", header="R_HH  AFQMC  QED-AFQMC  QED-HF")


    PHOT_AFQMC_QED = np.einsum( "TFaj,F,TFaj->T", WFN_AFQMC_QED.conj(), np.arange(NFock), WFN_AFQMC_QED )
    plt.plot(bond_list_coarse, PHOT_AFQMC_QED.real, '-', c='black', lw=2, label="Re[$\\langle \\hat{a}^\\dag \\hat{a} \\rangle$]")
    plt.plot(bond_list_coarse, PHOT_AFQMC_QED.imag, '--', c='red', lw=2, label="Im[$\\langle \\hat{a}^\\dag \\hat{a} \\rangle$]")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Average Photon Number, $\\langle \\hat{a}^\\dag \\hat{a} \\rangle$", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    plt.tight_layout()
    plt.savefig("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f_PHOTON_NUMBER.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), dpi=300)
    plt.clf()
    np.savetxt("%s/01-AFQMC_H2_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f_PHOTON_NUMBER.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), np.c_[bond_list_coarse, PHOT_AFQMC_QED], fmt="%1.6f", header="R_HH  PHOT_AFQMC_QED")
