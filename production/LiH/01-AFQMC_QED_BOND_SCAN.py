from pyscf import gto, scf, fci
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import sys
from time import time

from openms import mqed
from openms.qmc import afqmc



if __name__ == "__main__":

    DATA_DIR = "01-AFQMC_QED_BOND_SCAN"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    basis        = "321g"
    photon_basis = 'fock'
    NFock        = 2
    bmin = 1.0
    bmax = 10.0
    bond_list_coarse = np.arange( bmin,bmax+0.25,0.25 ) # Bohr
    #bond_list_coarse = [10.] # np.arange( bmin,bmax+0.2,0.2 ) # Bohr
    bond_list_fine   = np.arange( bmin,bmax+0.05,0.05 ) # Bohr
    #bond_list_fine   = np.arange( bmin,bmax+0.025,0.025 ) # Bohr

    time_list      = []
    E_rAFQMC       = []
    E_uAFQMC       = []
    E_rAFQMC_QED   = []
    E_uAFQMC_QED   = []
    E_rHF          = np.zeros( len(bond_list_fine) )
    E_uHF          = np.zeros( len(bond_list_fine) )
    E_FCI          = np.zeros( len(bond_list_fine) )
    E_rHF_QED      = np.zeros( len(bond_list_coarse) )


    # Cavity parameters
    cavity_freq     = np.array([0.1]) # a.u.
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

        atoms = [("Li", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

        # HF
        mf       = scf.RHF(mol)
        E_rHF[bi] = mf.kernel()

        # UHF -- BMW: Need to break symmetry of initial guess to get right solution
        mf1 = scf.UHF(mol)
        dm_alpha, dm_beta = mf1.get_init_guess()
        dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
        dm = (dm_alpha,dm_beta)
        mf1.kernel(dm) # BMW: Pass in modified initial guess
        mf2 = scf.UHF(mol)
        dm_alpha, dm_beta = mf2.get_init_guess()
        dm_beta[:,:] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
        dm = (dm_alpha,dm_beta)
        mf2.kernel(dm) # BMW: Pass in modified initial guess

        if ( mf1.e_tot < mf2.e_tot ): # BMW: Check which symmetry breaking works... H2 is mf1 but LiH is mf2
            E_uHF[bi] = mf1.e_tot
        else:
            E_uHF[bi] = mf2.e_tot

        # FCI
        fcisolver = fci.FCI(mf)
        E_FCI[bi] = fcisolver.kernel()[0]

    np.savetxt("%s/01-AFQMC_QED_BOND_SCAN_FCI_HF_basis_%s.dat" % (DATA_DIR,basis), np.c_[bond_list_fine, E_rHF, E_uHF, E_FCI], fmt="%1.6f", header="R_HH  E_rHF  E_uHF  E_FCI")



    print("\n\tDoing AFQMC calculations on a coarse grid.")
    for bi,b in enumerate(bond_list_coarse):
        print("\n\n\tDoing calculations for R(H-H) = %1.3f Bohr." % b)
        atoms = [("Li", -b/2, 0, 0), ("H", b/2, 0, 0)]
        mol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=3)

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
                E_rHF_QED[bi] = float('nan')
            else:
                E_rHF_QED[bi] = qedmf.e_tot
                print( "Energy QED-HF:", E_rHF_QED[bi] )
        except np.linalg.LinAlgError:
            print("   Warning! QED-HF encountered LinAlgError. Setting energy to NaN.")
            E_rHF_QED[bi] = float('nan')

        # QMC params
        num_walkers     = 5000 # 5000 is converged for this system
        dt              = 0.01 # 0.01 is converged for this system
        total_time      = 10.0 # 10.0 is converged for this system
        
    
        # rAFQMC
        afqmc_obj          = afqmc.AFQMC(mol, trial="RHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, e_rAFQMC, _ = afqmc_obj.kernel()
        E_rAFQMC.append(e_rAFQMC)

        # uAFQMC
        afqmc_obj          = afqmc.AFQMC(mol, trial="UHF", compute_wavefunction=True, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        times, e_uAFQMC, _ = afqmc_obj.kernel()
        E_uAFQMC.append(e_uAFQMC)

        # # QED-rAFQMC
        # afqmc_obj = afqmc.QEDAFQMC(mol, trial="RHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        # times, e_rAFQMC_QED, _ = afqmc_obj.kernel()
        # E_rAFQMC_QED.append(e_rAFQMC_QED)
        
        # # QED-uAFQMC
        # afqmc_obj = afqmc.QEDAFQMC(mol, trial="UHF", compute_wavefunction=True, NFock=NFock, photon_basis=photon_basis, cavity_freq=cavity_freq, cavity_coupling=cavity_coupling, cavity_vec=cavity_vec, numdets=1, dt=dt, total_time=total_time, num_walkers=num_walkers, energy_scheme="hybrid")
        # times, e_uAFQMC_QED, _ = afqmc_obj.kernel()
        # E_uAFQMC_QED.append(e_uAFQMC_QED)
    

        if ( bi == 0 ):
            time_list = np.array(times)


    E_rAFQMC       = np.array(E_rAFQMC).real
    E_uAFQMC       = np.array(E_uAFQMC).real
    # E_rAFQMC_QED   = np.array(E_rAFQMC_QED).real
    # E_uAFQMC_QED   = np.array(E_uAFQMC_QED).real

    EQ_TIME = len(times)//4 # Choose to be first 25% of the projection time
    
    # Remove equilibration time and perform average
    E_rAFQMC       = E_rAFQMC[:,EQ_TIME:]
    E_uAFQMC       = E_uAFQMC[:,EQ_TIME:]
    # E_rAFQMC_QED   = E_rAFQMC_QED[:,EQ_TIME:]
    # E_uAFQMC_QED   = E_uAFQMC_QED[:,EQ_TIME:]
    E_rAFQMC_AVE   = np.average(E_rAFQMC, axis=1)
    E_uAFQMC_AVE   = np.average(E_uAFQMC, axis=1)
    # E_rAFQMC_AVE_QED = np.average(E_rAFQMC_QED, axis=1)
    # E_uAFQMC_AVE_QED = np.average(E_uAFQMC_QED, axis=1)
    E_rAFQMC_STD     = np.std(E_rAFQMC, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    E_uAFQMC_STD     = np.std(E_uAFQMC, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    # E_rAFQMC_STD_QED = np.std(E_rAFQMC_QED, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    # E_uAFQMC_STD_QED = np.std(E_uAFQMC_QED, axis=1) # Since this is a biased walk, we also need to add correlated error TODO
    

    plt.plot(bond_list_fine,   E_FCI, '-', c='green', lw=8, alpha=0.5, label="FCI")
    plt.plot(bond_list_fine,   E_rHF, '-', c='blue' , lw=8, alpha=0.5, label="RHF")
    plt.plot(bond_list_fine,   E_uHF, '-', c='blue' , lw=2,            label="UHF")
    plt.errorbar(bond_list_coarse, E_rAFQMC_AVE, yerr=E_rAFQMC_STD, fmt='o', linestyle="solid", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="AFQMC@RHF")
    plt.errorbar(bond_list_coarse, E_uAFQMC_AVE, yerr=E_uAFQMC_STD, fmt='o', linestyle="solid", elinewidth=1, ecolor='red', capsize=4, c='red', mfc='none', label="AFQMC@UHF")

    #plt.plot(bond_list_coarse, E_rHF_QED , '--', c='red', lw=2, label="QED-HF")
    #plt.plot(bond_list_fine,   E_FCI + 0.5 * np.sum(cavity_freq), '--', c='blue', lw=2, label="FCI + $\\frac{1}{2}\\omega_\\mathrm{c}$")
    # plt.errorbar(bond_list_coarse, E_rAFQMC_AVE_QED, yerr=E_rAFQMC_STD_QED, fmt='o', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@RHF")
    # plt.errorbar(bond_list_coarse, E_uAFQMC_AVE_QED, yerr=E_uAFQMC_STD_QED, fmt='s', linestyle="dashed", elinewidth=1, ecolor='black', capsize=4, c='black', mfc='none', label="QED-AFQMC@UHF")
    plt.xlabel("H-H Bond Length (Bohr)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$T$ = %1.3f a.u.,  $d\\tau$ = %1.3f a.u.,  $N_\\mathrm{w}$ = %1.0f" % (total_time, dt, num_walkers), fontsize=15)
    plt.legend()
    plt.xlim(bmin,bmax)
    #plt.ylim(-1.2,-0.8) # Good bare H-H Setting
    plt.tight_layout()
    plt.savefig("%s/01-AFQMC_QED_BOND_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.jpg" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), dpi=300)
    plt.clf()


    np.savetxt("%s/01-AFQMC_QED_BOND_SCAN_T_%1.0f_dt_%1.4f_Nw_%1.0f_basis_%s_LAM_%1.3f.dat" % (DATA_DIR,total_time,dt,num_walkers,basis,cavity_coupling[0]), np.c_[bond_list_coarse, E_rAFQMC_AVE, E_rAFQMC_STD, E_uAFQMC_AVE, E_uAFQMC_STD], fmt="%1.6f", header="R_HH  AFQMC@RHF  STD(AFQMC@RHF)  AFQMC@UHF  STD(AFQMC@UHF)")

