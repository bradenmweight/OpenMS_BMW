#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#


r"""
Auxiliary-Field Quantum Monte Carlo
===================================

Theoretical background
----------------------

Reference: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf


TBA.

"""

import sys, os
from pyscf import tools, lo, scf, fci, ao2mo
import numpy as np
import scipy
import h5py


from openms.mqed.qedhf import RHF as QEDRHF
from openms.lib.boson import Photon, get_dipole_ao, get_quadrupole_ao

from openms.qmc import qmc
#from openms.mqed.qedhf import RHF as QEDRHF


class AFQMC(qmc.QMCbase):


    def __init__(self, mol, *args, **kwargs):

        super().__init__(mol, *args, **kwargs)

    def dump_flags(self):
        r"""
        Dump flags
        """
        print(f"\n========  AFQMC simulation using OpenMS package ========\n")

    def hs_transform(self, h1e):
        r"""
        Perform Hubbard-Stratonovich (HS) decomposition

        .. math::

            e^{-\Delta\tau \hat{H}} = \int d\boldsymbol{x} p(\boldsymbol{x})\hat{B}(\boldsymbol{x}).

        """
        hs_fields = None
        return hs_fields

    def get_wfn(self):
        """
        Get the wavefunction
        """
        coeffs = self.walker_coeff
        wfns   = self.walker_tensors
        wfn    = np.einsum('z,zFSak->FSak', coeffs, wfns)
        norm   = np.linalg.det( np.einsum('FSaj,FSak->jk', wfn.conj(), wfn ) )
        norm   = np.prod( norm )
        return wfn / np.sqrt( norm )

    def propagation(self, h1e, F, ltensor):
        r"""
        Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf
        Eqs 50 - 51
        """
        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power   = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors)

        # 2-body propagator propagation
        # exp[(x-F) * L], F = sqrt(-dt) <L_n>
        xi = np.random.normal(0, 1.0, size=(self.num_walkers, self.nfields) )
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn,nab->zab', xi - F, ltensor)
        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('zab,zFSbj->zFSaj', two_body_op_power, temp) / (order_i + 1.0)
            self.walker_tensors += temp
        
        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors)

        # (x*\bar{x} - \bar{x}^2/2)
        N_I = np.einsum("zn, zn->z", xi, F)-0.5*np.einsum("zn, zn->z", F, F)
        cmf = -np.sqrt(self.dt)*np.einsum('zn,n->z', xi-F, self.L_mf)

        return N_I, cmf




class QEDAFQMC(AFQMC):
    def __init__(self, 
                 mol, 
                 cavity_freq = None,
                 cavity_coupling = None,
                 cavity_vec = None,
                 photon_basis = 'fock',
                 NFock = 2,
                 do_coherent_state = True,
                 **kwargs):
        
        assert ( cavity_freq     is not None ), "Cavity frequency           is not provided to QED-AFQMC class."
        assert ( cavity_coupling is not None ), "Cavity coupling            is not provided to QED-AFQMC class."
        assert ( cavity_vec      is not None ), "Cavity polarization vector is not provided to QED-AFQMC class."
        
        cavity = {}
        cavity['cavity_freq']     = cavity_freq
        cavity['cavity_coupling'] = cavity_coupling
        cavity['cavity_vec']      = cavity_vec
        cavity['photon_basis']    = photon_basis.lower()
        cavity['NFock']           = NFock

        super().__init__(mol, cavity=cavity, **kwargs)

        # Cavity Parameters
        self.do_coherent_state = do_coherent_state
        self.cavity_freq       = cavity_freq
        self.cavity_coupling   = cavity_coupling
        self.cavity_vec        = cavity_vec / np.linalg.norm( cavity_vec )
        self.cavity_mode       = cavity_coupling * cavity_vec # To match with definition in qedhf.py -- my opinion is that coupling and vector should be separated.
        self.NMODE             = len(cavity_freq)
        self.qedmf             = QEDRHF(self.mol, cavity_mode=self.cavity_mode, cavity_freq=self.cavity_freq)
        self.qedmf.kernel()  
        self.photon            = Photon(self.mol,self.qedmf,omega=self.cavity_freq,vec=self.cavity_vec,gfac=self.cavity_coupling)
        

        # Define photon parameters
        self.photon_basis    = photon_basis
        self.NFock           = NFock
        self.a               = np.diag( np.sqrt(np.arange(1,self.NFock)), k=1 ) # Define photon operator
        self.aTa             = self.a.T @ self.a
        self.aT_plus_a       = self.a.T + self.a
        self.bilinear_factor = np.sqrt(self.cavity_freq/2) * self.cavity_coupling
        self.DSE_factor      = self.cavity_coupling**2 / 2


    def make_dipole_octopole(self):

        self.dipole_ao_polarized = []
        for mode in range( self.NMODE ):
            self.dipole_ao_polarized.append( self.photon.get_polarized_dipole_ao(mode) )
        self.dipole_ao_polarized     = np.array(self.dipole_ao_polarized)    
        self.NAO                     = self.dipole_ao_polarized.shape[-1]
        self.quadrupole_ao           = get_quadrupole_ao(self.mol, add_nuc_dipole=True).reshape( (3,3,self.NAO,self.NAO) )
        self.quadrupole_ao_polarized = np.einsum("mx,xyab,my->mab", self.cavity_vec, self.quadrupole_ao, self.cavity_vec)
        
        # Orthogonalize the dipole_ao_polarized and quadrupole_ao_polarized by Lowdin
        ao_overlap                   = self.mol.intor('int1e_ovlp')
        self.ao_coeff                = lo.orth.lowdin(ao_overlap) # Need this to be defined as self for fcidum line. DO NOT REMOVE.
        xinv                         = np.linalg.inv(self.ao_coeff)
        self.dipole_ao_polarized     = np.einsum( "ab,mbc,cd->mad", xinv, self.dipole_ao_polarized, xinv.T )
        self.quadrupole_ao_polarized = np.einsum( "ab,mbc,cd->mad", xinv, self.quadrupole_ao_polarized, xinv.T )

    def get_integrals(self):

        self.make_dipole_octopole() # Make in orthogonal AO basis
        self.MuQc = np.einsum("m,FG,mab->FGab", self.bilinear_factor, self.aT_plus_a, self.dipole_ao_polarized)



        # This is the fcidump way of doing things. Everything here is in AO basis
        h1e, eri = self.make_read_fcidump( self.NAO )

        ### Do bare QED terms
        h1e_DSE   = np.einsum("m,mab->ab", self.DSE_factor, -1*self.quadrupole_ao_polarized ) # TODO -- How to do CS on this ?
        # BMW: YZ says that eri should have factor 2 from HS transform
        eri_DSE   = 2 * np.einsum("m,mab,mcd->abcd", self.DSE_factor, self.dipole_ao_polarized, self.dipole_ao_polarized )

        if ( self.do_coherent_state == True ):
            # Modify QED terms if coherent state shift is to be applied
            print( "Performing coherent state shift based on trial wavefunction." )
            rho_mf          = np.einsum( "FSaj,FSbj->ab", self.trial.wf.conj(), self.trial.wf ) # rho_mf in AO Basis (electronic subspace only)
            self.mu_mf_pol  = np.einsum( "mab,ab->m", self.dipole_ao_polarized, rho_mf )
            self.mu_mf_pol  = np.array([ np.identity( self.NAO ) * self.mu_mf_pol[m] for m in range(self.NMODE) ]) # (NMode, NAO, NAO)
            self.mu_shifted = self.dipole_ao_polarized - self.mu_mf_pol # \hat{\mu} - <\mu>
            self.MuQc       = np.einsum("m,FG,mab->FGab", self.bilinear_factor, self.aT_plus_a, self.mu_shifted) # Replace with shifted version
            eri_DSE         = 2 * np.einsum("m,mab,mcd->abcd", self.DSE_factor, self.mu_shifted, self.mu_shifted ) # Replace with shifted version
            print( "<\\mu>_Trial =", self.mu_mf_pol[0,0,0] )

        h1e     += h1e_DSE
        eri     += eri_DSE

        ltensor = self.make_ltensor( eri, self.NAO )
        return h1e, eri, ltensor

    def local_energy(self, h1e, eri, G1p):
        r"""Compute local energy
             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}

        Note: eri and h1e include the DSE terms
            If do_coherent_state == True, eri and MuQC will also be generated based on shifted dipole
            Therefore, the energy computed here would be directly the CS Hamiltonian energy. 
            No need to unshift.
        """

        # E_coul
        tmp  = 2.0 * np.einsum("prqs,zFFSpr->zqs", eri, G1p) * self.spin_fac
        ecoul = np.einsum("zqs,zFFSqs->z", tmp, G1p)
        # E_xx
        tmp =  np.einsum("prqs,zFFSps->zSqr", eri, G1p)
        exx = np.einsum("zSqs,zFFSqs->z", tmp, G1p)
        e2  = (ecoul - exx) * self.spin_fac

        e1       = 2 * np.einsum("pq,zFFSpq->z", h1e, G1p )          * self.spin_fac
        Eph      = 2 * np.einsum( "m,FF,zFFSaa->z", self.cavity_freq, self.aTa, G1p ) * self.spin_fac
        bilinear = 2 * np.einsum( "FGab,zFGSab->z", self.MuQc, G1p ) * self.spin_fac

        ZPE = 0.5 * np.sum(self.cavity_freq) # Zero-point energy of the cavity modes
        energy = e1 + e2 + self.energy_nuc + Eph + ZPE + bilinear

        return energy



    def propagate_photon_hamiltonian( self ):
        # Half-step photon propagation

        # exp_Hph is diagonal in the Fock basis
        wcaTa    = np.einsum("m,F->F", self.cavity_freq, np.arange(self.NFock)) # (NFock)
        evol_Hph = np.exp( -0.5 * self.dt * wcaTa ) # (NFock)
        # print( wcaTa )
        # print( evol_Hph ) # This looks correct.

        """
        aa1 checks the Fock-state probability
        """
        #aa1 = np.einsum( "zFSaj,zFSak->zFSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, NFock, Spin, NMO, NMO)
        #aa1 = np.linalg.det( aa1 ).real # (w, NFock, Spin)
        #aa1 = np.prod( aa1, axis=-1 ) # (w, NFock)
        #aa1 = np.average( aa1, axis=0 ) # (NFock)
        #print("AAA Prob. per Fock State =", aa1)
        self.walker_tensors = np.einsum( "F,zFSaj->zFSaj", evol_Hph, self.walker_tensors )
        """
        aa1 checks the Fock-state probability
        """
        #aa1 = np.einsum( "zFSaj,zFSak->zFSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, NFock, Spin, NMO, NMO)
        #aa1 = np.linalg.det( aa1 ).real # (w, NFock, Spin)
        #aa1 = np.prod( aa1, axis=-1 ) # (w, NFock)
        #aa1 = np.average( aa1, axis=0 ) # (NFock)
        #print("BBB Prob. per Fock State =", aa1)

    def propagate_bilinear_coupling( self ):

        """
        OVLP checks the total wfn overlap
        """
        # OVLP = np.einsum( "zFSaj,zFSak->zSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, Spin, NMO, NMO)
        # OVLP = np.linalg.det( OVLP ) # (w, Spin)
        # OVLP = np.prod( OVLP, axis=-1 ) # (w)
        # print( "AAA", np.average(OVLP).real )

        """
        aa1 checks the Fock-state probability
        """
        # aa1 = np.einsum( "zFSaj,zFSak->zFSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, NFock, Spin, NMO, NMO)
        # aa1 = np.linalg.det( aa1 ).real # (w, NFock, Spin)
        # aa1 = np.prod( aa1, axis=-1 ) # (w, NFock)
        # aa1 = np.average( aa1, axis=0 ) # (NFock)
        # print("CCC Prob. per Fock State =", aa1)

        ####### Half-step Bilinear propagation #######

        # Evolution by Taylor Series Expansion of the Bilinear Operator
        # temp = self.walker_tensors.copy()
        # for order_i in range(self.taylor_order):
        #     temp = np.einsum('FGab,zGSbj->zFSaj', -0.5 * self.dt * self.MuQc, temp, optimize=True) / (order_i + 1.0)
        #     self.walker_tensors += temp

        """
        As a check, I rewrote the bilinear propagation in a different way.
        Here, I first diagonalize the dipole operator in AO basis and then propagate the bilinear operator.
        It gives me the same answer. Will be expensive for large N_AO. 
        Though we could store the diagonalized dipole at the beginning of the simulation.
        """
        # Evolution by Diagonalization of the Bilinear Operator
        Ea,Ua               = np.linalg.eigh( self.aT_plus_a )
        mu,Umu              = np.linalg.eigh( self.dipole_ao_polarized[0] )
        exp_mua             = np.exp( -0.500 * self.dt * self.bilinear_factor[0] * np.einsum("F,a->Fa", Ea, mu) )
        exp_bilinear        = np.einsum("FG,ab,Gb,bc,GH->FHac", Ua.T, Umu.T, exp_mua, Umu, Ua) # exp_mua is interpreted as a diagaonal matrix (GGbb)
        self.walker_tensors = np.einsum('FGab,zGSbj->zFSaj', exp_bilinear, self.walker_tensors, optimize=True)

        """
        aa1 checks the Fock-state probability
        """
        # aa1 = np.einsum( "zFSaj,zFSak->zFSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, NFock, Spin, NMO, NMO)
        # aa1 = np.linalg.det( aa1 ).real # (w, NFock, Spin)
        # aa1 = np.prod( aa1, axis=-1 ) # (w, NFock)
        # aa1 = np.average( aa1, axis=0 ) # (NFock)
        # print("DDD Prob. per Fock State =", aa1)

        """
        OVLP checks the total wfn overlap
        """
        # OVLP = np.einsum( "zFSaj,zFSak->zSjk", self.walker_tensors.conj(), self.walker_tensors ) # (w, Spin, NMO, NMO)
        # OVLP = np.linalg.det( OVLP ) # (w, Spin)
        # OVLP = np.prod( OVLP, axis=-1 ) # (w)
        #print( "OOOO", np.average(OVLP).real )

    def get_apply_background_mf_shift(self, h1e, eri, ltensor):
        from itertools import product
        shifted_h1e    = np.zeros_like( h1e )
        rho_mf         = np.einsum( "FSaj,FSbj->ab", self.trial.wf, self.trial.wf ) # rho_mf in AO Basis (electronic subspace only)
        self.L_mf      = 1j * np.einsum("nab,ab->n", ltensor, rho_mf) # Compute <L_n>_mf

        for p, q in product(range(h1e.shape[0]), repeat=2):
            shifted_h1e[p, q]  = h1e[p, q] - 0.5 * np.trace(eri[p, :, :, q])

        shifted_h1e = shifted_h1e - np.einsum("n,nab->ab", self.L_mf, 1j*ltensor)
        return shifted_h1e


    def propagation(self, h1e, F, ltensor):
        r"""
        Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf
        Eqs 50 - 51
        """
        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power   = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors)

        #### PROPAGATE QED TERMS BY HALF STEP ####
        self.propagate_photon_hamiltonian()
        self.propagate_bilinear_coupling()
        #############################

        # # 2-body propagator propagation
        # # exp[(x-F) * L], F = sqrt(-dt) <L_n>
        xi = np.random.normal(0, 1.0, size=(self.num_walkers, self.nfields) )
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn,nab->zab', xi - F, ltensor)
        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('zab,zFSbj->zFSaj', two_body_op_power, temp) / (order_i + 1.0)
            self.walker_tensors += temp
        
        #### PROPAGATE QED TERMS BY HALF STEP ####
        self.propagate_bilinear_coupling()
        self.propagate_photon_hamiltonian()
        #############################

        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors)

        # (x*\bar{x} - \bar{x}^2/2)
        N_I = np.einsum("zn, zn->z", xi, F)-0.5*np.einsum("zn, zn->z", F, F)
        cmf = -np.sqrt(self.dt)*np.einsum('zn,n->z', xi-F, self.L_mf)

        return N_I, cmf



    def dump_flags(self):
        r"""
        Dump flags
        """
        print(f"\n========  QED-AFQMC simulation using OpenMS package ========\n")




if __name__ == "__main__":
    # TODO: Not updated for QED yet
    from pyscf import gto, scf, fci
    bond = 1.6
    natoms = 2
    atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
    mol = gto.M(atom=atoms, basis='sto-6g', unit='Bohr', verbose=3)

    num_walkers = 500
    afqmc = AFQMC(mol, dt=0.005, total_time=5.0, num_walkers=num_walkers, energy_scheme="hybrid")

    times, energies = afqmc.kernel()

    # HF energy
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()

    # FCI energy
    fcisolver = fci.FCI(mf)
    fci_energy = fcisolver.kernel()[0]

    print(fci_energy)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # time = np.arange(0, 5, 0.)
    ax.plot(times, energies, '--', label='afqmc (my code)')
    ax.plot(times, [hf_energy] * len(times), '--')
    ax.plot(times, [fci_energy] * len(times), '--')
    ax.set_ylabel("ground state energy")
    ax.set_xlabel("imaginary time")
    plt.savefig("afqmc_gs1.pdf")
    #plt.show()