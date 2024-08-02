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
from openms.lib.boson import Photon

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

    def propagation(self, h1e, xbar, ltensor):
        r"""
        Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf
        Eqs 50 - 51
        """
        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power   = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('ab,zFbk->zFak', one_body_op_power, self.walker_tensors)

        # 2-body propagator propagation
        # exp[(x-\bar{x}) * L]
        xi = np.random.normal(0.0, 1.0, self.nfields * self.num_walkers)
        xi = xi.reshape(self.num_walkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn,nab->zab', xi-xbar, ltensor)

        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('zab,zFbj->zFaj', two_body_op_power, temp) / (order_i + 1.0)
            self.walker_tensors += temp


        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('ab,zFbj->zFaj', one_body_op_power, self.walker_tensors)
        # self.walker_tensosr = np.exp(-self.dt * nuc) * self.walker_tensors

        # (x*\bar{x} - \bar{x}^2/2)
        cfb = np.einsum("zn,zn->z", xi, xbar) - 0.5 * np.einsum("zn,zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt) * np.einsum('zn,n->z', xi-xbar, self.mf_shift)
        return cfb, cmf


class QEDAFQMC(AFQMC):
    def __init__(self, 
                 mol, 
                 cavity_freq = None,
                 cavity_coupling = None,
                 cavity_vec = None,
                 photon_basis = None,
                 NFock = None,
                 **kwargs):
        
        cavity = {}
        cavity['cavity_freq'] = cavity_freq
        cavity['cavity_coupling'] = cavity_coupling
        cavity['cavity_vec'] = cavity_vec
        cavity['photon_basis'] = photon_basis.lower()
        cavity['NFock'] = NFock

        super().__init__(mol, cavity=cavity, **kwargs)

        # Cavity Parameters
        if ( cavity_freq is not None ):
            self.cavity_freq     = cavity_freq
            self.cavity_coupling = cavity_coupling
            self.cavity_vec      = cavity_vec / np.linalg.norm( cavity_vec )
            self.cavity_mode     = cavity_coupling * cavity_vec # To match with definition in qedhf.py -- I think coupling and vector should be separated.
            self.qedmf           = QEDRHF(self.mol, cavity_mode=self.cavity_mode, cavity_freq=self.cavity_freq)
            self.qedmf.kernel()
            self.photon          = Photon(self.mol,self.qedmf,omega=self.cavity_freq,vec=self.cavity_vec,gfac=self.cavity_coupling)
            self.dipole_ao       = mol.intor_symmetric("int1e_r", comp=3) #- np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())[:,None,None] / mol.tot_electrons() # self.photon.get_dipole_ao()
            self.dipole_ao       = np.einsum("dab,d->ab", self.dipole_ao, self.cavity_vec[0]) # For now, just do first mode.
            self.photon_basis    = photon_basis
            self.NFock           = NFock
            
            # print("\n")
            # print( "\tCavity Frequency = %1.4f a.u." % self.cavity_freq[0])
            # print( "\tLight-Matter Coupling (\\lambda = 1/\sqrt(2 wc) A0) = %1.4f a.u." % self.cavity_coupling[0])
            # print( "\tCavity Polarization Direction: %1.3f %1.3f %1.3f" % (self.cavity_vec[0,0], self.cavity_vec[0,1], self.cavity_vec[0,2]) )
            # print("\n")

            # create qed mf object
            #self.qedmf = QEDRHF(mol, *args, **kwargs)

    def local_energy(self, h1e, eri, G1p):
        tmp  = 2 * np.einsum("prqs,zFpr->zqs", eri, G1p)
        tmp -=     np.einsum("prqs,zFps->zqr", eri, G1p)
        
        e1   = 2 * np.einsum("zFpq,pq->z",   G1p, h1e)
        e2   =     np.einsum("zqs,zFqs->z", tmp, G1p)

        """
        NOCC = G1p.shape[-1]
        self.photon.get_gmat_so()        # Construct gmat in AO basis
        gmat       = self.photon.gmatso  # Make local variable for it
        gmat       = gmat[0,:NOCC,:NOCC] # Take only first mode for now
        # Is this the projected dipole operator in AO basis
        dipole_ao = np.array([ gmat for _ in range(self.NFock) ]) # Put dipole operator in product basis
        """
        dipole_ao        = self.dipole_ao
        dipole_ao        = np.array([ dipole_ao for _ in range(self.NFock) ]) # Put dipole operator in product basis
        cavity_freq      = self.cavity_freq
        cavity_coupling  = self.cavity_coupling
        bilinear_factor  = np.sqrt(cavity_freq/2) * cavity_coupling
        DSE_factor       = cavity_coupling**2 / 2
        a         = np.diag( np.sqrt(np.arange(self.NFock-1)), k=1 )
        aTa       = a.T + a
        bilinear  = bilinear_factor * np.einsum( "Fab,FG,zGba->z", dipole_ao, aTa, G1p )
        DSE       = DSE_factor      * np.einsum( "Fab,Fbc,F,zFca->z", dipole_ao, dipole_ao, np.ones(self.NFock), G1p )

        energy = e1 + e2 + self.energy_nuc + bilinear + DSE

        return energy

    def get_integrals(self):
        r"""
        TODO: 1) add DSE-mediated eri and oei
              2) bilinear coupling term (gmat)
        """
        overlap         = self.mol.intor('int1e_ovlp')
        self.ao_coeff   = lo.orth.lowdin(overlap)
        norb            = self.ao_coeff.shape[0]

        h1e_QED = self.qedmf.get_hcore( mol=self.mol, dm=self.trial.mf.dm )                       # BMW: This is he1_BARE + h1e_QED 
                                                                                                  # BMW: What if we wanted the dm in the QED-HF basis ? qedmf.mf.dm --> trial.mf.dm
        h1e_QED = self.ao_coeff @ h1e_QED @ self.ao_coeff.T # BMW: Rotate from AO to MO basis

        # This is the fcidump way of doing things. Everything here is already in A) basis
        self.photon.get_gmat_so() # Construct gmat in AO basis
        gmat       = self.photon.gmatso # Make local variable for it
        h1e, eri_QED = self.make_read_fcidump( norb )
        NKEEP_MOs  = eri_QED.shape[0]
        for mode in range( self.qedmf.qed.nmodes ):
            eri_QED += np.einsum("pq,rs->pqrs", gmat[mode,:NKEEP_MOs,:NKEEP_MOs], gmat[mode,:NKEEP_MOs,:NKEEP_MOs])


        # ltensor = self.make_ltensor( eri_QED, norb )
        # return h1e_QED, eri_QED, ltensor

        # FOR DEBUGGING #
        h1e, eri = self.make_read_fcidump( norb )
        ltensor  = self.make_ltensor( eri, norb )
        return h1e, eri, ltensor



    def propagate_bilinear_coupling( self ):
        # BMW:
        # Insert photon bilinear propagation here
        NAO = self.walker_tensors.shape[-2]
        NMO = self.walker_tensors.shape[-1]
        """
        self.photon.get_gmat_so()  # Construct gmat in MO basis -- Is this the projected dipole operator in MO basis ?
        gmat = self.photon.gmatso  # Make local variable for it
        gmat = self.ao_coeff @ gmat[0] @ self.ao_coeff.T # Rotate from MO to AO basis
        gmat = gmat[0,:NAO,:NAO]   # What is the first axis here ??? Mode label ??? -- Yes, mode label
        """
        dipole_ao        = self.dipole_ao
        cavity_freq      = self.cavity_freq
        cavity_coupling  = self.cavity_coupling
        factor           = np.sqrt(cavity_freq/2) * cavity_coupling
        a         = np.diag( np.sqrt(np.arange(self.NFock-1)), k=1 ) # Define photon operator
        aTa       = a.T + a
        MuQc      = np.zeros( (self.NFock * NAO, self.NFock * NAO) )
        for ao1 in range( NAO ):
            for ao2 in range( NAO ):
                for n1 in range( self.NFock ):
                    for n2 in range( self.NFock ):
                        index1 = n1 * NAO + ao1
                        index2 = n2 * NAO + ao2
                        MuQc[index1,index2] = dipole_ao[ao1,ao2] * aTa[n1,n2]
        MuQc = factor * MuQc
        exp_MuQc            = scipy.linalg.expm( -self.dt/2 * MuQc ).reshape( (self.NFock, NAO, self.NFock, NAO) ) # In full Hilbert space
        self.walker_tensors = np.einsum('FaGb,zGbk->zFak', exp_MuQc, self.walker_tensors)


    def propagate_dipole_self_energy( self ):
        # BMW:
        # Insert photon bilinear propagation here
        NAO = self.walker_tensors.shape[-2]
        NMO = self.walker_tensors.shape[-1]
        """
        self.photon.get_gmat_so()  # Construct gmat in MO basis -- Is this the projected dipole operator in MO basis ?
        gmat = self.photon.gmatso  # Make local variable for it
        gmat = gmat[0,:NAO,:NAO]   # What is the first axis here ???
        """
        dipole_ao        = self.dipole_ao
        cavity_freq      = self.cavity_freq
        cavity_coupling  = self.cavity_coupling
        factor           = cavity_coupling**2 / 2
        """
        DSE       = np.zeros( (self.NFock * NAO, self.NFock * NAO) )
        for ao1 in range( NAO ):
            for ao2 in range( NAO ):
                for ao3 in range( NAO ):
                    for n1 in range( self.NFock ):
                        index1 = n1 * NAO + ao1
                        index2 = n1 * NAO + ao2
                        DSE[index1,index2] += dipole_ao[ao1,ao3] * dipole_ao[ao3,ao2] #* Iph[n1,n1]
        exp_DSE             = scipy.linalg.expm( -self.dt/2 * DSE ).reshape( (self.NFock, NAO, self.NFock, NAO) ) # In full Hilbert space
        """
        DSE  = dipole_ao @ dipole_ao
        DSE  = np.kron( DSE, np.identity(self.NFock) ) # In full Hilbert space
        DSE *= factor
        exp_DSE             = scipy.linalg.expm( -self.dt/2 * DSE ).reshape( (self.NFock, NAO, self.NFock, NAO) ) # In full Hilbert space
        self.walker_tensors = np.einsum('FaGb,zGbk->zFak', exp_DSE, self.walker_tensors)

    def propagation(self, h1e, xbar, ltensor):
        r"""
        Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf
        Eqs 50 - 51
        """
        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power   = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('ab,zFbk->zFak', one_body_op_power, self.walker_tensors)


        # 2-body propagator propagation
        # exp[(x-\bar{x}) * L]
        xi = np.random.normal(0.0, 1.0, self.nfields * self.num_walkers)
        xi = xi.reshape(self.num_walkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn,nab->zab', xi-xbar, ltensor)

        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('zab,zFbj->zFaj', two_body_op_power, temp) / (order_i + 1.0)
            self.walker_tensors += temp
        
        #### PROPAGATE QED TERMS ####
        self.propagate_bilinear_coupling()
        self.propagate_dipole_self_energy()
        #############################

        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        self.walker_tensors = np.einsum('ab,zFbk->zFak', one_body_op_power, self.walker_tensors) # one_body_op_power defined already

        # (x*\bar{x} - \bar{x}^2/2)
        cfb = np.einsum("zn, zn->z", xi, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt)*np.einsum('zn, n->z', xi-xbar, self.mf_shift)
        return cfb, cmf


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