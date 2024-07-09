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
        self.walker_tensors = np.einsum('pq,zqr->zpr', one_body_op_power, self.walker_tensors)

        # 2-body propagator propagation
        # exp[(x-\bar{x}) * L]
        xi = np.random.normal(0.0, 1.0, self.nfields * self.num_walkers)
        xi = xi.reshape(self.num_walkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn,npq->zpq', xi-xbar, ltensor)

        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('zpq, zqr->zpr', two_body_op_power, temp) / (order_i + 1.0)
            self.walker_tensors += temp

        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e)
        self.walker_tensors = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensors)
        # self.walker_tensosr = np.exp(-self.dt * nuc) * self.walker_tensors

        # (x*\bar{x} - \bar{x}^2/2)
        cfb = np.einsum("zn, zn->z", xi, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt)*np.einsum('zn, n->z', xi-xbar, self.mf_shift)
        return cfb, cmf


class QEDAFQMC(AFQMC):
    def __init__(self, 
                 mol, 
                 cavity_freq = None,
                 cavity_coupling = None,
                 cavity_vec = None,
                 **kwargs):

        super().__init__(mol, **kwargs)

        # Cavity Parameters
        if ( cavity_freq is not None ):
            self.cavity_freq     = cavity_freq
            self.cavity_coupling = cavity_coupling
            self.cavity_vec      = cavity_vec / np.linalg.norm( cavity_vec )
            self.cavity_mode     = cavity_coupling * cavity_vec # To match with definition in qedhf.py -- I think coupling and vector should be separated.
            self.qedmf           = QEDRHF(self.mol, cavity_mode=self.cavity_mode, cavity_freq=self.cavity_freq)
            self.qedmf.kernel()
            self.photon          = Photon(self.mol,self.qedmf,omega=self.cavity_freq,vec=self.cavity_vec,gfac=self.cavity_coupling)

            # print("\n")
            # print( "\tCavity Frequency = %1.4f a.u." % self.cavity_freq[0])
            # print( "\tLight-Matter Coupling (\\lambda = 1/\sqrt(2 wc) A0) = %1.4f a.u." % self.cavity_coupling[0])
            # print( "\tCavity Polarization Direction: %1.3f %1.3f %1.3f" % (self.cavity_vec[0,0], self.cavity_vec[0,1], self.cavity_vec[0,2]) )
            # print("\n")

            # create qed mf object
            #self.qedmf = QEDRHF(mol, *args, **kwargs)


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


        """ # This is with built-in eri generator in AO basis. Then use gmat in AO basis to augment the eri with QED parts
        # Add the DSE-mediated eri to the eri
        eri_QED =  self.qedmf.mol.intor("int2e", aosym="s1") # Returns bare eri in AO basis (?)
        for mode in range( self.qedmf.qed.nmodes ):
            eri_QED += np.einsum("pq,rs->pqrs", self.qedmf.gmat[mode], self.qedmf.gmat[mode]) # gmat are in AO basis (?)
        eri_QED = ao2mo.general(eri_QED, np.array([self.trial.mf.mo_coeff,]*4), compact=False) # BMW: What if we wanted the gmat in the QED-HF basis ? self.qedmf.mo_coeff --> trial.wf
        eri_QED = eri_QED.reshape( [norb,]*4 )
        """

        # This is the fcidump way of doing things. Everything here is already in MO basis
        self.photon.get_gmat_so() # Construct gmat in MO basis
        gmat       = self.photon.gmatso # Make local variable for it
        h1e, eri_QED = self.make_read_fcidump( norb )
        NKEEP_MOs  = eri_QED.shape[0]
        for mode in range( self.qedmf.qed.nmodes ):
            eri_QED += np.einsum("pq,rs->pqrs", gmat[mode,:NKEEP_MOs,:NKEEP_MOs], gmat[mode,:NKEEP_MOs,:NKEEP_MOs])


        ltensor = self.make_ltensor( eri_QED, norb )
        return h1e_QED, eri_QED, ltensor





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