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
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors) # one_body_op_power defined already

        # (x*\bar{x} - \bar{x}^2/2)
        N_I = np.einsum("zn, zn->z", xi, F)-0.5*np.einsum("zn, zn->z", F, F)
        cmf = -np.sqrt(self.dt)*np.einsum('zn,n->z', xi-F, self.mf_shift)

        return N_I, cmf

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
            self.NMODE           = len(cavity_freq)
            self.qedmf           = QEDRHF(self.mol, cavity_mode=self.cavity_mode, cavity_freq=self.cavity_freq)
            self.qedmf.kernel()
            self.photon          = Photon(self.mol,self.qedmf,omega=self.cavity_freq,vec=self.cavity_vec,gfac=self.cavity_coupling)
            
            self.dipole_ao_polarized = []
            for mode in range( self.NMODE ):
                self.dipole_ao_polarized.append( self.photon.get_polarized_dipole_ao(mode) )
            self.dipole_ao_polarized     = np.array(self.dipole_ao_polarized)        
            self.NAO                     = self.dipole_ao_polarized.shape[-1]
            self.quadrupole_ao           = get_quadrupole_ao(mol, add_nuc_dipole=True).reshape( (3,3,self.NAO,self.NAO) )
            self.quadrupole_ao_polarized = np.einsum("mx,xyab,my->mab", self.cavity_vec, self.quadrupole_ao, self.cavity_vec)
            


            # Define photon parameters
            self.photon_basis    = photon_basis
            self.NFock           = NFock
            self.a               = np.diag( np.sqrt(np.arange(1,self.NFock)), k=1 ) # Define photon operator
            self.aTa             = self.a.T + self.a
            self.bilinear_factor = np.sqrt(self.cavity_freq/2) * self.cavity_coupling
            self.DSE_factor      = self.cavity_coupling**2 / 2
            self.MuQc            = np.einsum("m,FG,mab->FGab", self.bilinear_factor, self.aTa, self.dipole_ao_polarized)
            
            self.h1e_DSE         = np.einsum("m,mab->ab", self.DSE_factor, -1*self.quadrupole_ao_polarized )
            self.exp_h1e_DSE     = scipy.linalg.expm( -self.dt/2 * self.h1e_DSE )
            self.eri_DSE         = np.einsum("m,mab,mcd->abcd", self.DSE_factor, self.dipole_ao_polarized, self.dipole_ao_polarized )




    def get_integrals(self):
        ao_overlap      = self.mol.intor('int1e_ovlp')
        self.ao_coeff   = lo.orth.lowdin(ao_overlap)

        # This is the fcidump way of doing things. Everything here is in AO basis
        h1e, eri = self.make_read_fcidump( self.NAO )
        h1e     += self.h1e_DSE
        eri     += self.eri_DSE

        ltensor = self.make_ltensor( eri, self.NAO )
        return h1e, eri, ltensor


    def local_energy(self, h1e, eri, G1p):
        r"""Compute local energy
             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}
        """
        # E_coul
        tmp  = 2.0 * np.einsum("prqs,zFFSpr->zqs", eri, G1p) * self.spin_fac
        ecoul = np.einsum("zqs,zFFSqs->z", tmp, G1p)
        # E_xx
        tmp =  np.einsum("prqs,zFFSps->zSqr", eri, G1p)
        exx  = np.einsum("zSqs,zFFSqs->z", tmp, G1p)
        e2 = (ecoul - exx) * self.spin_fac

        e1   = 2 * np.einsum("zFFSpq,pq->z",   G1p, h1e) * self.spin_fac

        #bilinear = 2 * np.einsum( "FGab,zFGab->z", self.MuQc, G1p )

        ZPE = 0.5 * np.sum(self.cavity_freq) # Zero-point energy of the cavity mode
        energy = e1 + e2 + self.energy_nuc + ZPE #+ bilinear

        return energy


    def propagate_bilinear_coupling( self ):
        # BMW:
        # Bilinear propagation

        # BMW:
        # I put Taylor expansion here to keep the four-index matrix notation for einsum. 
        # We could reshape, then use expm(MuQc) if done properly
        temp = self.walker_tensors.copy()
        for order_i in range(self.taylor_order):
            temp = np.einsum('FGab,zGSbj->zFSaj', -self.dt * self.MuQc, temp) / (order_i + 1.0)
            self.walker_tensors += temp




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
        
        #### PROPAGATE QED TERMS ####
        self.propagate_bilinear_coupling()
        #############################

        # 1-body propagator propagation
        # e^{-dt/2*H1e}
        self.walker_tensors = np.einsum('ab,zFSbk->zFSak', one_body_op_power, self.walker_tensors) # one_body_op_power defined already

        # (x*\bar{x} - \bar{x}^2/2)
        N_I = np.einsum("zn, zn->z", xi, F)-0.5*np.einsum("zn, zn->z", F, F)
        cmf = -np.sqrt(self.dt)*np.einsum('zn,n->z', xi-F, self.mf_shift)

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