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

# QMC module (TBA)

import sys
from pyscf import tools, lo, scf, fci
from openms.mqed.qedhf import RHF as QEDRHF
import numpy as np
import h5py

# files for trial WF and random walkers

class TrialWFBase(object):
    r"""
    Base class for trial wavefunction
    """

    def __init__(self,
                 mol,
                 cavity,
                 #ne: Tuple[int, int],
                 #n_mo : int,
                 mf = None,
                 numdets = 1,
                 numdets_props = 1,
                 numdets_chunks = 1,
                 verbose = 1):

        self.mol = mol
        if mf is None:
            mf = scf.RHF(self.mol)
        mf.kernel()
        self.mf = mf
        self.mf.dm = mf.make_rdm1()

        #self.num_elec = num_elec # number of electrons
        #self.n_mo = n_mo
        # only works for spin-restricted reference at this moment
        #self.nalpha = self.nbeta = self.num_elec // 2

        self.numdets = numdets
        self.numdets_props = numdets_props
        self.numdets_chunks = numdets_chunks

        self.build( cavity )

    def build(self):
        r"""
        build initial trial wave function
        """
        pass

# single determinant HF trial wavefunction
class TrialHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def build(self, cavity):

        overlap = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(overlap)
        xinv = np.linalg.inv(ao_coeff)

        self.wf = self.mf.mo_coeff
        self.wf = xinv.dot(self.mf.mo_coeff[:, :self.mol.nelec[0]])

        # Define tensor product basis for electron-boson DOFs
        if ( cavity is not None ):
            print( "wc = ", cavity['cavity_freq'] )
            if ( cavity['photon_basis'] == 'fock' ):
                print( "photon_basis = ", cavity['photon_basis'] )
                if ( cavity['NFock'] is not None ): # Introduce Fock state basis for quantized cavity field
                    print( "NFock = ", cavity['NFock'] )
                    fock_basis    = np.zeros( (cavity['NFock']) )
                    fock_basis[0] = 1.0 # Start from vacuum state as initial guess: |PSI> = |HF> \otimes |n=0>
                    # BMW: 
                    # How to do a tensor product basis with HF if it is a matrix ? It only makes sense with determinents...
                    #self.wf = np.kron( self.wf, fock_basis )
                    self.wf = np.array([ self.wf * fock_basis[i] for i in range(cavity['NFock']) ])
                    print( "Constructing the tensor-product electron-photon basis" )
                    print( "Polariton WFN Shape =", self.wf.shape )
                    print( "Trial Polariton WFN: |TRIAL> = |HF> \otimes |n = 0>\n", self.wf )
                else:
                    raise ValueError("Number of Fock states NFock >= 2 must be specified.")
            else:
                raise NotImplementedError("Only the Fock basis is implemented for photon DOFs. Choose photon_basis='fock'.")
        else:
            self.wf = self.wf[None,:,:] # Add single Fock basis state for computational ease -- No added scaling due to this




# define walker class
class WalkerBase(object):
    r"""
    Walker Base class
    """
    def __init__(self, trial):

        self.trial = trial

    def build(self):

        pass