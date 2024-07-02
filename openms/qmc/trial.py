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
import numpy as np
import h5py

# files for trial WF and random walkers

class TrialWFBase(object):
    r"""
    Base class for trial wavefunction
    """

    def __init__(self,
                 mol,
                 #ne: Tuple[int, int],
                 #n_mo : int,
                 mf = None,
                 trial = None,
                 numdets = 1,
                 numdets_props = 1,
                 numdets_chunks = 1,
                 verbose = 1):

        self.mol = mol
        if mf is None:
            print( "trial", trial )
            if ( trial == "RHF" ):
                print("Doing restricted RHF calculation.")
                mf = scf.RHF(self.mol)
            elif ( trial == "UHF" ):
                print("Doing unrestricted UHF calculation.")
                mf = scf.UHF(self.mol)
            else:
                print("No trial wavefunction selected. Defaulting to RHF.")
                mf = scf.RHF(self.mol)
            mf.kernel()
        self.mf = mf
        

        #self.num_elec = num_elec # number of electrons
        #self.n_mo = n_mo
        # only works for spin-restricted reference at this moment
        #self.nalpha = self.nbeta = self.num_elec // 2

        self.numdets = numdets
        self.numdets_props = numdets_props
        self.numdets_chunks = numdets_chunks

        self.build()

    def build(self):
        r"""
        build initial trial wave function
        """
        pass

# single determinant HF trial wavefunction
class TrialHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        overlap = self.mol.intor('int1e_ovlp') # AO Overlap Matrix, S
        ao_coeff = lo.orth.lowdin(overlap) # Eigenvectors of S**(1/2)
        xinv = np.linalg.inv(ao_coeff) # S**(-1/2)

        # Compute orthogonalized MO coefficients, \tilde{C} = S**(-1/2) @ C
        self.wf  = np.zeros_like( self.mf.mo_coeff )
        self.wf += np.dot( xinv, self.mf.mo_coeff[:, :self.mol.nelec[0]] ) # Include only occupied orbitals

# single determinant unrestricted HF trial wavefunction
class TrialUHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        overlap = self.mol.intor('int1e_ovlp') # AO Overlap Matrix, S
        ao_coeff = lo.orth.lowdin(overlap) # Eigenvectors of S**(1/2)
        xinv = np.linalg.inv(ao_coeff) # S**(-1/2)

        self.wf    = np.zeros_like( self.mf.mo_coeff )
        self.wf[0] = np.dot( xinv, self.mf.mo_coeff[0, :, :self.mol.nelec[0]] ) # ALPHA ORBITALS
        self.wf[1] = np.dot( xinv, self.mf.mo_coeff[1, :, :self.mol.nelec[1]] ) # BETA ORBITALS



# define walker class
class WalkerBase(object):
    r"""
    Walker Base class
    """
    def __init__(self, trial):

        self.trial = trial

    def build(self):

        pass



