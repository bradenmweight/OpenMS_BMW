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
                mf = scf.RHF(self.mol)
                mf.kernel()
            elif ( trial == "UHF" ):
                # UHF -- BMW: Need to break symmetry of initial guess to get right solution
                mf = scf.UHF(mol)
                dm_alpha, dm_beta = mf.get_init_guess()
                dm_beta[:2,:2] = 0 # BMW: Set some of the beta coefficients to zero to break alpha/beta symmetry
                dm = (dm_alpha,dm_beta)
                mf.kernel(dm) # BMW: Pass in modified initial guess
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
        overlap  = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(overlap)
        xinv     = np.linalg.inv(ao_coeff)

        self.wf  = self.mf.mo_coeff
        self.wf  = xinv.dot(self.mf.mo_coeff[:, :self.mol.nelec[0]])

        self.wf  = self.wf[None,:,:] # BMW: Add dummy dimension for spin


### Braden M. Weight ###
# single determinant unrestricted HF trial wavefunction
class TrialUHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        overlap = self.mol.intor('int1e_ovlp') # AO Overlap Matrix, S
        ao_coeff = lo.orth.lowdin(overlap) # Eigenvectors of S**(1/2)
        xinv = np.linalg.inv(ao_coeff) # S**(-1/2)

        MO_ALPHA = self.mf.mo_coeff[0, :, :self.mol.nelec[0]] # Occupied ALPHA MO Coeffs
        MO_BETA  = self.mf.mo_coeff[1, :, :self.mol.nelec[1]] # Occupied BETA MO Coeffs
        self.wf  = [np.dot( xinv, MO_ALPHA )] # ALPHA ORBITALS AFTER LOWDIN ORTHOGONALIZATION
        self.wf.append(np.dot( xinv, MO_BETA )) # BETA ORBITALS AFTER LOWDIN ORTHOGONALIZATION
        self.wf  = np.array( self.wf ) # self.wf.shape = (spin, nocc mos per spin, nAOs)




# define walker class
class WalkerBase(object):
    r"""
    Walker Base class
    """
    def __init__(self, trial):

        self.trial = trial

    def build(self):

        pass