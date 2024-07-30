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

import sys
from pyscf import tools, lo, scf, fci
import numpy as np
import scipy
import itertools
import logging

from openms.qmc.trial import TrialHF, TrialUHF
from pyscf.lib import logger

def read_fcidump(fname, norb):
    """
    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization with chemist's notation
    """
    eri = np.zeros((norb, norb, norb, norb))
    h1e = np.zeros((norb, norb))

    with open(fname, "r") as f:
        lines = f.readlines()
        for line, info in enumerate(lines):
            if line < 4:
                continue
            line_content = info.split()
            integral = float(line_content[0])
            p, q, r, s = [int(i_index) for i_index in line_content[1:5]]
            if r != 0:
                # eri[p,q,r,s] is with chemist notation (pq|rs)=(qp|rs)=(pq|sr)=(qp|sr)
                eri[p-1, q-1, r-1, s-1] = integral
                eri[q-1, p-1, r-1, s-1] = integral
                eri[p-1, q-1, s-1, r-1] = integral
                eri[q-1, p-1, s-1, r-1] = integral
            elif p != 0:
                h1e[p-1, q-1] = integral
                h1e[q-1, p-1] = integral
            else:
                nuc = integral
    return h1e, eri, nuc


class QMCbase(object):
    r"""
    Basic QMC class
    """
    def __init__(self,
        system, # or molecule
        mf = None,
        dt = 0.005,
        nsteps = 25,
        trial = None,
        total_time = 5.0,
        num_walkers = 100,
        renorm_freq = 5,
        random_seed = 1,
        taylor_order = 20,
        energy_scheme = None,
        batched = False,
        *args, **kwargs):
        r"""

        Args:

           system:      (or molecule) that contains the information about
                        number of electrons, orbitals, Hamiltonain, etc.
           propagator:  propagator class that deals with the way of propagating walkers.
           walker:      Walkers used for open ended random walk.
           renorm_freq: renormalization frequency
           nblocks:     Number of blocks
           nsteps:      Number of steps per block
        """

        self.system = self.mol = system

        # propagator params
        self.dt = dt
        self.total_time = total_time
        self.propagator = None
        self.nsteps = nsteps   #
        self.nblocks = 500     #
        self.pop_control_freq = 5                 # population control frequency
        self.pop_control_method = "pair_brach"    # populaiton control method
        self.eq_time = 2.0                        # time of equilibration phase
        self.eq_steps = int(self.eq_time/self.dt) # Number of time steps for the equilibration phase
        self.stablize_freq = 5                    # Frequency of stablization(re-normalization) steps
        self.energy_scheme = energy_scheme
        self.verbose = 1
        self.stdout = sys.stdout

        self.trial = trial


        # walker parameters
        # TODO: move these variables into walker object
        self.walker = None
        self.__dict__.update(kwargs)

        self.taylor_order = taylor_order
        self.num_walkers = num_walkers
        self.renorm_freq = renorm_freq
        self.random_seed = random_seed
        self.walker_coeff = None
        self.walker_tensors = None

        self.mf_shift = None
        self.print_freq = 10

        self.hybrid_energy = None

        self.batched = batched

        self.build() # setup calculations


    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.info(self, "\n========  Initialize Trial WF and Walker  ======== \n")
        if ( self.trial == "RHF" ):
            print( "A", self.trial )
            self.trial = TrialHF(self.mol, trial=self.trial)
            self.spin_fac = 1.0
        elif ( self.trial == "UHF" ):
            self.trial = TrialUHF(self.mol, trial=self.trial)
            self.spin_fac = 0.5
        elif ( self.trial is None ):
            print("No trial wfn selected. Defaulting to RHF.")
            self.trial = TrialHF(self.mol)
            self.spin_fac = 1.0

        # set up walkers
        # TODO: move this into walker class
        temp = self.trial.wf.copy()
        self.walker_tensors = np.array([temp] * self.num_walkers, dtype=np.complex128)
        self.walker_coeff = np.array([1.] * self.num_walkers)


    def get_integrals(self):
        r"""
        TODO:
        return oei and eri in MO
        """

        overlap = self.mol.intor('int1e_ovlp')
        self.ao_coeff = lo.orth.lowdin(overlap)
        norb = self.ao_coeff.shape[0]

        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, self.ao_coeff)
        h1e, eri, self.nuc_energy = read_fcidump(ftmp.name, norb)

        # Cholesky decomposition of eri
        eri_2d = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(eri_2d)
        ltensor = u * np.sqrt(s)
        ltensor = ltensor.T
        ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        self.nfields = ltensor.shape[0]

        return h1e, eri, ltensor


    def propagation(self, h1e, xbar, ltensor):
        pass

    def measure_observables(self, operator):
        observables = None
        return observables

    def walker_trial_overlap(self):
        # BMW:
        # I originally wanted this to sum over spin, 
        #   but we need to do another multiplication with wavefunction immediately after this: 
        #   e.g., theta = np.einsum("zSqp,zSpr->zSqr", self.walker_tensors, inv_overlap)
        # (j,k) are MO labels; (a,b) are AO labels; z is the walker label; S is the spin label
        
        # BMW:
        # Also, I believe that overlaps of determinants are determinants of the overlaps
        # <TRIAL|psi> = det(TRIAL_UP psi_UP) * det(TRIAL_DOWN psi_DOWN)
        # So how can we compute overlaps with only the knowledge of the wavefunction dot products ?
        # Does it have something to do with our Lowdin orthogonalization, which eliminates the need for determinants ?

        #return np.einsum('pr,zpq->zrq', self.trial.wf.conj(), self.walker_tensors) # Original: Yu Zhang
        return np.einsum('Saj,zSak->zSjk', self.trial.wf.conj(), self.walker_tensors) # shape = (Nwalkers,spin,NMO,NMO)

    def renormalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """

        ortho_walkers = np.zeros_like(self.walker_tensors)
        for idx in range(self.walker_tensors.shape[0]):
            ortho_walkers[idx] = np.linalg.qr(self.walker_tensors[idx])[0]
        self.walker_tensors = ortho_walkers

    def local_energy_YuZhang(self, h1e, eri, G1p):
        tmp  = 2 * np.einsum("prqs,zSpr->zSqs", eri, G1p)
        tmp -=     np.einsum("prqs,zSps->zSqr", eri, G1p)
        
        e1   = 2 * np.einsum("zSpq,pq->z",   G1p, h1e)
        e2   =     np.einsum("zSqs,zSqs->z", tmp, G1p)

        energy = e1 + e2 + self.nuc_energy
        return energy

    def local_energy(self, h1e, eri, G1p, trace_lTheta, lTheta, ltensor):
        # BMW:
        # Is this the same factors with and without spin ?
        # Eq. 72-74, J. Chem. Phys. 154, 024107 (2021)

        # One-body Terms
        e1   = 2 * np.einsum("zSjk,jk->z", G1p, h1e)

        # Two-body Terms
        Hartree   = np.einsum("zn->z", trace_lTheta**2 )
        tmp       = np.einsum("zSnjk,zSnkl->zSnjl", lTheta, lTheta ) # Eq. 74, J. Chem. Phys. 154, 024107 (2021)
        Exchange  = np.einsum("zSnjj->z", tmp)
        e2        = 0.5 * (Hartree - Exchange)

        energy = e1 + e2 + self.nuc_energy
        return energy


    def update_weight(self, overlap, cfb, cmf, local_energy):
        r"""
        Update the walker coefficients
        """
        newoverlap = self.walker_trial_overlap() # shape = (walker,spin,NMO,NMO)
        # be cautious! power of 2 was neglected before.
        overlap_ratio = (np.linalg.det(newoverlap) / np.linalg.det(overlap))**2
        #overlap_ratio = np.sum( overlap_ratio, axis=1 ) # BMW: Sum over spin here -- Is this right ?
        overlap_ratio = np.prod( overlap_ratio, axis=1 ) # BMW: Sum over spin here -- Is this right ?

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            self.ebound = (2.0 / self.dt) ** 0.5
            hybrid_energy = -(np.log(overlap_ratio) + cfb + cmf) / self.dt
            hybrid_energy = np.clip(hybrid_energy.real, a_min=-self.ebound, a_max=self.ebound, out=hybrid_energy.real)
            self.hybrid_energy = hybrid_energy if self.hybrid_energy is None else self.hybrid_energy

            importance_func = np.exp(-self.dt * 0.5 * (hybrid_energy + self.hybrid_energy))
            self.hybrid_energy = hybrid_energy
            phase = (-self.dt * self.hybrid_energy-cfb).imag
            phase_factor = np.array([max(0, np.cos(iphase)) for iphase in phase])
            importance_func = np.abs(importance_func) * phase_factor

        elif self.energy_scheme == "local":
            # The local energy formalism
            overlap_ratio = overlap_ratio * np.exp(cmf)
            phase_factor = np.array([max(0, np.cos(np.angle(iovlp))) for iovlp in overlap_ratio])
            importance_func = np.exp(-self.dt * np.real(local_energy)) * phase_factor

        else:
            raise ValueError(f'scheme {self.energy_scheme} is not available!!!')

        self.walker_coeff *= importance_func

    def kernel(self, trial_wf=None):
        r"""
        trial_wf: trial wavefunction
        TBA
        """

        np.random.seed(self.random_seed)

        logger.info(self, "\n======== get integrals ========")

        # BMW 
        # The values of h1e, eri, and ltensor do not depend on spin-polarization
        # However, do we need to duplicate the above integrals for spin-dependent shape ?
        # The off-diagonal blocks of each should be zero, so we only need one additional label S
        # e.g., h1e.shape = (NAO,NAO) --> h1e.shape = (S,NAO,NAO)
        # e.g., eri.shape = (NAO,NAO,NAO,NAO) --> eri.shape = (S,NAO,NAO,NAO,NAO)
        # If we are just duplicating the values, we don't need to add the label. Einsum can handle it.
        h1e, eri, ltensor = self.get_integrals() # (NAO,NAO), (NAO,NAO,NAO,NAO), (NAO**2,NAO,NAO)

        # BMW
        # Defined spin-resolved density matrix in AO basis: <a|rho_mf|b>
        # We enforce that the off-diagonal blocks (between spins) are zero by including only one spin label S
        # Note: .conj() is not required since we only have real-valued wavefunctions in PySCF
        # Note: Here, and in future, (j,k) are MOs while (a,b) are AO basis
        rho_mf = np.einsum( "Saj,Sbj->Sab", self.trial.wf.conj(), self.trial.wf )
        
        # BMW:
        # Contruct the mean-field shift term (Eq. 7,8 from J. Chem. Phys. 124, 224101 2006)
        # <v_n> = <TRIAL| \hat{v}_n |TRIAL> = Tr[ \hat{v}_n \rho_mf ]
        # \hat{v} are the effective one-body operators of the two-body term
        # \rho_mf is the mean-field density matrix
        # "ab,ab" = Tr[A.T @ B]
        self.mf_shift = 1j * np.einsum("nab,Sab->n", ltensor, rho_mf) # Should we sum over spin S ? Yes, <...> is a number


        # BMW
        # I don't understand the purpose of this operation. Maybe because I am not electronic structure person.
        # I don't see this step in any of the AFQMC references that I am looking at.
        # Furthermore, why are we using the eri at all anymore ?
        # If I remove this line, the code breaks. 
        # Are we not double counting the eri/ltensor terms somehow ?
        shifted_h1e = np.zeros(h1e.shape)
        for p, q in itertools.product(range(h1e.shape[0]), repeat=2): # Equivalent to nested for-loop
            shifted_h1e[p, q] = h1e[p, q] - 0.5 * np.trace(eri[p, :, :, q])
        
        # BMW
        # Eqns. 18-19, J. Chem. Phys. 154, 024107 (2021)
        # Here, we relocate the one-body terms of the two-body Hamiltonian into the one-body Hamiltonian
        shifted_h1e = shifted_h1e - np.einsum("n,npq->pq", self.mf_shift, 1j*ltensor)

        # BMW:
        # Precompute the action of the trial wfn on the ltensor -- not a time-dependent thing
        # We need this object to construct the one-particle density matrix and the force bias
        # Eq. 65-69, J. Chem. Phys. 154, 024107 (2021)
        # G_ij^S = <TRIAL|a_j^\dag a_k|psi_k> / <TRIAL|psi_k>      
        #        = [ <psi_k| (<TRIAL|psi_k>)^{-1}|TRIAL>]_{ji}
        #        = [ |THETA><TRIAL| ]_{ji}
        # <F_n>  = \sqrt(dt) Tr[\hat{L}_n \hat{G}]
        #        = \sqrt(dt) Tr[(<TRIAL|\hat{L}_n) THETA]
        # THETA  = <psi_k|(<TRIAL|psi_k>)^{-1}
        # The final shape of this operation is seemingly (spin,Ntensors,NMO,NAO), a rectangular matrix
        # wf.shape = (NAO,NMO), ltensor.shape = (S,NAO**2,NAO,NAO)
        # Note: (j,k) are MOs while (a,b) are AO basis
        #self.precomputed_ltensor = np.einsum("Spr,npq->Snrq", self.trial.wf.conj(), ltensor) # Original by Yu Zhang, shape = (spin,Ntensors,NMO,NAO)
        self.precomputed_ltensor = np.einsum("Saj,nab->Snjb", self.trial.wf.conj(), ltensor) # Same as original


        time = 0.0
        energy_list = []
        time_list = []
        while time <= self.total_time:
            dump_result = (int(time/self.dt) % self.print_freq  == 0)

            # pre-processing: prepare walker tensor
            overlap      = self.walker_trial_overlap() # shape = (Nwalkers,spin,NMO,NMO)
            inv_overlap  = np.linalg.inv(overlap)      # shape = (Nwalkers,spin,NMO,NMO)
            
            # BMW:
            # Compute the force bias: F_n = \sqrt(dt) <L_n> = \sqrt(dt) Tr[\hat{L}_n \hat{G}]
            # Eq. 65-69, J. Chem. Phys. 154, 024107 (2021)
            Theta        = np.einsum("zSaj,zSjk->zSak", self.walker_tensors, inv_overlap) # shape = (Nwalkers,spin,NAO,NMO)
            # Compute one-particle density matrix: G1p = <a_j^\dag a_k> = THETA TRIAL
            G1p          = np.einsum("zSaj,Sbj->zSab", Theta, self.trial.wf.conj()) # shape = (Nwalkers,spin,NAO,NAO)
            # Compute action of the precomputed ltensor (spin,Ntensors,NMO,NAO) on the current Theta(Nwalkers,spin,NAO,NMO)
            lTheta       = np.einsum('Snja,zSak->zSnjk', self.precomputed_ltensor, Theta) # shape = (Nwalkers,spin,Ntensors,NMO,NMO)
            trace_lTheta = np.einsum('zSnjj->zn', lTheta) # shape = (Nwalkers,Ntensors)

            # compute local energy for each walker
            local_energy = self.local_energy_YuZhang(h1e, eri, G1p)
            #local_energy = self.local_energy(h1e, eri, G1p, trace_lTheta, lTheta, ltensor)
            energy = np.sum([self.walker_coeff[i]*local_energy[i] for i in range(len(local_energy))])
            energy = energy / np.sum(self.walker_coeff)

            # imaginary time propagation
            xbar = -np.sqrt(self.dt) * (1j * 2 * trace_lTheta - self.mf_shift) # shape = (Nwalkers,Ntensors)
            cfb, cmf = self.propagation(shifted_h1e, xbar, ltensor)
            self.update_weight(overlap, cfb, cmf, local_energy)

            # re-orthogonalization
            if int(time / self.dt) == self.renorm_freq:
                self.renormalization()

            # print energy and time
            if dump_result:
                time_list.append(time)
                energy_list.append(energy)
                logger.info(self, f" Time: {time:9.3f}    Energy: {energy:15.8f}")

            time += self.dt

        return time_list, energy_list