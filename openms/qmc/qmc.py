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
import random

from openms.qmc.trial import TrialHF, TrialUHF
from openms import runtime_refs, _citations
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
        mol, # or molecule
        mf = None,
        dt = 0.005,
        cavity_freq = None,
        trial = None,
        nsteps = 25,
        total_time = 5.0,
        num_walkers = 100,
        renorm_freq = 10,
        random_seed = random.randint(1,1_000_000), #1,
        taylor_order = 6,
        energy_scheme = None,
        batched = False,
        cavity = None,
        compute_wavefunction = False,
        **kwargs):
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
        if ( _citations["pra2024"] not in runtime_refs ):
            runtime_refs.append(_citations["pra2024"])

        self.system = self.mol = mol
        self.energy_nuc = self.mol.energy_nuc() # Store nuclear energy for entire calculation

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
        self.compute_wavefunction = compute_wavefunction

        self.trial = trial


        self.mf_shift = None
        self.print_freq = 1

        self.hybrid_energy = None

        self.batched = batched

        # cavity parameters
        self.cavity = cavity

        self.build() # setup calculations



    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.info(self, "\n========  Initialize Trial WF and Walker  ======== \n")
        if ( self.trial == "RHF" ):
            self.trial = TrialHF(self.mol, cavity=self.cavity, trial=self.trial)
            self.spin_fac = 1.0
        elif ( self.trial == "UHF" ):
            self.trial = TrialUHF(self.mol, cavity=self.cavity, trial=self.trial)
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


    



    def make_read_fcidump(self, norb):
        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, self.ao_coeff)
        h1e, eri, self.nuc_energy = read_fcidump(ftmp.name, norb) # BMW: Nuclear energy is done elsewhere
        return h1e, eri

    def make_ltensor(self, eri, norb):
        """
        BMW: Make sure to do this decomposition after the QED part runs
        Cholesky decomposition of eri
        """
        eri_2d = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(eri_2d)
        ltensor = u * np.sqrt(s)
        ltensor = ltensor.T
        ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        self.nfields = ltensor.shape[0]
        return ltensor

        
    def get_integrals(self):
        r"""
        TODO:
        return oei and eri in MO
        """

        overlap = self.mol.intor('int1e_ovlp')
        self.ao_coeff = lo.orth.lowdin(overlap)
        norb = self.ao_coeff.shape[0]
        h1e, eri = self.make_read_fcidump(norb)

        ltensor = self.make_ltensor(eri, norb)
        return h1e, eri, ltensor


    def propagation(self, h1e, xbar, ltensor):
        pass

    def measure_observables(self, operator):
        observables = None
        return observables

    def walker_trial_overlap(self):
        O = np.einsum('FSaj,zFSak->zSjk', self.trial.wf.conj(), self.walker_tensors) # (Walker,Spin, MO,MO)
        O = np.linalg.det( O ) #  # (Walker,Spin)
        return np.prod( O, axis=1) # Product over determinants of MO overlaps for each spin
    
    def renormalization_YuZhang(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """
        ortho_walkers = np.zeros_like(self.walker_tensors)
        for idx in range(self.walker_tensors.shape[0]):
            ortho_walkers[idx] = np.linalg.qr(self.walker_tensors[idx])[0]
        self.walker_tensors = ortho_walkers


    def renormalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """

        # Orthogonalize the walkers
        shape         = self.walker_tensors[0].shape # (Fock, Spin, AO, MO)
        self.walker_tensors = self.walker_tensors.swapaxes(1,2) # (walkers,Fock,Spin,AO,MO) --> (walkers,Spin,Fock,AO,MO)
        ortho_walkers = np.zeros_like(self.walker_tensors)
        for zi in range( self.num_walkers ):
            for s in range( shape[1] ):
                tmp = self.walker_tensors[zi,s,:,:,:].reshape( -1, shape[-1] ) # (Fock,AO,MO) --> (Fock*AO,MO)
                ortho_walkers[zi,s] = np.linalg.qr( tmp )[0].reshape( (shape[0],shape[2],shape[3]) ) # (Fock*AO,MO) --> (Fock,AO,MO)
        self.walker_tensors = ortho_walkers.swapaxes(1,2) # (walkers,Spin,Fock,AO,MO) --> (walkers,Fock,Spin,AO,MO)

        # BMW:
        # I noticed that the coefficients become huge numbers during propagation.
        # e.g., 155232006980.58905
        # Here, we renormalize the coefficients to avoid numerical instability.
        # This does not seem to affect the results, but there could be numerical 
        #   instability without this correction
        #print( "Norm of Walker Coeffs:", np.sum( self.walker_coeff ) )
        #self.walker_coeff = self.walker_coeff / np.sum( np.abs(self.walker_coeff) )

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

        energy = e1 + e2 + self.nuc_energy
        return energy



    def update_weight(self, oldoverlap, cfb, cmf, local_energy):
        r"""
        Update the walker coefficients
        oldoverlap (float): old overlap
        """
        newoverlap = self.walker_trial_overlap()

        # be cautious! power of 2 was neglected before.
        overlap_ratio = (newoverlap / oldoverlap)**2

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

    def get_wfn(self):
        """
        Get the wavefunction
        """
        pass

    def kernel(self, trial_wf=None):
        r"""
        trial_wf: trial wavefunction
        TBA
        """

        np.random.seed(self.random_seed)

        logger.info(self, "\n======== get integrals ========")
        h1e, eri, ltensor = self.get_integrals()
        shifted_h1e   = np.zeros(h1e.shape)
        rho_mf        = np.einsum( "FSaj,FSbj->Sab", self.trial.wf, self.trial.wf ) # rho_mf in AO Basis (electronic subspace only)
        self.mf_shift = 1j * np.einsum("nab,Sab->n", ltensor, rho_mf) # Compute <L_n>

        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            shifted_h1e[p, q] = h1e[p, q] - 0.5 * np.trace(eri[p, :, :, q])

        #shifted_h1e = shifted_h1e - np.einsum("n,nab->ab", self.mf_shift, 1j*ltensor)
        shifted_h1e     = shifted_h1e + np.einsum("n,nab->ab", self.mf_shift, ltensor)

        self.precomputed_ltensor = np.einsum("FSaj,nab->FSnjb", self.trial.wf.conj(), ltensor) # shape = (Fock,Ntensors,NMO,NAO)

        time = 0.0
        energy_list = []
        time_list = []
        wavefunction_list = []
        while time <= self.total_time:
            #print( time )
            dump_result = (int(time/self.dt) % self.print_freq  == 0)

            # pre-processing: prepare walker tensor
            overlap     = self.walker_trial_overlap() # (Walker) # Photon and spin taken care of already
            ### VERSION 1 BMW ###
            #overlap     = np.linalg.det( overlap ) # Det in electronic sub-space
            Theta       = np.einsum("zFSaj,z->zFSaj", self.walker_tensors, 1/overlap)  # (Walker,Fock,Spin,AO,MO)
            ### VERSION 2 YZ ###
            #inv_overlap = np.linalg.inv(overlap) # (Walker,MO,MO)
            #Theta       = np.einsum("zFSaj,zSjk->zFSak", self.walker_tensors, inv_overlap)  # (Walker,Fock,AO,MO)

            G1p          = np.einsum("zFSaj,GSbj->zFGSab", Theta, self.trial.wf.conj())       # shape = (Nwalkers,Fock,NAO,NAO)
            lTheta       = np.einsum('FSnja,zFSak->zFSnjk', self.precomputed_ltensor, Theta)  # shape = (Nwalkers,Fock,Ntensors,NMO,NMO)
            trace_lTheta = np.einsum('zFSnjj->zn', lTheta)                                 # shape = (Nwalkers,Ntensors)

            # compute local energy for each walker
            local_energy = self.local_energy(h1e, eri, G1p)
            energy       = np.sum([self.walker_coeff[i]*local_energy[i] for i in range(len(local_energy))])
            energy       = energy / np.sum(self.walker_coeff)

            # imaginary time propagation
            F        = -np.sqrt(self.dt) * (1j * 2 * trace_lTheta - self.mf_shift) # shape = (Nwalkers,Ntensors)
            N_I, cmf = self.propagation(shifted_h1e, F, ltensor)
            self.update_weight(overlap, N_I, cmf, local_energy)


            if ( int(time / self.dt) % self.renorm_freq == 0 ):
               #print("Renormalizing at time %1.3f" % time)
               self.renormalization()
            #print( time, np.min(S), np.average(S), np.max(S) )

            # print energy and time
            if dump_result:
                time_list.append(time)
                energy_list.append(energy)
                if ( self.compute_wavefunction ):
                    wavefunction_list.append( self.get_wfn() )
                logger.info(self, f" Time: {time:9.3f}    Energy: {energy:15.8f}")

            time += self.dt
        
        #self.post_kernel()

        
            
        if ( self.compute_wavefunction ):
            return time_list, energy_list, wavefunction_list
        else:
            return time_list, energy_list

