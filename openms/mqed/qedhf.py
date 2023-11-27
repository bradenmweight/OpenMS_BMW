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
import copy
import numpy
from openms import __config__
from pyscf import lib
from pyscf.scf import hf
from pyscf.dft import rks
# from mqed.lib      import logger


r"""
Theoretical background
^^^^^^^^^^^^^^^^^^^^^^

Within the Coherent State (CS) representation (for photonic DOF), the
QEDHF wavefunction ansatz is

.. math::

   \ket{\Psi} = & \prod_\alpha e^{z_\alpha b^\dagger_\alpha - z^*_\alpha b_\alpha } \ket{HF}\otimes{0_p} \\
              = & U(\mathbf{z}) \ket{HF}\otimes{0_p}.

where :math:`z_\alpha=-\frac{\lambda_\alpha\cdot\langle\boldsymbol{D}\rangle}{\sqrt{2\omega_\alpha}}` denotes
the photon displacement due to the coupling with electrons.

Consequently, we can use :math:`U(\mathbf{z})` to transform the original PF Hamiltonian
into CS representation

.. math::

    H_{CS} = & U^\dagger(\mathbf{z}) H U(\mathbf{z}) \\
           = & H_e+\sum_\alpha\Big\{\omega_\alpha b^\dagger_\alpha b_\alpha
               +\frac{1}{2}[\lambda_\alpha\cdot(\boldsymbol{D}-\langle\boldsymbol{D}\rangle)]^2  \\
             & -\sqrt{\frac{\omega_\alpha}{2}}[\lambda_\alpha\cdot(\boldsymbol{D} -
             \langle\boldsymbol{D}\rangle)](b^\dagger_\alpha+ b_\alpha)\Big\}.


With the ansatz, the QEDHF energy is

.. math::

  E_{QEDHF}= E_{HF} + \frac{1}{2}\langle \boldsymbol{lambda}\cdot [\boldsymbol{D}-\langle \boldsymbol{D}\rangle)]^2\rangle,


"""

from pyscf.lib import logger
from pyscf.scf import diis
from pyscf.scf import _vhf
from pyscf.scf import chkfile
from pyscf.data import nist
from pyscf import __config__


WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)
MUTE_CHKFILE = getattr(__config__, 'scf_hf_SCF_mute_chkfile', False)

# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str


"""
 the only difference against bare hf kernel is that we include get_hcore within
 the scf cycle because qedhf has DSE-mediated hcore which depends on DM
"""

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    '''kernel: the SCF driver.

    Args:
        mf : an instance of SCF class
            mf object holds all parameters to control SCF.  One can modify its
            member functions to change the behavior of SCF.  The member
            functions which are called in kernel are

            | mf.get_init_guess
            | mf.get_hcore
            | mf.get_ovlp
            | mf.get_veff
            | mf.get_fock
            | mf.get_grad
            | mf.eig
            | mf.get_occ
            | mf.make_rdm1
            | mf.energy_tot
            | mf.dump_chk

    Kwargs:
        conv_tol : float
            converge threshold.
        conv_tol_grad : float
            gradients converge threshold.
        dump_chk : bool
            Whether to save SCF intermediate results in the checkpoint file
        dm0 : ndarray
            Initial guess density matrix.  If not given (the default), the kernel
            takes the density matrix generated by ``mf.get_init_guess``.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.

    Returns:
        A list :   scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

        scf_conv : bool
            True means SCF converged
        e_tot : float
            Hartree-Fock energy of last iteration
        mo_energy : 1D float array
            Orbital energies.  Depending the eig function provided by mf
            object, the orbital energies may NOT be sorted.
        mo_coeff : 2D array
            Orbital coefficients.
        mo_occ : 1D array
            Orbital occupancies.  The occupancies may NOT be sorted from large
            to small.

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=numpy.eye(mol.nao_nr()))
    >>> print('conv = %s, E(HF) = %.12f' % (conv, e))
    conv = True, E(HF) = -1.081170784378
    '''
    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    if mf.qed.use_cs:
        mf.qed.update_cs(dm)
    vhf = mf.get_veff(mol, dm)
    h1e = mf.get_hcore(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    from openms import mqed

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot
        if isinstance(mf, mqed.qedhf.RHF):
            h1e = mf.get_hcore(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        if isinstance(mf, mqed.qedhf.RHF):
            h1e = mf.get_hcore(mol, dm) # seems converg slower in thic as diis does not apply to hcore
        e_tot = mf.energy_tot(dm, h1e, vhf)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        if isinstance(mf, mqed.qedhf.RHF):
            h1e = mf.get_hcore(mol, dm)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)

# in the future, replace it with our own object?
class TDMixin(lib.StreamObject):
    conv_tol = getattr(__config__, "tdscf_rhf_TDA_conv_tol", 1e-9)
    nstates = getattr(__config__, "tdscf_rhf_TDA_nstates", 3)
    singlet = getattr(__config__, "tdscf_rhf_TDA_singlet", True)
    lindep = getattr(__config__, "tdscf_rhf_TDA_lindep", 1e-12)
    level_shift = getattr(__config__, "tdscf_rhf_TDA_level_shift", 0)
    max_space = getattr(__config__, "tdscf_rhf_TDA_max_space", 50)
    max_cycle = getattr(__config__, "tdscf_rhf_TDA_max_cycle", 100)


from openms.lib.boson import Photon

class RHF(hf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    """

    def __init__(self, mol, xc=None, **kwargs):
        hf.RHF.__init__(self, mol)
        # if xc is not None:
        #    rks.KohnShamDFT.__init__(self, xc)

        cavity = None
        qed = None
        add_nuc_dipole = False

        if "cavity" in kwargs:
            cavity = kwargs["cavity"]
        if "add_nuc_dipole" in kwargs:
            add_nuc_dipole = kwargs["add_nuc_dipole"]
        if "qed" in kwargs:
            qed = kwargs["qed"]
        else:
            if "cavity_mode" in kwargs:
                cavity_mode = kwargs["cavity_mode"]
            else:
                raise ValueError("The required keyword argument 'cavity_mode' is missing")

            if "cavity_freq" in kwargs:
                cavity_freq = kwargs["cavity_freq"]
            else:
                raise ValueError("The required keyword argument 'cavity_freq' is missing")
            print("cavity_freq=", cavity_freq)
            print("cavity_mode=", cavity_mode)

            nmode = len(cavity_freq)
            gfac = numpy.zeros(nmode)
            for i in range(nmode):
                gfac[i] = numpy.sqrt(numpy.dot(cavity_mode[i], cavity_mode[i]))
                if gfac[i] != 0:  # Prevent division by zero
                    cavity_mode[i] /= gfac[i]
            qed = Photon(mol, mf=self, omega=cavity_freq, vec=cavity_mode, gfac=gfac, add_nuc_dipole=add_nuc_dipole, shift=False)
        # end of define qed object

        self.qed = qed

        # make dipole matrix in AO
        #self.make_dipolematrix() # replaced by qed functions
        self.qed.get_gmatao()
        self.qed.get_q_dot_lambda()

        self.gmat = self.qed.gmat
        self.qd2 = self.qed.q_dot_lambda

        # print(f"{cavity} cavity mode is used!")
        # self.verbose    = mf.verbose
        # self.stdout     = mf.stdout
        # self.mol        = mf.mol
        # self.max_memory = mf.max_memory
        # self.chkfile    = mf.chkfile
        # self.wfnsym     = None
        self.dip_ao = mol.intor("int1e_r", comp=3)
        self.bare_h1e = None # bare oei

    #def make_dipolematrix(self):
    #    """
    #    return dipole and quadrupole matrix in AO

    #    Quarupole:
    #    # | xx, xy, xz |
    #    # | yx, yy, yz |
    #    # | zx, zy, zz |
    #    # xx <-> rrmat[0], xy <-> rrmat[3], xz <-> rrmat[6]
    #    #                  yy <-> rrmat[4], yz <-> rrmat[7]
    #    #                                   zz <-> rrmat[8]
    #    """

    #    self.mu_mo = None
    #    charges = self.mol.atom_charges()
    #    coords = self.mol.atom_coords()
    #    charge_center = (0, 0, 0)  # numpy.einsum('i,ix->x', charges, coords)
    #    with self.mol.with_common_orig(charge_center):
    #        self.mu_mat_ao = self.mol.intor_symmetric("int1e_r", comp=3)
    #        self.qmat = -self.mol.intor("int1e_rr")


    """
    def get_hcore(self, mol=None, dm=None):
        #
        # DSE-mediated oei: -<\lambda\cdot D> g^\alpha_{uv} - 0.5 q^\alpha_{uv}
        #                = -Tr[\rho g^\alpha] g^\alpha_{uv} - 0.5 q^\alpha_{uv}
        #

        if mol is None: mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.bare_h1e is None:
            self.bare_h1e = hf.get_hcore(mol)

        self.oei = - lib.einsum("Xpq, X->pq", self.gmat, self.z_lambda)
        self.oei -= numpy.sum(self.qd2, axis=0)
        return self.bare_h1e + self.oei
    """

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if not omega and (
            self._eri is not None or mol.incore_anyway or self._is_mem_enough()
        ):
            if self._eri is None:
                self._eri = mol.intor("int2e", aosym="s8")
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = RHF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        # note this term exists even if we don't use CS representation
        # don't simply replace this term with z since z can be zero without CS.

        dm_shape = dm.shape
        nao = dm_shape[-1]
        dm = dm.reshape(-1,nao,nao)
        n_dm = dm.shape[0]
        logger.debug(self, "No. of dm is %d", n_dm)

        vj_dse = numpy.zeros((n_dm,nao,nao))
        vk_dse = numpy.zeros((n_dm,nao,nao))
        for i in range(n_dm):
            scaled_mu = 0.0
            for imode in range(self.qed.nmodes):
                scaled_mu += numpy.einsum("pq, pq ->", dm[i], self.gmat[imode])# <\lambada * D>

            # DSE-medaited J
            for imode in range(self.qed.nmodes):
                vj_dse[i] += scaled_mu * self.gmat[imode]

            # DSE-mediated K
            vk_dse[i] += numpy.einsum("Xpr, Xqs, rs -> pq", self.gmat, self.gmat, dm[i])
            #gdm = numpy.einsum("Xqs, rs -> Xqr", self.gmat, dm)
            #vk += numpy.einsum("Xpr, Xqr -> pq", self.gmat, gdm)

        vj += vj_dse.reshape(dm_shape)
        vk += vk_dse.reshape(dm_shape)

        return vj, vk

    # get_veff = get_veff
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""QED Hartree-Fock potential matrix for the given density matrix

        .. math::

            V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        if self.qed.use_cs:
            self.qed.update_cs(dm)

        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * 0.5
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * 0.5
            vhf += numpy.asarray(vhf_last)

        # add photon contribution
        #if self.mu_mat_ao is None:
        #    self.make_dipolematrix()

        # DSE-mediated oei
        self.oei = self.qed.add_oei_ao(dm)
        vhf += self.oei

        return vhf

    def dump_flags(self, verbose=None):
        return hf.RHF.dump_flags(self, verbose)

    def dse(self, dm):
        r"""
        compute dipole self-energy
        """
        dip = self.dip_moment(dm=dm)
        # print("dipole_moment=", dip)
        e_dse = 0.0
        e_dse += 0.5 * numpy.dot(self.qed.z_lambda, self.qed.z_lambda)

        print("dipole self-energy=", e_dse)
        return e_dse

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        r"""Total QED Hartree-Fock energy, electronic part plus nuclear repulstion
        See :func:`scf.hf.energy_elec` for the electron part

        Note this function has side effects which cause mf.scf_summary updated.
        """

        nuc = self.energy_nuc()
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc
        e_tot += 0.5 * numpy.einsum("pq,pq->", self.oei, dm)
        dse = self.dse(dm)  # dipole sefl-energy(0.5*z^2)
        e_tot += dse
        self.scf_summary["nuc"] = nuc.real
        return e_tot

    """
    def scf(self, dm0=None, **kwargs):

        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    """

#----------------------------------------------

    """
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        #print("debug-zy: qed get_jk")
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        # add photon contribution, not done yet!!!!!!! (todo)
        # update molecular-cavity couplings
        vp = numpy.zeros_like(vj)

        vj += vp
        vk += vp
        return vj, vk
    """


class RKS(rks.KohnShamDFT, RHF):
    def __init__(self, mol, xc="LDA,VWN", **kwargs):
        RHF.__init__(self, mol, **kwargs)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    get_veff = rks.get_veff
    get_vsap = rks.get_vsap
    energy_elec = rks.energy_elec


"""
class RKS(rks.RKS):

    def __init__(self, mol, xc=None, **kwargs):
        print("debug- DFT driver is used!")
        print("xc=", xc)
        rks.RKS.__init__(self, mol, xc=xc)

        cavity = None
        if "cavity" in kwargs:
            cavity = kwargs['cavity']
            if "cavity_mode" in kwargs:
                cavity_mode = kwargs['cavity_mode']
            else:
                raise ValueError("The required keyword argument 'cavity_mode' is missing")

            if "cavity_freq" in kwargs:
                cavity_freq = kwargs['cavity_freq']
            else:
                raise ValueError("The required keyword argument 'cavity_freq' is missing")

            print('cavity_freq=', cavity_freq)
            print('cavity_mode=', cavity_mode)

        print(f"{cavity} cavity mode is used!")

    def dump_flags(self, verbose=None):
        return rks.RKS.dump_flags(self, verbose)

"""


# this eventually will moved to qedscf/rhf class
#

"""
def get_veff(mf, dm, dm_last=None):
  veff = None

  return veff
"""


def qedrhf(model, options):
    # restricted qed hf
    # make a copy for qedhf
    mf = copy.copy(model.mf)
    conv_tol = 1.0e-10
    conv_tol_grad = None
    dump_chk = False
    callback = None
    conv_check = False
    noscf = False
    if "noscf" in options:
        noscf = options["noscf"]

    # converged bare HF coefficients
    na = int(mf.mo_occ.sum() // 2)
    ca = mf.mo_coeff
    dm = 2.0 * numpy.einsum("ai,bi->ab", ca[:, :na], ca[:, :na])
    mu_ao = model.dmat

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    mol = mf.mol

    # initial guess
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)

    e_tot = mf.energy_tot(dm, h1e, vhf)
    nuc = mf.energy_nuc()
    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    print("converged Tr[D]=", numpy.trace(dm) / 2.0)
    nmode = model.vec.shape[0]

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        # fock = mf.get_fock(h1e, s1e, vhf, dm)
        fock = h1e + vhf

        """
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        """

        mu_mo = lib.einsum("pq, Xpq ->X", dm, mu_ao)

        scaled_mu = 0.0
        z_lambda = 0.0
        for imode in range(nmode):
            z_lambda -= numpy.dot(mu_mo, model.vec[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, model.gmat[imode])

        dse = 0.5 * z_lambda * z_lambda

        # oei = numpy.zeros((h1e.shape[0], h1e.shape[1]))
        oei = model.gmat * z_lambda
        oei -= model.qd2
        oei = numpy.sum(oei, axis=0)
        fock += oei

        #  <>
        for imode in range(nmode):
            fock += scaled_mu * model.gmat[imode]
        fock -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", model.gmat, model.gmat, dm)

        # e_tot = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse
        e_tot = 0.5 * numpy.einsum("pq,pq->", (oei + h1e + fock), dm) + nuc + dse

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        # factor of 2 is applied (via mo_occ)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        """
      # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
      # instead of the statement "fock = h1e + vhf" because Fock matrix may
      # be modified in some methods.

      fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
      oei = model.gmat * model.gmat
      oei -= model.qd2
      oei = numpy.sum(oei, axis=0)

      fock += oei

      mu_mo = lib.einsum('pq, Xpq ->X', 2 * dm, mu_ao)
      z_lambda = 0.0
      scaled_mu = 0.0
      for imode in range(nmode):
          z_lambda += -numpy.dot(mu_mo, model.vec[imode])
          scaled_mu += numpy.einsum('pq, pq ->', dm, model.gmat[imode])
      dse = 0.5 * z_lambda * z_lambda

      for imode in range(nmode):
          fock += 2 * scaled_mu * model.gmat[imode]
      fock -= numpy.einsum('Xpr, Xqs, rs -> pq', model.gmat, model.gmat, dm)
      """

        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)
        print(
            "cycle= %3d E= %.12g  delta_E= %4.3g |g|= %4.3g |ddm|= %4.3g |d*u|= %4.3g dse= %4.3g"
            % (cycle + 1, e_tot, e_tot - last_hf_e, norm_gorb, norm_ddm, z_lambda, dse)
        )
        if noscf:
            break

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    # to be updated
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        # fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        # e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse, e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        print(
            "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
            e_tot,
            e_tot - last_hf_e,
            norm_gorb,
            norm_ddm,
        )
        if dump_chk:
            mf.dump_chk(locals())

    # A post-processing hook before return
    mf.post_kernel(locals())
    print("HOMO-LUMO gap=", mo_energy[na] - mo_energy[na - 1])
    print("QEDHF energy=", e_tot)

    return scf_conv, e_tot, dse, mo_energy, mo_coeff, mo_occ


if __name__ == "__main__":
    # will add a loop
    import numpy
    from pyscf import gto, scf

    itest = 1
    zshift = itest * 2.0

    atom = f"C   0.00000000   0.00000000    {zshift};\
             O   0.00000000   1.23456800    {zshift};\
             H   0.97075033  -0.54577032    {zshift};\
             C  -1.21509881  -0.80991169    {zshift};\
             H  -1.15288176  -1.89931439    {zshift};\
             C  -2.43440063  -0.19144555    {zshift};\
             H  -3.37262777  -0.75937214    {zshift};\
             O  -2.62194056   1.12501165    {zshift};\
             H  -1.71446384   1.51627790    {zshift}"

    mol = gto.M(
        atom = atom,
        basis="sto3g",
        #basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=3,
    )
    print("mol coordinates=\n", mol.atom_coords())

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run(verbose=4)

    print("electronic energies=", mf.energy_elec())
    print("nuclear energy=     ", mf.energy_nuc())
    dm = mf.make_rdm1()

    print("\n=========== QED-HF calculation  ======================\n")

    from openms.mqed import qedhf

    nmode = 2 # create a zero (second) mode to test the code works for multiple modes
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel(dm0=dm)
    print(f"Total energy:     {qedmf.e_tot:.10f}")

    qed = qedmf.qed

    # get I
    qed.kernel()
    I = qed.get_I()
    F = qed.g_fock()

