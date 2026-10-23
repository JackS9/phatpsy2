"""Type stubs for _phatpsy_core compiled Fortran extension module."""
from typing import Any
import numpy as np
import numpy.typing as npt

class phatpsy_state_mod:
    natom: int
    nelem: int
    norbt: int
    mstot: int
    ncycl: int
    tmole: float
    totke: float
    totpe: float
    zzpot: float
    deltae: float
    cdamp: float
    vdamp: float
    qcheck: bool
    qatom: bool
    qopen: bool
    qfirst: bool
    qlast: bool
    qprint: bool
    qewmo: bool
    qdebug: bool
    qtrace: bool

    @staticmethod
    def init_block_data_and_defaults() -> None: ...
    @staticmethod
    def compute_derived_dimensions_and_allocate() -> None: ...
    @staticmethod
    def get_evalx(n: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_occx(n: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_evecx(m: int, n: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_ovlp(n2: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_ewmat(n2: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_coord(n3: int, n: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_charge(n3: int, n: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_popul(n: int, nspin: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_iatomx(m: int) -> npt.NDArray[np.int32]: ...
    @staticmethod
    def get_nx(m: int) -> npt.NDArray[np.int32]: ...
    @staticmethod
    def get_lx(m: int) -> npt.NDArray[np.int32]: ...
    @staticmethod
    def get_mlx(m: int) -> npt.NDArray[np.int32]: ...
    @staticmethod
    def get_etax(m: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def get_anormx(m: int) -> npt.NDArray[np.float64]: ...

class phatpsy_contrl_mod:
    @staticmethod
    def contrl_read_control_input() -> bool: ...
    @staticmethod
    def contrl_init_tables() -> None: ...
    @staticmethod
    def contrl_init_elements() -> bool: ...
    @staticmethod
    def contrl_step_aloop_and_ewmo() -> bool: ...
    @staticmethod
    def contrl_step_scf() -> bool: ...

class phatpsy_scf_mod:
    use_lapack_eig: bool
    fock_mix_alpha: float

class phatpsy_analys_mod:
    use_lapack_fit: bool

class phatpsy_integrals_mod:
    use_in_memory_tei: bool
    @staticmethod
    def clear_tei_cache() -> None: ...

class phatpsy_ewmo_mod:
    qnopop_ewmo: bool
    use_lowdin_ewmo: bool

class phatpsy_modelpot_mod:
    @staticmethod
    def genfac(nfact: int) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def genbc(nbinom: int, nrows: int, index_arr: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def gennrm_default(nplm: int, fact: npt.NDArray[np.float64], lmax: int, qrealy: bool) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def intnrm(evec: npt.NDArray[np.float64], projec: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @staticmethod
    def dampv(epot2: npt.NDArray[np.float64], pkpot: npt.NDArray[np.float64], zpot2: npt.NDArray[np.float64], vpot2: npt.NDArray[np.float64], vdamp: float) -> None: ...
