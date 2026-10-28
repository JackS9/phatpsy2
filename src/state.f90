module state
  use iso_fortran_env, only: real64, real32, int32
  implicit none
  public

  !---------------------------------------------------------------------
  ! 1. Legacy /PARMS/ Scalars & Equivalenced Control Flags
  !    Kept in COMMON /PARMS/ during transition so atomic.f, aloop.f,
  !    bloop.f, ewmo.f, and scf.f see identical memory layouts.
  !---------------------------------------------------------------------
  real(real64)   :: APARM(20)
  integer(int32) :: IPARM(50)
  logical(kind=1):: QPARM(50)
  common /PARMS/ APARM, IPARM, QPARM

  ! QPARM equivalences (complete union of main.f and contrl.f)
  logical(kind=1) :: QRSTRT, QBAKUP, QNWBAS, QOPEN,  QFIRST, QLAST,  &
                     QDEBUG, QSEPAT, QCMPTD, QEWMO,  QGSTAT, QPLOT,  &
                     QCORE,  QPRINT, QAUDMP, QATOM,  QMODVO, QMNPRT, &
                     QCHECK, QANGST, QGEOM,  QC2FIT, QX2FIT, QPSMET, &
                     QTRACE, QRERUN, QCSTFT, QGFERM, QDYNEX, QDFTEX
  equivalence (QPARM(1),  QRSTRT), (QPARM(2),  QBAKUP), &
              (QPARM(3),  QNWBAS), (QPARM(4),  QOPEN),  &
              (QPARM(5),  QFIRST), (QPARM(6),  QLAST),  &
              (QPARM(7),  QDEBUG), (QPARM(8),  QSEPAT), &
              (QPARM(9),  QCMPTD), (QPARM(10), QEWMO),  &
              (QPARM(19), QGSTAT), (QPARM(20), QPLOT),  &
              (QPARM(21), QCORE),  (QPARM(22), QPRINT), &
              (QPARM(23), QAUDMP), (QPARM(24), QATOM),  &
              (QPARM(25), QMODVO), (QPARM(28), QMNPRT), &
              (QPARM(29), QCHECK), (QPARM(30), QANGST), &
              (QPARM(31), QGEOM),  (QPARM(32), QC2FIT), &
              (QPARM(33), QX2FIT), (QPARM(34), QPSMET), &
              (QPARM(35), QTRACE), (QPARM(36), QCSTFT), &
              (QPARM(37), QGFERM), (QPARM(38), QDYNEX), &
              (QPARM(39), QDFTEX)

  ! Per-atom flags equivalenced to QPARM(11:18) in atomic.f and scf.f
  logical(kind=1) :: QFLAGS(8)
  logical(kind=1) :: QFIXED, QVIRTL, QNWVEC, QMOVE, QFLIP, QPNCHV, QPLOT_AT, QNWPOT
  equivalence (QPARM(11), QFLAGS(1), QFIXED), &
              (QPARM(12), QFLAGS(2), QVIRTL), &
              (QPARM(13), QFLAGS(3), QNWVEC), &
              (QPARM(14), QFLAGS(4), QMOVE),  &
              (QPARM(15), QFLAGS(5), QFLIP),  &
              (QPARM(16), QFLAGS(6), QPNCHV), &
              (QPARM(17), QFLAGS(7), QPLOT_AT), &
              (QPARM(18), QFLAGS(8), QNWPOT)

  ! IPARM equivalences (complete union of main.f and contrl.f)
  integer(int32) :: NATOM,  NELEM,  MSTO,   NORB,   IORBT,  NORBT,  &
                    N2ORBT, IATOM,  MXMSTO, N2ATOM, MXNORB, MXNSTO, &
                    MDIFAT, NMAX,   NNMX,   NNMXM1, LMAX,   LMXP1,  &
                    LLMXP1, LLMXP2, NCGC,   MXITER, IELEM,  M2STO,  &
                    L3MX,   NFACT,  NABDIM, NVTERM, NSTO,   ISPIN,  &
                    NYLM,   N2ORB,  MXCYCL, NCYCL,  IUNIT0, M4STO,  &
                    JSTOT,  MSTOT
  equivalence (IPARM(1),  NATOM),  (IPARM(2),  NELEM),  &
              (IPARM(5),  MSTO),   (IPARM(6),  NORB),   &
              (IPARM(7),  IORBT),  (IPARM(8),  NORBT),  &
              (IPARM(9),  N2ORBT), (IPARM(10), IATOM),  &
              (IPARM(12), MXMSTO), (IPARM(13), N2ATOM), &
              (IPARM(14), MXNORB), (IPARM(15), MXNSTO), &
              (IPARM(16), MDIFAT), (IPARM(17), NMAX),   &
              (IPARM(18), NNMX),   (IPARM(19), NNMXM1), &
              (IPARM(20), LMAX),   (IPARM(21), LMXP1),  &
              (IPARM(22), LLMXP1), (IPARM(23), LLMXP2), &
              (IPARM(24), NCGC),   (IPARM(25), MXITER), &
              (IPARM(26), IELEM),  (IPARM(27), M2STO),  &
              (IPARM(28), L3MX),   (IPARM(29), NFACT),  &
              (IPARM(30), NABDIM), (IPARM(31), NVTERM), &
              (IPARM(32), NSTO),   (IPARM(33), ISPIN),  &
              (IPARM(34), NYLM),   (IPARM(35), N2ORB),  &
              (IPARM(37), MXCYCL), (IPARM(38), NCYCL),  &
              (IPARM(39), IUNIT0), (IPARM(40), M4STO),  &
              (IPARM(41), JSTOT),  (IPARM(42), MSTOT)

  ! APARM equivalences
  real(real64) :: EXPMIN, CONVRG, THETA, CDAMP, CACCEL, RANGE, PKSCAL, PKE
  equivalence (APARM(1), EXPMIN), (APARM(2), CONVRG), &
              (APARM(3), THETA),  (APARM(4), CDAMP),  &
              (APARM(5), CACCEL), (APARM(6), RANGE),  &
              (APARM(7), PKSCAL), (APARM(8), PKE)

  !---------------------------------------------------------------------
  ! 2. Secondary COMMON Blocks (BPARMS, IODATA, DAUNIT, STATE, ENERGY,
  !    CHARGS, MODPOT, PUNCHK, TIMES, NDXTEI, TABLES, DYNAMC, FIXED)
  !---------------------------------------------------------------------
  integer(int32) :: IELEMB, IUNITB, MDFATB, JDFAT1, MEQATB, JATOM, &
                    NSTOB,  MSTOB,  NORBB,  N2ORBB, IORBTB, M2STOB, &
                    JSTOB,  NVTRMB
  common /BPARMS/ IELEMB, IUNITB, MDFATB, JDFAT1, MEQATB, JATOM, &
                  NSTOB,  MSTOB,  NORBB,  N2ORBB, IORBTB, M2STOB, &
                  JSTOB,  NVTRMB

  integer(int32) :: IUNIT(20), LENBUF
  integer(int32) :: IUNIT1, IUNIT2, IUNIT3, IUNIT4, IR, IW, IP, &
                    IUDUMP, IUTEMP, IUCSF
  common /IODATA/ IUNIT, LENBUF
  equivalence (IUNIT(1), IUNIT1), (IUNIT(2), IUNIT2), &
              (IUNIT(3), IUNIT3), (IUNIT(4), IUNIT4), &
              (IUNIT(5), IR),     (IUNIT(6), IW),     &
              (IUNIT(7), IP),     (IUNIT(8), IUDUMP), &
              (IUNIT(9), IUTEMP), (IUNIT(10), IUCSF)

  integer(int32)  :: NRECS, LENREC, NEXREC, NLRECS, NBYTES
  logical(kind=1) :: QMXREC
  common /DAUNIT/ NRECS, LENREC, NEXREC, NLRECS, NBYTES, QMXREC

  integer(int32)  :: NCREAT, NANNIH, ICREAT(7), IANNIH(7)
  logical(kind=1) :: QTRANS

  real(real64) :: TOTPE, TOTKE, TMOLE, ZZPOT, ETOT, VIRTHM, DELTAE
  common /ENERGY/ TOTPE, TOTKE, TMOLE, ZZPOT, ETOT, VIRTHM, DELTAE

  real(real64)      :: ZNSUM, ELSUM, ELDIF, ELECS(2), ORBOCC, CHTRAN
  character(len=8)  :: SPIN(2)
  common /CHARGS/ ZNSUM, ELSUM, ELDIF, ELECS, SPIN, ORBOCC, CHTRAN

  real(real64)   :: WGT0, WGT1, WGT2, WGTC, WGTV, VDAMP, VACCEL, &
                    TFEXP(4), TFCOEF(4)
  integer(int32) :: LTF(4), MXTF
  common /MODPOT/ WGT0, WGT1, WGT2, WGTC, WGTV, VDAMP, VACCEL, &
                  TFEXP, TFCOEF, LTF, MXTF

  real(real64)    :: RENRMA, RENRMB
  integer(int32)  :: KATOM, KORB, KSP
  logical(kind=1) :: QPNCHK
  common /PUNCHK/ RENRMA, RENRMB, KATOM, KORB, KSP, QPNCHK

  real(real32) :: BEGIN, START, FINISH, SECS, TIMEND
  common /TIMES/ BEGIN, START, FINISH, SECS, TIMEND

  integer(int32) :: IAC, KAC, IBD, KBD, IAD, KAD, IBC, KBC, IAB, KAB, ICD, KCD
  common /NDXTEI/ IAC, KAC, IBD, KBD, IAD, KAD, IBC, KBC, IAB, KAB, ICD, KCD

  real(real64) :: REALS(10), FACT(22), FFAC(19), BINOM(91), HALFPI(8), &
                  ZERO, HALF, ROOT(10), ROOTPI, CONST(10), CONVRT(10)
  common /TABLES/ REALS, FACT, FFAC, BINOM, HALFPI, &
                  ZERO, HALF, ROOT, ROOTPI, CONST, CONVRT

  integer(int32)   :: LISTAR(500), IPTR(10), NNIA(10), NID(10), NXARG(10), NROUT
  character(len=8) :: ROUTIN(10)
  common /DYNAMC/ LISTAR, IPTR, NNIA, NID, NXARG, ROUTIN, NROUT

  integer(int32) :: MSPTR
  common /MOLSTR/ MSPTR

  ! Legacy sub-arena tail for unrefactored inner routines (ATOMIC/SCF)
  integer(int32), parameter :: ICORE_SUB = 500000
  integer(int32) :: MXCORE, NWCORE, NWORDS
  common /FIXED/ MXCORE, NWCORE, NWORDS
  real(real64), allocatable :: SUB_CORE(:)

  ! Additional IPARM slots and A-atom aliases from aloop.f / bloop.f
  integer(int32) :: NUCZA, IEQATA, JDFATA
  equivalence (IPARM(3),  NUCZA),  &
              (IPARM(4),  IEQATA), &
              (IPARM(11), JDFATA)

  ! Additional APARM slots (Euler angles ALPHA, BETA used by RBETA & GENDC)
  real(real64) :: ALPHA_ANG, BETA_ANG
  equivalence (APARM(9),  ALPHA_ANG), &
              (APARM(10), BETA_ANG)

  !---------------------------------------------------------------------
  ! 3. Modern Allocatable System Workspace (Replaces IC(1)..IC(28) in main.f)
  !---------------------------------------------------------------------
  type :: PairMat
    logical :: populated = .false.
    real(real64), allocatable :: mat(:,:)
  end type PairMat

  type :: AtomRecord
    logical(kind=1) :: qflags(8) = .false.
    integer(int32)  :: neqat = 0, isym = 0, ixyz = 0
    real(real64)    :: xyz(3) = 0.0d0
    real(real64), allocatable :: stevec(:,:,:)
    real(real64), allocatable :: eval(:,:)
    real(real64), allocatable :: occ(:,:)
    real(real64), allocatable :: fock(:,:)
    real(real64), allocatable :: gamma(:,:)
    real(real64), allocatable :: cpot2(:)
    real(real64), allocatable :: vfit2(:)
    real(real64), allocatable :: vcoef(:)
    real(real64), allocatable :: pkpot(:,:,:)
    real(real64), allocatable :: zpot2(:)
    real(real64)    :: toten = 0.0d0, zzrep = 0.0d0, rmax = 0.0d0
  end type AtomRecord

  type :: AtomStream
    integer(int32) :: pos = 0
    integer(int32) :: mdifat_pos = 0
    integer(int32) :: mdifat_count = 0
    integer(int32), allocatable :: mdifat_arr(:)
    type(AtomRecord), allocatable :: records(:)
  end type AtomStream

  type :: ElemStream
    integer(int32) :: pos = 0
    integer(int32) :: nsto = 0, msto = 0, norb = 0, nvterm = 0
    character(len=2) :: latom = '  '
    integer(int32) :: nucz = 0
    integer(int32), allocatable :: n(:), l(:), lv(:), mv(:)
    real(real64),   allocatable :: eta(:), anorm(:), vexp(:), vexp_base(:)
    real(real64),   allocatable :: stovl(:), stkin(:), stpot(:,:), sthmat(:), ontran(:,:)
  end type ElemStream

  type :: MolecularWorkspace
    integer(int32), allocatable :: index_arr(:)      ! Formerly JCORE(J1)
    integer(int32), allocatable :: krec(:)          ! Formerly JCORE(J2)
    real(real64),   allocatable :: cgc(:)           ! Formerly CORE(IC(3))
    real(real64),   allocatable :: ylmnrm(:)        ! Formerly CORE(IC(4))
    real(real64),   allocatable :: s(:)             ! Formerly CORE(IC(5))
    real(real64),   allocatable :: w(:)             ! Formerly CORE(IC(6))
    real(real64),   allocatable :: omegai(:,:)      ! Formerly CORE(IC(7))
    real(real64),   allocatable :: omegaj(:,:)      ! Formerly CORE(IC(8))
    real(real64),   allocatable :: a(:)             ! Formerly CORE(IC(9))
    real(real64),   allocatable :: b(:)             ! Formerly CORE(IC(10))
    real(real64),   allocatable :: d(:,:)           ! Formerly CORE(IC(11))
    real(real64),   allocatable :: buffer(:)        ! Formerly CORE(IC(12))
    real(real64),   allocatable :: ovlp(:,:)        ! Formerly CORE(IC(13))
    real(real64),   allocatable :: ewmat(:,:)       ! Formerly CORE(IC(14))
    real(real64),   allocatable :: popul(:,:)       ! Formerly CORE(IC(15))
    real(real64),   allocatable :: coord(:,:)       ! Formerly CORE(IC(16))
    character(len=2), allocatable :: label(:)       ! Formerly LCORE(L17)
    real(real64),   allocatable :: charge(:,:)      ! Formerly CORE(IC(18))
    real(real64),   allocatable :: scrat(:,:)       ! Formerly CORE(IC(19))
    integer(int32), allocatable :: iatomx(:)        ! Formerly CORE(IC(20))
    integer(int32), allocatable :: nx(:)            ! Formerly CORE(J21) [FIXED]
    integer(int32), allocatable :: lx(:)            ! Formerly CORE(J22) [FIXED]
    integer(int32), allocatable :: mlx(:)           ! Formerly CORE(J23) [FIXED]
    real(real64),   allocatable :: etax(:)          ! Formerly CORE(J24) [FIXED]
    real(real64),   allocatable :: anormx(:)        ! Formerly CORE(IC(25))
    real(real64),   allocatable :: evalx(:,:)       ! Formerly CORE(IC(26))
    real(real64),   allocatable :: occx(:,:)        ! Formerly CORE(IC(27))
    real(real64),   allocatable :: evecx(:,:,:)     ! Formerly CORE(IC(28))

    ! In-memory caches replacing legacy scratch units (Phatpsy.DAUnit1 and Phatpsy.Unit2)
    type(PairMat), allocatable :: pair_stovlp(:)
    type(PairMat), allocatable :: pair_stvab(:)
    integer(int32) :: stvab_count = 0
    integer(int32) :: stvab_pos   = 0

    ! In-memory streams replacing legacy Units 3, 4, and 0 (Elem)
    type(AtomStream) :: unit3_stream
    type(AtomStream) :: unit4_stream
    type(ElemStream), allocatable :: elem_stream(:)

    ! Atomic multipole moments for anisotropic potential fitting
    real(real64), allocatable :: atom_dipoles(:,:)       ! (3, NATOM)
    real(real64), allocatable :: atom_quadrupoles(:,:,:) ! (3, 3, NATOM)

    ! Granular cycle diagnostics
    real(real64) :: last_pk_penalty   = 0.0d0
    real(real64) :: last_rms_coulomb  = 0.0d0
    real(real64) :: last_rms_exchange = 0.0d0
  end type MolecularWorkspace

  type(MolecularWorkspace), target :: mol_ws
  !f2py intent(hide) :: mol_ws
  real(real64) :: FERMI_MU(2) = 0.0d0
  integer(int32) :: EXCH_TYPE = 1
  real(real64) :: EXCH_ALPHA = 2.0d0 / 3.0d0
  integer(int32) :: CORR_TYPE = 0
  character(len=1) :: TITLE(80)

contains

  subroutine init_block_data_and_defaults()
    integer(int32) :: i, nr, nrm1
    real(real64), parameter :: EXPTF(4)  = [0.0d0, 3.519d0, 1.938d0, 0.357d0]
    real(real64), parameter :: COEFTF(4) = [1.0d0, -0.125d0, -0.35d0, -0.525d0]
    integer(int32), parameter :: LTF0(4) = [0, 0, 0, 0]

    ! Initialize former BLOCK DATA tables
    NNIA   = [19, 25, 22, 5, 35, 29, 0, 0, 0, 0]
    NID    = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0]
    NXARG  = [0, 12, 12, 3, 5, 28, 0, 0, 0, 0]
    ROUTIN = ['CONTRL .', 'ATOMIC .', 'ALOOP ..', 'EWMO ...', &
              'SCF ....', 'BLOOP ..', '        ', '        ', &
              '        ', '        ']
    NROUT  = 6
    SPIN   = ['ALPHA...', 'BETA... ']
    ZERO   = 0.0d0
    HALF   = 0.5d0
    ROOTPI = 1.7724539d0
    do i = 1, 10
      REALS(i) = real(i, real64)
      ROOT(i)  = sqrt(REALS(i))
    end do
    call GENFAC(FACT, size(FACT))
    do i = 1, 8
      HALFPI(i) = HALF * REALS(i) * 3.14159265358979323846d0
    end do

    IPTR(1) = 1
    do nr = 2, NROUT
      nrm1 = nr - 1
      IPTR(nr) = IPTR(nrm1) + 3*NNIA(nrm1) + NID(nrm1) + NXARG(nrm1)
    end do

    ! Reset parameter blocks
    APARM = 0.0d0
    IPARM = 0
    QPARM = .false.
    ICREAT = 0
    IANNIH = 0

    MXTF = 4
    TFEXP  = EXPTF
    TFCOEF = COEFTF
    LTF    = LTF0

    ! Default control parameters from main.f
    NATOM   = 1;      NELEM   = 1;      NORBT   = 5;      MSTOT   = 5
    QTRACE  = .false.; QDEBUG = .false.; QRSTRT = .false.; QBAKUP = .false.
    QCMPTD  = .false.; QMXREC = .true.;  QOPEN  = .true.;  QSEPAT = .true.
    QPRINT  = .false.; QMNPRT = .false.; QCHECK = .false.; QATOM  = .false.
    QEWMO   = .false.; QAUDMP = .false.; QMODVO = .false.; QANGST = .false.
    QGEOM   = .false.; QRERUN = .false.; QTRANS = .false.; QGSTAT = .true.
    QPLOT   = .false.; QCORE  = .false.; QC2FIT = .true.;  QX2FIT = .true.
    QPSMET  = .true.
    QCSTFT  = .false.
    QGFERM  = .false.
    QDYNEX  = .false.
    QDFTEX  = .false.
    EXCH_TYPE = 1
    EXCH_ALPHA = 2.0d0 / 3.0d0
    CORR_TYPE = 0
    FERMI_MU = 0.0d0

    mol_ws%last_pk_penalty   = 0.0d0
    mol_ws%last_rms_coulomb  = 0.0d0
    mol_ws%last_rms_exchange = 0.0d0

    do i = 1, 20
      IUNIT(i) = i
    end do
    IUNIT(9) = 21

    LENBUF = 804;    NRECS  = 200;    LENREC = 50
    NMAX   = 2;      LMAX   = 1
    MXNSTO = 3;      MXMSTO = 5;      MXNORB = 5
    MXITER = 20;     MXCYCL = 1;      TIMEND = 9.0e4_real32
    EXPMIN = -60.0d0; THETA = 1.0d5;  CONVRG = 1.0d-3
    CDAMP  = 1.0d0;  CACCEL = 1.0d0
    WGT0   = 1.0d0;  WGT1   = 1.0d0;  WGT2   = 1.0d0
    WGTC   = 1.0d0;  WGTV   = 1.0d0
    VDAMP  = 0.0d0;  VACCEL = 1.0d0
    RANGE  = 10.0d0; PKSCAL = 1.0d0;  PKE    = 0.0d0
    NCREAT = 0;      NANNIH = 0
    KATOM  = 0;      KORB   = 0;      KSP    = 0

    MXCORE = ICORE_SUB
    NWCORE = MXCORE
    if (.not. allocated(SUB_CORE)) allocate(SUB_CORE(ICORE_SUB))
  end subroutine init_block_data_and_defaults

  subroutine compute_derived_dimensions_and_allocate()
    integer(int32) :: ndxdim, llmx
    integer(int32), external :: NDXMAX

    ISPIN = merge(2, 1, logical(QOPEN))
    N2ATOM = (NATOM * (NATOM + 1)) / 2
    N2ORBT = (NORBT * (NORBT + 1)) / 2
    NNMX   = 2 * NMAX
    NNMXM1 = NNMX - 1
    NABDIM = max(128, 10 * NMAX)
    LMXP1  = LMAX + 1
    llmx   = 2 * LMAX
    LLMXP1 = llmx + 1
    LLMXP2 = LLMXP1 + 1
    L3MX   = ((llmx + 1) * (llmx + 2) * (llmx + 3)) / 6
    NYLM   = LLMXP1 * LMXP1
    NCGC   = NDXMAX(LMAX)
    ndxdim = max(MXMSTO, NATOM, NNMX)

    if (allocated(mol_ws%index_arr)) deallocate( &
      mol_ws%index_arr, mol_ws%krec, mol_ws%cgc, mol_ws%ylmnrm, &
      mol_ws%s, mol_ws%w, mol_ws%omegai, mol_ws%omegaj, &
      mol_ws%a, mol_ws%b, mol_ws%d, mol_ws%buffer, &
      mol_ws%ovlp, mol_ws%ewmat, mol_ws%popul, mol_ws%coord, &
      mol_ws%label, mol_ws%charge, mol_ws%atom_dipoles, mol_ws%atom_quadrupoles, &
      mol_ws%scrat, mol_ws%iatomx, &
      mol_ws%nx, mol_ws%lx, mol_ws%mlx, mol_ws%etax, &
      mol_ws%anormx, mol_ws%evalx, mol_ws%occx, mol_ws%evecx, &
      mol_ws%pair_stovlp, mol_ws%pair_stvab)

    allocate(mol_ws%index_arr(ndxdim))
    allocate(mol_ws%krec(N2ATOM))
    allocate(mol_ws%cgc(NCGC))
    allocate(mol_ws%ylmnrm(NYLM))
    allocate(mol_ws%s(NNMX))
    allocate(mol_ws%w(NNMXM1))
    allocate(mol_ws%omegai(NNMXM1, NNMXM1))
    allocate(mol_ws%omegaj(NNMXM1, NNMXM1))
    allocate(mol_ws%a(NABDIM))
    allocate(mol_ws%b(NABDIM))
    allocate(mol_ws%d(L3MX, 2))
    allocate(mol_ws%buffer(LENBUF))
    allocate(mol_ws%ovlp(N2ORBT, ISPIN))
    allocate(mol_ws%ewmat(N2ORBT, ISPIN))
    allocate(mol_ws%popul(NORBT, ISPIN))
    allocate(mol_ws%coord(3, NATOM))
    allocate(mol_ws%label(NATOM))
    allocate(mol_ws%charge(3, NATOM))
    allocate(mol_ws%atom_dipoles(3, NATOM))
    allocate(mol_ws%atom_quadrupoles(3, 3, NATOM))
    allocate(mol_ws%scrat(LLMXP2, 2))
    allocate(mol_ws%iatomx(MSTOT))
    allocate(mol_ws%nx(MSTOT))
    allocate(mol_ws%lx(MSTOT))
    allocate(mol_ws%mlx(MSTOT))
    allocate(mol_ws%etax(MSTOT))
    allocate(mol_ws%anormx(MSTOT))
    allocate(mol_ws%evalx(NORBT, ISPIN))
    allocate(mol_ws%occx(NORBT, ISPIN))
    allocate(mol_ws%evecx(MSTOT, NORBT, ISPIN))
    allocate(mol_ws%pair_stovlp(N2ATOM))
    allocate(mol_ws%pair_stvab(max(10, NATOM * NATOM * 2)))

    mol_ws%ovlp = 0.0d0; mol_ws%ewmat = 0.0d0; mol_ws%popul = 0.0d0
    mol_ws%coord = 0.0d0; mol_ws%charge = 0.0d0
    mol_ws%atom_dipoles = 0.0d0; mol_ws%atom_quadrupoles = 0.0d0
    mol_ws%iatomx = 0; mol_ws%nx = 0; mol_ws%lx = 0; mol_ws%mlx = 0
    mol_ws%etax = 0.0d0; mol_ws%anormx = 0.0d0
    mol_ws%evalx = 0.0d0; mol_ws%occx = 0.0d0; mol_ws%evecx = 0.0d0

    mol_ws%stvab_count = 0
    mol_ws%stvab_pos   = 0

    if (allocated(mol_ws%elem_stream)) deallocate(mol_ws%elem_stream)
    allocate(mol_ws%elem_stream(max(10, NELEM)))
    if (allocated(mol_ws%unit3_stream%records)) deallocate(mol_ws%unit3_stream%records)
    if (allocated(mol_ws%unit4_stream%records)) deallocate(mol_ws%unit4_stream%records)
    allocate(mol_ws%unit3_stream%records(max(10, NATOM * 2)))
    allocate(mol_ws%unit4_stream%records(max(10, NATOM * 2)))
    if (allocated(mol_ws%unit3_stream%mdifat_arr)) deallocate(mol_ws%unit3_stream%mdifat_arr)
    if (allocated(mol_ws%unit4_stream%mdifat_arr)) deallocate(mol_ws%unit4_stream%mdifat_arr)
    allocate(mol_ws%unit3_stream%mdifat_arr(max(10, NELEM * 4)))
    allocate(mol_ws%unit4_stream%mdifat_arr(max(10, NELEM * 4)))
    mol_ws%unit3_stream%pos = 0
    mol_ws%unit4_stream%pos = 0
    mol_ws%unit3_stream%mdifat_pos = 0
    mol_ws%unit4_stream%mdifat_pos = 0
    mol_ws%unit3_stream%mdifat_count = 0
    mol_ws%unit4_stream%mdifat_count = 0
  end subroutine compute_derived_dimensions_and_allocate

  subroutine get_evalx(out_eval, n, nspin)
    integer(int32), intent(in) :: n, nspin
    real(real64), intent(out) :: out_eval(n, nspin)
    if (allocated(mol_ws%evalx)) then
      out_eval(1:min(n, NORBT), 1:min(nspin, ISPIN)) = mol_ws%evalx(1:min(n, NORBT), 1:min(nspin, ISPIN))
    else
      out_eval = 0.0d0
    end if
  end subroutine get_evalx

  subroutine get_occx(out_occ, n, nspin)
    integer(int32), intent(in) :: n, nspin
    real(real64), intent(out) :: out_occ(n, nspin)
    if (allocated(mol_ws%occx)) then
      out_occ(1:min(n, NORBT), 1:min(nspin, ISPIN)) = mol_ws%occx(1:min(n, NORBT), 1:min(nspin, ISPIN))
    else
      out_occ = 0.0d0
    end if
  end subroutine get_occx

  subroutine get_evecx(out_evec, m, n, nspin)
    integer(int32), intent(in) :: m, n, nspin
    real(real64), intent(out) :: out_evec(m, n, nspin)
    if (allocated(mol_ws%evecx)) then
      out_evec(1:min(m, MSTOT), 1:min(n, NORBT), 1:min(nspin, ISPIN)) = &
        mol_ws%evecx(1:min(m, MSTOT), 1:min(n, NORBT), 1:min(nspin, ISPIN))
    else
      out_evec = 0.0d0
    end if
  end subroutine get_evecx

  subroutine get_ovlp(out_ovlp, n2, nspin)
    integer(int32), intent(in) :: n2, nspin
    real(real64), intent(out) :: out_ovlp(n2, nspin)
    if (allocated(mol_ws%ovlp)) then
      out_ovlp(1:min(n2, N2ORBT), 1:min(nspin, ISPIN)) = mol_ws%ovlp(1:min(n2, N2ORBT), 1:min(nspin, ISPIN))
    else
      out_ovlp = 0.0d0
    end if
  end subroutine get_ovlp

  subroutine get_ewmat(out_ewmat, n2, nspin)
    integer(int32), intent(in) :: n2, nspin
    real(real64), intent(out) :: out_ewmat(n2, nspin)
    if (allocated(mol_ws%ewmat)) then
      out_ewmat(1:min(n2, N2ORBT), 1:min(nspin, ISPIN)) = mol_ws%ewmat(1:min(n2, N2ORBT), 1:min(nspin, ISPIN))
    else
      out_ewmat = 0.0d0
    end if
  end subroutine get_ewmat

  subroutine get_coord(out_coord, n3, n)
    integer(int32), intent(in) :: n3, n
    real(real64), intent(out) :: out_coord(n3, n)
    if (allocated(mol_ws%coord)) then
      out_coord(1:min(n3, 3), 1:min(n, NATOM)) = mol_ws%coord(1:min(n3, 3), 1:min(n, NATOM))
    else
      out_coord = 0.0d0
    end if
  end subroutine get_coord

  subroutine get_charge(out_charge, n3, n)
    integer(int32), intent(in) :: n3, n
    real(real64), intent(out) :: out_charge(n3, n)
    if (allocated(mol_ws%charge)) then
      out_charge(1:min(n3, 3), 1:min(n, NATOM)) = mol_ws%charge(1:min(n3, 3), 1:min(n, NATOM))
    else
      out_charge = 0.0d0
    end if
  end subroutine get_charge

  subroutine set_charge(in_charge, n3, n)
    integer(int32), intent(in) :: n3, n
    real(real64), intent(in) :: in_charge(n3, n)
    if (allocated(mol_ws%charge)) then
      mol_ws%charge(1:min(n3, 3), 1:min(n, NATOM)) = in_charge(1:min(n3, 3), 1:min(n, NATOM))
    end if
  end subroutine set_charge

  subroutine get_atom_dipoles(out_dipoles, n3, n)
    integer(int32), intent(in) :: n3, n
    real(real64), intent(out) :: out_dipoles(n3, n)
    if (allocated(mol_ws%atom_dipoles)) then
      out_dipoles(1:min(n3, 3), 1:min(n, NATOM)) = mol_ws%atom_dipoles(1:min(n3, 3), 1:min(n, NATOM))
    else
      out_dipoles = 0.0d0
    end if
  end subroutine get_atom_dipoles

  subroutine set_atom_dipoles(in_dipoles, n3, n)
    integer(int32), intent(in) :: n3, n
    real(real64), intent(in) :: in_dipoles(n3, n)
    if (allocated(mol_ws%atom_dipoles)) then
      mol_ws%atom_dipoles(1:min(n3, 3), 1:min(n, NATOM)) = in_dipoles(1:min(n3, 3), 1:min(n, NATOM))
    end if
  end subroutine set_atom_dipoles

  subroutine get_popul(out_popul, n, nspin)
    integer(int32), intent(in) :: n, nspin
    real(real64), intent(out) :: out_popul(n, nspin)
    if (allocated(mol_ws%popul)) then
      out_popul(1:min(n, NORBT), 1:min(nspin, ISPIN)) = mol_ws%popul(1:min(n, NORBT), 1:min(nspin, ISPIN))
    else
      out_popul = 0.0d0
    end if
  end subroutine get_popul

  subroutine set_popul(in_popul, n, nspin)
    integer(int32), intent(in) :: n, nspin
    real(real64), intent(in) :: in_popul(n, nspin)
    if (allocated(mol_ws%popul)) then
      mol_ws%popul(1:min(n, NORBT), 1:min(nspin, ISPIN)) = in_popul(1:min(n, NORBT), 1:min(nspin, ISPIN))
    end if
  end subroutine set_popul

  subroutine get_iatomx(out_iatom, m)
    integer(int32), intent(in) :: m
    integer(int32), intent(out) :: out_iatom(m)
    out_iatom = 0
    if (allocated(mol_ws%iatomx)) then
      out_iatom(1:min(m, MSTOT)) = mol_ws%iatomx(1:min(m, MSTOT))
    end if
  end subroutine get_iatomx

  subroutine get_nx(out_n, m)
    integer(int32), intent(in) :: m
    integer(int32), intent(out) :: out_n(m)
    out_n = 0
    if (allocated(mol_ws%nx)) then
      out_n(1:min(m, MSTOT)) = mol_ws%nx(1:min(m, MSTOT))
    end if
  end subroutine get_nx

  subroutine get_lx(out_l, m)
    integer(int32), intent(in) :: m
    integer(int32), intent(out) :: out_l(m)
    out_l = 0
    if (allocated(mol_ws%lx)) then
      out_l(1:min(m, MSTOT)) = mol_ws%lx(1:min(m, MSTOT))
    end if
  end subroutine get_lx

  subroutine get_mlx(out_ml, m)
    integer(int32), intent(in) :: m
    integer(int32), intent(out) :: out_ml(m)
    out_ml = 0
    if (allocated(mol_ws%mlx)) then
      out_ml(1:min(m, MSTOT)) = mol_ws%mlx(1:min(m, MSTOT))
    end if
  end subroutine get_mlx

  subroutine get_etax(out_eta, m)
    integer(int32), intent(in) :: m
    real(real64), intent(out) :: out_eta(m)
    out_eta = 0.0d0
    if (allocated(mol_ws%etax)) then
      out_eta(1:min(m, MSTOT)) = mol_ws%etax(1:min(m, MSTOT))
    end if
  end subroutine get_etax

  subroutine get_anormx(out_anorm, m)
    integer(int32), intent(in) :: m
    real(real64), intent(out) :: out_anorm(m)
    out_anorm = 0.0d0
    if (allocated(mol_ws%anormx)) then
      out_anorm(1:min(m, MSTOT)) = mol_ws%anormx(1:min(m, MSTOT))
    end if
  end subroutine get_anormx

  subroutine reset_cycle_diagnostics()
    mol_ws%last_pk_penalty   = 0.0d0
    mol_ws%last_rms_coulomb  = 0.0d0
    mol_ws%last_rms_exchange = 0.0d0
  end subroutine reset_cycle_diagnostics

  subroutine get_cycle_diagnostics(pk_penalty, rms_coulomb, rms_exchange, charge_transfer)
    real(real64), intent(out) :: pk_penalty, rms_coulomb, rms_exchange, charge_transfer
    pk_penalty      = mol_ws%last_pk_penalty
    rms_coulomb     = mol_ws%last_rms_coulomb
    rms_exchange    = mol_ws%last_rms_exchange
    charge_transfer = CHTRAN
  end subroutine get_cycle_diagnostics

  !---------------------------------------------------------------------
  ! In-Memory Atomic Stream Handlers (replacing Units 3 and 4)
  !---------------------------------------------------------------------
  subroutine rewind_atom_stream(unit_id)
    integer(int32), intent(in) :: unit_id
    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%pos = 0
      mol_ws%unit3_stream%mdifat_pos = 0
    end if
    if (unit_id == IUNIT4) then
      mol_ws%unit4_stream%pos = 0
      mol_ws%unit4_stream%mdifat_pos = 0
    end if
  end subroutine rewind_atom_stream

  subroutine write_atom_stream_mdifat(unit_id, mdifat)
    integer(int32), intent(in) :: unit_id, mdifat
    if (unit_id == IUNIT3) then
      if (mol_ws%unit3_stream%pos == 0) then
        mol_ws%unit3_stream%mdifat_count = 0
        mol_ws%unit3_stream%mdifat_pos = 0
      end if
      mol_ws%unit3_stream%mdifat_count = mol_ws%unit3_stream%mdifat_count + 1
      if (mol_ws%unit3_stream%mdifat_count > size(mol_ws%unit3_stream%mdifat_arr)) then
        block
          integer(int32), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit3_stream%mdifat_arr)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit3_stream%mdifat_arr
          call move_alloc(tmp, mol_ws%unit3_stream%mdifat_arr)
        end block
      end if
      mol_ws%unit3_stream%mdifat_arr(mol_ws%unit3_stream%mdifat_count) = mdifat
    end if
    if (unit_id == IUNIT4) then
      if (mol_ws%unit4_stream%pos == 0) then
        mol_ws%unit4_stream%mdifat_count = 0
        mol_ws%unit4_stream%mdifat_pos = 0
      end if
      mol_ws%unit4_stream%mdifat_count = mol_ws%unit4_stream%mdifat_count + 1
      if (mol_ws%unit4_stream%mdifat_count > size(mol_ws%unit4_stream%mdifat_arr)) then
        block
          integer(int32), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit4_stream%mdifat_arr)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit4_stream%mdifat_arr
          call move_alloc(tmp, mol_ws%unit4_stream%mdifat_arr)
        end block
      end if
      mol_ws%unit4_stream%mdifat_arr(mol_ws%unit4_stream%mdifat_count) = mdifat
    end if
  end subroutine write_atom_stream_mdifat

  subroutine read_atom_stream_mdifat(unit_id, mdifat)
    integer(int32), intent(in) :: unit_id
    integer(int32), intent(out) :: mdifat
    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%mdifat_pos = mol_ws%unit3_stream%mdifat_pos + 1
      if (mol_ws%unit3_stream%mdifat_pos <= mol_ws%unit3_stream%mdifat_count) then
        mdifat = mol_ws%unit3_stream%mdifat_arr(mol_ws%unit3_stream%mdifat_pos)
      else
        mdifat = 0
      end if
    end if
    if (unit_id == IUNIT4) then
      mol_ws%unit4_stream%mdifat_pos = mol_ws%unit4_stream%mdifat_pos + 1
      if (mol_ws%unit4_stream%mdifat_pos <= mol_ws%unit4_stream%mdifat_count) then
        mdifat = mol_ws%unit4_stream%mdifat_arr(mol_ws%unit4_stream%mdifat_pos)
      else
        mdifat = 0
      end if
    end if
  end subroutine read_atom_stream_mdifat

  subroutine write_atom_record_ref(unit_id, qflags, neqat, x, y, z, stevec, eval, occ, fock, &
                                   gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmax)
    integer(int32), intent(in) :: unit_id, neqat
    logical(kind=1), intent(in) :: qflags(8)
    real(real64), intent(in) :: x, y, z, toten, zzrep, rmax
    real(real64), intent(in) :: stevec(:,:,:), eval(:,:), occ(:,:), fock(:,:), gamma(:,:)
    real(real64), intent(in) :: cpot2(:), vfit2(:), vcoef(:), pkpot(:,:,:), zpot2(:)

    integer(int32) :: idx

    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%pos = mol_ws%unit3_stream%pos + 1
      idx = mol_ws%unit3_stream%pos
      if (idx > size(mol_ws%unit3_stream%records)) then
        block
          type(AtomRecord), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit3_stream%records)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit3_stream%records
          call move_alloc(tmp, mol_ws%unit3_stream%records)
        end block
      end if
      mol_ws%unit3_stream%records(idx)%qflags = qflags
      mol_ws%unit3_stream%records(idx)%neqat  = neqat
      mol_ws%unit3_stream%records(idx)%isym   = 0
      mol_ws%unit3_stream%records(idx)%ixyz   = 0
      mol_ws%unit3_stream%records(idx)%xyz    = [x, y, z]
      mol_ws%unit3_stream%records(idx)%stevec = stevec
      mol_ws%unit3_stream%records(idx)%eval   = eval
      mol_ws%unit3_stream%records(idx)%occ    = occ
      mol_ws%unit3_stream%records(idx)%fock   = fock
      mol_ws%unit3_stream%records(idx)%gamma  = gamma
      mol_ws%unit3_stream%records(idx)%cpot2  = cpot2
      mol_ws%unit3_stream%records(idx)%vfit2  = vfit2
      mol_ws%unit3_stream%records(idx)%vcoef  = vcoef
      mol_ws%unit3_stream%records(idx)%pkpot  = pkpot
      mol_ws%unit3_stream%records(idx)%zpot2  = zpot2
      mol_ws%unit3_stream%records(idx)%toten  = toten
      mol_ws%unit3_stream%records(idx)%zzrep  = zzrep
      mol_ws%unit3_stream%records(idx)%rmax   = rmax
    else
      mol_ws%unit4_stream%pos = mol_ws%unit4_stream%pos + 1
      idx = mol_ws%unit4_stream%pos
      if (idx > size(mol_ws%unit4_stream%records)) then
        block
          type(AtomRecord), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit4_stream%records)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit4_stream%records
          call move_alloc(tmp, mol_ws%unit4_stream%records)
        end block
      end if
      mol_ws%unit4_stream%records(idx)%qflags = qflags
      mol_ws%unit4_stream%records(idx)%neqat  = neqat
      mol_ws%unit4_stream%records(idx)%isym   = 0
      mol_ws%unit4_stream%records(idx)%ixyz   = 0
      mol_ws%unit4_stream%records(idx)%xyz    = [x, y, z]
      mol_ws%unit4_stream%records(idx)%stevec = stevec
      mol_ws%unit4_stream%records(idx)%eval   = eval
      mol_ws%unit4_stream%records(idx)%occ    = occ
      mol_ws%unit4_stream%records(idx)%fock   = fock
      mol_ws%unit4_stream%records(idx)%gamma  = gamma
      mol_ws%unit4_stream%records(idx)%cpot2  = cpot2
      mol_ws%unit4_stream%records(idx)%vfit2  = vfit2
      mol_ws%unit4_stream%records(idx)%vcoef  = vcoef
      mol_ws%unit4_stream%records(idx)%pkpot  = pkpot
      mol_ws%unit4_stream%records(idx)%zpot2  = zpot2
      mol_ws%unit4_stream%records(idx)%toten  = toten
      mol_ws%unit4_stream%records(idx)%zzrep  = zzrep
      mol_ws%unit4_stream%records(idx)%rmax   = rmax
    end if
  end subroutine write_atom_record_ref

  subroutine write_atom_record_equiv(unit_id, qflags, isym, ixyz, x, y, z, stevec, eval, occ, &
                                     fock, gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmax)
    integer(int32), intent(in) :: unit_id, isym, ixyz
    logical(kind=1), intent(in) :: qflags(8)
    real(real64), intent(in) :: x, y, z, toten, zzrep, rmax
    real(real64), intent(in) :: stevec(:,:,:), eval(:,:), occ(:,:), fock(:,:), gamma(:,:)
    real(real64), intent(in) :: cpot2(:), vfit2(:), vcoef(:), pkpot(:,:,:), zpot2(:)

    integer(int32) :: idx

    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%pos = mol_ws%unit3_stream%pos + 1
      idx = mol_ws%unit3_stream%pos
      if (idx > size(mol_ws%unit3_stream%records)) then
        block
          type(AtomRecord), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit3_stream%records)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit3_stream%records
          call move_alloc(tmp, mol_ws%unit3_stream%records)
        end block
      end if
      mol_ws%unit3_stream%records(idx)%qflags = qflags
      mol_ws%unit3_stream%records(idx)%neqat  = 0
      mol_ws%unit3_stream%records(idx)%isym   = isym
      mol_ws%unit3_stream%records(idx)%ixyz   = ixyz
      mol_ws%unit3_stream%records(idx)%xyz    = [x, y, z]
      mol_ws%unit3_stream%records(idx)%stevec = stevec
      mol_ws%unit3_stream%records(idx)%eval   = eval
      mol_ws%unit3_stream%records(idx)%occ    = occ
      mol_ws%unit3_stream%records(idx)%fock   = fock
      mol_ws%unit3_stream%records(idx)%gamma  = gamma
      mol_ws%unit3_stream%records(idx)%cpot2  = cpot2
      mol_ws%unit3_stream%records(idx)%vfit2  = vfit2
      mol_ws%unit3_stream%records(idx)%vcoef  = vcoef
      mol_ws%unit3_stream%records(idx)%pkpot  = pkpot
      mol_ws%unit3_stream%records(idx)%zpot2  = zpot2
      mol_ws%unit3_stream%records(idx)%toten  = toten
      mol_ws%unit3_stream%records(idx)%zzrep  = zzrep
      mol_ws%unit3_stream%records(idx)%rmax   = rmax
    else
      mol_ws%unit4_stream%pos = mol_ws%unit4_stream%pos + 1
      idx = mol_ws%unit4_stream%pos
      if (idx > size(mol_ws%unit4_stream%records)) then
        block
          type(AtomRecord), allocatable :: tmp(:)
          integer(int32) :: old_sz
          old_sz = size(mol_ws%unit4_stream%records)
          allocate(tmp(old_sz * 2))
          tmp(1:old_sz) = mol_ws%unit4_stream%records
          call move_alloc(tmp, mol_ws%unit4_stream%records)
        end block
      end if
      mol_ws%unit4_stream%records(idx)%qflags = qflags
      mol_ws%unit4_stream%records(idx)%neqat  = 0
      mol_ws%unit4_stream%records(idx)%isym   = isym
      mol_ws%unit4_stream%records(idx)%ixyz   = ixyz
      mol_ws%unit4_stream%records(idx)%xyz    = [x, y, z]
      mol_ws%unit4_stream%records(idx)%stevec = stevec
      mol_ws%unit4_stream%records(idx)%eval   = eval
      mol_ws%unit4_stream%records(idx)%occ    = occ
      mol_ws%unit4_stream%records(idx)%fock   = fock
      mol_ws%unit4_stream%records(idx)%gamma  = gamma
      mol_ws%unit4_stream%records(idx)%cpot2  = cpot2
      mol_ws%unit4_stream%records(idx)%vfit2  = vfit2
      mol_ws%unit4_stream%records(idx)%vcoef  = vcoef
      mol_ws%unit4_stream%records(idx)%pkpot  = pkpot
      mol_ws%unit4_stream%records(idx)%zpot2  = zpot2
      mol_ws%unit4_stream%records(idx)%toten  = toten
      mol_ws%unit4_stream%records(idx)%zzrep  = zzrep
      mol_ws%unit4_stream%records(idx)%rmax   = rmax
    end if
  end subroutine write_atom_record_equiv

  subroutine read_atom_record_ref(unit_id, qflags, neqat, x, y, z, stevec, eval, occ, fock, &
                                  gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmax)
    integer(int32), intent(in) :: unit_id
    integer(int32), intent(out) :: neqat
    logical(kind=1), intent(out) :: qflags(8)
    real(real64), intent(out) :: x, y, z, toten, zzrep, rmax
    real(real64), intent(out) :: stevec(:,:,:), eval(:,:), occ(:,:), fock(:,:), gamma(:,:)
    real(real64), intent(out) :: cpot2(:), vfit2(:), vcoef(:), pkpot(:,:,:), zpot2(:)

    integer(int32) :: idx

    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%pos = mol_ws%unit3_stream%pos + 1
      idx = mol_ws%unit3_stream%pos
      qflags = mol_ws%unit3_stream%records(idx)%qflags
      neqat  = mol_ws%unit3_stream%records(idx)%neqat
      x      = mol_ws%unit3_stream%records(idx)%xyz(1)
      y      = mol_ws%unit3_stream%records(idx)%xyz(2)
      z      = mol_ws%unit3_stream%records(idx)%xyz(3)
      stevec = mol_ws%unit3_stream%records(idx)%stevec
      eval   = mol_ws%unit3_stream%records(idx)%eval
      occ    = mol_ws%unit3_stream%records(idx)%occ
      fock   = mol_ws%unit3_stream%records(idx)%fock
      gamma  = mol_ws%unit3_stream%records(idx)%gamma
      cpot2  = mol_ws%unit3_stream%records(idx)%cpot2
      vfit2  = mol_ws%unit3_stream%records(idx)%vfit2
      vcoef  = mol_ws%unit3_stream%records(idx)%vcoef
      pkpot  = mol_ws%unit3_stream%records(idx)%pkpot
      zpot2  = mol_ws%unit3_stream%records(idx)%zpot2
      toten  = mol_ws%unit3_stream%records(idx)%toten
      zzrep  = mol_ws%unit3_stream%records(idx)%zzrep
      rmax   = mol_ws%unit3_stream%records(idx)%rmax
    else
      mol_ws%unit4_stream%pos = mol_ws%unit4_stream%pos + 1
      idx = mol_ws%unit4_stream%pos
      qflags = mol_ws%unit4_stream%records(idx)%qflags
      neqat  = mol_ws%unit4_stream%records(idx)%neqat
      x      = mol_ws%unit4_stream%records(idx)%xyz(1)
      y      = mol_ws%unit4_stream%records(idx)%xyz(2)
      z      = mol_ws%unit4_stream%records(idx)%xyz(3)
      stevec = mol_ws%unit4_stream%records(idx)%stevec
      eval   = mol_ws%unit4_stream%records(idx)%eval
      occ    = mol_ws%unit4_stream%records(idx)%occ
      fock   = mol_ws%unit4_stream%records(idx)%fock
      gamma  = mol_ws%unit4_stream%records(idx)%gamma
      cpot2  = mol_ws%unit4_stream%records(idx)%cpot2
      vfit2  = mol_ws%unit4_stream%records(idx)%vfit2
      vcoef  = mol_ws%unit4_stream%records(idx)%vcoef
      pkpot  = mol_ws%unit4_stream%records(idx)%pkpot
      zpot2  = mol_ws%unit4_stream%records(idx)%zpot2
      toten  = mol_ws%unit4_stream%records(idx)%toten
      zzrep  = mol_ws%unit4_stream%records(idx)%zzrep
      rmax   = mol_ws%unit4_stream%records(idx)%rmax
    end if
  end subroutine read_atom_record_ref

  subroutine read_atom_record_equiv(unit_id, qflags, isym, ixyz, x, y, z, stevec, eval, occ, &
                                    fock, gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmax)
    integer(int32), intent(in) :: unit_id
    integer(int32), intent(out) :: isym, ixyz
    logical(kind=1), intent(out) :: qflags(8)
    real(real64), intent(out) :: x, y, z, toten, zzrep, rmax
    real(real64), intent(out) :: stevec(:,:,:), eval(:,:), occ(:,:), fock(:,:), gamma(:,:)
    real(real64), intent(out) :: cpot2(:), vfit2(:), vcoef(:), pkpot(:,:,:), zpot2(:)

    integer(int32) :: idx

    if (unit_id == IUNIT3) then
      mol_ws%unit3_stream%pos = mol_ws%unit3_stream%pos + 1
      idx = mol_ws%unit3_stream%pos
      qflags = mol_ws%unit3_stream%records(idx)%qflags
      isym   = mol_ws%unit3_stream%records(idx)%isym
      ixyz   = mol_ws%unit3_stream%records(idx)%ixyz
      x      = mol_ws%unit3_stream%records(idx)%xyz(1)
      y      = mol_ws%unit3_stream%records(idx)%xyz(2)
      z      = mol_ws%unit3_stream%records(idx)%xyz(3)
      stevec = mol_ws%unit3_stream%records(idx)%stevec
      eval   = mol_ws%unit3_stream%records(idx)%eval
      occ    = mol_ws%unit3_stream%records(idx)%occ
      fock   = mol_ws%unit3_stream%records(idx)%fock
      gamma  = mol_ws%unit3_stream%records(idx)%gamma
      cpot2  = mol_ws%unit3_stream%records(idx)%cpot2
      vfit2  = mol_ws%unit3_stream%records(idx)%vfit2
      vcoef  = mol_ws%unit3_stream%records(idx)%vcoef
      pkpot  = mol_ws%unit3_stream%records(idx)%pkpot
      zpot2  = mol_ws%unit3_stream%records(idx)%zpot2
      toten  = mol_ws%unit3_stream%records(idx)%toten
      zzrep  = mol_ws%unit3_stream%records(idx)%zzrep
      rmax   = mol_ws%unit3_stream%records(idx)%rmax
    else
      mol_ws%unit4_stream%pos = mol_ws%unit4_stream%pos + 1
      idx = mol_ws%unit4_stream%pos
      qflags = mol_ws%unit4_stream%records(idx)%qflags
      isym   = mol_ws%unit4_stream%records(idx)%isym
      ixyz   = mol_ws%unit4_stream%records(idx)%ixyz
      x      = mol_ws%unit4_stream%records(idx)%xyz(1)
      y      = mol_ws%unit4_stream%records(idx)%xyz(2)
      z      = mol_ws%unit4_stream%records(idx)%xyz(3)
      stevec = mol_ws%unit4_stream%records(idx)%stevec
      eval   = mol_ws%unit4_stream%records(idx)%eval
      occ    = mol_ws%unit4_stream%records(idx)%occ
      fock   = mol_ws%unit4_stream%records(idx)%fock
      gamma  = mol_ws%unit4_stream%records(idx)%gamma
      cpot2  = mol_ws%unit4_stream%records(idx)%cpot2
      vfit2  = mol_ws%unit4_stream%records(idx)%vfit2
      vcoef  = mol_ws%unit4_stream%records(idx)%vcoef
      pkpot  = mol_ws%unit4_stream%records(idx)%pkpot
      zpot2  = mol_ws%unit4_stream%records(idx)%zpot2
      toten  = mol_ws%unit4_stream%records(idx)%toten
      zzrep  = mol_ws%unit4_stream%records(idx)%zzrep
      rmax   = mol_ws%unit4_stream%records(idx)%rmax
    end if
  end subroutine read_atom_record_equiv

  !---------------------------------------------------------------------
  ! In-Memory Element Stream Handlers (replacing Phatpsy.Elem files)
  !---------------------------------------------------------------------
  subroutine rewind_elem_stream(ielem)
    integer(int32), intent(in) :: ielem
    if (allocated(mol_ws%elem_stream)) then
      if (ielem >= 1 .and. ielem <= size(mol_ws%elem_stream)) then
        mol_ws%elem_stream(ielem)%pos = 0
      end if
    end if
  end subroutine rewind_elem_stream

  subroutine ensure_elem_stream_capacity(ielem)
    integer(int32), intent(in) :: ielem
    if (.not. allocated(mol_ws%elem_stream)) then
      allocate(mol_ws%elem_stream(max(10, ielem)))
    else if (ielem > size(mol_ws%elem_stream)) then
      block
        type(ElemStream), allocatable :: tmp(:)
        integer(int32) :: old_sz
        old_sz = size(mol_ws%elem_stream)
        allocate(tmp(max(old_sz * 2, ielem)))
        tmp(1:old_sz) = mol_ws%elem_stream
        call move_alloc(tmp, mol_ws%elem_stream)
      end block
    end if
  end subroutine ensure_elem_stream_capacity

  subroutine write_elem_dims(ielem, nsto, msto, norb, nvterm)
    integer(int32), intent(in) :: ielem, nsto, msto, norb, nvterm
    call ensure_elem_stream_capacity(ielem)
    mol_ws%elem_stream(ielem)%nsto   = nsto
    mol_ws%elem_stream(ielem)%msto   = msto
    mol_ws%elem_stream(ielem)%norb   = norb
    mol_ws%elem_stream(ielem)%nvterm = nvterm
  end subroutine write_elem_dims

  subroutine read_elem_dims(ielem, nsto, msto, norb, nvterm)
    integer(int32), intent(in) :: ielem
    integer(int32), intent(out) :: nsto, msto, norb, nvterm
    nsto   = mol_ws%elem_stream(ielem)%nsto
    msto   = mol_ws%elem_stream(ielem)%msto
    norb   = mol_ws%elem_stream(ielem)%norb
    nvterm = mol_ws%elem_stream(ielem)%nvterm
  end subroutine read_elem_dims

  subroutine write_elem_basis(ielem, latom, nucz, n, l, eta, anorm)
    integer(int32), intent(in) :: ielem, nucz
    character(len=2), intent(in) :: latom
    integer(int32), intent(in) :: n(:), l(:)
    real(real64), intent(in) :: eta(:), anorm(:)

    mol_ws%elem_stream(ielem)%latom = latom
    mol_ws%elem_stream(ielem)%nucz  = nucz
    mol_ws%elem_stream(ielem)%n     = n
    mol_ws%elem_stream(ielem)%l     = l
    mol_ws%elem_stream(ielem)%eta   = eta
    mol_ws%elem_stream(ielem)%anorm = anorm
  end subroutine write_elem_basis

  subroutine read_elem_basis(ielem, latom, nucz, n, l, eta, anorm)
    integer(int32), intent(in) :: ielem
    character(len=2), intent(out) :: latom
    integer(int32), intent(out) :: nucz
    integer(int32), intent(out) :: n(:), l(:)
    real(real64), intent(out) :: eta(:), anorm(:)

    latom = mol_ws%elem_stream(ielem)%latom
    nucz  = mol_ws%elem_stream(ielem)%nucz
    n     = mol_ws%elem_stream(ielem)%n
    l     = mol_ws%elem_stream(ielem)%l
    eta   = mol_ws%elem_stream(ielem)%eta
    anorm = mol_ws%elem_stream(ielem)%anorm
  end subroutine read_elem_basis

  subroutine write_elem_potentials(ielem, stovl, stkin, stpot, lv, mv, vexp)
    integer(int32), intent(in) :: ielem
    integer(int32), intent(in) :: lv(:), mv(:)
    real(real64), intent(in) :: stovl(:), stkin(:), stpot(:,:), vexp(:)

    mol_ws%elem_stream(ielem)%stovl = stovl
    mol_ws%elem_stream(ielem)%stkin = stkin
    mol_ws%elem_stream(ielem)%stpot = stpot
    mol_ws%elem_stream(ielem)%lv    = lv
    mol_ws%elem_stream(ielem)%mv    = mv
    mol_ws%elem_stream(ielem)%vexp  = vexp
    if (.not. allocated(mol_ws%elem_stream(ielem)%vexp_base)) then
      mol_ws%elem_stream(ielem)%vexp_base = vexp
    end if
  end subroutine write_elem_potentials

  subroutine read_elem_potentials(ielem, stovl, stkin, stpot, lv, mv, vexp)
    integer(int32), intent(in) :: ielem
    integer(int32), intent(out) :: lv(:), mv(:)
    real(real64), intent(out) :: stovl(:), stkin(:), stpot(:,:), vexp(:)

    stovl = mol_ws%elem_stream(ielem)%stovl
    stkin = mol_ws%elem_stream(ielem)%stkin
    stpot = mol_ws%elem_stream(ielem)%stpot
    lv    = mol_ws%elem_stream(ielem)%lv
    mv    = mol_ws%elem_stream(ielem)%mv
    vexp  = mol_ws%elem_stream(ielem)%vexp
  end subroutine read_elem_potentials

  subroutine write_elem_core_matrices(ielem, sthmat, ontran)
    integer(int32), intent(in) :: ielem
    real(real64), intent(in) :: sthmat(:), ontran(:,:)

    mol_ws%elem_stream(ielem)%sthmat = sthmat
    mol_ws%elem_stream(ielem)%ontran = ontran
  end subroutine write_elem_core_matrices

  subroutine read_elem_core_matrices(ielem, sthmat, ontran)
    integer(int32), intent(in) :: ielem
    real(real64), intent(out) :: sthmat(:), ontran(:,:)

    sthmat = mol_ws%elem_stream(ielem)%sthmat
    ontran = mol_ws%elem_stream(ielem)%ontran
  end subroutine read_elem_core_matrices

end module state