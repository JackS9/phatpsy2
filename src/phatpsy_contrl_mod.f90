module phatpsy_contrl_mod
  use iso_fortran_env, only: real64, real32, int32
  use phatpsy_state_mod
  use phatpsy_atomic_mod, only: run_atomic_element
  use phatpsy_aloop_mod,  only: run_aloop_element
  use phatpsy_scf_mod,    only: run_scf_element
  use phatpsy_ewmo_mod,   only: run_ewmo
  implicit none
  public

  ! Module-level cycle state variables (formerly local to CONTRL)
  integer(int32)  :: IWSAV, NRATOM, NREMCY
  real(real64)    :: OLDTE, CTWOD, EXTE1, EXTE2, TEDIFF
  real(real32)    :: TIMREM, OLDTIM = 0.0_real32, CYCTIM
  logical(kind=1) :: QREALY = .true., QPRNT, QSTALL, QEWMOS
  real(real64), parameter :: CDMP90 = 0.9d0

contains

  !---------------------------------------------------------------------
  ! Phase 1: Initialize mathematical tables (Factorials, Binomials, CGC)
  !---------------------------------------------------------------------
  subroutine contrl_init_tables()
    integer(int32) :: i, ndxdim, mxfact, nbinom
    real(real64)   :: pi

    IWSAV = IW
    if (QMNPRT) IW = IUDUMP
    if (QDEBUG) call ARRMAP(1)
    ORBOCC = 3 - ISPIN

    ZERO = 0.0d0
    HALF = 0.5d0
    do i = 1, 10
      REALS(i) = real(i, real64)
      ROOT(i)  = sqrt(REALS(i))
    end do

    pi = atan2(REALS(1), ZERO) * REALS(2)
    ROOTPI = sqrt(pi)
    do i = 1, 8
      HALFPI(i) = HALF * REALS(i) * pi
    end do

    ndxdim = max(MXMSTO, NATOM, NNMX)
    do i = 1, ndxdim
      mol_ws%index_arr(i) = (i * (i - 1)) / 2
    end do

    mxfact = max(4 * LMAX + 1, NNMX)
    NFACT  = mxfact + 1
    call GENFAC(FACT, size(FACT))
    nbinom = mol_ws%index_arr(NNMX)
    call GENBC(BINOM, nbinom, NNMXM1, mol_ws%index_arr)
    call GENCGC(mol_ws%cgc, NCGC, mol_ws%scrat, LLMXP2, &
                mol_ws%ylmnrm, NYLM, FACT, NFACT, QREALY)
  end subroutine contrl_init_tables

  !---------------------------------------------------------------------
  ! Phase 2: Read &BASDIM for each element, run ATOMIC, and open DAUnit1
  !---------------------------------------------------------------------
  subroutine contrl_init_elements(eof_reached)
    logical, intent(out) :: eof_reached
    logical        :: eof_elem, isOpen
    integer(int32) :: isp, ios, blkfac, nprecs
    real(real64)   :: x0, y0, z0
    real(real64), external :: ASUM
    namelist /BASDIM/ QNWBAS, NSTO, MSTO, NORB, NVTERM

    ! Scoped allocatable arrays replacing IC(1)..IC(26) for ATOMIC
    integer(int32), allocatable :: at_n(:), at_l(:), at_ml(:)
    real(real64),   allocatable :: at_4(:), at_5(:), at_6(:), at_7(:), &
      at_8(:), at_9(:), at_10(:), at_11(:), at_12(:), at_13(:),        &
      at_14(:), at_15(:), at_16(:), at_17(:), at_18(:), at_19(:),      &
      at_20(:), at_21(:), at_22(:), at_23(:), at_24(:), at_25(:),      &
      at_tail(:)

    eof_reached = .false.
    mol_ws%evalx = 0.0d0
    mol_ws%occx  = 0.0d0
    mol_ws%evecx = 0.0d0

    IELEM  = 0; IATOM  = 0; IORBT  = 0; JSTOT  = 0
    NRATOM = 0; NCYCL  = 0; TMOLE  = ZERO
    QFIRST = (MXCYCL /= 0)
    QLAST  = (MXCYCL <= 1)

    elem_loop: do
      QNWBAS = (.not. QRSTRT)
      NSTO   = MXNSTO
      MSTO   = MXMSTO
      NORB   = MXNORB
      NVTERM = merge(1, MXTF, logical(QATOM))

      read(IR, nml=BASDIM, iostat=ios)
      if (ios < 0) then
        eof_reached = .true.
        return
      end if

      IELEM = IELEM + 1
      write(IW, 5000) IELEM
      if (QDEBUG) write(IUDUMP, 5000) IELEM
      if (IELEM > NELEM) call BOMB(7)
      IUNIT0 = IUNIT(IELEM + 10)

      if (.not. QNWBAS) write(IW, 6000) IUNIT0
      if (QNWBAS) then
        call write_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
      else
        call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
      end if
      write(IW, 7000) NSTO, MSTO
      write(IW, 8000) NORB
      if (MSTO > MXMSTO) call BOMB(5)

      M2STO = mol_ws%index_arr(MSTO) + MSTO
      M4STO = (M2STO * (M2STO + 1)) / 2
      N2ORB = mol_ws%index_arr(NORB) + NORB
      if (NVTERM > 1) write(IW, 9000) NVTERM
      write(IW, 10000) 4
      call run_atomic_element(NRATOM, QREALY, eof_elem)
      call rewind_elem_stream(IELEM)

      if (QATOM .or. IELEM >= NELEM) exit elem_loop
    end do elem_loop

    write(IW, 10200)
    do JSTOT = 1, MSTOT
      write(IW, 10300) mol_ws%iatomx(JSTOT), mol_ws%nx(JSTOT), &
                       mol_ws%lx(JSTOT), mol_ws%mlx(JSTOT), mol_ws%etax(JSTOT)
    end do
    write(IW, 10400)
    do isp = 1, ISPIN
      if (QOPEN) write(IW, 16000) SPIN(isp)
      call OUTVEC(mol_ws%evecx(1, 1, isp), mol_ws%evalx(1, isp), MSTOT, NORBT, IW)
    end do

    ELSUM = mol_ws%charge(1, 1)
    ELDIF = mol_ws%charge(2, 1)
    ELECS(ISPIN) = ZERO
    ELECS(1)     = HALF * (ELSUM + ELDIF)
    ELECS(ISPIN) = ELECS(ISPIN) + HALF * (ELSUM - ELDIF)

    if (.not. QATOM) then
      write(IW, 10500) RANGE, RANGE
      x0 = ASUM(mol_ws%coord(1, 1), NATOM, 3) / NATOM
      y0 = ASUM(mol_ws%coord(2, 1), NATOM, 3) / NATOM
      z0 = ASUM(mol_ws%coord(3, 1), NATOM, 3) / NATOM
      call XYZMAP(mol_ws%coord, mol_ws%label, NATOM, IW, RANGE, x0, y0, z0)
    end if

    write(IW, 11000)
    call rewind_atom_stream(IUNIT3)
    if (.not. QATOM) call rewind_atom_stream(IUNIT4)
    write(IW, 12000) IUNIT3
    call TIMOUT(-2)
    NELEM = IELEM
    NATOM = IATOM
    NORBT = IORBT

    if (.not. QATOM) then
      if (QMXREC) then
        blkfac = 1
        NLRECS = blkfac * (NRATOM * (2 * NATOM - NRATOM - 1)) / 2
        LENREC = 8 * MXMSTO * MXMSTO / blkfac
        nprecs = (N2ATOM + 2) / LENREC + 1
        NRECS  = nprecs + NLRECS
        NBYTES = NRECS * LENREC
      end if

      write(IW, 13000) NRECS, LENREC, IUNIT1

      OLDTE  = TMOLE
      TOTKE  = -TMOLE
      QPRNT  = QPRINT
      if (QSEPAT) QPRINT = .true.
      if (.not. QAUDMP) then
        QSTALL = ((CDAMP >= CDMP90) .and. (CDAMP < REALS(1)))
        if (QSTALL) CDAMP = REALS(1)
      end if
      NEXREC = 2
      QFIRST = (MXCYCL /= 0)
      QLAST  = (MXCYCL <= 1)
    end if
5000  format(//' === ELEMENT',i3,' ==='//)
6000  format(' THE BASIS FOR THIS ELEMENT AND THE ONE-CENTER INTEGRALS', &
             ' WILL BE READ FROM UNIT',i3,'.')
7000  format(' THE BASIS CONTAINS',i3,' STO''S (',i3,' COUNTING', &
             ' ML-VALUES).')
8000  format(' THIS ELEMENT IS DESCRIBED BY',i3,' ATOMIC ORBITALS.')
9000  format(' THE MODEL POTENTIAL CONTAINS',i3,' TERMS.')
10000 format(' ...',i6,'K BYTES OF CORE TO BE ALLOCATED FOR ATOMIC ...')
10200 format(//' COMPLETE MOLECULAR BASIS ...'// &
             ' ATOM  N  L  ML       ETA'/ &
             ' ----  -  -  --       ---')
10300 format(i4,1x,i3,i3,i4,2x,f10.5)
10400 format(//'ATOMIC ORBITALS OVER MOLECULAR BASIS...')
10500 format(//' COORDINATE PROJECTIONS...'/' (EACH FRAME IS ', &
             f4.1,' BY ',f4.1,')'//)
11000 format(//' ... ALL INPUT IS COMPLETE ...')
12000 format(' ... WRITES TO UNIT',i3,' ARE COMPLETE ...')
13000 format(' ...',i4,' DIRECT-ACCESS RECORDS, 4*',i5,' BYTES EACH', &
             ' HAVE BEEN ALLOCATED ON UNIT',i3,' ...')
16000 format(/' ',a8)
  end subroutine contrl_init_elements

  !---------------------------------------------------------------------
  ! Phase 3: Single Polyatomic Cycle (ALOOP -> EWMO -> Convergence -> SCF)
  !          Callable from Fortran loop OR step-by-step from Python!
  !---------------------------------------------------------------------
  subroutine contrl_step_aloop_and_ewmo(converged_or_done)
    logical, intent(out) :: converged_or_done
    ! Scoped allocatable workspaces for ALOOP and EWMO
    integer(int32), allocatable :: al_1(:), al_2(:)
    real(real64),   allocatable :: al_3(:), al_4(:), al_5(:), al_6(:),  &
      al_7(:), al_8(:), al_9(:), al_10(:), al_11(:), al_12(:),          &
      al_13(:), al_14(:), al_15(:), al_16(:), al_17(:), al_18(:),       &
      al_19(:), al_20(:), al_21(:), al_22(:), al_23(:)
    real(real64),   allocatable :: ew_1(:), ew_2(:), ew_3(:), ew_4(:),  &
      ew_5(:), ew_6(:)

    converged_or_done = .false.
    if (QATOM) return

    call TIMOUT(0)
    mol_ws%ovlp  = 0.0d0; mol_ws%ewmat = 0.0d0; mol_ws%popul = 0.0d0
    mol_ws%evalx = 0.0d0; mol_ws%occx  = 0.0d0; mol_ws%evecx = 0.0d0
    IATOM  = 0; IORBT = 0; JSTOT = 0
    NBYTES = 8 * N2ATOM
    ZNSUM  = ZERO; ELSUM = ZERO; ELDIF = ZERO

    ! Run ALOOP across all elements
    do IELEM = 1, NELEM
      IUNIT0 = IUNIT(10 + IELEM)
      call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
      M2STO = mol_ws%index_arr(MSTO) + MSTO
      call read_atom_stream_mdifat(IUNIT3, MDIFAT)
      N2ORB = mol_ws%index_arr(NORB) + NORB
      call write_atom_stream_mdifat(IUNIT4, MDIFAT)
      call run_aloop_element(QREALY)
      call rewind_elem_stream(IELEM)
    end do

    mol_ws%stvab_pos = 0; call rewind_atom_stream(IUNIT3); call rewind_atom_stream(IUNIT4)
    if (QPRINT) write(IW, 12000) IUNIT4
    ELECS(ISPIN) = ZERO
    ELECS(1)     = HALF * (ELSUM + ELDIF)
    ELECS(ISPIN) = ELECS(ISPIN) + HALF * (ELSUM - ELDIF)

    if (.not. QCMPTD) then
      write(IW, 13500) NEXREC - 1, LENREC, LENREC * (NEXREC - 1) / 1024, IUNIT1
      write(IW, 13600) NLRECS, NBYTES / 1024
      write(IW, 12000) IUNIT2
      call TIMOUT(8)
    end if

    ! Run EWMO if not first cycle
    if (.not. QFIRST) then
      QEWMOS = QEWMO
      if (QMNPRT) IW = IWSAV
      if ((CDAMP == REALS(1)) .and. (.not. QPRINT)) QEWMO = .false.
      call run_ewmo()
      QEWMO  = QEWMOS
      QSEPAT = .false.

      CTWOD = CHTRAN
      EXTE1 = TMOLE + DELTAE
      if (QAUDMP) CDAMP = REALS(1)
      if (QAUDMP .and. (DELTAE < ZERO)) CDAMP = (CHTRAN / ELSUM)**CACCEL
      CHTRAN = (REALS(1) - CDAMP) * CHTRAN
      EXTE2  = EXTE1 - CDAMP * DELTAE
    end if

    call TIMOUT(0)
    NREMCY = MXCYCL - NCYCL
    TIMREM = TIMEND - FINISH
    CYCTIM = FINISH - OLDTIM
    OLDTIM = FINISH
    TEDIFF = abs(OLDTE - TMOLE)

    if (QFIRST) then
      if (.not. QPRINT) write(IW, 22000)
    else
      if (QPRINT) write(IW, 18000) TEDIFF, TIMREM, NREMCY
      if (.not. QPRINT) write(IW, 19000) NCYCL, TMOLE, CYCTIM, CHTRAN, VDAMP
      if (QPRINT) write(IW, 20000) CYCTIM

      if (QLAST) then
        converged_or_done = .true.
        return
      end if
      if (TEDIFF < CONVRG) QLAST = .true.
    end if

    if (NREMCY <= 1) QLAST = .true.
    if (TIMREM < REALS(2) * CYCTIM) QLAST = .true.
    if (QLAST) write(IW, 21000)
    if (QMNPRT) IW = IUDUMP
12000 format(' ... WRITES TO UNIT',i3,' ARE COMPLETE ...')
13500 format(' ...',i4,i5,'-BYTE RECORDS [',i4,'K] WRITTEN TO UNIT',i3)
13600 format('    ',i4,'   LOGICAL RECORDS [',i4,'K]...')
18000 format(// &
     ' -------------------------------------'/ &
     ' TOTAL ENERGY CONVERGENCE...',f10.5/ &
     ' CPU SECONDS REMAINING......',f10.3/ &
     ' ALLOWED CYCLES REMAINING...',i10/ &
     ' -------------------------------------')
19000 format(' ',i3,f16.4,f13.4,f17.4,f14.4,t76,'<te>')
20000 format(' ... IT TOOK',f6.2,' SECONDS TO COMPLETE THIS CYCLE ...')
21000 format(//' ... LAST CYCLE FOLLOWS ...'/)
22000 format(// &
     ' CYCLE   TOTAL ENERGY    CYCLE TIME    CHARGE TRANSFER', &
     '     DAMPING   ',t76,'<te>'/ &
     '   #      THIS CYCLE     (SECONDS)      FOR NEXT CYCLE', &
     '     FACTOR    ',t76,'<te>'/ &
     ' -----   ------------    ----------    ---------------', &
     '     -------   ',t76,'<te>'/)
  end subroutine contrl_step_aloop_and_ewmo

  !---------------------------------------------------------------------
  ! Phase 4: Execute Atomic SCF Step Across All Elements (IC(1)..IC(35))
  !---------------------------------------------------------------------
  subroutine contrl_step_scf(done_after_scf)
    logical, intent(out) :: done_after_scf
    integer(int32), allocatable :: scf_1(:), scf_2(:), scf_8(:), scf_9(:), scf_29(:)
    real(real64),   allocatable :: scf_3(:), scf_4(:), scf_5(:), scf_6(:),  &
      scf_7(:), scf_10(:), scf_11(:), scf_12(:), scf_13(:), scf_14(:),      &
      scf_15(:), scf_16(:), scf_17(:), scf_18(:), scf_19(:), scf_20(:),     &
      scf_21(:), scf_22(:), scf_23(:), scf_24(:), scf_25(:), scf_26(:),     &
      scf_27(:), scf_28(:), scf_30(:), scf_31(:), scf_32(:), scf_33(:),     &
      scf_34(:), scf_tail(:)

    done_after_scf = .false.
    if (QLAST) QPRINT = .true.
    IORBT = 0
    IATOM = 1
    NCYCL = NCYCL + 1
    OLDTE = TMOLE
    TMOLE = ZERO; TOTKE = ZERO; ZZPOT = ZERO

    do IELEM = 1, NELEM
      IUNIT0 = IUNIT(IELEM + 10)
      call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
      M2STO = mol_ws%index_arr(MSTO) + MSTO

      if (QATOM) then
        call read_atom_stream_mdifat(IUNIT3, MDIFAT)
      else
        call read_atom_stream_mdifat(IUNIT4, MDIFAT)
        call write_atom_stream_mdifat(IUNIT3, MDIFAT)
      end if
      N2ORB = mol_ws%index_arr(NORB) + NORB

      allocate(scf_1(NSTO), scf_2(NSTO), scf_3(NSTO), scf_4(NSTO))
      allocate(scf_5(M2STO), scf_6(M2STO), scf_7(M2STO * NVTERM))
      allocate(scf_8(NVTERM), scf_9(NVTERM), scf_10(NVTERM))
      allocate(scf_11(M2STO), scf_12(MSTO * MSTO), scf_13(M2STO))
      call run_scf_element()
      deallocate(scf_1, scf_2, scf_3, scf_4, scf_5, scf_6, scf_7, &
                 scf_8, scf_9, scf_10, scf_11, scf_12, scf_13)
      call rewind_elem_stream(IELEM)
    end do

    call rewind_atom_stream(IUNIT3)
    if (.not. QATOM) call rewind_atom_stream(IUNIT4)
    if (QATOM) then
      done_after_scf = .true.
      return
    end if

    QCMPTD = .true.
    if (.not. QAUDMP) then
      if (QRSTRT .or. (.not. QFIRST)) CDAMP = CDAMP**CACCEL
      if (QSTALL .and. (NCYCL == 2)) CDAMP = CDMP90
    end if
    VDAMP  = VDAMP**VACCEL
    if (QMNPRT) IW = IWSAV
    QFIRST = .false.
    if (QSEPAT) QPRINT = QPRNT
  end subroutine contrl_step_scf

  !---------------------------------------------------------------------
  ! Phase 0: Read Title and &CTLINP from unit IR (fd 0/stdin)
  !---------------------------------------------------------------------
  subroutine contrl_read_control_input(eof_reached)
    logical, intent(out) :: eof_reached
    real(real64), parameter :: BOLTZ = 3.166728d-6
    integer(int32) :: ios
    logical        :: is_open

    namelist /CTLINP/ QRSTRT, QBAKUP, QOPEN,  QDEBUG, QSEPAT, QCMPTD, QEWMO,  &
                      QATOM,  QMODVO, QMXREC, QMNPRT, QCHECK, QGEOM,  QRERUN, &
                      QANGST, TIMEND, MXNSTO, NRECS,  LENREC, PKSCAL, PKE,    &
                      NATOM,  NELEM,  NORBT,  MXNORB, MSTOT,  MXMSTO, NMAX,   &
                      LMAX,   MXITER, MXCYCL, IUNIT,  LENBUF, KATOM,  KORB,   &
                      KSP,    WGT0,   WGT1,   WGT2,   WGTV,   WGTC,   VDAMP,  &
                      VACCEL, TFEXP,  TFCOEF, LTF,    QC2FIT, QX2FIT, QPSMET, &
                      THETA,  RANGE,  EXPMIN, CONVRG, CDAMP,  CACCEL, NCREAT, &
                      NANNIH, ICREAT, IANNIH, QTRANS, QPLOT,  QCORE,  QPRINT, &
                      QAUDMP, QTRACE

    inquire(unit=9, opened=is_open)
    if (.not. is_open) open(9, file='Phatpsy.Stderr')
    if (IUDUMP /= IW) then
      inquire(unit=IUDUMP, opened=is_open)
      if (.not. is_open) open(IUDUMP, file='Phatpsy.Debug')
    end if
    inquire(unit=IP, opened=is_open)
    if (.not. is_open) open(unit=IP, file='Phatpsy.Punch')

    eof_reached = .false.
    read(IR, '(80A1)', iostat=ios) TITLE
    if (ios < 0) then
      eof_reached = .true.
      return
    end if
    if (QTRACE) write(9, *) TITLE
    write(IW, *) TITLE

    read(IR, nml=CTLINP, iostat=ios)
    if (ios < 0) then
      eof_reached = .true.
      return
    end if

    call GETSET()
    call SETBOM()
    if (QBAKUP) QRSTRT = .true.
    if (QRSTRT) QSEPAT = .false.
    if (QDEBUG) QPRINT = .true.
    if (QATOM .and. QRSTRT) QBAKUP = .true.
    if (QATOM) NORBT = MXNORB
    if (NORBT < MXNORB) NORBT = MXNORB * NATOM
    if (MSTOT < MXMSTO) MSTOT = MXMSTO * NATOM
    if (MSTOT < NORBT)  MSTOT = NORBT
    if ((NCREAT > 0) .or. (NANNIH > 0)) QGSTAT = .false.
    if (QRERUN) then
      QRSTRT = .true.
      QCMPTD = .true.
      MXCYCL = merge(1, 0, logical(QBAKUP))
    end if
    call print_banner_and_control_input()
    THETA = BOLTZ * THETA
  end subroutine contrl_read_control_input

  subroutine print_banner_and_control_input()
    integer(int32) :: icr, ian

    write(IW, 2000)
    if (QCHECK) write(IW, 2500)
    if (QRERUN) write(IW, 2600)
    if (QRSTRT .and. .not. QBAKUP) write(IW, 3000) IUNIT(4)
    if (QBAKUP) write(IW, 3000) IUNIT(3)
    if (QOPEN) write(IW, 4000)
    if (QDEBUG) write(IW, 5000) IUNIT(8)
    if (QGEOM) write(IW, 5500)
    if (.not. QATOM) then
      if (QSEPAT) write(IW, 6000)
      if (QCMPTD) write(IW, 7000) IUNIT(1), IUNIT(2)
    end if
    if (QATOM .or. .not. QEWMO) write(IW, 8000)
    if (.not. QATOM) then
      if (QGSTAT) write(IW, 9000)
      if (.not. QGSTAT) write(IW, 10000)
      if (NCREAT > 0) write(IW, 11000) (ICREAT(icr), icr = 1, NCREAT)
      if (NANNIH > 0) write(IW, 12000) (IANNIH(ian), ian = 1, NANNIH)
      if (.not. QGSTAT) write(IW, 13000)
      if (QTRANS) write(IW, 14000)
    end if
    if (QCORE) write(IW, 15000)
    if (QPLOT) write(IW, 16000)
    if (.not. QATOM) then
      if (QPRINT) write(IW, 17000)
      if (QMNPRT) write(IW, 17500) IUNIT(6)
      write(IW, 18000) NATOM, NELEM
      write(IW, 19000) NORBT, MXNORB
      write(IW, 20000) MSTOT, MXMSTO, NMAX, LMAX
      write(IW, 21000) MXCYCL, MXITER
    end if
    write(IW, 21500) TIMEND
    write(IW, 22000) LENBUF
    write(IW, 23000) EXPMIN
    write(IW, 24000) CONVRG
    write(IW, 24500) THETA
    if (QMODVO) write(IW, 25000)
    if (.not. QATOM) then
      if (QANGST) write(IW, 25500)
      if (.not. QAUDMP) write(IW, 28000) CDAMP, CACCEL
      if (QC2FIT) write(IW, 29000) WGT0, WGT1, WGTC, WGTV, WGT2
      if (.not. QC2FIT) write(IW, 29010)
      if (.not. QX2FIT) write(IW, 29020)
      if (QPSMET) write(IW, 29030)
      if (VDAMP /= ZERO) write(IW, 30000) VDAMP, VACCEL
      write(IW, 30500) PKSCAL
      if (.not. QPSMET) write(IW, 30600) PKE
      write(IW, 31000) IUNIT(1)
      if (.not. QMXREC) write(IW, 32000) IUNIT(1), LENREC, NRECS
      write(IW, 33000) IUNIT(2)
      write(IW, 34000) IUNIT(3), IUNIT(4)
      write(IW, 35000) IUCSF
      if (KATOM /= 0) write(IW, 36000) KORB, KSP, KATOM
      if (QAUDMP) write(IW, 37000) CACCEL
    end if
    if (QATOM) write(IW, 38000) IUNIT(3)
    write(IW, 39000) 11

2000 format(// &
     ' ',t25,' ---------------------------------------------------- '/ &
     ' ',t25,'|                                                    |'/ &
     ' ',t25,'|                 P H A T P S Y                      |'/ &
     ' ',t25,'|                                                    |'/ &
     ' ',t25,'|                    PROGRAM                         |'/ &
     ' ',t25,'|                                                    |'/ &
     ' ',t25,'|                      BY                            |'/ &
     ' ',t25,'|                                                    |'/ &
     ' ',t25,'|                 JACK A. SMITH                      |'/ &
     ' ',t25,'|             QUANTUM THEORY PROJECT                 |'/ &
     ' ',t25,'|             UNIVERSITY OF FLORIDA                  |'/ &
     ' ',t25,'|                      -                             |'/ &
     ' ',t25,'|           UNION CARBIDE CORPORATION                |'/ &
     ' ',t25,'|               S. CHARLESTON, WV                    |'/ &
     ' ',t25,'|                                                    |'/ &
     ' ',t25,' ---------------------------------------------------- '// &
     ' === CONTROL INPUT ==='//)
2500 format(' ONLY THE INPUT WILL BE CHECKED ON THIS RUN.')
2600 format(' THIS IS A RERUN - NO SCF, OR BACKUP AND REPEAT.')
3000 format(' THIS IS A RESTART BEGINNING WITH DATA ON UNIT',i3,'.')
4000 format(' THIS IS A SPIN-UNRESTRICTED CALCULATION.')
5000 format(' THIS IS A DEBUG RUN WITH EXTRA OUTPUT ON UNIT',i3,'.')
5500 format(' THIS IS A GEOMETRY RUN.')
6000 format(' THE 1ST CYCLE WILL BE A SEPARATED-ATOM CALCULATION.')
7000 format(' THE TWO-CENTER INTEGRALS HAVE ALREADY BEEN COMPUTED', &
            ' AND ARE ON UNITS',i3,' AND',i3,'.')
8000 format(' NO EWMO CALCULATION WILL BE PERFORMED - ONLY ATOMIC SCF', &
            ' WILL BE PERFORMED.')
9000 format(' THIS IS A GROUND STATE CALCULATION.')
10000 format(' THIS CALCULATION WILL BE ON THE STATE GIVEN BY'/)
11000 format('    ICREAT =',7i4)
12000 format('    IANNIH =',7i4)
13000 format('    WORKING ON THE INITIAL "GROUND" STATE.')
14000 format(' THIS WILL BE A TRANSITION STATE CALCULATION.')
15000 format(' THE ATOMIC CORE ELECTRONS WILL BE TREATED LIKE VALENCE.')
16000 format(' TOTAL ATOMIC DENSITY PLOTS WILL BE GENERATED.')
17000 format(' THE INTERMEDIATE RESULTS FOR EACH CYCLE WILL BE PRINTED.')
17500 format(' ONLY MINIMAL PRINTING ON UNIT',i3,' WILL BE PERFORMED.')
18000 format(' THIS MOLECULE CONTAINS',i3,' ATOMS MADE UP FROM',i3, &
            ' DIFFERENT ELEMENTS.')
19000 format(' THIS MOLECULE CONTAINS',i4,' ATOMIC ORBITALS WITH A', &
            ' MAXIMUM OF',i3,' ON ANY ATOM.')
20000 format(' THE BASIS CONTAINS',i4,' STO''S WITH THE LARGEST', &
            ' ATOM HAVING',i3,' WITH (N,L) UP TO (',i1,',',i1,').')
21000 format(' THE MAXIMUM NUMBER CYCLES TO BE ALLOWED IS',i3,' WITH', &
            i3,' ITERATIONS PER CYCLE.')
21500 format(' THE MAXIMUM CPU TIME ALLOWED IS',f7.0,' SECONDS.')
22000 format(' THE TWO-ELECTRON INTEGRAL BUFFER IS',i5,' DOUBLE WORDS LONG.')
23000 format(' THE SMALLEST EXPONENT EVALUATED DURING INTEGRAL COMPUTATION IS',f7.1)
24000 format(' THE TOTAL ENERGY CONVERGENCE THRESHOLD IS',1pd9.1)
24500 format(' THE FERMI "TEMPERATURE" FOR CHARGE EQUILIBRATION IS',1pd9.1)
25000 format(' THE VIRTUAL ORBITAL ENERGIES WILL BE MODIFIED.')
25500 format(' ALL DISTANCES WILL BE GIVEN IN ANGSTROMS AND CONVERTED', &
            ' TO ATOMIC UNITS (BOHRS) WITHIN.')
28000 format(' THE CHARGE TRANSFER IN THE MOLECULE WILL BE DAMPED BY', &
            f6.2,' **',f6.2,'.')
29000 format(' THE FOLLOWING CONDITIONS WILL BE SATISFIED ACCORDING TO'/ &
            '    THE GIVEN WEIGHTS IN FITTING THE MODEL POTENTIALS.'/ &
            '       BEHAVIOR AT THE NUCLEUS........',f9.4/ &
            '       HF COUL POTENTIAL AT NUCLEUS...',f9.4/ &
            '       CORE HF COULOMB POTENTIALS.....',f9.4/ &
            '       VALENCE HF COULOMB POTENTIALS..',f9.4/ &
            '       EXTERNAL NUC. ATT. POTENTIAL...',f9.4)
29010 format(' THE 2-CENTER COULOMB POTENTIALS WILL NOT BE FIT EACH CYCLE.')
29020 format(' THE 2-CENTER EXCHANGE POTENTIALS WILL NOT BE FIT EACH CYCLE.')
29030 format(' THE ENERGY DEPENDENCE OF THE PSEUDOPOTENTIAL WILL BE', &
            ' HANDLED BY PSEUDOMETRIC.')
30000 format(' THE CHANGE IN THE EXTERNAL POTENTIAL WILL BE DAMPED BY', &
            f6.2,' **',f6.2,'.')
30500 format(' THE P-K PROJECTION MATRIX WILL BE SCALED BY',f6.2,'.')
30600 format(' THE ENERGY USED IN THE P-K PSEUDOPOTENTIAL IS',f7.2,'.')
31000 format(' THE PRIMITIVE TWO-CENTER OVERLAP INTEGRALS WILL BE ON', &
            ' DIRECT-ACCESS UNIT',i3,'.')
32000 format(' THE SPACE PARAMETER FOR UNIT',i3,' SHOULD BE (4*',i4,',',i4,').')
33000 format(' THE PRIMITIVE TWO-CENTER MODEL POTENTIAL INTEGRALS WILL BE ON UNIT',i3,'.')
34000 format(' THE INTERMEDIATE AND FINAL ATOMIC DATA WILL BE ON UNITS',i3,' AND',i3,' ALTERNATELY.')
35000 format(' THE FINAL CSF (MOLSTRUCT) OUTPUT WILL BE ON UNIT',i3,'.')
36000 format(' THE',i3,'-',i1,' ORBITAL ON ATOM',i3,' WILL BE PUNCHED.')
37000 format(' THE CHARGE TRANSFER WILL BE AUTOMATICALLY DAMPED ACCORDING TO THE RELATIVE CHARGE TRANSFER (RCT)'/ &
            '    CDAMP = RCT**',f6.2)
38000 format(' THE INTERMEDIATE AND FINAL ATOMIC DATA WILL BE ON UNIT',i3,'.')
39000 format(' ...',i6,'K BYTES OF CORE TO BE ALLOCATED FOR CONTRL ...')
  end subroutine print_banner_and_control_input

end module phatpsy_contrl_mod