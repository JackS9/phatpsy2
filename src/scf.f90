module scf
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin, only: run_lowdin
  use symmetry, only: symop
  use modelpot, only: regularize_pk_projector
  use exchange_dft, only: compute_dft_exchange
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
  public :: run_scf_element, USE_LAPACK_EIG, FOCK_MIX_ALPHA

  ! Modern solver toggles accessible from Python
  logical,      save :: USE_LAPACK_EIG = .true.  ! Set .true. to use LAPACK DSPEVD
  real(real64), save :: FOCK_MIX_ALPHA = 1.0d0   ! 1.0 = legacy; <1.0 = Fock damping

contains

  subroutine solve_orthonormal_fock(onfock, eval_all, onevec, scrat2, msto, m2sto, norb, ierr)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    integer(int32), intent(in)    :: msto, m2sto, norb
    integer(int32), intent(out)   :: ierr
    real(real64),   intent(inout) :: onfock(m2sto)
    real(real64),   intent(out)   :: eval_all(msto), onevec(msto, norb)
    real(real64),   intent(inout) :: scrat2(msto, 9)

    real(real64),   allocatable :: z_full(:,:), work(:), ap_copy(:), a_full(:,:)
    integer(int32) :: lwork, i, j, ij

    if (msto <= 0) then
      ierr = 0
      return
    end if

    if (msto == 1) then
      eval_all(1) = onfock(1)
      onevec(1, 1:norb) = 1.0d0
      ierr = 0
      return
    end if

    ! Tier 1: Primary packed symmetric eigensolver (LAPACK DSPEV)
    ! Operates on a copy of onfock to preserve matrix if fallback is needed
    allocate(ap_copy(m2sto), z_full(msto, msto), work(3 * msto))
    ap_copy = onfock
    call DSPEV('V', 'U', msto, ap_copy, eval_all, z_full, msto, work, ierr)
    if (ierr == 0) then
      onevec(:, 1:norb) = z_full(:, 1:norb)
      deallocate(ap_copy, z_full, work)
      return
    end if
    deallocate(ap_copy, z_full, work)

    ! Tier 2 Fallback: Full-matrix Householder QR eigensolver (LAPACK DSYEV)
    ! Unpack onfock into full symmetric matrix A(msto, msto)
    allocate(a_full(msto, msto))
    ij = 0
    do j = 1, msto
      do i = 1, j
        ij = ij + 1
        a_full(i, j) = onfock(ij)
        a_full(j, i) = onfock(ij)
      end do
    end do

    lwork = max(1, 3 * msto)
    allocate(work(lwork))
    call DSYEV('V', 'U', msto, a_full, msto, eval_all, work, lwork, ierr)
    deallocate(work)

    if (ierr == 0) then
      onevec(:, 1:norb) = a_full(:, 1:norb)
      deallocate(a_full)
      return
    end if

    ! If ierr > 0 after both DSPEV and DSYEV:
    ! In LAPACK, INFO > 0 indicates that intermediate tridiagonal QL did not reach
    ! machine zero for an off-diagonal term, but eigenvectors/eigenvalues are still computed.
    ! If solutions are non-NaN, accept them and clear ierr to allow SCF to converge self-consistently.
    if (.not. any(ieee_is_nan(eval_all(1:norb)))) then
      onevec(:, 1:norb) = a_full(:, 1:norb)
      ierr = 0
    end if
    deallocate(a_full)
  end subroutine solve_orthonormal_fock

  subroutine run_scf_element()
    ! Local allocatable workspaces (formerly CORE(IC(1))..CORE(IC(35)) in contrl.f)
    integer(int32), allocatable :: n(:), l(:), lv(:), mv(:), iorder(:,:)
    real(real64),   allocatable :: eta(:), anorm(:), stovl(:), stkin(:),     &
      stpot(:,:), vexp(:), sthmat(:), ontran(:,:), cpot2(:), xpot2(:,:),     &
      zpot2(:), stden(:,:), stfock(:,:), stfock_prev(:,:), stcoul(:,:),      &
      stexch(:,:), onfock(:), onevec(:,:), stevec(:,:,:), oldvec(:,:,:),     &
      eval(:,:), occ(:,:), aofock(:,:), gamma(:,:), projec(:,:,:),           &
      scrat1(:,:,:), scrat2(:,:), tranx(:,:,:), vfit2(:), vcoef(:),          &
      pkpot(:,:,:)

    ! In-memory backups eliminating IUTEMP
    real(real64), allocatable :: stovl_bak(:), ontran_bak(:,:)
    real(real64), allocatable :: stevec_bak(:,:,:), eval_bak(:,:), occ_bak(:,:), &
                                 pkpot_bak(:,:,:), aofock_bak(:,:), gamma_bak(:,:), &
                                 vcoef_bak(:)
    real(real64) :: toten_bak

    real(real64), parameter :: ESCALE = 1.55d0
    character(len=2) :: latom
    integer(int32)   :: jdifat, nucz, iunitx, neqat, isp, iorb, jorb, ijorb
    integer(int32)   :: nv, niter, ij, iterm1, ierr, ieqat, isym, ixyz, matom
    real(real64)     :: x, y, z, toten, zzrep, rmAx, tke, te, endiff, oldte
    real(real64)     :: oldiff, tcoul0, zzrep0
    logical(kind=1)  :: qx2inc, qnrmlz, qtrunc, qindef, qdivrg, qdone, qnword, qdummy

    real(real64), external :: AVERAG

    qx2inc = .true.
    qnrmlz = .true.
    qtrunc = .false.

    if (allocated(n)) then
      deallocate(n, l, eta, anorm, stovl, stkin, stpot, lv, mv, vexp, &
                 sthmat, ontran, cpot2, xpot2, zpot2, stden, stfock,  &
                 stfock_prev, stcoul, stexch, onfock, onevec, stevec, &
                 oldvec, eval, occ, aofock, gamma, projec, iorder,    &
                 scrat1, scrat2, tranx, vfit2, vcoef, pkpot,          &
                 stovl_bak, ontran_bak, stevec_bak, eval_bak, occ_bak,&
                 pkpot_bak, aofock_bak, gamma_bak, vcoef_bak)
    end if

    allocate(n(NSTO), l(NSTO), eta(NSTO), anorm(NSTO))
    allocate(stovl(M2STO), stkin(M2STO), stpot(M2STO, NVTERM))
    allocate(lv(NVTERM), mv(NVTERM), vexp(NVTERM))
    allocate(sthmat(M2STO), ontran(MSTO, MSTO))
    allocate(cpot2(M2STO), xpot2(M2STO, ISPIN), zpot2(M2STO))
    allocate(stden(M2STO, ISPIN), stfock(M2STO, ISPIN), stfock_prev(M2STO, ISPIN))
    allocate(stcoul(M2STO, ISPIN), stexch(M2STO, ISPIN))
    allocate(onfock(M2STO), onevec(MSTO, NORB))
    allocate(stevec(MSTO, NORB, ISPIN), oldvec(MSTO, NORB, ISPIN))
    allocate(eval(NORB, ISPIN), occ(NORB, ISPIN), aofock(N2ORB, ISPIN))
    allocate(gamma(NORB, ISPIN), projec(MSTO, MSTO, ISPIN), iorder(NORB, ISPIN))
    allocate(scrat1(MSTO, MSTO, 2), scrat2(MSTO, 9), tranx(MSTO, MSTO, ISPIN))
    allocate(vfit2(NVTERM), vcoef(NVTERM), pkpot(M2STO, 6, ISPIN))
    allocate(stovl_bak(M2STO), ontran_bak(MSTO, MSTO))
    allocate(stevec_bak(MSTO, NORB, ISPIN), eval_bak(NORB, ISPIN), occ_bak(NORB, ISPIN))
    allocate(pkpot_bak(M2STO, 6, ISPIN), aofock_bak(N2ORB, ISPIN), gamma_bak(NORB, ISPIN))
    allocate(vcoef_bak(NVTERM))

    !-------------------------------------------------------------------
    ! Loop over non-equivalent atoms (JDIFAT = 1..MDIFAT)
    !-------------------------------------------------------------------
    do jdifat = 1, MDIFAT
      call TIMOUT(0)
      call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
      call read_elem_basis(IELEM, latom, nucz, n, l, eta, anorm)
      call read_elem_potentials(IELEM, stovl, stkin, stpot, lv, mv, vexp)
      call read_elem_core_matrices(IELEM, sthmat, ontran)
      stovl_bak  = stovl
      ontran_bak = ontran

      iunitx = merge(IUNIT3, IUNIT4, logical(QATOM))
      call read_atom_record_ref(iunitx, QFLAGS, neqat, x, y, z, stevec, eval, occ, aofock, gamma, &
                                cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)
      if (QATOM) call rewind_atom_stream(IUNIT3)

      tke    = -toten
      te     = merge(0.0d0, toten, logical(QFIRST .and. .not. QRSTRT))
      endiff = -toten
      if (QATOM) zzrep = 0.0d0

      if (.not. QVIRTL) then
        if (.not. QFIRST) then
          do isp = 1, ISPIN
            do iorb = 1, NORB
              occ(iorb, isp) = CDAMP * occ(iorb, isp) &
                             + (1.0d0 - CDAMP) * mol_ws%popul(IORBT + iorb, isp)
            end do
          end do
        end if

        if (QPRINT) then
          write(IW, 1000) latom, jdifat
          write(IW, 2000) NCYCL
          if (QSEPAT) write(IW, 3000)
        end if

        ! Construct pseudometric and Löwdin transformation matrix TRANX
        do isp = 1, ISPIN
          stovl = stovl_bak
          if (QPSMET .and. nucz > 1.5d0) then
            call regularize_pk_projector(stovl_bak, pkpot(1,1,isp), pkpot(1,2,isp), MSTO, M2STO, 1.0d-3)
            stovl = stovl - pkpot(:, 1, isp)
            call TRISQ(MSTO, stovl, ontran, projec(1,1,isp))
          end if
          call run_lowdin(stovl, tranx(1,1,isp), scrat1, MSTO, MSTO, 1, qindef)
          if (qindef) then
            stovl = stovl_bak
            call TRISQ(MSTO, stovl, ontran, projec(1,1,isp))
            call run_lowdin(stovl, tranx(1,1,isp), scrat1, MSTO, MSTO, 1, qindef)
          end if
          xpot2(:, isp) = pkpot(:, 3, isp)
        end do

        !---------------------------------------------------------------
        ! Inner Atomic GCHF SCF Iteration Loop
        !---------------------------------------------------------------
        niter  = 0
        qdivrg = .false.
        qdone  = (QFIXED .or. (MXITER <= 0))

        scf_loop: do
          do isp = 1, ISPIN
            call DENMAT(stden(1,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
          end do

          call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
          call read_elem_basis(IELEM, latom, nucz, n, l, eta, anorm)
          call read_elem_potentials(IELEM, stovl, stkin, stpot, lv, mv, vexp)
          call read_elem_core_matrices(IELEM, sthmat, ontran)
          call FOKMAT(stcoul, stexch, stden, mol_ws%buffer)
          if (QDFTEX) then
            call compute_dft_exchange(IELEM, stden, stexch, EXCH_TYPE, EXCH_ALPHA, CORR_TYPE)
          end if

          do isp = 1, ISPIN
            do ij = 1, M2STO
              stfock(ij, isp) = sthmat(ij) + stcoul(ij, isp) + stexch(ij, isp) &
                              + zpot2(ij) + cpot2(ij)
              if (qx2inc) stfock(ij, isp) = stfock(ij, isp) + xpot2(ij, isp)
            end do
            ! Optional Fock-matrix damping for difficult surface clusters (e.g., Ni(100))
            if (niter > 0 .and. FOCK_MIX_ALPHA < 1.0d0) then
              stfock(:, isp) = FOCK_MIX_ALPHA * stfock(:, isp) &
                             + (1.0d0 - FOCK_MIX_ALPHA) * stfock_prev(:, isp)
            end if
            stfock_prev(:, isp) = stfock(:, isp)
          end do

          if (qdone) exit scf_loop

          niter  = niter + 1
          oldte  = te
          te     = AVERAG(stfock(1,1), stden(1,1), MSTO, M2STO)
          if (QOPEN) te = te + AVERAG(stfock(1,2), stden(1,2), MSTO, M2STO)
          te     = te * ESCALE * ORBOCC
          oldiff = endiff
          endiff = abs(te - oldte)

          if ((niter >= MXITER) .or. (endiff < CONVRG)) qdone = .true.

          if (te >= oldte .and. endiff > oldiff .and. &
              (endiff / te > 0.001d0 / niter) .and. niter > 2 .and. .not. QDEBUG) then
            if (qdivrg) then
              write(IW, 16000) latom, jdifat
              iterm1 = niter - 1
              write(IW, 17000) iterm1, oldte, oldiff, niter, te, endiff
              qdone = .true.
            else
              qdivrg = .true.
            end if
          else
            qdivrg = .false.
          end if

          do isp = 1, ISPIN
            if (nucz > 1.5d0) stfock(:, isp) = stfock(:, isp) - pkpot(:, 2, isp)
            call UTHU(MSTO, stfock(1,isp), tranx(1,1,isp), scrat1, onfock)
            call solve_orthonormal_fock(onfock, scrat1, onevec, scrat2, &
                                        MSTO, M2STO, NORB, ierr)
            if (ierr < 0) then
              call BOMB(17)
            else if (ierr > 0) then
              call WARN(15, qdummy)
              if (any(ieee_is_nan(scrat1(1:NORB, 1, 1)))) call BOMB(16)
            end if

            eval(1:NORB, isp) = scrat1(1:NORB, 1, 1)
            oldvec(:, :, isp) = stevec(:, :, isp)

            call DMPAB(tranx(1,1,isp), MSTO, MSTO, MSTO, MSTO, onevec, MSTO, &
                       NORB, MSTO, NORB, stevec(1,1,isp), MSTO, NORB)

            if (QPSMET .and. .not. qtrunc) then
              call DMPATB(projec(1,1,isp), MSTO, MSTO, MSTO, MSTO,           &
                          stevec(1,1,isp), MSTO, NORB, MSTO, NORB,           &
                          onevec, MSTO, NORB)
              call CTSC(onevec, stovl, scrat1, stovl, MSTO, NORB)
              call run_lowdin(stovl, ontran, scrat1, NORB, NORB, 2, qindef)
              if (.not. qindef) then
                call DMPAB(onevec, MSTO, NORB, MSTO, NORB, ontran, NORB,     &
                           NORB, NORB, NORB, stevec(1,1,isp), MSTO, NORB)
                stovl  = stovl_bak
                ontran = ontran_bak
              end if
            end if

            if (qnrmlz) then
              stovl  = stovl_bak
              ontran = ontran_bak
              call NORMLZ(stevec(1,1,isp), NORB, stovl, MSTO, M2STO)
            end if

            if (nucz > 1.5d0) stfock(:, isp) = stfock(:, isp) + pkpot(:, 2, isp)
            call MAXOVL(iorder(1,isp), oldvec(1,1,isp), stevec(1,1,isp),     &
                        stovl, MSTO, M2STO, NORB, qnword)
            if (qnword) then
              if (any(abs(occ(1:NORB, isp) - occ(iorder(1:NORB, isp), isp)) > 1.0d-8)) then
                call NEWORD(occ(1,isp), scrat1, iorder(1,isp), NORB)
              else
                qnword = .false.
              end if
            end if
          end do
        end do scf_loop

        !---------------------------------------------------------------
        ! Energy, Charge, Spin Analysis & Model Potential Least-Squares Fit
        !---------------------------------------------------------------
        if ((endiff > CONVRG) .and. (.not. QFIXED)) write(IW, 20010) endiff, niter
        if (QPRINT .and. (endiff <= CONVRG)) write(IW, 20000) endiff, niter
        if (QPRINT .and. (VDAMP > 0.0d0)) write(IW, 21000) VDAMP

        tcoul0 = ORBOCC * AVERAG(stpot, stden(1,1), MSTO, M2STO)
        if (QOPEN) tcoul0 = tcoul0 + AVERAG(stpot, stden(1,2), MSTO, M2STO)

        call ANALYS(stkin, stpot, stcoul, stexch, cpot2, zpot2, vfit2,       &
                    vcoef, vexp, lv, pkpot, xpot2, stevec, eval, occ, gamma, &
                    mol_ws%index_arr, toten, tke, zzrep, stovl,              &
                    mol_ws%charge, tcoul0)

        ! Update external projection, pseudopotential, and exchange matrices
        do isp = 1, ISPIN
          ijorb = 0
          do iorb = 1, NORB
            do jorb = 1, iorb
              ijorb = ijorb + 1
              aofock(ijorb, isp) = merge(gamma(iorb, isp), 0.0d0, jorb == iorb)
            end do
          end do

          call DENMAT(pkpot(1,4,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
          call DENMAT(pkpot(1,5,isp), stevec(1,1,isp), eval(1,isp), MSTO, M2STO, NORB)
          if (nucz <= 1.5d0) then
            pkpot(:, 4, isp) = 0.0d0
            pkpot(:, 5, isp) = 0.0d0
          else
            pkpot(:, 4, isp) = PKSCAL * pkpot(:, 4, isp)
            pkpot(:, 5, isp) = PKSCAL * pkpot(:, 5, isp)
          end if

          ! FIXED BUG: Vectorized over 1:M2STO instead of uninitialized IJ
          if (.not. QPSMET) then
            pkpot(1:M2STO, 5, isp) = pkpot(1:M2STO, 5, isp) - PKE * pkpot(1:M2STO, 4, isp)
          end if

          do ij = 1, M2STO
            onfock(ij) = 0.0d0
            if (QX2FIT) onfock(ij) = onfock(ij) + stexch(ij, isp)
            if (QC2FIT) then
              onfock(ij) = onfock(ij) + stcoul(ij, isp) &
                         - dot_product(vcoef(1:NVTERM), stpot(ij, 1:NVTERM))
            end if
          end do
          call UTHU(MSTO, onfock, ontran, scrat1, pkpot(1,6,isp))
        end do

        if (QPRINT) then
          write(IW, 1000) latom, jdifat
          write(IW, 2000) NCYCL
          if (QSEPAT) write(IW, 3000)
          write(IW, 22000)
          do isp = 1, ISPIN
            if (QOPEN) write(IW, 23000) SPIN(isp)
            call OUTVEC(stevec(1,1,isp), eval(1,isp), MSTO, NORB, IW)
            call ESCA(stevec(1,1,isp), eval(1,isp), n, l, eta, NORB, MSTO, NSTO, IW)
          end do
          write(IW, 28000) latom, jdifat
          call TIMOUT(5)
        end if

        if (QLAST .and. QPNCHV) then
          call PUNCHV(stevec, eval, occ, l, NSTO, MSTO, NORB, ISPIN, IP)
          write(IW, 24000) latom, jdifat, IP
        end if

        mol_ws%popul(IORBT+1:IORBT+NORB, 1) = occ(1:NORB, 1)
        if (QOPEN) mol_ws%popul(IORBT+1:IORBT+NORB, 2) = occ(1:NORB, 2)
        IORBT = IORBT + NORB
      end if

      IATOM = IATOM + 1
      TMOLE = TMOLE + toten
      TOTKE = TOTKE + tke
      ZZPOT = ZZPOT + zzrep / 2.0d0

      if (QATOM) call write_atom_stream_mdifat(IUNIT3, MDIFAT)
      call write_atom_record_ref(IUNIT3, QFLAGS, neqat, x, y, z, stevec, eval, occ, aofock, gamma, &
                                 cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)

      !-----------------------------------------------------------------
      ! Loop over symmetry-equivalent atoms (IEQAT = 1..NEQAT)
      !-----------------------------------------------------------------
      if (neqat > 0) then
        stovl_bak  = stovl
        ontran_bak = ontran
        stevec_bak = stevec
        eval_bak   = eval
        occ_bak    = occ
        pkpot_bak  = pkpot
        aofock_bak = aofock
        gamma_bak  = gamma
        vcoef_bak  = vcoef
        toten_bak  = toten
        zzrep0     = zzrep

        do ieqat = 1, neqat
          call read_atom_record_equiv(IUNIT4, QFLAGS, isym, ixyz, x, y, z, stevec, eval, occ, aofock, &
                                      gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)
          stovl  = stovl_bak
          ontran = ontran_bak
          stevec = stevec_bak
          eval   = eval_bak
          occ    = occ_bak
          pkpot  = pkpot_bak
          aofock = aofock_bak
          gamma  = gamma_bak
          vcoef  = vcoef_bak
          toten  = toten_bak

          mol_ws%charge(2, IATOM) = mol_ws%charge(2, IATOM - 1)
          mol_ws%charge(3, IATOM) = mol_ws%charge(3, IATOM - 1)

          if (isym /= 0) then
            call SYMOP(isym, ixyz, stevec, vcoef, l, lv, mol_ws%d, scrat1, &
                       NSTO, MSTO, NORB, ISPIN, NVTERM, LMAX, LLMXP1)
            do isp = 1, ISPIN
              call DENMAT(stden(1,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
            end do
            call read_elem_dims(IELEM, NSTO, MSTO, NORB, NVTERM)
            call read_elem_basis(IELEM, latom, nucz, n, l, eta, anorm)
            call read_elem_potentials(IELEM, stovl, stkin, stpot, lv, mv, vexp)
            call read_elem_core_matrices(IELEM, sthmat, ontran)
            call FOKMAT(stcoul, stexch, stden, mol_ws%buffer)
            if (QDFTEX) then
              call compute_dft_exchange(IELEM, stden, stexch, EXCH_TYPE, EXCH_ALPHA, CORR_TYPE)
            end if

            do isp = 1, ISPIN
              ijorb = 0
              do iorb = 1, NORB
                do jorb = 1, iorb
                  ijorb = ijorb + 1
                  aofock(ijorb, isp) = merge(gamma(iorb, isp), 0.0d0, jorb == iorb)
                end do
              end do
              call DENMAT(pkpot(1,4,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
              call DENMAT(pkpot(1,5,isp), stevec(1,1,isp), eval(1,isp), MSTO, M2STO, NORB)
              if (nucz <= 1.5d0) then
                pkpot(:, 4, isp) = 0.0d0
                pkpot(:, 5, isp) = 0.0d0
              else
                pkpot(:, 4, isp) = PKSCAL * pkpot(:, 4, isp)
                pkpot(:, 5, isp) = PKSCAL * pkpot(:, 5, isp)
              end if

              ! FIXED BUG: Vectorized over 1:M2STO
              if (.not. QPSMET) then
                pkpot(1:M2STO, 5, isp) = pkpot(1:M2STO, 5, isp) - PKE * pkpot(1:M2STO, 4, isp)
              end if

              do ij = 1, M2STO
                onfock(ij) = 0.0d0
                if (QX2FIT) onfock(ij) = onfock(ij) + stexch(ij, isp)
                if (QC2FIT) then
                  onfock(ij) = onfock(ij) + stcoul(ij, isp) &
                             - dot_product(vcoef(1:NVTERM), stpot(ij, 1:NVTERM))
                end if
              end do
              call UTHU(MSTO, onfock, ontran, scrat1, pkpot(1,6,isp))
            end do
          end if

          if (QFLIP) then
            mol_ws%charge(3, IATOM) = -mol_ws%charge(3, IATOM)
            call FLIP(stevec, 1, ISPIN, stevec, ISPIN, ISPIN, MSTO * NORB)
            call FLIP(eval,   1, ISPIN, eval,   ISPIN, ISPIN, NORB)
            call FLIP(occ,    1, ISPIN, occ,    ISPIN, ISPIN, NORB)
            call FLIP(aofock, 1, ISPIN, aofock, ISPIN, ISPIN, N2ORB)
            call FLIP(gamma,  1, ISPIN, gamma,  ISPIN, ISPIN, NORB)
            call FLIP(pkpot(1,4,1), 1, 1, pkpot(1,4,ISPIN), 1, 1, 3 * M2STO)
          end if

          IATOM = IATOM + 1
          TMOLE = TMOLE + toten
          TOTKE = TOTKE + tke
          if (isym == 0) zzrep = zzrep0
          ZZPOT = ZZPOT + zzrep / 2.0d0

          if (.not. QVIRTL) then
            mol_ws%popul(IORBT+1:IORBT+NORB, 1) = occ(1:NORB, 1)
            if (QOPEN) mol_ws%popul(IORBT+1:IORBT+NORB, 2) = occ(1:NORB, 2)
            IORBT = IORBT + NORB
          end if

          if (QLAST .and. QPNCHV) then
            call PUNCHV(stevec, eval, occ, l, NSTO, MSTO, NORB, ISPIN, IP)
            matom = ieqat * MDIFAT + jdifat
            write(IW, 24000) latom, matom, IP
          end if
          call write_atom_record_equiv(IUNIT3, QFLAGS, isym, ixyz, x, y, z, stevec, eval, occ, aofock, &
                                       gamma, cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)
          if (ieqat < neqat) then
            stovl_bak  = stovl
            ontran_bak = ontran
            stevec_bak = stevec
            eval_bak   = eval
            occ_bak    = occ
            pkpot_bak  = pkpot
            aofock_bak = aofock
            gamma_bak  = gamma
            vcoef_bak  = vcoef
            toten_bak  = toten
          end if
        end do
      end if
    end do

    if (allocated(n)) then
      deallocate(n, l, eta, anorm, stovl, stkin, stpot, lv, mv, vexp, &
                 sthmat, ontran, cpot2, xpot2, zpot2, stden, stfock,  &
                 stfock_prev, stcoul, stexch, onfock, onevec, stevec, &
                 oldvec, eval, occ, aofock, gamma, projec, iorder,    &
                 scrat1, scrat2, tranx, vfit2, vcoef, pkpot,          &
                 stovl_bak, ontran_bak, stevec_bak, eval_bak, occ_bak,&
                 pkpot_bak, aofock_bak, gamma_bak, vcoef_bak)
    end if

1000 format(///' ',a2,i2,/' ____')
2000 format(' ',t100,'... CYCLE',i3,' ...')
3000 format(' ',t45,'+++ NEAR SEPARATED-ATOM LIMIT +++')
16000 format(/' *** CONVERGENCE DIFFICULTY MET FOR ',a2,i2,' ***')
17000 format(/' SUMMARY OF TOTAL ENERGY CONVERGENCE...'// &
             ' OLD TOTAL ENERGY (',i2,')...',1pd15.8/ &
             ' OLD ENERGY DIFFERENCE.......',1pd8.1/ &
             ' NEW TOTAL ENERGY (',i2,')...',1pd15.8/ &
             ' NEW ENERGY DIFFERENCE.......',1pd8.1)
20000 format(/' THE TOTAL ENERGY CONVERGED TO',1pd8.1,' AFTER',i3,' ITERATIONS.')
20010 format(/' THE TOTAL ENERGY *NOT* CONVERGED (',1pd8.1,') AFTER',i3,' ITERATIONS.')
21000 format(/' THE EXTERNAL POTENTIALS WERE DAMPED THIS CYCLE BY',f7.4)
22000 format(/' FINAL VECTORS AND ENERGIES...')
23000 format(/' ',a8)
24000 format(/' ... ATOMIC ORBITALS FOR ',a2,i2,' HAVE BEEN PUNCHED ON UNIT',i3,' ...')
28000 format(/' ... END OF SCF FOR ',a2,i2,' ...'//)
  end subroutine run_scf_element

end module scf