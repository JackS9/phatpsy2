module scf
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin, only: run_lowdin
  use symmetry, only: symop
  use modelpot, only: regularize_pk_projector
  use exchange_dft, only: compute_dft_exchange
  implicit none
  private
  public :: run_scf_element, USE_LAPACK_EIG, FOCK_MIX_ALPHA

  ! Modern solver toggles accessible from Python
  logical,      save :: USE_LAPACK_EIG = .true.  ! Set .true. to use LAPACK DSPEVD
  real(real64), save :: FOCK_MIX_ALPHA = 1.0d0   ! 1.0 = legacy; <1.0 = Fock damping

contains

  subroutine solve_orthonormal_fock(onfock, eval_all, onevec, scrat2, msto, m2sto, norb, ierr)
    integer(int32), intent(in)    :: msto, m2sto, norb
    integer(int32), intent(out)   :: ierr
    real(real64),   intent(inout) :: onfock(m2sto)
    real(real64),   intent(out)   :: eval_all(msto), onevec(msto, norb)
    real(real64),   intent(inout) :: scrat2(msto, 9)

    real(real64),   allocatable :: a_full(:,:), work(:)
    integer(int32)              :: i, j, k, lwork

    allocate(a_full(msto, msto))
    lwork = max(1, 3 * msto)
    allocate(work(lwork))

    ! Unpack packed upper triangular matrix onfock into full symmetric a_full
    k = 0
    do j = 1, msto
      do i = 1, j
        k = k + 1
        a_full(i, j) = onfock(k)
        a_full(j, i) = onfock(k)
      end do
    end do

    call DSYEV('V', 'U', msto, a_full, msto, eval_all, work, lwork, ierr)

    if (ierr == 0) then
      onevec(:, 1:norb) = a_full(:, 1:norb)
    else
      ! Fallback to DSPEV if needed
      call DSPEV('V', 'U', msto, onfock, eval_all, a_full, msto, work, ierr)
      onevec(:, 1:norb) = a_full(:, 1:norb)
    end if

    deallocate(a_full, work)
  end subroutine solve_orthonormal_fock

  !---------------------------------------------------------------------
  ! SMEAR_ATOMIC_OCCUPATIONS: replace the occupation numbers of one spin
  ! channel by Fermi-Dirac occupations f_i = 1 / (1 + exp((eps_i - mu)/w)),
  ! with the chemical potential mu chosen (bisection) so that the channel keeps
  ! its electron count sum(occ). Occupation numbers are per spin orbital, so
  ! 0 <= f_i <= 1 for both the open- and closed-shell cases.
  !---------------------------------------------------------------------
  subroutine smear_atomic_occupations(eps, occ, norb, width)
    integer(int32), intent(in)    :: norb
    real(real64),   intent(in)    :: eps(norb), width
    real(real64),   intent(inout) :: occ(norb)

    real(real64)   :: nel, mu_lo, mu_hi, mu_mid, fsum
    integer(int32) :: ib, io

    nel   = sum(occ(1:norb))
    mu_lo = minval(eps(1:norb)) - 5.0d0
    mu_hi = maxval(eps(1:norb)) + 5.0d0
    mu_mid = 0.5d0 * (mu_lo + mu_hi)
    do ib = 1, 200
      mu_mid = 0.5d0 * (mu_lo + mu_hi)
      fsum = 0.0d0
      do io = 1, norb
        fsum = fsum + fermi_factor(eps(io), mu_mid, width)
      end do
      if (fsum > nel) then
        mu_hi = mu_mid
      else
        mu_lo = mu_mid
      end if
    end do
    do io = 1, norb
      occ(io) = fermi_factor(eps(io), mu_mid, width)
    end do
  end subroutine smear_atomic_occupations

  pure function fermi_factor(e, mu, width) result(f)
    real(real64), intent(in) :: e, mu, width
    real(real64)             :: f
    f = 1.0d0 / (1.0d0 + exp(max(-200.0d0, min(200.0d0, (e - mu) / width))))
  end function fermi_factor

  !---------------------------------------------------------------------
  ! EFFECTIVE_ATOMIC_SMEARING:
  ! Determines the effective Fermi-Dirac smearing width w_eff (in Hartree)
  ! for one spin channel of an atom:
  !   1. If SMEARW > 0 (explicit user setting), returns SMEARW.
  !   2. If SMEARW < 0 (explicitly disabled), QFIRST (Cycle 1 separated
  !      atoms), QNWVEC (explicit hybrid / valence_config seed),
  !      .not. QOPEN (RHF half-occupied open shells), QGFERM (global
  !      molecular Fermi smearing), or CDAMP < 1.0 (interatomic BLOOP
  !      Fermi equilibration active), returns 0.0d0.
  !   3. Otherwise (SMEARW == 0.0d0, automatic mode on unseeded UHF atoms
  !      with integer spin-channel electron count):
  !      Combines:
  !      (a) Early-cycle annealing on Cycles 2..4:
  !          w_anneal = 0.02 * 0.5**(NCYCL - 2) when 1 <= n_occ < nact_orb
  !          and the HOMO-LUMO gap is within an open p/d subshell (< 0.15 Ha),
  !      (b) HOMO-LUMO gap-triggered floor:
  !          w_gap = max(1e-4, 0.5 * (1e-3 - gap_hl)) whenever gap_hl < 1e-3 Ha
  !          (genuine frontier orbital degeneracy),
  !      (c) T -> 0+ snap (1e-4 Ha) once annealing ends and gap_hl >= 1e-3 Ha
  !          if occ still holds fractional values from earlier smearing cycles.
  !---------------------------------------------------------------------
  function effective_atomic_smearing(eps, occ, norb, nact_orb, smearw_in, &
                                     cdamp_in, ncycl_in, qfirst_in,       &
                                     qnwvec_in, qopen_in, qgferm_in) result(w_eff)
    integer(int32), intent(in) :: norb, nact_orb, ncycl_in
    real(real64),   intent(in) :: eps(norb), occ(norb), smearw_in, cdamp_in
    logical,        intent(in) :: qfirst_in, qnwvec_in, qopen_in, qgferm_in
    real(real64)               :: w_eff

    real(real64), parameter :: W_ANNEAL_INIT  = 0.02d0
    real(real64), parameter :: GAP_MIN        = 1.0d-3
    real(real64), parameter :: GAP_SUBSHELL   = 0.15d0
    real(real64), parameter :: W_SNAP         = 1.0d-4
    integer(int32), parameter :: N_ANNEAL_MAX = 4

    real(real64)   :: nel, gap_hl, w_anneal, w_gap
    integer(int32) :: n_occ, io
    logical        :: has_frac

    if (smearw_in > 0.0d0) then
      w_eff = smearw_in
      return
    end if

    w_eff = 0.0d0
    if (smearw_in < 0.0d0 .or. qfirst_in .or. qnwvec_in .or. (.not. qopen_in) .or. qgferm_in) return

    nel   = sum(occ(1:norb))
    n_occ = nint(nel)
    if (abs(nel - real(n_occ, real64)) > 1.0d-4) return
    if (n_occ < 1 .or. n_occ >= nact_orb) return

    ! If CDAMP < 1.0 and occ already holds fractional populations from interatomic
    ! BLOOP equilibration (on cycles after the initial annealing window), do not
    ! overwrite the BLOOP-equilibrated occupations with intra-atomic Fermi smearing.
    if (cdamp_in < 1.0d0 - 1.0d-6 .and. ncycl_in > N_ANNEAL_MAX + 1) then
      do io = 1, norb
        if (occ(io) > 1.0d-6 .and. occ(io) < 1.0d0 - 1.0d-6) return
      end do
    end if

    gap_hl = max(0.0d0, eps(n_occ + 1) - eps(n_occ))

    w_anneal = 0.0d0
    if (ncycl_in >= 2 .and. ncycl_in <= N_ANNEAL_MAX .and. gap_hl < GAP_SUBSHELL) then
      w_anneal = W_ANNEAL_INIT * (0.5d0 ** (ncycl_in - 2))
    end if

    w_gap = 0.0d0
    if (gap_hl < GAP_MIN) then
      w_gap = max(W_SNAP, 0.5d0 * (GAP_MIN - gap_hl))
    end if

    w_eff = max(w_anneal, w_gap)

    if (w_eff <= 0.0d0 .and. ncycl_in == N_ANNEAL_MAX + 1) then
      has_frac = .false.
      do io = 1, norb
        if (occ(io) > 1.0d-6 .and. occ(io) < 1.0d0 - 1.0d-6) then
          has_frac = .true.
          exit
        end if
      end do
      if (has_frac) w_eff = W_SNAP
    end if
  end function effective_atomic_smearing

  !---------------------------------------------------------------------
  ! ORTHONORMALIZE_PROJECTED_ORBITALS:
  ! After PK back-projection onevec = (I - S_A^{-1} P_B) * stevec,
  ! orthonormalize the occupied orbitals (occ > 0) among themselves via
  ! symmetric Löwdin (ITYPE = 2), then project the occupied subspace out
  ! of any unoccupied orbitals (occ == 0) in the S_A metric and
  ! orthonormalize the unoccupied orbitals among themselves.
  ! This prevents Pauli-annihilated unoccupied virtual residuals from
  ! rotating into the occupied valence/core orbitals.
  !---------------------------------------------------------------------
  subroutine orthonormalize_projected_orbitals(onevec, stevec_out, stovl_s, &
                                               occ_col, msto, m2sto, norb, qindef)
    integer(int32),  intent(in)    :: msto, m2sto, norb
    real(real64),    intent(in)    :: onevec(msto, norb), stovl_s(m2sto), occ_col(norb)
    real(real64),    intent(inout) :: stevec_out(msto, norb)
    logical(kind=1), intent(out)   :: qindef

    integer(int32) :: iorb, uorb, i, k, ii, ik, nocc, nunocc
    integer(int32), allocatable :: idx_occ(:), idx_unocc(:)
    real(real64),   allocatable :: vec_occ(:,:), vec_unocc(:,:),               &
                                   s_sub(:), u_occ(:,:), u_unocc(:,:),         &
                                   scrat(:,:), s_vec(:)
    logical(kind=1) :: qindef_u
    real(real64)    :: ovl_ou, sum_val

    allocate(idx_occ(norb), idx_unocc(norb))
    nocc   = 0
    nunocc = 0
    do iorb = 1, norb
      if (occ_col(iorb) > 1.0d-6) then
        nocc = nocc + 1
        idx_occ(nocc) = iorb
      else
        nunocc = nunocc + 1
        idx_unocc(nunocc) = iorb
      end if
    end do

    allocate(s_sub(m2sto), scrat(msto, norb))

    if (nocc == norb .or. nocc == 0) then
      allocate(u_occ(norb, norb))
      call CTSC(onevec, stovl_s, scrat, s_sub, msto, norb)
      call run_lowdin(s_sub, u_occ, scrat, norb, norb, 2, qindef)
      if (.not. qindef) then
        stevec_out(:, 1:norb) = matmul(onevec(:, 1:norb), u_occ)
      end if
      deallocate(u_occ)
    else
      allocate(vec_occ(msto, nocc), vec_unocc(msto, nunocc), &
               u_occ(nocc, nocc), u_unocc(nunocc, nunocc), s_vec(msto))
      do iorb = 1, nocc
        vec_occ(:, iorb) = onevec(:, idx_occ(iorb))
      end do
      do uorb = 1, nunocc
        vec_unocc(:, uorb) = onevec(:, idx_unocc(uorb))
      end do

      call CTSC(vec_occ, stovl_s, scrat, s_sub, msto, nocc)
      call run_lowdin(s_sub, u_occ, scrat, nocc, nocc, 2, qindef)
      if (.not. qindef) then
        vec_occ = matmul(vec_occ, u_occ)

        ! Project occupied subspace out of unoccupied orbitals in S_A metric
        do iorb = 1, nocc
          do i = 1, msto
            ii = (i * (i - 1)) / 2
            sum_val = 0.0d0
            do k = 1, msto
              ik = ii + k
              if (k > i) ik = (k * (k - 1)) / 2 + i
              sum_val = sum_val + stovl_s(ik) * vec_occ(k, iorb)
            end do
            s_vec(i) = sum_val
          end do
          do uorb = 1, nunocc
            ovl_ou = dot_product(s_vec, vec_unocc(:, uorb))
            vec_unocc(:, uorb) = vec_unocc(:, uorb) - ovl_ou * vec_occ(:, iorb)
          end do
        end do

        call CTSC(vec_unocc, stovl_s, scrat, s_sub, msto, nunocc)
        call run_lowdin(s_sub, u_unocc, scrat, nunocc, nunocc, 2, qindef_u)
        if (.not. qindef_u) then
          vec_unocc = matmul(vec_unocc, u_unocc)
        else
          call NORMLZ(vec_unocc, nunocc, stovl_s, msto, m2sto)
        end if

        do iorb = 1, nocc
          stevec_out(:, idx_occ(iorb)) = vec_occ(:, iorb)
        end do
        do uorb = 1, nunocc
          stevec_out(:, idx_unocc(uorb)) = vec_unocc(:, uorb)
        end do
      end if
      deallocate(vec_occ, vec_unocc, u_occ, u_unocc, s_vec)
    end if

    deallocate(idx_occ, idx_unocc, s_sub, scrat)
  end subroutine orthonormalize_projected_orbitals

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
    integer(int32)   :: jdifat, nucz, iunitx, neqat, isp, iorb, jorb, ijorb, nact_orb
    integer(int32)   :: nv, niter, ij, iterm1, ierr, ieqat, isym, ixyz, matom
    real(real64)     :: x, y, z, toten, zzrep, rmAx, tke, te, endiff, oldte
    real(real64)     :: oldiff, tcoul0, zzrep0, w_eff
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
        if (.not. QFIRST .and. .not. QNWVEC) then
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
          pkpot_bak(:, 1:2, isp) = pkpot(:, 1:2, isp)
          ! Hydrogen exemption (thesis Eq. III-24): H 1s is valence-like, so the
          ! energy-weighted projector is negligible; the unweighted pseudometric
          ! would unphysically remove the bonding orbital -> M = S for H.
          if (QPSMET .and. nucz > 1.5d0) then
            call regularize_pk_projector(stovl_bak, pkpot(1,1,isp), pkpot(1,2,isp), MSTO, M2STO, &
                                         merge(0.42d0, 1.0d-3, logical(QNWVEC)))
            stovl = stovl_bak - pkpot(:, 1, isp)
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
        qdone  = (QFIXED .or. (MXITER <= 0) .or. (QFIRST .and. QSEPAT .and. QNWVEC))

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

          if (qdone) then
            if (QFIRST .and. QSEPAT .and. QNWVEC) then
              if (maxval(abs(eval(1:NORB, 1) - eval(1:NORB, ISPIN))) <= 1.0d-4) then
                do isp = 1, ISPIN
                  call CTSC(stevec(1,1,isp), stfock(1,isp), scrat1, aofock(1,isp), MSTO, NORB)
                  do iorb = 1, NORB
                    eval(iorb, isp) = aofock((iorb * (iorb + 1)) / 2, isp)
                  end do
                end do
              end if
            end if
            exit scf_loop
          end if

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
            if (nucz > 1.5d0) stfock(:, isp) = stfock(:, isp) - pkpot(:, 2, isp)  ! H exempt: no PK projection
            call UTHU(MSTO, stfock(1,isp), tranx(1,1,isp), scrat1, onfock)
            call solve_orthonormal_fock(onfock, scrat1, onevec, scrat2, &
                                        MSTO, M2STO, NORB, ierr)
            if (ierr < 0) then
              call BOMB(17)
            else if (ierr > 0) then
              call WARN(15, qdummy)
            end if

            eval(1:NORB, isp) = scrat1(1:NORB, 1, 1)
            ! Identify active core+valence orbital subspace (1..nact_orb)
            ! excluding trailing purely virtual polarization slots (occ == 0 in both spins).
            nact_orb = 0
            do iorb = 1, NORB
              if (sum(occ(iorb, 1:ISPIN)) > 1.0d-6) nact_orb = iorb
            end do
            if (nact_orb <= 0) nact_orb = NORB

            ! Fermi-Dirac atomic occupations: either explicit (SMEARW > 0) or
            ! automatic early-cycle annealing + HOMO-LUMO gap floor on unseeded atoms.
            ! Restrict automatic smearing to the active valence subspace (1..nact_orb)
            ! so virtual polarization shells remain strictly unoccupied.
            w_eff = effective_atomic_smearing(eval(1, isp), occ(1, isp), NORB, nact_orb, &
                                              SMEARW, CDAMP, NCYCL, logical(QFIRST),     &
                                              logical(QNWVEC), logical(QOPEN), logical(QGFERM))
            if (w_eff > 0.0d0) then
              if (SMEARW > 0.0d0) then
                call smear_atomic_occupations(eval(1, isp), occ(1, isp), NORB, w_eff)
              else
                call smear_atomic_occupations(eval(1, isp), occ(1, isp), nact_orb, w_eff)
                if (nact_orb < NORB) occ(nact_orb + 1:NORB, isp) = 0.0d0
              end if
            end if
            oldvec(:, :, isp) = stevec(:, :, isp)

            call DMPAB(tranx(1,1,isp), MSTO, MSTO, MSTO, MSTO, onevec, MSTO, &
                       NORB, MSTO, NORB, stevec(1,1,isp), MSTO, NORB)

            if (QNWVEC) then
              if (QPSMET .and. .not. qtrunc .and. nucz > 1.5d0) then
                call DMPATB(projec(1,1,isp), MSTO, MSTO, MSTO, MSTO,           &
                            stevec(1,1,isp), MSTO, NORB, MSTO, NORB,           &
                            onevec, MSTO, NORB)
                stevec(:, 1:NORB, isp) = onevec(:, 1:NORB)
              end if

              ! Normalize stevec in S_A metric before MAXOVL so Bessel's inequality
              ! (sum_j |<old_i | S_A | new_j>|^2 <= 1) holds in MAXOVL.
              onevec(:, 1:NORB) = stevec(:, 1:NORB, isp)
              call NORMLZ(onevec, NORB, stovl_bak, MSTO, M2STO)
              iorder(1:NORB, isp) = [(iorb, iorb = 1, NORB)]
              call MAXOVL(iorder(1,isp), oldvec(1,1,isp), onevec,              &
                          stovl_bak, MSTO, M2STO, nact_orb, qnword)
              ! With smeared occupations the occupation follows the energy ordering,
              ! so no overlap-based reordering of occupation numbers is wanted.
              if (qnword .and. w_eff <= 0.0d0) then
                if (any(abs(occ(1:nact_orb, isp) - occ(iorder(1:nact_orb, isp), isp)) > 1.0d-8)) then
                  call NEWORD(occ(1,isp), scrat1, iorder(1:nact_orb, isp), nact_orb)
                else
                  qnword = .false.
                end if
              end if

              if (QPSMET .and. .not. qtrunc .and. nucz > 1.5d0) then
                onevec(:, 1:NORB) = stevec(:, 1:NORB, isp)
                call orthonormalize_projected_orbitals(onevec, stevec(1,1,isp), &
                                                       stovl_bak, occ(1,isp),   &
                                                       MSTO, M2STO, nact_orb, qindef)
                stovl  = stovl_bak
                ontran = ontran_bak
              end if

              if (qnrmlz) then
                stovl  = stovl_bak
                ontran = ontran_bak
                call NORMLZ(stevec(1,1,isp), NORB, stovl, MSTO, M2STO)
              end if

              if (nucz > 1.5d0) stfock(:, isp) = stfock(:, isp) + pkpot(:, 2, isp)  ! H exempt: no PK projection
            else
              if (QPSMET .and. .not. qtrunc) then
                if (nucz > 1.5d0) then
                  call DMPATB(projec(1,1,isp), MSTO, MSTO, MSTO, MSTO,         &
                              stevec(1,1,isp), MSTO, NORB, MSTO, NORB,         &
                              onevec, MSTO, NORB)
                  call CTSC(onevec, stovl, scrat1, stovl, MSTO, NORB)
                  call run_lowdin(stovl, ontran, scrat1, NORB, NORB, 2, qindef)
                  if (.not. qindef) then
                    call DMPAB(onevec, MSTO, NORB, MSTO, NORB, ontran, NORB,   &
                               NORB, NORB, NORB, stevec(1,1,isp), MSTO, NORB)
                    stovl  = stovl_bak
                    ontran = ontran_bak
                  end if
                else
                  stevec(:, 1:NORB, isp) = 0.0d0
                end if
              end if

              if (qnrmlz) then
                stovl  = stovl_bak
                ontran = ontran_bak
                call NORMLZ(stevec(1,1,isp), NORB, stovl, MSTO, M2STO)
              end if

              if (nucz > 1.5d0) stfock(:, isp) = stfock(:, isp) + pkpot(:, 2, isp)  ! H exempt: no PK projection
              iorder(1:NORB, isp) = [(iorb, iorb = 1, NORB)]
              call MAXOVL(iorder(1,isp), oldvec(1,1,isp), stevec(1,1,isp),     &
                          stovl, MSTO, M2STO, nact_orb, qnword)
              ! With smeared occupations the occupation follows the energy ordering,
              ! so no overlap-based reordering of occupation numbers is wanted.
              if (qnword .and. w_eff <= 0.0d0) then
                if (any(abs(occ(1:nact_orb, isp) - occ(iorder(1:nact_orb, isp), isp)) > 1.0d-8)) then
                  call NEWORD(occ(1,isp), scrat1, iorder(1:nact_orb, isp), nact_orb)
                else
                  qnword = .false.
                end if
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
          if (QNWVEC) pkpot(:, 1:2, isp) = pkpot_bak(:, 1:2, isp)
          ijorb = 0
          do iorb = 1, NORB
            do jorb = 1, iorb
              ijorb = ijorb + 1
              aofock(ijorb, isp) = merge(gamma(iorb, isp), 0.0d0, jorb == iorb)
            end do
          end do

          call DENMAT(pkpot(1,4,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
          call DENMAT(pkpot(1,5,isp), stevec(1,1,isp), &
                      merge(occ(1:NORB,isp) * eval(1:NORB,isp), eval(1:NORB,isp), logical(QNWVEC)), MSTO, M2STO, NORB)
          ! Hydrogen exemption: H emits no projector onto neighboring atoms.
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
              call DENMAT(pkpot(1,5,isp), stevec(1,1,isp), &
                          merge(occ(1:NORB,isp) * eval(1:NORB,isp), eval(1:NORB,isp), logical(QNWVEC)), MSTO, M2STO, NORB)
              ! Hydrogen exemption: H emits no projector onto neighboring atoms.
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