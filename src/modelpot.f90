!=======================================================================
! PHATPSY 2.0: Model Potential & Normalization Kernels Module
! Module: modelpot (src/modelpot.f90)
! Modern Fortran 2018 replacement for legacy:
!   - genfac.f: Factorial array generation
!   - genbc.f:  Binomial coefficients table (Pascal's triangle)
!   - gennrm.f: Spherical harmonic normalization factors
!   - intnrm.f: Intermediate vector normalization <i|(1-P)|i> = 1
!   - pkepot.f: Phillips-Kleinman energy-dependent pseudopotential Fock matrix
!   - epkpot.f: Extended Phillips-Kleinman pseudopotential back-transformation
!   - dampv.f:  Model potential and pseudopotential matrix damping
!=======================================================================
module modelpot
  use iso_fortran_env, only: real64, int32
  use state,           only: MSTO, NORB, M2STO, ISPIN, NATOM, mol_ws, QDYNEX
  implicit none

  private

  public :: genfac, genbc, gennrm, gennrm_default, intnrm, pkepot, epkpot, epkpot_core, dampv, &
            regularize_pk_projector, scale_screening_exponents, update_dynamic_exponents, &
            eval_atomic_dipole, eval_multipole_field

  interface gennrm
    module procedure gennrm_default
    module procedure gennrm_kind1
  end interface gennrm

  interface
    subroutine trisq(n, h, u, result)
      import :: real64, int32
      integer(int32), intent(in)  :: n
      real(real64),   intent(in)  :: h(*)
      real(real64),   intent(in)  :: u(n, n)
      real(real64),   intent(out) :: result(n, n)
    end subroutine trisq
  end interface

contains

  !---------------------------------------------------------------------
  ! GENFAC: Generate factorial table up to nfact:
  !   fact(1)   = 1.0 (0-factorial)
  !   fact(n+1) = fact(n) * n
  !---------------------------------------------------------------------
  subroutine genfac(fact, nfact)
    integer(int32), intent(in)  :: nfact
    real(real64),   intent(out) :: fact(nfact)

    integer(int32) :: n, nmax

    fact(1) = 1.0d0
    nmax = nfact - 1
    if (nmax < 1) return
    do n = 1, nmax
      fact(n + 1) = fact(n) * real(n, real64)
    end do
  end subroutine genfac

  !---------------------------------------------------------------------
  ! GENBC: Generate binomial coefficients up to specified order using
  ! Pascal's triangle.
  !   binom(index(n) + m) = (n-1)! / ((m-1)! * (n-m)!)
  !---------------------------------------------------------------------
  subroutine genbc(binom, nbinom, nrows, index_arr)
    integer(int32), intent(in)  :: nbinom, nrows
    integer(int32), intent(in)  :: index_arr(nrows)
    real(real64),   intent(out) :: binom(nbinom)

    integer(int32) :: n, nn, nm1, m, nm

    binom(1) = 1.0d0
    if (nrows < 2) return
    binom(2) = 1.0d0
    binom(3) = 1.0d0
    if (nrows < 3) return
    do n = 3, nrows
      nn = index_arr(n)
      nm1 = n - 1
      binom(nn + 1) = 1.0d0
      do m = 2, nm1
        nm = nn + m
        binom(nm) = binom(nm - n) + binom(nm - n + 1)
      end do
      binom(nn + n) = 1.0d0
    end do
  end subroutine genbc

  !---------------------------------------------------------------------
  ! GENNRM: Generate normalization constants for spherical harmonics
  ! up to maximum l-value:
  !   plmnrm(lm) = sqrt((2*l + 1) * (l - m)! / (area * (l + m)!))
  !   where area = 2*pi (or 4*pi for m=0 or complex harmonics)
  !---------------------------------------------------------------------
  subroutine gennrm_default(plmnrm, fact, nplm, nfact, lmax, qrealy)
    integer(int32), intent(in)  :: nplm, nfact, lmax
    logical,        intent(in)  :: qrealy
    real(real64),   intent(in)  :: fact(nfact)
    real(real64),   intent(out) :: plmnrm(nplm)

    integer(int32) :: il, im, lmxp1, ll, m, lm
    real(real64)   :: pi, area

    pi = 2.0d0 * atan2(1.0d0, 0.0d0)
    lmxp1 = lmax + 1
    do il = 1, lmxp1
      ll = (il * (il - 1)) / 2
      do im = 1, il
        m = im - 1
        lm = ll + im
        area = 2.0d0 * pi
        if (.not. qrealy .or. (m == 0)) area = 2.0d0 * area
        plmnrm(lm) = sqrt(real(2 * il - 1, real64) * fact(il - m) / (area * fact(il + m)))
      end do
    end do
  end subroutine gennrm_default

  subroutine gennrm_kind1(plmnrm, fact, nplm, nfact, lmax, qrealy)
    integer(int32),  intent(in)  :: nplm, nfact, lmax
    logical(kind=1), intent(in)  :: qrealy
    real(real64),    intent(in)  :: fact(nfact)
    real(real64),    intent(out) :: plmnrm(nplm)

    call gennrm_default(plmnrm, fact, nplm, nfact, lmax, logical(qrealy))
  end subroutine gennrm_kind1

  !---------------------------------------------------------------------
  ! INTNRM: Intermediate normalization of eigenvectors:
  !   <i | (1 - P) | i> = 1
  ! where P is a projection operator in packed form.
  !---------------------------------------------------------------------
  subroutine intnrm(evec, projec, scrat, nbas, nvec, n2bas)
    integer(int32), intent(in)    :: nbas, nvec, n2bas
    real(real64),   intent(in)    :: projec(n2bas)
    real(real64),   intent(inout) :: evec(nbas, nvec)
    real(real64),   intent(out)   :: scrat(nbas, nbas)

    integer(int32) :: n, i, ii, j, ij
    real(real64)   :: sum_val, vnorm

    do n = 1, nvec
      do i = 1, nbas
        ii = (i * (i - 1)) / 2
        sum_val = 0.0d0
        do j = 1, nbas
          if (j <= i) then
            ij = ii + j
          else
            ij = i + (j * (j - 1)) / 2
          end if
          sum_val = sum_val + projec(ij) * evec(j, n)
        end do
        scrat(i, n) = sum_val
      end do
    end do

    do n = 1, nvec
      sum_val = 1.0d0
      do i = 1, nbas
        sum_val = sum_val - evec(i, n) * scrat(i, n)
      end do
      vnorm = 1.0d0 / sqrt(sum_val)
      do i = 1, nbas
        evec(i, n) = evec(i, n) * vnorm
      end do
    end do
  end subroutine intnrm

  !---------------------------------------------------------------------
  ! PKEPOT: Phillips-Kleinman energy-dependent pseudopotential Fock matrix
  !   STFOCK = STFOCK + 0.5 * (P S^-1 F + F S^-1 P)
  ! expanded through second order in overlap and Hermitian averaged.
  !---------------------------------------------------------------------
  subroutine pkepot(pkpot, stfock, ontran, scrat1, scrat2, msto, m2sto)
    integer(int32), intent(in)    :: msto, m2sto
    real(real64),   intent(in)    :: pkpot(m2sto)
    real(real64),   intent(inout) :: stfock(m2sto)
    real(real64),   intent(in)    :: ontran(msto, msto)
    real(real64),   intent(out)   :: scrat1(msto, msto), scrat2(msto, msto)

    integer(int32) :: i, j, k, ij
    real(real64)   :: sum_val

    call trisq(msto, stfock, ontran, scrat1)
    call trisq(msto, pkpot, ontran, scrat2)

    ij = 0
    do i = 1, msto
      do j = 1, i
        ij = ij + 1
        sum_val = 0.0d0
        do k = 1, msto
          sum_val = sum_val + scrat1(i, k) * scrat2(j, k) + scrat2(i, k) * scrat1(j, k)
        end do
        stfock(ij) = stfock(ij) + sum_val * 0.5d0
      end do
    end do
  end subroutine pkepot

  !---------------------------------------------------------------------
  ! EPKPOT_CORE: Extended Phillips-Kleinman pseudopotential back-transformation
  ! from SCF eigenvalues and eigenvectors of the previous cycle.
  !---------------------------------------------------------------------
  subroutine epkpot_core(pkpot, evec, eval, stovl, scrat1, scrat2, msto_in, m2sto_in, norb_in, ispin_in)
    integer(int32), intent(in)    :: msto_in, m2sto_in, norb_in, ispin_in
    real(real64),   intent(inout) :: pkpot(m2sto_in, 4, ispin_in)
    real(real64),   intent(in)    :: evec(msto_in, norb_in, ispin_in)
    real(real64),   intent(in)    :: eval(norb_in, ispin_in)
    real(real64),   intent(in)    :: stovl(m2sto_in)
    real(real64),   intent(out)   :: scrat1(msto_in, msto_in)
    real(real64),   intent(out)   :: scrat2(msto_in, msto_in)

    integer(int32) :: isp, isto, jorb, ksto, iorb, korb, jsto
    integer(int32) :: iksto, indxi, indxj, ijsto, jksto, jkmx, jkmn
    real(real64)   :: sumk

    do isp = 1, ispin_in
      do isto = 1, msto_in
        indxi = (isto * (isto - 1)) / 2
        do jorb = 1, norb_in
          sumk = 0.0d0
          do ksto = 1, msto_in
            if (ksto <= isto) then
              iksto = indxi + ksto
            else
              iksto = (ksto * (ksto - 1)) / 2 + isto
            end if
            sumk = sumk + evec(ksto, jorb, isp) * pkpot(iksto, 1, isp)
          end do
          scrat1(isto, jorb) = sumk
        end do
      end do

      do iorb = 1, norb_in
        sumk = 0.0d0
        do ksto = 1, msto_in
          sumk = sumk + evec(ksto, iorb, isp) * scrat1(ksto, iorb)
        end do
        scrat2(iorb, iorb) = eval(iorb, isp) * sumk
      end do

      do isto = 1, msto_in
        do jsto = 1, isto
          sumk = 0.0d0
          do korb = 1, norb_in
            sumk = sumk + scrat2(korb, korb) * evec(isto, korb, isp) * evec(jsto, korb, isp)
          end do
          scrat1(isto, jsto) = sumk
        end do
      end do

      do isto = 1, msto_in
        indxi = (isto * (isto - 1)) / 2
        do jsto = 1, msto_in
          sumk = 0.0d0
          do ksto = 1, msto_in
            jkmx = max(jsto, ksto)
            jkmn = min(jsto, ksto)
            if (ksto <= isto) then
              iksto = indxi + ksto
            else
              iksto = (ksto * (ksto - 1)) / 2 + isto
            end if
            sumk = sumk + stovl(iksto) * scrat1(jkmx, jkmn)
          end do
          scrat2(isto, jsto) = sumk
        end do
      end do

      do isto = 1, msto_in
        indxi = (isto * (isto - 1)) / 2
        do jsto = 1, isto
          indxj = (jsto * (jsto - 1)) / 2
          ijsto = indxi + jsto
          sumk = 0.0d0
          do ksto = 1, msto_in
            if (ksto <= jsto) then
              jksto = indxj + ksto
            else
              jksto = (ksto * (ksto - 1)) / 2 + jsto
            end if
            sumk = sumk + scrat2(isto, ksto) * stovl(jksto)
          end do
          pkpot(ijsto, 2, isp) = pkpot(ijsto, 2, isp) + sumk
        end do
      end do
    end do
  end subroutine epkpot_core

  !---------------------------------------------------------------------
  ! EPKPOT: Interface reading dimensions from state module
  !---------------------------------------------------------------------
  subroutine epkpot(pkpot, evec, eval, stovl, scrat1, scrat2, index_arr)
    real(real64),   intent(inout) :: pkpot(M2STO, 4, ISPIN)
    real(real64),   intent(in)    :: evec(MSTO, NORB, ISPIN)
    real(real64),   intent(in)    :: eval(NORB, ISPIN)
    real(real64),   intent(in)    :: stovl(M2STO)
    real(real64),   intent(out)   :: scrat1(MSTO, MSTO)
    real(real64),   intent(out)   :: scrat2(MSTO, MSTO)
    integer(int32), intent(in)    :: index_arr(*)

    call epkpot_core(pkpot, evec, eval, stovl, scrat1, scrat2, MSTO, M2STO, NORB, ISPIN)
  end subroutine epkpot

  !---------------------------------------------------------------------
  ! DAMPV: Damp model potential and pseudopotential matrix elements
  !---------------------------------------------------------------------
  subroutine dampv(epot2, pkpot, zpot2, vpot2, msto, m2sto, nvterm, ispin, vdamp)
    integer(int32), intent(in)    :: msto, m2sto, nvterm, ispin
    real(real64),   intent(in)    :: vdamp
    real(real64),   intent(inout) :: epot2(m2sto, ispin)
    real(real64),   intent(inout) :: pkpot(m2sto, 4, ispin)
    real(real64),   intent(inout) :: zpot2(m2sto)
    real(real64),   intent(inout) :: vpot2(nvterm)

    integer(int32) :: isp

    do isp = 1, ispin
      epot2(1:m2sto, isp) = vdamp * epot2(1:m2sto, isp)
      pkpot(1:m2sto, 1:2, isp) = vdamp * pkpot(1:m2sto, 1:2, isp)
    end do
    zpot2(1:m2sto) = vdamp * zpot2(1:m2sto)
    vpot2(1:nvterm) = vdamp * vpot2(1:nvterm)
  end subroutine dampv

  !---------------------------------------------------------------------
  ! REGULARIZE_PK_PROJECTOR: Canonical multi-center projection bounding
  ! Ensures the environment projector P <= (1 - delta) S in the Löwdin basis,
  ! bounding P so that M = S - P is strictly positive-definite.
  ! Simultaneously scales the energy projector B by the same mode factors.
  !---------------------------------------------------------------------
  subroutine regularize_pk_projector(stovl_packed, pkpot_p, pkpot_b, msto, m2sto, delta_in)
    integer(int32), intent(in)    :: msto, m2sto
    real(real64),   intent(in)    :: stovl_packed(m2sto)
    real(real64),   intent(inout) :: pkpot_p(m2sto)
    real(real64),   intent(inout) :: pkpot_b(m2sto)
    real(real64),   intent(in), optional :: delta_in

    real(real64) :: delta, max_lam, tol
    real(real64), allocatable :: s_mat(:,:), s_half(:,:), s_mhalf(:,:)
    real(real64), allocatable :: p_mat(:,:), b_mat(:,:)
    real(real64), allocatable :: p_lowdin(:,:), b_lowdin(:,:)
    real(real64), allocatable :: v_p(:,:), lam_p(:), s_modes(:)
    real(real64), allocatable :: b_vp(:,:), tmp1(:,:)
    real(real64), allocatable :: w_s(:), work(:)
    real(real64), allocatable :: ovl_copy(:)
    integer(int32) :: info, i, j, k, ij, lwork
    logical :: needs_reg

    delta = 1.0d-3
    if (present(delta_in)) delta = delta_in
    tol = 1.0d-10

    ! Quick check: if norm of P is negligible, exit immediately
    if (maxval(abs(pkpot_p(1:m2sto))) < tol) return

    allocate(ovl_copy(m2sto), w_s(msto), work(3 * msto))
    allocate(s_mat(msto, msto), s_half(msto, msto), s_mhalf(msto, msto))
    allocate(p_mat(msto, msto), b_mat(msto, msto))
    allocate(p_lowdin(msto, msto), b_lowdin(msto, msto))
    allocate(v_p(msto, msto), lam_p(msto), s_modes(msto))
    allocate(tmp1(msto, msto))

    ! 1. Diagonalize atomic overlap S_A via DSPEV
    ovl_copy(1:m2sto) = stovl_packed(1:m2sto)
    call DSPEV('V', 'U', msto, ovl_copy, w_s, s_mat, msto, work, info)
    deallocate(ovl_copy, work)
    if (info /= 0) goto 999

    ! Form S^(1/2) and S^(-1/2)
    s_half = 0.0d0
    s_mhalf = 0.0d0
    do k = 1, msto
      if (w_s(k) > tol) then
        do j = 1, msto
          do i = 1, msto
            s_half(i, j)  = s_half(i, j)  + s_mat(i, k) * sqrt(w_s(k)) * s_mat(j, k)
            s_mhalf(i, j) = s_mhalf(i, j) + s_mat(i, k) * (1.0d0 / sqrt(w_s(k))) * s_mat(j, k)
          end do
        end do
      end if
    end do

    ! 2. Unpack P and B from packed upper triangular
    ij = 0
    do i = 1, msto
      do j = 1, i
        ij = ij + 1
        p_mat(i, j) = pkpot_p(ij)
        p_mat(j, i) = pkpot_p(ij)
        b_mat(i, j) = pkpot_b(ij)
        b_mat(j, i) = pkpot_b(ij)
      end do
    end do

    ! 3. Transform P and B to orthonormal Löwdin basis:
    ! P_lowdin = S^(-1/2) * P * S^(-1/2)
    ! B_lowdin = S^(-1/2) * B * S^(-1/2)
    tmp1 = matmul(p_mat, s_mhalf)
    p_lowdin = matmul(s_mhalf, tmp1)

    tmp1 = matmul(b_mat, s_mhalf)
    b_lowdin = matmul(s_mhalf, tmp1)

    ! 4. Diagonalize P_lowdin via DSYEV
    lwork = max(1, 3 * msto - 1)
    allocate(work(lwork))
    v_p = p_lowdin
    call DSYEV('V', 'U', msto, v_p, msto, lam_p, work, lwork, info)
    deallocate(work)
    if (info /= 0) goto 999

    ! 5. Check if any eigenvalue exceeds physical bound (1 - delta)
    max_lam = maxval(lam_p)
    needs_reg = (max_lam > (1.0d0 - delta))

    if (needs_reg) then
      ! Calculate mode scaling factors s_k <= 1.0
      do k = 1, msto
        if (lam_p(k) > (1.0d0 - delta)) then
          s_modes(k) = (1.0d0 - delta) / lam_p(k)
        else
          s_modes(k) = 1.0d0
        end if
      end do

      ! Reconstruct regularized P in Löwdin basis:
      ! P_lowdin_reg = V_P * diag(s_k * lam_k) * V_P^T
      tmp1 = 0.0d0
      do j = 1, msto
        do i = 1, msto
          tmp1(i, j) = v_p(i, j) * (s_modes(j) * lam_p(j))
        end do
      end do
      p_lowdin = matmul(tmp1, transpose(v_p))

      ! Scale B in the V_P basis:
      ! B_vp = V_P^T * B_lowdin * V_P
      allocate(b_vp(msto, msto))
      tmp1 = matmul(b_lowdin, v_p)
      b_vp = matmul(transpose(v_p), tmp1)

      do j = 1, msto
        do i = 1, msto
          b_vp(i, j) = b_vp(i, j) * sqrt(s_modes(i) * s_modes(j))
        end do
      end do

      tmp1 = matmul(b_vp, transpose(v_p))
      b_lowdin = matmul(v_p, tmp1)
      deallocate(b_vp)

      ! Transform regularized P and B back to STO basis:
      ! P_reg = S^(1/2) * P_lowdin_reg * S^(1/2)
      ! B_reg = S^(1/2) * B_lowdin_reg * S^(1/2)
      tmp1 = matmul(p_lowdin, s_half)
      p_mat = matmul(s_half, tmp1)

      tmp1 = matmul(b_lowdin, s_half)
      b_mat = matmul(s_half, tmp1)

      ! Pack back into pkpot_p and pkpot_b
      ij = 0
      do i = 1, msto
        do j = 1, i
          ij = ij + 1
          pkpot_p(ij) = 0.5d0 * (p_mat(i, j) + p_mat(j, i))
          pkpot_b(ij) = 0.5d0 * (b_mat(i, j) + b_mat(j, i))
        end do
      end do
    end if

999 continue
    if (allocated(ovl_copy)) deallocate(ovl_copy)
    if (allocated(w_s)) deallocate(w_s)
    if (allocated(work)) deallocate(work)
    if (allocated(s_mat)) deallocate(s_mat)
    if (allocated(s_half)) deallocate(s_half)
    if (allocated(s_mhalf)) deallocate(s_mhalf)
    if (allocated(p_mat)) deallocate(p_mat)
    if (allocated(b_mat)) deallocate(b_mat)
    if (allocated(p_lowdin)) deallocate(p_lowdin)
    if (allocated(b_lowdin)) deallocate(b_lowdin)
    if (allocated(v_p)) deallocate(v_p)
    if (allocated(lam_p)) deallocate(lam_p)
    if (allocated(s_modes)) deallocate(s_modes)
    if (allocated(b_vp)) deallocate(b_vp)
    if (allocated(tmp1)) deallocate(tmp1)
  end subroutine regularize_pk_projector

  !---------------------------------------------------------------------
  ! SCALE_SCREENING_EXPONENTS: Computes charge-dependent model potential
  ! screening exponents:
  !   zeta_k(Z, q) = zeta_k^0 * (1 + gamma * q / Z)^(1/3)
  ! where q = Z - N is the net atomic charge.
  ! Exponents are bounded to prevent unphysical contraction/collapse.
  !---------------------------------------------------------------------
  subroutine scale_screening_exponents(vexp_base, vexp_scaled, nvterm, znuc, q_charge, gamma_param)
    integer(int32), intent(in)  :: nvterm
    real(real64),   intent(in)  :: vexp_base(nvterm)
    real(real64),   intent(out) :: vexp_scaled(nvterm)
    real(real64),   intent(in)  :: znuc, q_charge
    real(real64),   intent(in), optional :: gamma_param

    real(real64) :: gamma_val, zeff_ratio, scale_factor
    integer(int32) :: k

    gamma_val = 1.0d0
    if (present(gamma_param)) then
      if (gamma_param > 0.0d0) gamma_val = gamma_param
    end if

    vexp_scaled = vexp_base
    if (nvterm <= 1) return

    if (znuc > 0.0d0) then
      zeff_ratio = 1.0d0 + gamma_val * (q_charge / znuc)
      ! Numerical bounds: ratio in [0.5, 2.0] keeps scale_factor in [0.7937, 1.2599]
      zeff_ratio = max(0.5d0, min(2.0d0, zeff_ratio))
      scale_factor = zeff_ratio**(1.0d0 / 3.0d0)
    else
      scale_factor = 1.0d0
    end if

    vexp_scaled(1) = 0.0d0 ! Coulomb -Z*/r term is never scaled
    do k = 2, nvterm
      vexp_scaled(k) = vexp_base(k) * scale_factor
    end do
  end subroutine scale_screening_exponents

  !---------------------------------------------------------------------
  ! UPDATE_DYNAMIC_EXPONENTS: Iterates over all active elements and adapts
  ! their model potential screening exponents vexp based on net atomic
  ! charges q_A = Z_A - N_A if QDYNEX = .true.
  !---------------------------------------------------------------------
  subroutine update_dynamic_exponents(gamma_param)
    real(real64), intent(in), optional :: gamma_param
    real(real64) :: gamma_val, znuc, q_net
    integer(int32) :: ielem, nvterm, iat, nat_elem, nucz

    gamma_val = 1.0d0
    if (present(gamma_param)) then
      if (gamma_param > 0.0d0) gamma_val = gamma_param
    end if

    if (.not. QDYNEX) return
    if (.not. allocated(mol_ws%elem_stream)) return
    if (.not. allocated(mol_ws%charge)) return

    do ielem = 1, size(mol_ws%elem_stream)
      nvterm = mol_ws%elem_stream(ielem)%nvterm
      if (nvterm <= 1) cycle
      if (.not. allocated(mol_ws%elem_stream(ielem)%vexp_base)) cycle

      nucz = mol_ws%elem_stream(ielem)%nucz
      znuc = real(nucz, real64)
      if (znuc <= 0.0d0) cycle

      ! Compute average net charge for atoms of this element
      q_net = 0.0d0
      nat_elem = 0
      do iat = 1, NATOM
        if (nint(mol_ws%charge(1, iat)) == nucz) then
          q_net = q_net + (mol_ws%charge(1, iat) - mol_ws%charge(2, iat))
          nat_elem = nat_elem + 1
        end if
      end do
      if (nat_elem > 0) q_net = q_net / real(nat_elem, real64)

      call scale_screening_exponents(mol_ws%elem_stream(ielem)%vexp_base, &
                                    mol_ws%elem_stream(ielem)%vexp,      &
                                    nvterm, znuc, q_net, gamma_val)
    end do
  end subroutine update_dynamic_exponents
  !---------------------------------------------------------------------
  ! EVAL_ATOMIC_DIPOLE:
  ! Evaluates the intra-atomic dipole moment p = [px, py, pz] from the
  ! atomic orbital eigenvectors and occupations using one-center transition
  ! integrals between s <-> p and p <-> d Slater-Type Orbitals.
  !---------------------------------------------------------------------
  subroutine eval_atomic_dipole(stevec, occ, n, l, ml, eta, anorm, msto, norb, ispin, p_dipole)
    integer(int32), intent(in)  :: msto, norb, ispin
    integer(int32), intent(in)  :: n(msto), l(msto), ml(msto)
    real(real64),   intent(in)  :: stevec(msto, norb, ispin)
    real(real64),   intent(in)  :: occ(norb, ispin)
    real(real64),   intent(in)  :: eta(msto), anorm(msto)
    real(real64),   intent(out) :: p_dipole(3)

    real(real64), allocatable :: p_mat(:,:), dx_mat(:,:), dy_mat(:,:), dz_mat(:,:)
    real(real64) :: mult, r_int, fact_term, eta_sum
    integer(int32) :: isp, iorb, i, j, ni, li, mi, nj, lj, mj, n_rad
    real(real64), parameter :: r_sqrt3 = 0.5773502691896257d0  ! 1 / sqrt(3)
    real(real64), parameter :: r_sqrt5 = 0.4472135954999579d0  ! 1 / sqrt(5)
    real(real64), parameter :: r_2sqrt15 = 0.5163977794943222d0 ! 2 / sqrt(15)

    p_dipole = 0.0d0
    if (msto <= 0 .or. norb <= 0) return

    allocate(p_mat(msto, msto), dx_mat(msto, msto), dy_mat(msto, msto), dz_mat(msto, msto))
    p_mat = 0.0d0
    dx_mat = 0.0d0
    dy_mat = 0.0d0
    dz_mat = 0.0d0

    ! 1. Construct atomic density matrix P_ij
    mult = merge(1.0d0, 2.0d0, ispin == 2)
    do isp = 1, ispin
      do iorb = 1, norb
        if (occ(iorb, isp) <= 0.0d0) cycle
        do i = 1, msto
          do j = 1, msto
            p_mat(i, j) = p_mat(i, j) + mult * occ(iorb, isp) * stevec(i, iorb, isp) * stevec(j, iorb, isp)
          end do
        end do
      end do
    end do

    ! 2. Evaluate analytical dipole transition matrices between s <-> p and p <-> d
    do i = 1, msto
      ni = n(i); li = l(i); mi = ml(i)
      do j = 1, msto
        nj = n(j); lj = l(j); mj = ml(j)
        if (abs(li - lj) /= 1) cycle

        n_rad = ni + nj + 1
        eta_sum = eta(i) + eta(j)
        if (eta_sum <= 0.0d0) cycle

        fact_term = 1.0d0
        do iorb = 1, n_rad
          fact_term = fact_term * real(iorb, real64)
        end do
        r_int = anorm(i) * anorm(j) * fact_term / (eta_sum ** (n_rad + 1))

        ! s <-> p transitions:
        if ((li == 0 .and. lj == 1) .or. (li == 1 .and. lj == 0)) then
          if ((li == 1 .and. mi == 0) .or. (lj == 1 .and. mj == 0)) dz_mat(i, j) = r_int * r_sqrt3
          if ((li == 1 .and. mi == 1) .or. (lj == 1 .and. mj == 1)) dx_mat(i, j) = r_int * r_sqrt3
          if ((li == 1 .and. mi == -1) .or. (lj == 1 .and. mj == -1)) dy_mat(i, j) = r_int * r_sqrt3
        end if

        ! p <-> d transitions:
        if ((li == 1 .and. lj == 2) .or. (li == 2 .and. lj == 1)) then
          if ((li == 1 .and. mi == 0 .and. mj == 0) .or. (lj == 1 .and. mj == 0 .and. mi == 0)) then
            dz_mat(i, j) = r_int * r_2sqrt15
          end if
          if ((li == 1 .and. mi == 1 .and. mj == 1) .or. (lj == 1 .and. mj == 1 .and. mi == 1)) then
            dz_mat(i, j) = r_int * r_sqrt5
            dx_mat(i, j) = r_int * r_sqrt5
          end if
          if ((li == 1 .and. mi == -1 .and. mj == -1) .or. (lj == 1 .and. mj == -1 .and. mi == -1)) then
            dz_mat(i, j) = r_int * r_sqrt5
            dy_mat(i, j) = r_int * r_sqrt5
          end if
        end if
      end do
    end do

    ! 3. Electronic dipole p_elec = -sum(P_ij * D_ij)
    p_dipole(1) = -sum(p_mat * dx_mat)
    p_dipole(2) = -sum(p_mat * dy_mat)
    p_dipole(3) = -sum(p_mat * dz_mat)
    deallocate(p_mat, dx_mat, dy_mat, dz_mat)
  end subroutine eval_atomic_dipole

  !---------------------------------------------------------------------
  ! EVAL_MULTIPOLE_FIELD:
  ! Evaluates the interatomic electric dipole reaction field E_A and
  ! potential Phi_A at atom A from all other polarized atomic centers B.
  !---------------------------------------------------------------------
  subroutine eval_multipole_field(coord, atom_dipoles, natom, iatom, e_field, phi_dipole)
    integer(int32), intent(in)  :: natom, iatom
    real(real64),   intent(in)  :: coord(3, natom)
    real(real64),   intent(in)  :: atom_dipoles(3, natom)
    real(real64),   intent(out) :: e_field(3), phi_dipole

    real(real64) :: rab(3), r_dist, r_inv, r_inv3, r_hat(3), p_dot_r, f_damp, alpha_damp, ar
    integer(int32) :: ib

    e_field = 0.0d0
    phi_dipole = 0.0d0
    alpha_damp = 1.50d0

    do ib = 1, natom
      if (ib == iatom) cycle
      rab = coord(:, iatom) - coord(:, ib)
      r_dist = sqrt(sum(rab**2))
      if (r_dist < 1.0d-5) cycle

      r_inv = 1.0d0 / r_dist
      r_hat = rab * r_inv
      r_inv3 = r_inv**3

      ! Tang-Toennies damping function f_1(R)
      ar = alpha_damp * r_dist
      f_damp = max(0.0d0, 1.0d0 - exp(-ar) * (1.0d0 + ar))

      p_dot_r = dot_product(atom_dipoles(:, ib), r_hat)
      phi_dipole = phi_dipole + f_damp * p_dot_r * (r_inv**2)
      e_field = e_field + f_damp * r_inv3 * (3.0d0 * p_dot_r * r_hat - atom_dipoles(:, ib))
    end do
  end subroutine eval_multipole_field

end module modelpot

!=======================================================================
! Legacy External ABI Compatibility Wrappers
!=======================================================================

subroutine genfac(fact, nfact)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_genfac => genfac
  implicit none
  integer(int32), intent(in)  :: nfact
  real(real64),   intent(out) :: fact(nfact)
  call mod_genfac(fact, nfact)
end subroutine genfac

subroutine genbc(binom, nbinom, nrows, index_arr)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_genbc => genbc
  implicit none
  integer(int32), intent(in)  :: nbinom, nrows
  integer(int32), intent(in)  :: index_arr(nrows)
  real(real64),   intent(out) :: binom(nbinom)
  call mod_genbc(binom, nbinom, nrows, index_arr)
end subroutine genbc

subroutine gennrm(plmnrm, fact, nplm, nfact, lmax, qrealy)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_gennrm => gennrm
  implicit none
  integer(int32),  intent(in)  :: nplm, nfact, lmax
  logical(kind=1), intent(in)  :: qrealy
  real(real64),    intent(in)  :: fact(nfact)
  real(real64),    intent(out) :: plmnrm(nplm)
  call mod_gennrm(plmnrm, fact, nplm, nfact, lmax, qrealy)
end subroutine gennrm

subroutine intnrm(evec, projec, scrat, nbas, nvec, n2bas)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_intnrm => intnrm
  implicit none
  integer(int32), intent(in)    :: nbas, nvec, n2bas
  real(real64),   intent(in)    :: projec(n2bas)
  real(real64),   intent(inout) :: evec(nbas, nvec)
  real(real64),   intent(out)   :: scrat(nbas, nbas)
  call mod_intnrm(evec, projec, scrat, nbas, nvec, n2bas)
end subroutine intnrm

subroutine pkepot(pkpot, stfock, ontran, scrat1, scrat2, msto, m2sto)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_pkepot => pkepot
  implicit none
  integer(int32), intent(in)    :: msto, m2sto
  real(real64),   intent(in)    :: pkpot(m2sto)
  real(real64),   intent(inout) :: stfock(m2sto)
  real(real64),   intent(in)    :: ontran(msto, msto)
  real(real64),   intent(out)   :: scrat1(msto, msto), scrat2(msto, msto)
  call mod_pkepot(pkpot, stfock, ontran, scrat1, scrat2, msto, m2sto)
end subroutine pkepot

subroutine epkpot(pkpot, evec, eval, stovl, scrat1, scrat2, index_arr)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_epkpot => epkpot
  implicit none
  real(real64),   intent(inout) :: pkpot(*)
  real(real64),   intent(in)    :: evec(*)
  real(real64),   intent(in)    :: eval(*)
  real(real64),   intent(in)    :: stovl(*)
  real(real64),   intent(out)   :: scrat1(*)
  real(real64),   intent(out)   :: scrat2(*)
  integer(int32), intent(in)    :: index_arr(*)
  call mod_epkpot(pkpot, evec, eval, stovl, scrat1, scrat2, index_arr)
end subroutine epkpot

subroutine dampv(epot2, pkpot, zpot2, vpot2, msto, m2sto, nvterm, ispin, vdamp)
  use iso_fortran_env, only: real64, int32
  use modelpot,        only: mod_dampv => dampv
  implicit none
  integer(int32), intent(in)    :: msto, m2sto, nvterm, ispin
  real(real64),   intent(in)    :: vdamp
  real(real64),   intent(inout) :: epot2(m2sto, ispin)
  real(real64),   intent(inout) :: pkpot(m2sto, 4, ispin)
  real(real64),   intent(inout) :: zpot2(m2sto)
  real(real64),   intent(inout) :: vpot2(nvterm)
  call mod_dampv(epot2, pkpot, zpot2, vpot2, msto, m2sto, nvterm, ispin, vdamp)
end subroutine dampv
