!=======================================================================
! PHATPSY 2.0: Modern Density-Functional Exchange Kernels (exchange_dft.f90)
! 
! Implements modern local and gradient-corrected exchange functionals:
!   - Slater-Dirac local density exchange (LDA, alpha = 2/3 Kohn-Sham or 1.0 Slater)
!   - Becke 88 (B88) Generalized Gradient Approximation (GGA)
!   - Perdew-Burke-Ernzerhof (PBE) Generalized Gradient Approximation (GGA)
!
! Provides:
!   1. Exact Gauss-Legendre mapped radial quadrature for STO basis integration.
!   2. Atomic spherical electron density rho(r) and radial gradient |drho/dr|.
!   3. Local exchange potential evaluation v_x(r) for Slater, Becke88, and PBE.
!   4. Construction of the packed STO exchange matrix stexch(M2STO, ISPIN).
!   5. Multi-exponential least-squares fitting of v_x(r) ~ sum_k c_k^X exp(-zeta_k r).
!=======================================================================
module exchange_dft
  use iso_fortran_env, only: real64, int32
  use state,           only: NSTO, MSTO, NORB, M2STO, ISPIN, mol_ws
  implicit none

  private

  public :: gauss_legendre
  public :: eval_atomic_rho
  public :: eval_exchange_potential_point
  public :: compute_dft_exchange
  public :: fit_exchange_exponential

  ! Physical constants and functional parameters in atomic units (Hartrees, Bohrs)
  real(real64), parameter :: PI = 3.141592653589793238462643383279502884197d0
  real(real64), parameter :: RHO_FLOOR = 1.0d-15
  real(real64), parameter :: BETA_B88  = 0.0042d0
  real(real64), parameter :: KAPPA_PBE = 0.804d0
  real(real64), parameter :: MU_PBE    = 0.2195149727645171d0

contains

  !---------------------------------------------------------------------
  ! 1. Gauss-Legendre Quadrature Generator
  ! Computes N roots and weights on [-1, 1] via Newton-Raphson on P_N(x).
  !---------------------------------------------------------------------
  subroutine gauss_legendre(n, x, w)
    integer(int32), intent(in)  :: n
    real(real64),   intent(out) :: x(n), w(n)

    integer(int32) :: i, j, m
    real(real64)   :: z, z1, p1, p2, p3, pp

    m = (n + 1) / 2
    do i = 1, m
      z = cos(PI * (real(i, real64) - 0.25d0) / (real(n, real64) + 0.5d0))
      do
        p1 = 1.0d0
        p2 = 0.0d0
        do j = 1, n
          p3 = p2
          p2 = p1
          p1 = ((2.0d0 * real(j, real64) - 1.0d0) * z * p2 - (real(j, real64) - 1.0d0) * p3) / real(j, real64)
        end do
        pp = real(n, real64) * (z * p1 - p2) / (z * z - 1.0d0)
        z1 = z
        z  = z1 - p1 / pp
        if (abs(z - z1) < 1.0d-15) exit
      end do
      x(i)         = -z
      x(n + 1 - i) = z
      w(i)         = 2.0d0 / ((1.0d0 - z * z) * pp * pp)
      w(n + 1 - i) = w(i)
    end do
  end subroutine gauss_legendre

  !---------------------------------------------------------------------
  ! 2. Pointwise Exchange Potential and Enhancement Evaluation
  ! Inputs:
  !   rho       : Electron density (Bohr^-3)
  !   drho      : Radial density gradient |drho/dr| (Bohr^-4)
  !   exch_type : 1 = Slater-Dirac, 2 = Becke 88, 3 = PBE
  !   alpha     : Exchange parameter (e.g. 2/3 for Kohn-Sham Dirac, 1.0 for Slater)
  ! Outputs:
  !   vx        : Local exchange potential (Hartree)
  !   fx        : Exchange enhancement factor F_x
  !---------------------------------------------------------------------
  subroutine eval_exchange_potential_point(rho, drho, exch_type, alpha, vx, fx)
    real(real64),   intent(in)  :: rho, drho
    integer(int32), intent(in)  :: exch_type
    real(real64),   intent(in)  :: alpha
    real(real64),   intent(out) :: vx, fx

    real(real64) :: rho_eff, k_f, cx, x_b88, asinh_x, s_pbe, vx_lda

    fx = 1.0d0
    vx = 0.0d0

    if (rho <= RHO_FLOOR) return

    rho_eff = max(rho, RHO_FLOOR)

    ! Slater-Dirac LDA potential: v_x^LDA = -alpha * (3/2) * (3/pi * rho)^(1/3)
    ! For alpha = 2/3, prefactor is 1.0: v_x = -(3/pi * rho)^(1/3)
    vx_lda = -alpha * 1.5d0 * (3.0d0 / PI * rho_eff)**(1.0d0 / 3.0d0)

    select case (exch_type)
    case (1)
      ! Standard Slater / Dirac LDA
      fx = 1.0d0
      vx = vx_lda

    case (2)
      ! Becke 88 (B88) GGA
      cx = 0.75d0 * (3.0d0 / PI)**(1.0d0 / 3.0d0)
      x_b88 = abs(drho) / (rho_eff**(4.0d0 / 3.0d0))
      asinh_x = log(x_b88 + sqrt(x_b88 * x_b88 + 1.0d0))
      fx = 1.0d0 + (BETA_B88 * x_b88 * x_b88 / cx) / (1.0d0 + 6.0d0 * BETA_B88 * x_b88 * asinh_x)
      vx = vx_lda * fx

    case (3)
      ! Perdew-Burke-Ernzerhof (PBE) GGA
      ! s = |grad rho| / (2 * k_F * rho) where k_F = (3*pi^2*rho)^(1/3)
      k_f   = (3.0d0 * PI * PI * rho_eff)**(1.0d0 / 3.0d0)
      s_pbe = abs(drho) / (2.0d0 * k_f * rho_eff)
      fx = 1.0d0 + KAPPA_PBE - KAPPA_PBE / (1.0d0 + MU_PBE * s_pbe * s_pbe / KAPPA_PBE)
      vx = vx_lda * fx

    case default
      ! Fallback to Slater LDA
      fx = 1.0d0
      vx = vx_lda
    end select
  end subroutine eval_exchange_potential_point

  !---------------------------------------------------------------------
  ! 3. Evaluate Spherically Averaged Atomic Density rho(r) and |drho/dr|
  ! Evaluates rho(r) and drho/dr over an arbitrary array of radial points.
  !---------------------------------------------------------------------
  subroutine eval_atomic_rho(ielem, stden, r_in, rho_out, drho_out, n_pts, ispin_in)
    integer(int32), intent(in)  :: ielem, n_pts, ispin_in
    real(real64),   intent(in)  :: stden(M2STO, ispin_in), r_in(n_pts)
    real(real64),   intent(out) :: rho_out(n_pts, ispin_in), drho_out(n_pts, ispin_in)

    integer(int32) :: nsto_l, msto_l, i, j, k, isp, li, lj, ni, nj
    integer(int32) :: mi, mj, m, mu, nu, p
    integer(int32), allocatable :: shell_offset(:)
    real(real64)   :: r_val, r_inv, etai, etaj, anormi, anormj
    real(real64)   :: ri_val, dri_val, rj_val, drj_val, d_bar
    real(real64), allocatable :: d_shell(:,:,:)

    rho_out  = 0.0d0
    drho_out = 0.0d0

    if (.not. allocated(mol_ws%elem_stream)) return
    if (ielem < 1 .or. ielem > size(mol_ws%elem_stream)) return
    nsto_l = mol_ws%elem_stream(ielem)%nsto
    msto_l = mol_ws%elem_stream(ielem)%msto
    if (nsto_l <= 0 .or. msto_l <= 0) return

    allocate(shell_offset(nsto_l))
    shell_offset(1) = 0
    do i = 2, nsto_l
      shell_offset(i) = shell_offset(i - 1) + 2 * mol_ws%elem_stream(ielem)%l(i - 1) + 1
    end do

    ! Compute spherically averaged shell density matrix D_shell(i, j, isp)
    allocate(d_shell(nsto_l, nsto_l, ispin_in))
    d_shell = 0.0d0

    do isp = 1, ispin_in
      do i = 1, nsto_l
        li = mol_ws%elem_stream(ielem)%l(i)
        do j = 1, i
          lj = mol_ws%elem_stream(ielem)%l(j)
          if (li /= lj) cycle
          d_bar = 0.0d0
          do m = -li, li
            mu = shell_offset(i) + m + li + 1
            nu = shell_offset(j) + m + lj + 1
            if (mu >= nu) then
              p = (mu * (mu - 1)) / 2 + nu
            else
              p = (nu * (nu - 1)) / 2 + mu
            end if
            if (p <= M2STO) d_bar = d_bar + stden(p, isp)
          end do
          d_shell(i, j, isp) = d_bar
          d_shell(j, i, isp) = d_bar
        end do
      end do
    end do

    ! Evaluate rho(r) and drho/dr over radial points
    do k = 1, n_pts
      r_val = r_in(k)
      r_inv = merge(1.0d0 / r_val, 0.0d0, r_val > 1.0d-14)

      do isp = 1, ispin_in
        do i = 1, nsto_l
          ni     = mol_ws%elem_stream(ielem)%n(i)
          li     = mol_ws%elem_stream(ielem)%l(i)
          etai   = mol_ws%elem_stream(ielem)%eta(i)
          anormi = mol_ws%elem_stream(ielem)%anorm(i)

          ! R_i(r) = anormi * r^(ni - 1) * exp(-etai * r)
          if (ni == 1) then
            ri_val  = anormi * exp(-etai * r_val)
            dri_val = -etai * ri_val
          else
            ri_val  = anormi * (r_val**(ni - 1)) * exp(-etai * r_val)
            dri_val = ri_val * (real(ni - 1, real64) * r_inv - etai)
          end if

          do j = 1, nsto_l
            if (mol_ws%elem_stream(ielem)%l(j) /= li) cycle
            d_bar = d_shell(i, j, isp)
            if (abs(d_bar) < 1.0d-16) cycle

            nj     = mol_ws%elem_stream(ielem)%n(j)
            etaj   = mol_ws%elem_stream(ielem)%eta(j)
            anormj = mol_ws%elem_stream(ielem)%anorm(j)

            if (nj == 1) then
              rj_val  = anormj * exp(-etaj * r_val)
              drj_val = -etaj * rj_val
            else
              rj_val  = anormj * (r_val**(nj - 1)) * exp(-etaj * r_val)
              drj_val = rj_val * (real(nj - 1, real64) * r_inv - etaj)
            end if

            ! Spherical density: rho(r) = sum_ij D_ij R_i R_j / (4*pi)
            rho_out(k, isp)  = rho_out(k, isp)  + (d_bar / (4.0d0 * PI)) * ri_val * rj_val
            drho_out(k, isp) = drho_out(k, isp) + (d_bar / (4.0d0 * PI)) * (dri_val * rj_val + ri_val * drj_val)
          end do
        end do
      end do
    end do

    deallocate(shell_offset, d_shell)
  end subroutine eval_atomic_rho

  !---------------------------------------------------------------------
  ! 4. Construct Packed DFT Exchange Matrix in STO Basis
  ! Computes:
  !   (V_X)_ij = delta_l_i,l_j delta_m_i,m_j int_0^inf R_i(r) v_x(r) R_j(r) r^2 dr
  ! via mapped Gauss-Legendre quadrature.
  !---------------------------------------------------------------------
  subroutine compute_dft_exchange(ielem, stden, stexch, exch_type_in, exch_alpha_in)
    integer(int32),           intent(in)    :: ielem
    real(real64),             intent(in)    :: stden(M2STO, ISPIN)
    real(real64),             intent(out)   :: stexch(M2STO, ISPIN)
    integer(int32), optional, intent(in)    :: exch_type_in
    real(real64),   optional, intent(in)    :: exch_alpha_in

    integer(int32), parameter :: N_QUAD = 64
    real(real64),   parameter :: R0 = 1.0d0

    integer(int32) :: exch_type, nsto_l, msto_l, k, isp, i, j, m, li, lj, ni, nj
    integer(int32) :: mu, nu, p
    integer(int32), allocatable :: shell_offset(:)
    real(real64)   :: alpha, t_nodes(N_QUAD), w_nodes(N_QUAD), r_grid(N_QUAD), w_metric(N_QUAD)
    real(real64)   :: rho_pts(N_QUAD, ISPIN), drho_pts(N_QUAD, ISPIN)
    real(real64)   :: vx_pts(N_QUAD, ISPIN), fx_pts(N_QUAD, ISPIN)
    real(real64)   :: r_val, r_inv, ri_val, rj_val, etai, etaj, anormi, anormj
    real(real64)   :: int_val, vx_val
    real(real64), allocatable :: r_rad(:,:), shell_vmat(:,:,:)

    stexch = 0.0d0

    exch_type = 1
    if (present(exch_type_in)) exch_type = exch_type_in

    alpha = 2.0d0 / 3.0d0
    if (present(exch_alpha_in)) then
      if (exch_alpha_in > 0.0d0) alpha = exch_alpha_in
    end if

    if (.not. allocated(mol_ws%elem_stream)) return
    if (ielem < 1 .or. ielem > size(mol_ws%elem_stream)) return
    nsto_l = mol_ws%elem_stream(ielem)%nsto
    msto_l = mol_ws%elem_stream(ielem)%msto
    if (nsto_l <= 0 .or. msto_l <= 0) return

    ! Generate Gauss-Legendre quadrature nodes and weights
    call gauss_legendre(N_QUAD, t_nodes, w_nodes)

    ! Map [-1, 1] to [0, inf) via r(t) = R0 * (1 + t) / (1 - t)
    ! dr = 2 * R0 / (1 - t)^2 dt
    do k = 1, N_QUAD
      r_grid(k)   = R0 * (1.0d0 + t_nodes(k)) / (1.0d0 - t_nodes(k))
      w_metric(k) = w_nodes(k) * (2.0d0 * R0 / ((1.0d0 - t_nodes(k))**2))
    end do

    ! Compute atomic electron density and gradient on the radial grid
    call eval_atomic_rho(ielem, stden, r_grid, rho_pts, drho_pts, N_QUAD, ISPIN)

    ! Evaluate exchange potential at each radial grid point
    do isp = 1, ISPIN
      do k = 1, N_QUAD
        call eval_exchange_potential_point(rho_pts(k, isp), drho_pts(k, isp), &
                                           exch_type, alpha, vx_pts(k, isp), fx_pts(k, isp))
      end do
    end do

    ! Precompute R_i(r_k) across all shells i and grid points k
    allocate(r_rad(nsto_l, N_QUAD))
    do k = 1, N_QUAD
      r_val = r_grid(k)
      do i = 1, nsto_l
        ni     = mol_ws%elem_stream(ielem)%n(i)
        etai   = mol_ws%elem_stream(ielem)%eta(i)
        anormi = mol_ws%elem_stream(ielem)%anorm(i)
        if (ni == 1) then
          r_rad(i, k) = anormi * exp(-etai * r_val)
        else
          r_rad(i, k) = anormi * (r_val**(ni - 1)) * exp(-etai * r_val)
        end if
      end do
    end do

    ! Integrate radial matrix elements between shells with matching l:
    ! V_ij^shell = int_0^inf R_i(r) v_x(r) R_j(r) r^2 dr
    allocate(shell_vmat(nsto_l, nsto_l, ISPIN))
    shell_vmat = 0.0d0

    do isp = 1, ISPIN
      do i = 1, nsto_l
        li = mol_ws%elem_stream(ielem)%l(i)
        do j = 1, i
          lj = mol_ws%elem_stream(ielem)%l(j)
          if (li /= lj) cycle
          int_val = 0.0d0
          do k = 1, N_QUAD
            int_val = int_val + w_metric(k) * r_rad(i, k) * r_rad(j, k) * &
                                vx_pts(k, isp) * (r_grid(k)**2)
          end do
          shell_vmat(i, j, isp) = int_val
          shell_vmat(j, i, isp) = int_val
        end do
      end do
    end do

    ! Assemble packed stexch(M2STO, ISPIN)
    allocate(shell_offset(nsto_l))
    shell_offset(1) = 0
    do i = 2, nsto_l
      shell_offset(i) = shell_offset(i - 1) + 2 * mol_ws%elem_stream(ielem)%l(i - 1) + 1
    end do

    do isp = 1, ISPIN
      do i = 1, nsto_l
        li = mol_ws%elem_stream(ielem)%l(i)
        do j = 1, i
          lj = mol_ws%elem_stream(ielem)%l(j)
          if (li /= lj) cycle
          int_val = shell_vmat(i, j, isp)
          do m = -li, li
            mu = shell_offset(i) + m + li + 1
            nu = shell_offset(j) + m + lj + 1
            if (mu >= nu) then
              p = (mu * (mu - 1)) / 2 + nu
            else
              p = (nu * (nu - 1)) / 2 + mu
            end if
            if (p <= M2STO) stexch(p, isp) = int_val
          end do
        end do
      end do
    end do

    deallocate(r_rad, shell_vmat, shell_offset)
  end subroutine compute_dft_exchange

  !---------------------------------------------------------------------
  ! 5. Fit Radial Exchange Potential to Multi-Exponential Form
  ! Fits v_x(r) ~ sum_k c_k^X exp(-zeta_k r) using weighted least squares.
  !---------------------------------------------------------------------
  subroutine fit_exchange_exponential(n_grid, r_grid, vx_grid, w_grid, n_terms, vexp, vcoef, rms)
    integer(int32), intent(in)  :: n_grid, n_terms
    real(real64),   intent(in)  :: r_grid(n_grid), vx_grid(n_grid), w_grid(n_grid), vexp(n_terms)
    real(real64),   intent(out) :: vcoef(n_terms), rms

    integer(int32) :: i, j, k, info
    real(real64)   :: a_mat(n_terms, n_terms), b_vec(n_terms), w_r2, r_val, v_val, v_fit
    real(real64)   :: sum_sq, sum_w
    real(real64)   :: work_gels(10 * n_terms)
    integer(int32) :: lwork

    a_mat = 0.0d0
    b_vec = 0.0d0
    vcoef = 0.0d0
    rms   = 0.0d0

    if (n_grid <= 0 .or. n_terms <= 0) return

    do k = 1, n_grid
      r_val = r_grid(k)
      w_r2  = w_grid(k) * (r_val**2)
      v_val = vx_grid(k)
      do i = 1, n_terms
        b_vec(i) = b_vec(i) + w_r2 * v_val * exp(-vexp(i) * r_val)
        do j = 1, i
          a_mat(i, j) = a_mat(i, j) + w_r2 * exp(-(vexp(i) + vexp(j)) * r_val)
        end do
      end do
    end do

    ! Symmetrize A
    do i = 1, n_terms
      do j = 1, i - 1
        a_mat(j, i) = a_mat(i, j)
      end do
    end do

    ! Solve normal equations via LAPACK DPOSV (Cholesky factorization)
    call dposv('U', n_terms, 1, a_mat, n_terms, b_vec, n_terms, info)
    if (info == 0) then
      vcoef = b_vec
    else
      ! Fallback: least squares via DGELS
      lwork = 10 * n_terms
      call dgels('N', n_terms, n_terms, 1, a_mat, n_terms, b_vec, n_terms, work_gels, lwork, info)
      if (info == 0) vcoef = b_vec
    end if

    ! Compute RMS deviation
    sum_sq = 0.0d0
    sum_w  = 0.0d0
    do k = 1, n_grid
      r_val = r_grid(k)
      w_r2  = w_grid(k) * (r_val**2)
      v_fit = dot_product(vcoef, exp(-vexp * r_val))
      sum_sq = sum_sq + w_r2 * ((vx_grid(k) - v_fit)**2)
      sum_w  = sum_w  + w_r2
    end do

    if (sum_w > 0.0d0) rms = sqrt(sum_sq / sum_w)
  end subroutine fit_exchange_exponential

end module exchange_dft
