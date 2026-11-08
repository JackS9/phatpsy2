module analys
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin,   only: run_lowdin
  use modelpot, only: eval_atomic_dipole
  use exchange_dft, only: eval_atomic_rho
  implicit none
  private
  public :: run_analys, run_chgan, USE_LAPACK_FIT
  !f2py private :: run_analys, run_chgan

  logical, save :: USE_LAPACK_FIT = .false. ! Set .true. to use LAPACK DGELSD SVD solver

contains

  !---------------------------------------------------------------------
  ! NNLS_SMALL: Lawson-Hanson non-negative least squares, min ||A x - b||,
  ! x >= 0, for small n (<= ~20). Passive-set sub-problems are solved with
  ! the normal equations (DSYSV).
  !---------------------------------------------------------------------
  subroutine nnls_small(a, m, n, b, x, ok)
    integer(int32), intent(in)  :: m, n
    real(real64),   intent(in)  :: a(m, n), b(m)
    real(real64),   intent(out) :: x(n)
    logical,        intent(out) :: ok

    real(real64)   :: w(n), z(n), ata(n, n), atb(n), sub(n, n), rhs(n), wk(max(1, 4 * n))
    real(real64)   :: alpha, wmax, tol, tmp
    integer(int32) :: iter, jmax, ip, jp, np, info, i, j, ipiv(n), inner
    logical        :: passive(n)
    integer(int32) :: idx(n)

    ok = .true.
    x = 0.0d0
    passive = .false.
    ata = matmul(transpose(a), a)
    atb = matmul(transpose(a), b)
    tol = 1.0d-12 * max(1.0d0, maxval(abs(atb)))

    do iter = 1, 5 * n + 10
      w = atb - matmul(ata, x)
      jmax = 0
      wmax = tol
      do j = 1, n
        if (.not. passive(j) .and. w(j) > wmax) then
          wmax = w(j)
          jmax = j
        end if
      end do
      if (jmax == 0) exit
      passive(jmax) = .true.

      do inner = 1, 5 * n + 10
        np = 0
        do j = 1, n
          if (passive(j)) then
            np = np + 1
            idx(np) = j
          end if
        end do
        sub = 0.0d0
        rhs = 0.0d0
        do ip = 1, np
          rhs(ip) = atb(idx(ip))
          do jp = 1, np
            sub(ip, jp) = ata(idx(ip), idx(jp))
          end do
        end do
        call DSYSV('U', np, 1, sub, n, ipiv, rhs, n, wk, size(wk), info)
        if (info /= 0) then
          ok = .false.
          return
        end if
        z = 0.0d0
        do ip = 1, np
          z(idx(ip)) = rhs(ip)
        end do
        if (all(z(idx(1:np)) > 0.0d0)) then
          x = z
          exit
        end if
        alpha = huge(1.0d0)
        do ip = 1, np
          i = idx(ip)
          if (z(i) <= 0.0d0) then
            tmp = x(i) / (x(i) - z(i))
            if (tmp < alpha) alpha = tmp
          end if
        end do
        x = x + alpha * (z - x)
        do ip = 1, np
          i = idx(ip)
          if (x(i) <= 1.0d-14) then
            x(i) = 0.0d0
            passive(i) = .false.
          end if
        end do
      end do
    end do
  end subroutine nnls_small

  !---------------------------------------------------------------------
  ! FIT_SCREENING_SIMPLEX: fits the atom's spherically averaged electronic
  ! Coulomb potential as
  !     V(r) = (N/r) * [ 1 - sum_k w_k exp(-zeta_k r) ],  w_k >= 0, sum_k w_k = 1
  ! against the potential obtained by Poisson integration of the current
  ! density on a logarithmic radial grid, instead of against a handful of
  ! orbital expectation values. Charge conservation (far field N/r) and the
  ! nuclear cusp (finite potential at r = 0) hold exactly by construction, and
  ! every weight lies in [0, 1], so the fitted coefficients cannot cancel one
  ! another. In model-potential coefficients: vcoef(1) = N and
  ! vcoef(k) = -N * w(k-1) for k = 2..nvterm.
  !
  ! Fit measure: weights h * r * exp(-r/10) on the grid (a uniform-in-r measure
  ! on the screening function, tapered beyond ~10 bohr), a heavily weighted
  ! sum-to-one row, and a tiny ridge toward a uniform mixture so the solution is
  ! unique when exponentials are nearly collinear.
  !---------------------------------------------------------------------
  subroutine fit_screening_simplex(vexp, vcoef, sumocc, nvterm, stevec, occ, ok)
    integer(int32), intent(in)    :: nvterm
    real(real64),   intent(in)    :: vexp(nvterm), sumocc
    real(real64),   intent(in)    :: stevec(MSTO, NORB, ISPIN), occ(NORB, ISPIN)
    real(real64),   intent(inout) :: vcoef(nvterm)
    logical,        intent(out)   :: ok

    integer(int32), parameter :: NG = 400
    real(real64),   parameter :: FOURPI = 12.566370614359172d0, R0 = 1.0d-4, RMAX = 40.0d0
    real(real64)   :: rg(NG), rho(NG, ISPIN), drho(NG, ISPIN), rtot(NG)
    real(real64)   :: stden(M2STO, ISPIN), f(NG), f2(NG), qcum(NG), ucum(NG), sg(NG)
    real(real64)   :: h, ngrid, scale, ridge, trace_ata, wsum, sqw
    real(real64), allocatable :: a(:,:), bb(:), x(:), wt(:)
    integer(int32) :: g, k, nk, isp, m

    ok = .false.
    nk = nvterm - 1
    if (nk < 1 .or. sumocc <= 0.0d0) return

    h = log(RMAX / R0) / real(NG - 1, real64)
    do g = 1, NG
      rg(g) = R0 * exp(h * real(g - 1, real64))
    end do

    do isp = 1, ISPIN
      call DENMAT(stden(1, isp), stevec(1, 1, isp), occ(1, isp), MSTO, M2STO, NORB)
    end do
    call eval_atomic_rho(IELEM, stden, rg, rho, drho, NG, ISPIN)
    rtot = ORBOCC * sum(rho, dim=2)

    ! Cumulative charge Q(r) = int_0^r 4 pi r'^2 rho dr' and outer integral
    ! U(r) = int_r^inf 4 pi r' rho dr' on the log grid (dr = r dx).
    f  = FOURPI * rg**3 * rtot
    f2 = FOURPI * rg**2 * rtot
    qcum(1) = f(1) / 3.0d0
    do g = 2, NG
      qcum(g) = qcum(g - 1) + 0.5d0 * h * (f(g - 1) + f(g))
    end do
    ucum(NG) = 0.0d0
    do g = NG - 1, 1, -1
      ucum(g) = ucum(g + 1) + 0.5d0 * h * (f2(g) + f2(g + 1))
    end do
    ngrid = qcum(NG)
    if (ngrid <= 1.0d-8) return
    scale = sumocc / ngrid

    ! Screening function S(r) = 1 - r V_el(r) / N with V_el = Q/r + U
    do g = 1, NG
      sg(g) = 1.0d0 - scale * (qcum(g) + rg(g) * ucum(g)) / sumocc
    end do

    m = NG + 1 + nk
    allocate(a(m, nk), bb(m), x(nk), wt(NG))
    a  = 0.0d0
    bb = 0.0d0
    do g = 1, NG
      wt(g) = h * rg(g) * exp(-rg(g) / 10.0d0)
      sqw = sqrt(wt(g))
      do k = 1, nk
        a(g, k) = sqw * exp(-vexp(k + 1) * rg(g))
      end do
      bb(g) = sqw * sg(g)
    end do
    do k = 1, nk
      a(NG + 1, k) = 1.0d3
    end do
    bb(NG + 1) = 1.0d3
    trace_ata = sum(a(1:NG, :)**2) / real(nk, real64)
    ridge = sqrt(1.0d-9 * trace_ata)
    do k = 1, nk
      a(NG + 1 + k, k) = ridge
      bb(NG + 1 + k) = ridge / real(nk, real64)
    end do

    call nnls_small(a, m, nk, bb, x, ok)
    if (ok) then
      wsum = sum(x)
      if (wsum > 0.0d0) then
        x = x / wsum          ! enforce the cusp sum rule exactly
        vcoef(1) = sumocc
        vcoef(2:nvterm) = -sumocc * x(1:nk)
      else
        ok = .false.
      end if
    end if
    deallocate(a, bb, x, wt)
  end subroutine fit_screening_simplex

  !---------------------------------------------------------------------
  ! SOLVE_HELMERT_NULLSPACE_FIT:
  ! Solves min ||G0 * c - g_rhs|| subject to exact linear cusp sum(c) = d1
  ! using an analytical Helmert orthonormal basis Z for the null space (1^T Z = 0).
  ! Guarantees the cusp condition is satisfied to exact machine precision
  ! without condition-number blowup from penalty weights.
  !---------------------------------------------------------------------
  subroutine solve_helmert_nullspace_fit(g0_packed, g0_rhs, d1, nc, c_out, qindef)
    integer(int32), intent(in)  :: nc
    real(real64),   intent(in)  :: g0_packed(210), g0_rhs(20), d1
    real(real64),   intent(out) :: c_out(nc)
    logical(kind=1), intent(out) :: qindef

    real(real64), allocatable :: g0(:,:), z(:,:), gz(:,:), g_tilde(:,:), g_sub_rhs(:)
    real(real64), allocatable :: x0(:), y(:), s_svd(:), work(:)
    integer(int32), allocatable :: iwork(:)
    real(real64) :: norm, inv_norm
    integer(int32) :: i, j, ij, k, info, rank, lwork, liwork

    qindef = .false.
    if (nc <= 1) then
      c_out(1) = d1
      return
    end if

    allocate(g0(nc, nc), z(nc, nc - 1), x0(nc))
    x0 = d1 / real(nc, real64)

    ! Construct Helmert orthonormal basis Z for 1^T z = 0
    z = 0.0d0
    do k = 1, nc - 1
      norm = sqrt(real(k * (k + 1), real64))
      inv_norm = 1.0d0 / norm
      z(1:k, k) = inv_norm
      z(k + 1, k) = -real(k, real64) * inv_norm
    end do

    ! Unpack symmetric G0
    ij = 0
    do i = 1, nc
      do j = 1, i
        ij = ij + 1
        g0(i, j) = g0_packed(ij)
        g0(j, i) = g0_packed(ij)
      end do
    end do

    ! Form G_tilde = Z^T * G0 * Z and g_sub_rhs = Z^T * (g0_rhs - G0 * x0)
    allocate(gz(nc, nc - 1), g_tilde(nc - 1, nc - 1), g_sub_rhs(nc - 1))
    gz = matmul(g0, z)
    g_tilde = matmul(transpose(z), gz)
    g_sub_rhs = matmul(transpose(z), g0_rhs(1:nc) - matmul(g0, x0))

    ! Solve G_tilde * y = g_sub_rhs using DGELSD (guaranteed robust even if rank-deficient)
    allocate(y(nc - 1), s_svd(nc - 1))
    y = g_sub_rhs
    lwork  = 12 * (nc - 1) + 2 * (nc - 1) * 25 + 8 * (nc - 1)**2 + 100
    liwork = max(1, 3 * (nc - 1) * 10 + 11 * (nc - 1))
    allocate(work(lwork), iwork(liwork))

    call DGELSD(nc - 1, nc - 1, 1, g_tilde, nc - 1, y, nc - 1, s_svd, 1.0d-12, &
                rank, work, lwork, iwork, info)

    if (info /= 0) then
      qindef = .true.
      c_out = x0
    else
      c_out = x0 + matmul(z, y)
    end if

    deallocate(g0, z, gz, g_tilde, g_sub_rhs, x0, y, s_svd, work, iwork)
  end subroutine solve_helmert_nullspace_fit

  !---------------------------------------------------------------------
  ! SOLVE_CONSTRAINED_MODEL_POTENTIAL_FIT:
  ! Equality-constrained linear fitting with KKT system and Helmert null-space fallback.
  ! Constraint 1: Exact nuclear cusp sum(c_2..c_nvterm) = -sumocc
  ! Constraint 2: External field matching (when WGT2 > 0 and tevz2 is active)
  !---------------------------------------------------------------------
  subroutine solve_constrained_model_potential_fit(g0_packed, g0_rhs, vfit2, tevz2, &
                                                   vcoef, sumocc, nvterm, wgt2, qindef)
    integer(int32), intent(in)    :: nvterm
    real(real64),   intent(in)    :: g0_packed(210), g0_rhs(20), vfit2(nvterm)
    real(real64),   intent(in)    :: tevz2, sumocc, wgt2
    real(real64),   intent(inout) :: vcoef(nvterm)
    logical(kind=1), intent(out)   :: qindef

    integer(int32) :: nc, p_const, n_kkt, i, j, ij, info, lwork
    real(real64), allocatable :: kkt(:,:), rhs(:), work(:)
    integer(int32), allocatable :: ipiv(:)
    real(real64) :: d1, d2
    logical :: use_wgt2

    nc = nvterm - 1
    qindef = .false.
    if (nc <= 0) return

    ! Constraint 1: Nuclear cusp sum(c_2..c_nvterm) = -sumocc
    d1 = -sumocc

    ! Constraint 2: External field matching
    use_wgt2 = (wgt2 > 1.0d-4 .and. abs(tevz2 - 0.5d0) > 1.0d-3)
    if (use_wgt2) then
      p_const = 2
      d2 = tevz2 - sumocc * vfit2(1)
    else
      p_const = 1
    end if

    n_kkt = nc + p_const
    allocate(kkt(n_kkt, n_kkt), rhs(n_kkt), ipiv(n_kkt))
    kkt = 0.0d0
    rhs = 0.0d0

    ! 1. Unpack G0 into upper-left block of KKT
    ij = 0
    do i = 1, nc
      do j = 1, i
        ij = ij + 1
        kkt(i, j) = g0_packed(ij)
        kkt(j, i) = g0_packed(ij)
      end do
      rhs(i) = g0_rhs(i)
    end do

    ! 2. Constraint rows/columns B and B^T
    ! Row 1: Cusp (all 1's)
    do j = 1, nc
      kkt(nc + 1, j) = 1.0d0
      kkt(j, nc + 1) = 1.0d0
    end do
    rhs(nc + 1) = d1

    ! Row 2: External field (vfit2(2..nvterm))
    if (use_wgt2) then
      do j = 1, nc
        kkt(nc + 2, j) = vfit2(j + 1)
        kkt(j, nc + 2) = vfit2(j + 1)
      end do
      rhs(nc + 2) = d2
    end if

    ! 3. Query optimal workspace and solve KKT system via LAPACK DSYSV
    lwork = -1
    allocate(work(1))
    call DSYSV('U', n_kkt, 1, kkt, n_kkt, ipiv, rhs, n_kkt, work, lwork, info)
    lwork = max(1, int(work(1)))
    deallocate(work)
    allocate(work(lwork))

    call DSYSV('U', n_kkt, 1, kkt, n_kkt, ipiv, rhs, n_kkt, work, lwork, info)

    if (info == 0) then
      vcoef(1) = sumocc
      vcoef(2:nvterm) = rhs(1:nc)
    else
      ! Null-space Helmert fallback for constraint 1
      call solve_helmert_nullspace_fit(g0_packed, g0_rhs, d1, nc, vcoef(2:nvterm), qindef)
      vcoef(1) = sumocc
    end if

    deallocate(kkt, rhs, ipiv, work)
  end subroutine solve_constrained_model_potential_fit

  subroutine solve_model_potential_fit(vmat_packed, vsum, vcoef, sumocc, nvterm, qindef)
    integer(int32),  intent(in)    :: nvterm
    real(real64),    intent(inout) :: vmat_packed(210), vsum(20)
    real(real64),    intent(inout) :: vcoef(nvterm)
    real(real64),    intent(in)    :: sumocc
    logical(kind=1), intent(out)   :: qindef

    real(real64)   :: vinv(20, 20), scrat1(20, 20)
    integer(int32) :: nfit, i, j, ij, info, rank, lwork, liwork
    real(real64),   allocatable :: a_full(:,:), rhs(:), s_svd(:), work(:)
    integer(int32), allocatable :: iwork(:)

    nfit   = nvterm - 1
    qindef = .false.

    if (.not. USE_LAPACK_FIT) then
      vinv   = 0.0d0
      scrat1 = 0.0d0
      call run_lowdin(vmat_packed, vinv, scrat1, nfit, 20, 0, qindef)
      if (qindef) return
      vcoef(1) = sumocc
      do i = 2, nvterm
        vcoef(i) = dot_product(vinv(i - 1, 1:nfit), vsum(1:nfit))
      end do
    else
      allocate(a_full(nfit, nfit), rhs(nfit), s_svd(nfit))
      ij = 0
      do i = 1, nfit
        do j = 1, i
          ij = ij + 1
          a_full(i, j) = vmat_packed(ij)
          a_full(j, i) = vmat_packed(ij)
        end do
        rhs(i) = vsum(i)
      end do
      lwork  = 12 * nfit + 2 * nfit * 25 + 8 * nfit * nfit + 100
      liwork = max(1, 3 * nfit * 10 + 11 * nfit)
      allocate(work(lwork), iwork(liwork))
      call DGELSD(nfit, nfit, 1, a_full, nfit, rhs, nfit, s_svd, 1.0d-10, &
                  rank, work, lwork, iwork, info)
      if (info /= 0) then
        qindef = .true.
      else
        vcoef(1)        = sumocc
        vcoef(2:nvterm) = rhs(1:nfit)
      end if
      deallocate(a_full, rhs, s_svd, work, iwork)
    end if
  end subroutine solve_model_potential_fit

  subroutine run_analys(stkin, stpot, stcoul, stexch, cpot2, zpot2, vfit2,    &
                        vcoef, vexp, lv, pkpot, xpot2, stevec, eval, occ,     &
                        gamma, index_arr, toten, tke, zzrep, stovl, charge, tcoul0)
    real(real64),   intent(in)    :: stkin(M2STO), stpot(M2STO, NVTERM),      &
                                     stcoul(M2STO, ISPIN), stexch(M2STO, ISPIN), &
                                     vexp(NVTERM), xpot2(M2STO, ISPIN),       &
                                     cpot2(M2STO), stevec(MSTO, NORB, ISPIN), &
                                     eval(NORB, ISPIN), occ(NORB, ISPIN),     &
                                     zpot2(M2STO), vfit2(NVTERM),             &
                                     stovl(M2STO), pkpot(M2STO, 6, ISPIN),    &
                                     tcoul0
    integer(int32), intent(in)    :: lv(NVTERM), index_arr(MSTO)
    real(real64),   intent(inout) :: vcoef(NVTERM), gamma(NORB, ISPIN),       &
                                     toten, tke, zzrep, charge(3, NATOM)

    real(real64)    :: oe(9), te(9), charg(6), epot(20), vsum(20), vmat(210)
    real(real64)    :: eorb, ekin, enuc, echf, exhf, evz2, evc2, evx2, pkv2
    real(real64)    :: teorB, tekin, tenuc, techf, texhf, tevz2, tevc2, tevx2, tpkv2
    real(real64)    :: aspin, bspin, spdiff, znuc, echarg, totchg
    real(real64)    :: sumocc, oldchg, nelec, vdevv, vdevc, vcalct, teval, tpkerr
    real(real64)    :: wgt2sv, c1c1, proj, ckck, ckcl2, occup, vdev, wgt, vcalc
    real(real64)    :: smax, hole, screen, schole, el2, pkerr, vcalc0, vcalc2
    real(real64)    :: vdev2, tetot, tpe, virial
    integer(int32)  :: ncore, nval_active, isp, iorb, nv, k, indxk, kk, km1, l, kl, i, j, ij
    integer(int32)  :: nelec_int
    integer(int32)  :: idx_sto, li_sto, ni_sto, m_sto
    real(real64)    :: etai_sto, anormi_sto
    integer(int32), allocatable :: n_msto(:), l_msto(:), ml_msto(:)
    real(real64),   allocatable :: eta_msto(:), anorm_msto(:)
    logical(kind=1) :: qpkinc, qnofit, qindef, qretry
    logical         :: qgrid_ok

    real(real64),   external :: ASUM
    integer(int32), external :: NOBLE

    qpkinc = .true.
    ORBOCC = 3 - ISPIN
    aspin  = ASUM(occ(1, 1), NORB, 1)
    bspin  = ASUM(occ(1, ISPIN), NORB, 1)
    sumocc = aspin + bspin
    znuc   = charge(1, IATOM)
    oldchg = charge(2, IATOM)
    nelec  = min(oldchg, znuc)
    ! NOBLE takes a default-integer atomic number. Pass a properly typed integer:
    ! passing the real(real64) NELEC by reference made NOBLE reinterpret the low
    ! 32 bits of the double's bit pattern as an integer, so NCORE was a
    ! pseudo-random function of the atom's electron count (e.g. flipping between
    ! 0 and 43 from cycle to cycle). The legacy code had NELEC as an implicitly
    ! typed INTEGER, i.e. truncation toward zero, which int() reproduces.
    nelec_int = int(nelec, int32)
    ncore  = NOBLE(nelec_int)
    nval_active = 0
    do iorb = ncore + 1, NORB
      if (sum(occ(iorb, 1:ISPIN)) > 1.0d-6) nval_active = nval_active + 1
    end do

    te     = 0.0d0
    vsum   = 0.0d0
    vmat   = 0.0d0
    vdevv  = 0.0d0; vdevc  = 0.0d0; vcalct = 0.0d0
    teval  = 0.0d0; tpkerr = 0.0d0

    wgt2sv = WGT2
    if (QATOM .or. QSEPAT) WGT2 = 0.0d0
    qnofit = (QFIXED .or. (.not. QC2FIT) .or. (WGTC + WGTV + WGT2 <= 0.1d0) .or. (NVTERM < 3))

    if (QPRINT) write(IW, 1000)

    do isp = 1, ISPIN
      if (QPRINT .and. QOPEN) write(IW, 2000) SPIN(isp)
      if (QPRINT) write(IW, 3000)

      do iorb = 1, NORB
        c1c1 = stevec(1, iorb, isp)**2
        ekin = stkin(1) * c1c1
        echf = stcoul(1, isp) * c1c1
        exhf = stexch(1, isp) * c1c1
        evz2 = zpot2(1) * c1c1
        evc2 = cpot2(1) * c1c1
        evx2 = xpot2(1, isp) * c1c1
        proj = pkpot(1, 1, isp) * c1c1
        pkv2 = pkpot(1, 2, isp) * c1c1
        epot(1:NVTERM) = stpot(1, 1:NVTERM) * c1c1

        if (MSTO >= 2) then
          do k = 2, MSTO
            indxk = index_arr(k)
            kk    = indxk + k
            km1   = k - 1
            ckck  = stevec(k, iorb, isp)**2
            ekin  = ekin + stkin(kk) * ckck
            echf  = echf + stcoul(kk, isp) * ckck
            exhf  = exhf + stexch(kk, isp) * ckck
            evz2  = evz2 + zpot2(kk) * ckck
            evc2  = evc2 + cpot2(kk) * ckck
            evx2  = evx2 + xpot2(kk, isp) * ckck
            proj  = proj + pkpot(kk, 1, isp) * ckck
            pkv2  = pkv2 + pkpot(kk, 2, isp) * ckck
            epot(1:NVTERM) = epot(1:NVTERM) + stpot(kk, 1:NVTERM) * ckck

            do l = 1, km1
              kl    = indxk + l
              ckcl2 = 2.0d0 * stevec(k, iorb, isp) * stevec(l, iorb, isp)
              ekin  = ekin + stkin(kl) * ckcl2
              echf  = echf + stcoul(kl, isp) * ckcl2
              exhf  = exhf + stexch(kl, isp) * ckcl2
              evz2  = evz2 + zpot2(kl) * ckcl2
              evc2  = evc2 + cpot2(kl) * ckcl2
              evx2  = evx2 + xpot2(kl, isp) * ckcl2
              proj  = proj + pkpot(kl, 1, isp) * ckcl2
              pkv2  = pkv2 + pkpot(kl, 2, isp) * ckcl2
              epot(1:NVTERM) = epot(1:NVTERM) + stpot(kl, 1:NVTERM) * ckcl2
            end do
          end do
        end if

        enuc  = -znuc * epot(1)
        occup = occ(iorb, isp) * ORBOCC

        vdev = ORBOCC * (echf - dot_product(vcoef(1:NVTERM), epot(1:NVTERM)))
        if (iorb <= ncore) then
          vdevc = vdevc + vdev**2
          wgt   = WGTC / (ORBOCC * ncore)
        else if (sum(occ(iorb, 1:ISPIN)) > 1.0d-6 .and. nval_active > 0) then
          vdevv = vdevv + vdev**2
          wgt   = WGTV / (ORBOCC * nval_active)
        else
          wgt   = 0.0d0
        end if

        vcalc = vcoef(1) * epot(1)
        ij    = 0
        do i = 2, NVTERM
          do j = 2, i
            ij = ij + 1
            vmat(ij) = vmat(ij) + epot(i) * wgt * epot(j)
          end do
          vsum(i - 1) = vsum(i - 1) + epot(i) * wgt * (echf - sumocc * epot(1))
          vcalc       = vcalc + vcoef(i) * epot(i)
        end do
        vcalct = vcalct + occup * vcalc

        ! N-1 virtual orbital potential modification (if QMODVO=T)
        if (QMODVO) then
          smax   = real(NATOM - 1, real64)
          hole   = 1.0d0 - occ(iorb, isp)
          if ((NATOM > 1) .and. (sumocc > 0.0d0) .and. (ELSUM > 0.0d0)) then
            screen = log(ELSUM / sumocc) / smax
          else
            screen = 0.0d0
          end if
          schole = hole * exp(-screen * proj)
          echf   = echf * (sumocc - schole) / sumocc
          el2    = ELSUM - sumocc
          if (el2 /= 0.0d0) evc2 = evc2 * (el2 - hole + schole) / el2

          charg(isp) = ASUM(occ(1, isp), NORB, 1)
          screen = 0.0d0
          if ((NATOM > 1) .and. (charg(isp) > 0.0d0) .and. (ELECS(isp) > 0.0d0)) then
            screen = log(ELECS(isp) / charg(isp)) / smax
          end if
          schole = hole * exp(-screen * proj)
          if (charg(isp) /= 0.0d0) exhf = exhf * (charg(isp) - schole) / charg(isp)
          el2 = ELECS(isp) - charg(isp)
          if (el2 /= 0.0d0) evx2 = evx2 * (el2 - hole + schole) / el2
        end if

        oe(2:8) = [ekin, enuc, echf, exhf, evz2, evc2, evx2]
        eorb    = sum(oe(2:8))
        pkv2    = eval(iorb, isp) * proj - pkv2
        if (qpkinc) eorb = eorb + pkv2
        oe(1)   = eorb
        oe(9)   = pkv2
        pkerr   = eval(iorb, isp) - eorb

        te     = te + occup * oe
        tpkerr = tpkerr + pkerr * occup
        teval  = teval + eval(iorb, isp) * occup

        if (QPRINT) then
          write(IW, 4000) iorb, oe, occup
          write(IW, 4500) eval(iorb, isp), vcalc, pkerr
        end if
        gamma(iorb, isp) = 0.5d0 * (eval(iorb, isp) + oe(1) + oe(2))
      end do
      charg(isp) = ASUM(occ(1, isp), NORB, 1)
    end do

    tekin = te(2); tenuc = te(3); techf = te(4); texhf = te(5)
    tevz2 = te(6); tevc2 = te(7); tevx2 = te(8); tpkv2 = te(9)

    !-------------------------------------------------------------------
    ! Solve for new local model potential coefficients VCOEF(1..NVTERM)
    !-------------------------------------------------------------------
    if (QCSTFT .and. (.not. qnofit) .and. (.not. QGRDFT)) then
      call solve_constrained_model_potential_fit(vmat, vsum, vfit2, tevz2, &
                                                 vcoef, sumocc, NVTERM, WGT2, qindef)
      vcalc0 = sum(vcoef(1:NVTERM))
      vcalc2 = dot_product(vcoef(1:NVTERM), vfit2(1:NVTERM))
      vdev2  = tevz2 - vcalc2
      if (abs(tevz2) < 0.1d0) vdev2 = -zzrep - vcalc2
      if (abs(zzrep) < 0.1d0) vdev2 = 0.0d0
    else
      vcalc0 = vcoef(1)
      vcalc2 = vcoef(1) * vfit2(1)
      ij     = 0
      do i = 2, NVTERM
        do j = 2, i
          ij = ij + 1
          vmat(ij) = vmat(ij) + WGT0 / sumocc &
                   + vfit2(i) * WGT2 * vfit2(j) / (tevz2 - 0.5d0)
        end do
        vsum(i - 1) = vsum(i - 1) - WGT0 &
                    + vfit2(i) * WGT2 * (tevz2 - sumocc * vfit2(1)) / (tevz2 - 0.5d0)
        vcalc0 = vcalc0 + vcoef(i)
        vcalc2 = vcalc2 + vcoef(i) * vfit2(i)
      end do

      vdev2 = tevz2 - vcalc2
      if (abs(tevz2) < 0.1d0) vdev2 = -zzrep - vcalc2
      if (abs(zzrep) < 0.1d0) vdev2 = 0.0d0

      if (.not. qnofit) then
        qindef = .false.
        if (QGRDFT) then
          ! Grid-based convex screening fit (see fit_screening_simplex); fall back
          ! to the weighted orbital-expectation fit only if it cannot be solved.
          call fit_screening_simplex(vexp, vcoef, sumocc, NVTERM, stevec, occ, qgrid_ok)
          qindef = .not. qgrid_ok
        end if
        if ((.not. QGRDFT) .or. qindef) then
          call solve_model_potential_fit(vmat, vsum, vcoef, sumocc, NVTERM, qindef)
        end if
        if (qindef) then
          call WARN(18, qretry)
          if (NVTERM >= 4) then
            vcoef(1)        = sumocc
            vcoef(2:NVTERM) = 0.0d0
            vcoef(3)        = -sumocc
          end if
        end if
      end if
    end if

    teorB = sum(te(2:8))
    if (qpkinc) teorB = teorB + tpkv2
    te(1) = teorB

    if (QPRINT) then
      write(IW, 5000) te, sumocc
      write(IW, 5500) teval, vcalct, vcalc2, tpkerr, sumocc - vcalc0
    end if

    tke    = tekin
    tetot  = teorB - 0.5d0 * (techf + texhf)
    tpe    = tetot - tke
    virial = tpe / tke
    toten  = tetot - 0.5d0 * (tevc2 + tevx2 - zzrep)
    if (qpkinc) toten = toten - 0.5d0 * tpkv2 + 0.5d0 * tpkerr

    if (QPRINT) write(IW, 6000) tetot, tke, tpe, virial, zzrep, toten

    aspin  = charg(1)
    bspin  = merge(charg(2), aspin, logical(QOPEN))
    echarg = aspin + bspin
    spdiff = aspin - bspin
    charge(2, IATOM) = echarg
    charge(3, IATOM) = spdiff
    totchg = znuc - echarg
    call run_chgan(charge)

    !-------------------------------------------------------------------
    ! Evaluate intra-atomic dipole moment for atom IATOM
    !-------------------------------------------------------------------
    if (allocated(mol_ws%elem_stream) .and. allocated(mol_ws%atom_dipoles)) then
      if (IELEM >= 1 .and. IELEM <= size(mol_ws%elem_stream)) then
        if (allocated(mol_ws%elem_stream(IELEM)%l) .and. MSTO > 0) then
          allocate(n_msto(MSTO), l_msto(MSTO), ml_msto(MSTO), eta_msto(MSTO), anorm_msto(MSTO))
          idx_sto = 0
          do i = 1, mol_ws%elem_stream(IELEM)%nsto
            li_sto     = mol_ws%elem_stream(IELEM)%l(i)
            ni_sto     = mol_ws%elem_stream(IELEM)%n(i)
            etai_sto   = mol_ws%elem_stream(IELEM)%eta(i)
            anormi_sto = mol_ws%elem_stream(IELEM)%anorm(i)
            do m_sto = -li_sto, li_sto
              idx_sto = idx_sto + 1
              if (idx_sto <= MSTO) then
                n_msto(idx_sto)     = ni_sto
                l_msto(idx_sto)     = li_sto
                ml_msto(idx_sto)    = m_sto
                eta_msto(idx_sto)   = etai_sto
                anorm_msto(idx_sto) = anormi_sto
              end if
            end do
          end do
          if (idx_sto == MSTO) then
            call eval_atomic_dipole(stevec, occ, n_msto, l_msto, ml_msto, eta_msto, anorm_msto, &
                                    MSTO, NORB, ISPIN, mol_ws%atom_dipoles(:, IATOM))
          end if
          deallocate(n_msto, l_msto, ml_msto, eta_msto, anorm_msto)
        end if
      end if
    end if

    vdevc = sqrt(vdevc)
    vdevv = sqrt(vdevv)
    mol_ws%last_pk_penalty  = mol_ws%last_pk_penalty + tpkerr
    mol_ws%last_rms_coulomb = max(mol_ws%last_rms_coulomb, max(vdevc, vdevv))
    if (QDFTEX) then
      mol_ws%last_rms_exchange = max(mol_ws%last_rms_exchange, abs(vdev2))
    end if

    if (QPRINT) then
      if (NORB * ISPIN > 20) write(IW, 8000)
      write(IW, 9000) vdevc, vdevv, vdev2
      if (qnofit) then
        write(IW, 9500)
      else
        write(IW, 10000) (nv, vcoef(nv), nv = 1, NVTERM)
      end if
    end if

    WGT2 = wgt2sv

1000 format(/' ENERGY ANALYSIS...')
2000 format(/' ',a8)
3000 format(/'  #',4x,'ORBITAL',3x,'KINETIC',4x,'INT',7x,'INT',7x,'INT',7x,'EXT',7x,'EXT',7x,'EXT',5x,'PSEUDO-',6x,'ORB'/ &
            ' ',6x,'ENERGY',4x,'ENERGY',3x,'NUC ATTR',2x,'COULOMB',3x,'EXCHANGE',2x, &
            'NUC ATTR',2x,'COULOMB',3x,'EXCHANGE',2x,'POTENTIAL',3x,'OCCUP'/)
4000 format(' ',i2,10f10.3)
4500 format(' ',3x,'(',f8.3,')',20x,'(',f8.3,')',42x,'(',f6.3,')')
5000 format(/' =>',10f10.3)
5500 format(' ',3x,'(',f8.3,')',20x,'(',f8.3,')',9x,'(',f9.3,')',22x,'(',f6.3,')','(',f8.3,')')
6000 format(/'    TOTAL ELECTRONIC ENERGY...',f11.4/ &
            '    TOTAL KINETIC ENERGY......',f11.4/ &
            '    TOTAL POTENTIAL ENERGY....',f11.4/ &
            '    VIRIAL THEOREM (V/T)......',f11.4// &
            '    NUCLEAR REPULSION.........',f11.4/ &
            '    CONTR. TO MOLEC. ENERGY...',f11.4)
8000 format(///' ')
9000 format(/' STD DEVS FOR THE LAST MODEL POTENTIAL FIT:'// &
            '    CORE HF COULOMB POTENTIAL........',f9.4/ &
            '    VALENCE HF COULOMB POTENTIAL.....',f9.4/ &
            '    EXTERNAL HF NUCLEAR ATTRACTION...',f9.4)
9500 format(/' MODEL POTENTIAL HELD FIXED...')
10000 format(/' NEW LOCAL MODEL POTENTIAL COEFFICIENTS ...'//(' ',i2,2x,f10.4))
  end subroutine run_analys

  subroutine run_chgan(charge)
    real(real64), intent(in) :: charge(3, NATOM)
    real(real64) :: znuc, echarg, spdiff, aspin, bspin, totchg

    znuc   = charge(1, IATOM)
    echarg = charge(2, IATOM)
    spdiff = charge(3, IATOM)
    aspin  = 0.5d0 * (echarg + spdiff)
    bspin  = 0.5d0 * (echarg - spdiff)
    totchg = znuc - echarg
    if (QPRINT) write(IW, 1000) aspin, bspin, spdiff, znuc, echarg, totchg

1000 format(/' SPIN ANALYSIS...'// &
            '    ALPHA SPIN...',f9.4/ &
            '    BETA SPIN....',f9.4/ &
            '    NET SPIN.....',f9.4// &
            ' CHARGE ANALYSIS...'// &
            '    NUCLEAR CHARGE......',f9.4/ &
            '    ELECTRONIC CHARGE...',f9.4/ &
            '    NET CHARGE..........',f9.4)
  end subroutine run_chgan

end module analys