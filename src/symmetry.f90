!=======================================================================
! PHATPSY 2.0: Symmetry Transformations Module (symmetry.f90)
! Modern Fortran 2018 replacement for legacy symop.f, symxyz.f, genrot.f,
! sympak.f, sympro.f, sympr1.f, sympr2.f, putsym.f, and rotsab.f
!=======================================================================
module symmetry
  use iso_fortran_env, only: real64, int32
  use state,           only: ALPHA_ANG, BETA_ANG
  use angular,         only: gendc, dcoef
  use linalg,          only: tritri, trisq
  implicit none

  private
  public :: symop, symxyz, genrot, sympak, sympro, sympr1, sympr2, &
            putsym, rotsab

contains

  !---------------------------------------------------------------------
  ! GENROT: Generate Cartesian 3x3 rotation matrix from Euler angles.
  !---------------------------------------------------------------------
  subroutine genrot(alpha, beta, gamma, rotate)
    real(real64), intent(in)  :: alpha, beta, gamma
    real(real64), intent(out) :: rotate(3, 3)

    real(real64) :: cosa, sina, cosb, sinb, cosg, sing

    cosa = cos(alpha)
    sina = sin(alpha)
    cosb = cos(beta)
    sinb = sin(beta)
    cosg = cos(gamma)
    sing = sin(gamma)

    rotate(1, 1) =  cosg * cosa - cosb * sina * sing
    rotate(1, 2) =  cosg * sina + cosb * cosa * sing
    rotate(1, 3) =  sing * sinb
    rotate(2, 1) = -sing * cosa - cosb * sina * cosg
    rotate(2, 2) = -sing * sina + cosb * cosa * cosg
    rotate(2, 3) =  cosg * sinb
    rotate(3, 1) =  sinb * sina
    rotate(3, 2) = -sinb * cosa
    rotate(3, 3) =  cosb
  end subroutine genrot

  !---------------------------------------------------------------------
  ! SYMXYZ: Determines Cartesian coordinates for symmetry-equivalent atom
  ! from its generator center (last = iatom - 1).
  !---------------------------------------------------------------------
  subroutine symxyz(isym, ixyz, xyz, coord, natom, iatom)
    integer(int32), intent(in)    :: isym, ixyz
    real(real64),   intent(inout) :: xyz(3)
    real(real64),   intent(in)    :: coord(3, *)
    integer(int32), intent(in)    :: natom, iatom

    real(real64)   :: pi, angle, parity, alpha, beta, gamma
    real(real64)   :: xlast, ylast, zlast
    real(real64)   :: rotate(3, 3)
    integer(int32) :: nfold, last
    external       :: bomb

    if (isym == 0 .or. isym == 1) return

    pi = 4.0d0 * atan(1.0d0)
    nfold = abs(isym)
    angle = 2.0d0 * pi / real(nfold, kind=real64)
    last = iatom - 1
    parity = 1.0d0

    if (isym < 0) then
      parity = -1.0d0
      angle = angle + pi
    end if

    xlast = coord(1, last)
    ylast = coord(2, last)
    zlast = coord(3, last)

    select case (ixyz)
    case (1, 0)
      alpha = angle
      beta  = 0.0d0
      gamma = 0.0d0
    case (10)
      alpha = pi / 2.0d0
      beta  = angle
      gamma = -pi / 2.0d0
    case (100)
      alpha = 0.0d0
      beta  = angle
      gamma = 0.0d0
    case default
      call bomb(4)
      alpha = angle
      beta  = 0.0d0
      gamma = 0.0d0
    end select

    call genrot(alpha, beta, gamma, rotate)

    xyz(1) = parity * (rotate(1, 1) * xlast + rotate(2, 1) * ylast + rotate(3, 1) * zlast)
    xyz(2) = parity * (rotate(1, 2) * xlast + rotate(2, 2) * ylast + rotate(3, 2) * zlast)
    xyz(3) = parity * (rotate(1, 3) * xlast + rotate(2, 3) * ylast + rotate(3, 3) * zlast)
  end subroutine symxyz

  !---------------------------------------------------------------------
  ! SYMOP: Transform atomic orbital and model potential coefficients to
  ! equivalent centers via Clebsch-Gordan / Wigner rotation matrices.
  !---------------------------------------------------------------------
  subroutine symop(isym, ixyz, stevec, vcoef, l, lv, d, scrat, &
                   nsto, msto, norb, ispin, nvterm, lmax, llmxp1)
    integer(int32), intent(in)    :: isym, ixyz
    integer(int32), intent(in)    :: nsto, msto, norb, ispin, nvterm, lmax, llmxp1
    real(real64),   intent(inout) :: stevec(msto, norb, ispin)
    real(real64),   intent(inout) :: vcoef(*)
    integer(int32), intent(in)    :: l(nsto), lv(*)
    real(real64),   intent(inout) :: d(:, :)
    real(real64),   intent(inout) :: scrat(*)

    real(real64)   :: pi, angle, parity, alpha, beta, gamma, coef
    real(real64), allocatable :: work(:)
    integer(int32) :: nfold, isp, iorb, isto, jsto, lq, llqp1, imlq, mlq, isig, isigma, nv, mv
    logical        :: qinvrt
    external       :: bomb

    if (isym == 0 .or. isym == 1) return

    pi = 4.0d0 * atan(1.0d0)
    nfold = isym
    angle = 2.0d0 * pi / real(abs(nfold), kind=real64)
    qinvrt = .false.

    if (isym < 0) then
      angle = angle + pi
      qinvrt = .true.
    end if

    select case (ixyz)
    case (1, 0)
      alpha = angle
      beta  = 0.0d0
      gamma = 0.0d0
    case (10)
      alpha = pi / 2.0d0
      beta  = angle
      gamma = -pi / 2.0d0
    case (100)
      alpha = 0.0d0
      beta  = angle
      gamma = 0.0d0
    case default
      call bomb(4)
      alpha = angle
      beta  = 0.0d0
      gamma = 0.0d0
    end select

    ALPHA_ANG = alpha
    BETA_ANG  = beta

    call gendc(d, lmax)

    if (allocated(work)) then
      if (size(work) < 2 * lmax + 1) then
        deallocate(work)
        allocate(work(2 * lmax + 1))
      end if
    else
      allocate(work(2 * lmax + 1))
    end if

    parity = 1.0d0
    do isp = 1, ispin
      do iorb = 1, norb
        jsto = 1
        do isto = 1, nsto
          lq = l(isto)
          if (qinvrt) parity = real(1 - 2 * mod(lq, 2), kind=real64)
          llqp1 = 2 * lq + 1
          work(1:llqp1) = stevec(jsto:jsto + llqp1 - 1, iorb, isp)
          do imlq = 1, llqp1
            mlq = imlq - lq - 1
            coef = 0.0d0
            do isig = 1, llqp1
              isigma = isig - lq - 1
              coef = coef + dcoef(d, lq, mlq, isigma) * work(isig)
            end do
            stevec(jsto, iorb, isp) = parity * coef
            jsto = jsto + 1
          end do
        end do
      end do
    end do

    if (nvterm <= 2) return

    mv = 1
    do nv = 1, nvterm
      if (mv > nvterm) exit
      lq = 0
      if (lv(mv) < 0) lq = -lv(mv)
      if (qinvrt) parity = real(1 - 2 * mod(lq, 2), kind=real64)
      llqp1 = 2 * lq + 1
      work(1:llqp1) = vcoef(mv:mv + llqp1 - 1)
      do imlq = 1, llqp1
        mlq = imlq - lq - 1
        coef = 0.0d0
        do isig = 1, llqp1
          isigma = isig - lq - 1
          coef = coef + dcoef(d, lq, mlq, isigma) * work(isig)
        end do
        vcoef(mv) = parity * coef
        mv = mv + 1
      end do
    end do
  end subroutine symop

  !---------------------------------------------------------------------
  ! SYMPAK: Packs an N x N square matrix A into lower-triangular packed
  !         form ASYM(N*(N+1)/2). If non-symmetric, uses (A_ij + A_ji)/2.
  !---------------------------------------------------------------------
  subroutine sympak(a, asym, n, n2)
    integer(int32), intent(in)  :: n, n2
    real(real64),   intent(in)  :: a(n, n)
    real(real64),   intent(out) :: asym(n2)
    integer(int32) :: i, j, ij
    real(real64), parameter :: TWO = 2.0d0

    ij = 0
    do i = 1, n
      do j = 1, i
        ij = ij + 1
        asym(ij) = (a(i, j) + a(j, i)) / TWO
      end do
    end do
  end subroutine sympak

  !---------------------------------------------------------------------
  ! SYMPRO: Modifies symmetric matrix A by adding/subtracting symmetric
  !         combinations with non-symmetric projection matrix B:
  !           icode =  4: A + BA
  !           icode =  3: A + AB
  !           icode =  2: A + 1/2(BA + AB^T)
  !           icode =  1: A + 1/2(AB + B^T A)
  !           icode =  0: A
  !           icode <  0: A - corresponding product
  !---------------------------------------------------------------------
  subroutine sympro(a, b, scrat, nbas, n2bas, icode)
    integer(int32), intent(in)    :: nbas, n2bas, icode
    real(real64),   intent(inout) :: a(n2bas)
    real(real64),   intent(in)    :: b(nbas, nbas)
    real(real64),   intent(out)   :: scrat(n2bas)

    real(real64), parameter :: ZERO = 0.0d0, TWO = 2.0d0
    integer(int32) :: i, j, k, ii, jj, ij, kk, ik, jk, igo
    real(real64)   :: sum_val

    igo = abs(icode)
    do i = 1, nbas
      ii = (i * (i - 1)) / 2
      do j = 1, i
        jj = (j * (j - 1)) / 2
        ij = ii + j
        sum_val = ZERO
        do k = 1, nbas
          kk = (k * (k - 1)) / 2
          if (k <= i) then
            ik = ii + k
          else
            ik = kk + i
          end if
          if (k <= j) then
            jk = jj + k
          else
            jk = kk + j
          end if
          select case (igo)
          case (0)
            ! No-op
          case (1)
            sum_val = sum_val + a(ik) * b(k, j) + a(jk) * b(k, i)
          case (2)
            sum_val = sum_val + a(ik) * b(j, k) + a(jk) * b(i, k)
          case (3)
            sum_val = sum_val + TWO * a(ik) * b(k, j)
          case (4)
            sum_val = sum_val + TWO * a(jk) * b(i, k)
          end select
        end do
        scrat(ij) = sum_val / TWO
      end do
    end do

    if (icode > 0) then
      a(1:n2bas) = a(1:n2bas) + scrat(1:n2bas)
    else if (icode < 0) then
      a(1:n2bas) = a(1:n2bas) - scrat(1:n2bas)
    end if
  end subroutine sympro

  !---------------------------------------------------------------------
  ! SYMPR1: Computes on-site contribution to Fock matrix:
  !         F = P*R*S + S*R*P - P*R*P
  !---------------------------------------------------------------------
  subroutine sympr1(p, r, s, f, scrat, n, n2)
    integer(int32), intent(in)  :: n, n2
    real(real64),   intent(in)  :: p(n2), r(n2), s(n2)
    real(real64),   intent(out) :: f(n2), scrat(n, n)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k, ij, ik, kj, ikmn, ikmx, kjmn, kjmx
    real(real64)   :: sum_val

    ! scrat = R * P
    call tritri(r, p, scrat, n, n2)

    ij = 0
    do i = 1, n
      do j = 1, i
        ij = ij + 1
        sum_val = ZERO
        do k = 1, n
          ikmn = min(i, k)
          ikmx = max(i, k)
          ik = (ikmx * (ikmx - 1)) / 2 + ikmn
          kjmn = min(k, j)
          kjmx = max(k, j)
          kj = (kjmx * (kjmx - 1)) / 2 + kjmn
          sum_val = sum_val + (s(ik) - p(ik)) * scrat(k, j) + scrat(k, i) * s(kj)
        end do
        f(ij) = sum_val
      end do
    end do
  end subroutine sympr1

  !---------------------------------------------------------------------
  ! SYMPR2: Generates off-site pseudopotential from on-site projection:
  !         R = R*S*P + P*S*R - P*S*R*S*P
  !---------------------------------------------------------------------
  subroutine sympr2(p, r, s, scrat1, scrat2, n, n2)
    integer(int32), intent(in)    :: n, n2
    real(real64),   intent(in)    :: p(n2), s(n2)
    real(real64),   intent(inout) :: r(n2)
    real(real64),   intent(out)   :: scrat1(n, n), scrat2(n, n)

    integer(int32) :: i, j, k, ij
    real(real64)   :: sum_val

    ! scrat1 = S * P
    call tritri(s, p, scrat1, n, n2)
    ! scrat2 = R * scrat1 = R * S * P
    call trisq(n, r, scrat1, scrat2)

    ij = 0
    do i = 1, n
      do j = 1, i
        ij = ij + 1
        sum_val = scrat2(i, j) + scrat2(j, i)
        do k = 1, n
          sum_val = sum_val - scrat1(k, i) * scrat2(k, j)
        end do
        r(ij) = sum_val
      end do
    end do
  end subroutine sympr2

  !---------------------------------------------------------------------
  ! PUTSYM: Writes out a symmetry packed matrix in lower triangular form.
  !---------------------------------------------------------------------
  subroutine putsym(title, symmat, nbas, iw)
    character(len=*), intent(in) :: title
    integer(int32),   intent(in) :: nbas, iw
    real(real64),     intent(in) :: symmat((nbas * (nbas + 1)) / 2)
    external :: putone

    write(iw, '(A,A)') '-', trim(title)
    call putone(symmat, nbas, iw)
  end subroutine putsym

  !---------------------------------------------------------------------
  ! ROTSAB: Rotates a set of overlaps between STOs on two centers from
  !         the common z-axis diatomic orientation to the molecular frame
  !         via Wigner D-matrices.
  !---------------------------------------------------------------------
  subroutine rotsab(stovlp, ovrlap, la, nstoa, mstoa, lb, nstob, mstob, d)
    integer(int32), intent(in)  :: nstoa, mstoa, nstob, mstob
    real(real64),   intent(out) :: stovlp(mstoa, mstob)
    real(real64),   intent(in)  :: ovrlap(:, :, :)
    integer(int32), intent(in)  :: la(nstoa), lb(nstob)
    real(real64),   intent(in)  :: d(:, :)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, ia, jb, lai, lbj, llai_p1, llbj_p1, lmin, llmn_p1
    integer(int32) :: im, jm, mai, mbj, iml, ml
    real(real64)   :: sum_val

    ia = 0
    do i = 1, nstoa
      lai = la(i)
      llai_p1 = 2 * lai + 1
      do im = 1, llai_p1
        mai = im - lai - 1
        ia = ia + 1
        jb = 0
        do j = 1, nstob
          lbj = lb(j)
          llbj_p1 = 2 * lbj + 1
          lmin = min(lai, lbj)
          llmn_p1 = 2 * lmin + 1
          do jm = 1, llbj_p1
            mbj = jm - lbj - 1
            jb = jb + 1
            sum_val = ZERO
            do iml = 1, llmn_p1
              ml = iml - lmin - 1
              sum_val = sum_val + dcoef(d, lai, mai, ml) * dcoef(d, lbj, mbj, ml) * ovrlap(iml, i, j)
            end do
            stovlp(ia, jb) = sum_val
          end do
        end do
      end do
    end do
  end subroutine rotsab

end module symmetry

!=======================================================================
! Global external subroutine wrappers for compatibility with legacy callers
!=======================================================================

subroutine genrot(alpha, beta, gamma, rotate)
  use iso_fortran_env, only: real64
  use symmetry,        only: mod_genrot => genrot
  implicit none
  real(real64), intent(in)  :: alpha, beta, gamma
  real(real64), intent(out) :: rotate(3, 3)
  call mod_genrot(alpha, beta, gamma, rotate)
end subroutine genrot

subroutine symxyz(isym, ixyz, xyz, coord, natom, iatom)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_symxyz => symxyz
  implicit none
  integer(int32), intent(in)    :: isym, ixyz
  real(real64),   intent(inout) :: xyz(3)
  real(real64),   intent(in)    :: coord(3, *)
  integer(int32), intent(in)    :: natom, iatom
  call mod_symxyz(isym, ixyz, xyz, coord, natom, iatom)
end subroutine symxyz

subroutine symop(isym, ixyz, stevec, vcoef, l, lv, d, scrat, &
                 nsto, msto, norb, ispin, nvterm, lmax, llmxp1)
  use iso_fortran_env, only: real64, int32
  use state,           only: L3MX
  use symmetry,        only: mod_symop => symop
  implicit none
  integer(int32), intent(in)    :: isym, ixyz
  integer(int32), intent(in)    :: nsto, msto, norb, ispin, nvterm, lmax, llmxp1
  real(real64),   intent(inout) :: stevec(msto, norb, ispin)
  real(real64),   intent(inout) :: vcoef(*)
  integer(int32), intent(in)    :: l(nsto), lv(*)
  real(real64),   intent(inout) :: d(L3MX, 2)
  real(real64),   intent(inout) :: scrat(*)
  call mod_symop(isym, ixyz, stevec, vcoef, l, lv, d, scrat, &
                 nsto, msto, norb, ispin, nvterm, lmax, llmxp1)
end subroutine symop

subroutine sympak(a, asym, n, n2)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_sympak => sympak
  implicit none
  integer(int32), intent(in)  :: n, n2
  real(real64),   intent(in)  :: a(n, n)
  real(real64),   intent(out) :: asym(n2)
  call mod_sympak(a, asym, n, n2)
end subroutine sympak

subroutine sympro(a, b, scrat, nbas, n2bas, icode)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_sympro => sympro
  implicit none
  integer(int32), intent(in)    :: nbas, n2bas, icode
  real(real64),   intent(inout) :: a(n2bas)
  real(real64),   intent(in)    :: b(nbas, nbas)
  real(real64),   intent(out)   :: scrat(n2bas)
  call mod_sympro(a, b, scrat, nbas, n2bas, icode)
end subroutine sympro

subroutine sympr1(p, r, s, f, scrat, n, n2)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_sympr1 => sympr1
  implicit none
  integer(int32), intent(in)  :: n, n2
  real(real64),   intent(in)  :: p(n2), r(n2), s(n2)
  real(real64),   intent(out) :: f(n2), scrat(n, n)
  call mod_sympr1(p, r, s, f, scrat, n, n2)
end subroutine sympr1

subroutine sympr2(p, r, s, scrat1, scrat2, n, n2)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_sympr2 => sympr2
  implicit none
  integer(int32), intent(in)    :: n, n2
  real(real64),   intent(in)    :: p(n2), s(n2)
  real(real64),   intent(inout) :: r(n2)
  real(real64),   intent(out)   :: scrat1(n, n), scrat2(n, n)
  call mod_sympr2(p, r, s, scrat1, scrat2, n, n2)
end subroutine sympr2

subroutine putsym(title, symmat, nbas, iw)
  use iso_fortran_env, only: real64, int32
  use symmetry,        only: mod_putsym => putsym
  implicit none
  character(len=*), intent(in) :: title
  integer(int32),   intent(in) :: nbas, iw
  real(real64),     intent(in) :: symmat((nbas * (nbas + 1)) / 2)
  call mod_putsym(title, symmat, nbas, iw)
end subroutine putsym

subroutine rotsab(stovlp, ovrlap, la, nstoa, mstoa, lb, nstob, mstob, d)
  use iso_fortran_env, only: real64, int32
  use state,           only: LLMXP1, L3MX
  use symmetry,        only: mod_rotsab => rotsab
  implicit none
  integer(int32), intent(in)  :: nstoa, mstoa, nstob, mstob
  real(real64),   intent(out) :: stovlp(mstoa, mstob)
  real(real64),   intent(in)  :: ovrlap(LLMXP1, nstoa, nstob)
  integer(int32), intent(in)  :: la(nstoa), lb(nstob)
  real(real64),   intent(in)  :: d(L3MX, 2)
  call mod_rotsab(stovlp, ovrlap, la, nstoa, mstoa, lb, nstob, mstob, d)
end subroutine rotsab
