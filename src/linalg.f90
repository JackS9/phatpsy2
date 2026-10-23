!=======================================================================
! PHATPSY 2.0: Linear Algebra & Matrix Utilities Module (linalg.f90)
! Modern Fortran 2018 replacement for legacy addmat.f, asum.f, sum.f,
! dcopy.f, dotprd.f, erase.f, flip.f, scmult.f, zero.f,
! trisq.f, tritri.f, uthu.f, invert.f, matinv.f, schmid.f, normlz.f,
! round.f, maxovl.f, hesbrg.f, norhes.f, heseig.f, and hesvec.f
!=======================================================================
module linalg
  use iso_fortran_env, only: real64, real32, int32
  implicit none

  private
  public :: addmat, submat, asum, dotprd, scmult, qzero, &
            erase, derase, ierase, qerase, dcopy, flip,  &
            trisq, tritri, uthu, invert, matinv, schmid, &
            normlz, round, maxovl, hesbrg, norhes, heseig, hesvec, &
            cdct, ctsc, csct, pcct, dmpab, dmpatb, insert, insrtd

contains

  !---------------------------------------------------------------------
  ! ADDMAT: Add two contiguous arrays: C = A + B
  !---------------------------------------------------------------------
  subroutine addmat(a, b, c, n)
    integer(int32), intent(in)  :: n
    real(real64),   intent(in)  :: a(n), b(n)
    real(real64),   intent(out) :: c(n)
    c(1:n) = a(1:n) + b(1:n)
  end subroutine addmat

  !---------------------------------------------------------------------
  ! SUBMAT: Subtract two contiguous arrays: C = A - B
  !---------------------------------------------------------------------
  subroutine submat(a, b, c, n)
    integer(int32), intent(in)  :: n
    real(real64),   intent(in)  :: a(n), b(n)
    real(real64),   intent(out) :: c(n)
    c(1:n) = a(1:n) - b(1:n)
  end subroutine submat

  !---------------------------------------------------------------------
  ! ASUM: Sum evenly spaced elements in an array
  !---------------------------------------------------------------------
  real(real64) function asum(start, nterms, incr)
    integer(int32), intent(in) :: nterms, incr
    real(real64),   intent(in) :: start(*)
    integer(int32) :: i, iend

    asum = 0.0d0
    iend = 1 + incr * (nterms - 1)
    do i = 1, iend, incr
      asum = asum + start(i)
    end do
  end function asum

  !---------------------------------------------------------------------
  ! DOTPRD: Dot product of two vectors
  !---------------------------------------------------------------------
  real(real64) function dotprd(a, b, n)
    integer(int32), intent(in) :: n
    real(real64),   intent(in) :: a(n), b(n)
    dotprd = dot_product(a(1:n), b(1:n))
  end function dotprd

  !---------------------------------------------------------------------
  ! SCMULT: Scalar multiplication of an array in-place
  !---------------------------------------------------------------------
  subroutine scmult(array, scalar, ndim)
    integer(int32), intent(in)    :: ndim
    real(real64),   intent(in)    :: scalar
    real(real64),   intent(inout) :: array(ndim)
    array(1:ndim) = scalar * array(1:ndim)
  end subroutine scmult

  !---------------------------------------------------------------------
  ! QZERO: Returns .true. if all elements in array are zero
  !---------------------------------------------------------------------
  logical function qzero(array, n)
    integer(int32), intent(in) :: n
    real(real64),   intent(in) :: array(n)
    qzero = all(array(1:n) == 0.0d0)
  end function qzero

  !---------------------------------------------------------------------
  ! ERASE / DERASE / IERASE / QERASE: Zero out arrays
  !---------------------------------------------------------------------
  subroutine erase(r4, n)
    integer(int32), intent(in)  :: n
    real(real32),   intent(out) :: r4(n)
    r4(1:n) = 0.0_real32
  end subroutine erase

  subroutine derase(r8, n)
    integer(int32), intent(in)  :: n
    real(real64),   intent(out) :: r8(n)
    r8(1:n) = 0.0d0
  end subroutine derase

  subroutine ierase(i4, n)
    integer(int32), intent(in)  :: n
    integer(int32), intent(out) :: i4(n)
    i4(1:n) = 0
  end subroutine ierase

  subroutine qerase(q1, n)
    integer(int32), intent(in)   :: n
    logical(kind=1), intent(out) :: q1(n)
    q1(1:n) = .false.
  end subroutine qerase

  !---------------------------------------------------------------------
  ! DCOPY: Copy array contiguously: B = A
  !---------------------------------------------------------------------
  subroutine dcopy(a, n, b)
    integer(int32), intent(in)  :: n
    real(real64),   intent(in)  :: a(n)
    real(real64),   intent(out) :: b(n)
    b(1:n) = a(1:n)
  end subroutine dcopy

  !---------------------------------------------------------------------
  ! FLIP: Interchange columns of two matrices (or within same matrix)
  !---------------------------------------------------------------------
  subroutine flip(array1, icol1, ncol1, array2, icol2, ncol2, nrow)
    integer(int32), intent(in)    :: icol1, ncol1, icol2, ncol2, nrow
    real(real64),   intent(inout) :: array1(nrow, ncol1), array2(nrow, ncol2)
    real(real64) :: tmp(nrow)

    tmp = array1(:, icol1)
    array1(:, icol1) = array2(:, icol2)
    array2(:, icol2) = tmp
  end subroutine flip

  !---------------------------------------------------------------------
  ! TRISQ: Multiplies a packed symmetric matrix times a square matrix:
  !        RESULT = H * U
  !---------------------------------------------------------------------
  subroutine trisq(n, h, u, result)
    integer(int32), intent(in)  :: n
    real(real64),   intent(in)  :: h((n * (n + 1)) / 2), u(n, n)
    real(real64),   intent(out) :: result(n, n)
    integer(int32) :: i, j, k, ik, ii
    real(real64)   :: s

    do i = 1, n
      ii = (i * (i - 1)) / 2
      do j = 1, n
        s = 0.0d0
        do k = 1, n
          if (k <= i) then
            ik = ii + k
          else
            ik = (k * (k - 1)) / 2 + i
          end if
          s = s + h(ik) * u(k, j)
        end do
        result(i, j) = s
      end do
    end do
  end subroutine trisq

  !---------------------------------------------------------------------
  ! TRITRI: Product of two packed triangular matrices: AB = A * B
  !---------------------------------------------------------------------
  subroutine tritri(a, b, ab, n, n2)
    integer(int32), intent(in)  :: n, n2
    real(real64),   intent(in)  :: a(n2), b(n2)
    real(real64),   intent(out) :: ab(n, n)
    integer(int32) :: i, j, k, ii, jj, ik, jk, kk
    real(real64)   :: s

    do i = 1, n
      ii = (i * (i - 1)) / 2
      do j = 1, n
        jj = (j * (j - 1)) / 2
        s = 0.0d0
        do k = 1, n
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
          s = s + a(ik) * b(jk)
        end do
        ab(i, j) = s
      end do
    end do
  end subroutine tritri

  !---------------------------------------------------------------------
  ! UTHU: Similarity transformation of a packed symmetric matrix:
  !       RESULT = U^T * H * U
  !---------------------------------------------------------------------
  subroutine uthu(n, h, u, scrat, result)
    integer(int32), intent(in)    :: n
    real(real64),   intent(in)    :: h((n * (n + 1)) / 2), u(n, n)
    real(real64),   intent(out)   :: scrat(n, n)
    real(real64),   intent(out)   :: result((n * (n + 1)) / 2)
    integer(int32) :: i, j, k, ii, ik, ij
    real(real64)   :: s

    ! Step 1: SCRAT = H * U
    do i = 1, n
      ii = (i * (i - 1)) / 2
      do j = 1, n
        s = 0.0d0
        do k = 1, n
          if (k <= i) then
            ik = ii + k
          else
            ik = (k * (k - 1)) / 2 + i
          end if
          s = s + h(ik) * u(k, j)
        end do
        scrat(i, j) = s
      end do
    end do

    ! Step 2: RESULT = U^T * SCRAT (lower triangle packed)
    do i = 1, n
      ii = (i * (i - 1)) / 2
      do j = 1, i
        ij = ii + j
        s = 0.0d0
        do k = 1, n
          s = s + u(k, i) * scrat(k, j)
        end do
        result(ij) = s
      end do
    end do
  end subroutine uthu

  !---------------------------------------------------------------------
  ! INVERT: Invert general matrix via standard LAPACK LU (DGETRF / DGETRI)
  !---------------------------------------------------------------------
  subroutine invert(matrix, invers, nsize, ierror, nrow, ncolum)
    integer(int32), intent(in)    :: nsize, nrow, ncolum
    real(real64),   intent(in)    :: matrix(nrow, ncolum)
    real(real64),   intent(inout) :: invers(nrow, ncolum)
    integer(int32), intent(out)   :: ierror

    integer(int32) :: info, lwork
    integer(int32), allocatable :: ipiv(:)
    real(real64), allocatable :: a_tmp(:, :), work(:)
    real(real64) :: work_opt(1)

    ierror = 0
    if (nsize <= 0) return
    if (nsize > nrow .or. nsize > ncolum) then
      ierror = 1
      return
    end if

    if (nsize == 1) then
      if (matrix(1, 1) == 0.0d0) then
        ierror = 1
      else
        invers(1, 1) = 1.0d0 / matrix(1, 1)
      end if
      return
    end if

    allocate(a_tmp(nsize, nsize), ipiv(nsize))
    a_tmp(1:nsize, 1:nsize) = matrix(1:nsize, 1:nsize)

    ! LU factorization
    call dgetrf(nsize, nsize, a_tmp, nsize, ipiv, info)
    if (info /= 0) then
      ierror = 1
      deallocate(a_tmp, ipiv)
      return
    end if

    ! Workspace query
    call dgetri(nsize, a_tmp, nsize, ipiv, work_opt, -1, info)
    if (info /= 0) then
      lwork = nsize * 64
    else
      lwork = max(nsize, int(work_opt(1), int32))
    end if

    allocate(work(lwork))
    call dgetri(nsize, a_tmp, nsize, ipiv, work, lwork, info)
    if (info /= 0) then
      ierror = 1
    else
      invers(1:nsize, 1:nsize) = a_tmp(1:nsize, 1:nsize)
    end if

    deallocate(a_tmp, ipiv, work)
  end subroutine invert

  !---------------------------------------------------------------------
  ! MATINV: In-place matrix inversion and linear system solution with
  !         determinant computation via LAPACK DGETRF / DGETRI / DGETRS
  !---------------------------------------------------------------------
  subroutine matinv(a, n, b, m, determ)
    integer(int32), intent(in)    :: n, m
    real(real64),   intent(inout) :: a(n, n)
    real(real64),   intent(inout) :: b(n, *)
    real(real64),   intent(out)   :: determ

    integer(int32) :: i, info, lwork
    integer(int32), allocatable :: ipiv(:)
    real(real64), allocatable :: work(:)
    real(real64) :: work_opt(1)

    determ = 1.0d0
    if (n <= 0) return

    allocate(ipiv(n))

    ! 1. LU factorization: P * A = L * U
    call dgetrf(n, n, a, n, ipiv, info)
    if (info /= 0) then
      determ = 0.0d0
      deallocate(ipiv)
      return
    end if

    ! 2. Compute determinant from diagonal of U and permutation parity
    do i = 1, n
      if (ipiv(i) /= i) determ = -determ
      determ = determ * a(i, i)
    end do

    ! 3. If m > 0, solve A * X = B in-place for B
    if (m > 0) then
      call dgetrs('N', n, m, a, n, ipiv, b, n, info)
      if (info /= 0) determ = 0.0d0
    end if

    ! 4. Invert A in-place using LU factors
    call dgetri(n, a, n, ipiv, work_opt, -1, info)
    if (info /= 0) then
      lwork = n * 64
    else
      lwork = max(n, int(work_opt(1), int32))
    end if

    allocate(work(lwork))
    call dgetri(n, a, n, ipiv, work, lwork, info)
    if (info /= 0) determ = 0.0d0

    deallocate(ipiv, work)
  end subroutine matinv

  !---------------------------------------------------------------------
  ! SCHMID: In-place Schmidt orthonormalization of packed positive-definite
  !         overlap matrix S(*), returning upper-triangular transformation T.
  !---------------------------------------------------------------------
  subroutine schmid(s, nrows)
    integer(int32), intent(in)    :: nrows
    real(real64),   intent(inout) :: s((nrows * (nrows + 1)) / 2)

    real(real64), parameter :: ZERO = 0.0d0, ONE = 1.0d0, TOLER = 1.0d-12
    integer(int32) :: n, nm1, k, j, kback, kpn, kpnp1, nk, kpk, jk, nj, nn, jn, kpj
    real(real64)   :: ovrlap, sum_val, factor

    if (nrows <= 0) return
    if (s(1) <= ZERO) then
      s(1) = ZERO
      return
    end if
    s(1) = ONE / sqrt(s(1))
    if (nrows <= 1) return

    kpn = 1
    do n = 2, nrows
      kpnp1 = kpn + n
      ovrlap = ZERO
      nm1 = n - 1
      nk = kpn + nm1

      kpk = (n * (n - 1)) / 2
      do kback = 1, nm1
        k = n - kback
        kpk = kpk - k
        jk = kpk
        nj = kpn
        sum_val = ZERO
        do j = 1, k
          nj = nj + 1
          jk = jk + 1
          sum_val = sum_val + s(nj) * s(jk)
        end do
        s(nk) = sum_val
        ovrlap = ovrlap + sum_val * sum_val
        nk = nk - 1
      end do
      nn = kpnp1
      ovrlap = s(nn) - ovrlap

      if (ovrlap <= TOLER) then
        write(6, '(A,I6,A,I6,A)') ' *** WARNING ***  OVERLAP MATRIX PASSED TO SCHMID SINGULAR AT ROW ', &
                                  n, '. OVERLAP MATRIX HAS ', nrows, ' ROWS.'
        jn = kpn
        do j = 1, n
          jn = jn + 1
          s(jn) = ZERO
        end do
      else
        factor = ONE / sqrt(ovrlap)
        jn = kpn
        kpj = 0
        do j = 1, nm1
          jn = jn + 1
          nk = kpn + j
          jk = kpj + j
          sum_val = ZERO
          do k = j, nm1
            sum_val = sum_val - s(jk) * s(nk)
            nk = nk + 1
            jk = jk + k
          end do
          s(jn) = sum_val * factor
          kpj = kpj + j
        end do
        s(jn + 1) = factor
      end if
      kpn = kpnp1
    end do
  end subroutine schmid

  !---------------------------------------------------------------------
  ! NORMLZ: Normalizes column eigenvectors such that C_i^T * S * C_i = 1
  !---------------------------------------------------------------------
  subroutine normlz(evec, nvec, ovrlap, nbas, n2bas)
    integer(int32), intent(in)    :: nvec, nbas, n2bas
    real(real64),   intent(inout) :: evec(nbas, nvec)
    real(real64),   intent(in)    :: ovrlap(n2bas)

    real(real64), parameter :: ZERO = 0.0d0, ONE = 1.0d0, TWO = 2.0d0
    integer(int32) :: i, j, k, jj, jk, jm1
    real(real64)   :: cji, sumjk, cinorm

    do i = 1, nvec
      sumjk = ZERO
      do j = 1, nbas
        jj = (j * (j - 1)) / 2
        cji = evec(j, i)
        sumjk = sumjk + cji * cji * ovrlap(jj + j)
        if (j > 1) then
          jm1 = j - 1
          do k = 1, jm1
            jk = jj + k
            sumjk = sumjk + TWO * evec(k, i) * cji * ovrlap(jk)
          end do
        end if
      end do
      if (sumjk /= ZERO) then
        cinorm = ONE / sqrt(sumjk)
        evec(1:nbas, i) = evec(1:nbas, i) * cinorm
      end if
    end do
  end subroutine normlz

  !---------------------------------------------------------------------
  ! ROUND: Round real number X to specified power of 10 (10**N)
  !---------------------------------------------------------------------
  real(real64) function round(x, n)
    real(real64),   intent(in) :: x
    integer(int32), intent(in) :: n
    real(real64) :: signif

    signif = 10.0d0**n
    round = anint(x / signif) * signif
  end function round

  !---------------------------------------------------------------------
  ! MAXOVL: Reorders columns in EVEC2 to maximize overlap with EVEC1
  !---------------------------------------------------------------------
  subroutine maxovl(iorder, evec1, evec2, ovlp, nbas, n2bas, nvec, qnword)
    integer(int32),  intent(in)    :: nbas, n2bas, nvec
    real(real64),    intent(in)    :: evec1(nbas, nvec), evec2(nbas, nvec)
    real(real64),    intent(in)    :: ovlp(n2bas)
    integer(int32),  intent(out)   :: iorder(nvec)
    logical(kind=1), intent(out)   :: qnword

    real(real64), parameter :: ZERO = 0.0d0, HALF = 0.5d0, ONE = 1.0d0
    integer(int32) :: i, j, k, l, kk, kl, jmax
    real(real64)   :: sii, sij, smax, remain
    logical        :: found_half

    iorder(1:nvec) = 0
    qnword = .false.

    do i = 1, nvec
      sii = ZERO
      if (iorder(i) == 0) then
        do k = 1, nbas
          kk = (k * (k - 1)) / 2
          do l = 1, nbas
            if (l <= k) then
              kl = kk + l
            else
              kl = (l * (l - 1)) / 2 + k
            end if
            sii = sii + evec1(k, i) * evec2(l, i) * ovlp(kl)
          end do
        end do
        sii = sii * sii
        jmax = i
        if (sii >= HALF) then
          iorder(i) = i
          cycle
        end if
      end if

      qnword = .true.
      smax = sii
      remain = ONE - sii
      found_half = .false.
      do j = 1, nvec
        if (iorder(j) /= 0 .or. j == i) cycle
        sij = ZERO
        do k = 1, nbas
          kk = (k * (k - 1)) / 2
          do l = 1, nbas
            if (l <= k) then
              kl = kk + l
            else
              kl = (l * (l - 1)) / 2 + k
            end if
            sij = sij + evec1(k, i) * evec2(l, j) * ovlp(kl)
          end do
        end do
        sij = sij * sij
        remain = remain - sij
        if (sij >= HALF) then
          iorder(j) = i
          found_half = .true.
          exit
        end if
        if (sij >= smax) then
          smax = sij
          jmax = j
        end if
        if (remain < smax) exit
      end do
      if (.not. found_half) iorder(jmax) = i
    end do
  end subroutine maxovl

  !---------------------------------------------------------------------
  ! HESBRG: Converts matrix A to upper Hessenberg form via similarity
  !         transformation with partial pivoting.
  !---------------------------------------------------------------------
  subroutine hesbrg(a, n, mpr)
    integer(int32), intent(in)    :: n
    real(real64),   intent(inout) :: a(n, n)
    integer(int32), intent(out)   :: mpr(n)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: m, i, k, kmax, mpiv, m2, n1
    real(real64)   :: piv, tem

    n1 = n - 1
    do m = 1, n
      piv = ZERO
      mpiv = m + 1
      do i = 1, n
        kmax = min(i - 1, m)
        if (kmax >= 2) then
          do k = 2, kmax
            a(i, m) = a(i, m) - a(i, k - 1) * a(k, m)
          end do
        end if
        if (m < n) then
          if (m > 1) then
            do k = m, n1
              a(i, m) = a(i, m) + a(i, k + 1) * a(k + 1, m - 1)
            end do
          end if
          if (i > m) then
            if (abs(a(i, m)) > abs(piv)) then
              mpiv = i
              piv = a(i, m)
            end if
          end if
        end if
      end do

      if (m == n) return
      mpr(m + 1) = mpiv
      if (mpiv /= m + 1) then
        do i = 1, n
          tem = a(mpiv, i)
          a(mpiv, i) = a(m + 1, i)
          a(m + 1, i) = tem
        end do
        do i = 1, n
          tem = a(i, mpiv)
          a(i, mpiv) = a(i, m + 1)
          a(i, m + 1) = tem
        end do
      end if

      m2 = m + 2
      if (m2 <= n) then
        if (piv == ZERO) then
          a(m2:n, m) = ZERO
        else
          a(m2:n, m) = a(m2:n, m) / piv
        end if
      end if
    end do
  end subroutine hesbrg

  !---------------------------------------------------------------------
  ! NORHES: Normalizes subdiagonal of upper Hessenberg matrix H.
  !---------------------------------------------------------------------
  subroutine norhes(h, n, fac, nn)
    integer(int32), intent(in)    :: n
    real(real64),   intent(inout) :: h(n, n)
    real(real64),   intent(out)   :: fac(n)
    integer(int32), intent(out)   :: nn

    real(real64), parameter :: ZERO = 0.0d0, ONE = 1.0d0
    integer(int32) :: j, k, n1
    real(real64)   :: a

    n1 = n - 1
    nn = n
    do j = 2, n
      a = h(j, j - 1)
      if (a == ZERO) then
        nn = j - 1
        return
      end if
      fac(j) = a
      h(j, j - 1) = ONE
      do k = 2, j
        h(k - 1, j) = h(k - 1, j) * a
      end do
      if (j < n) then
        h(j + 1, j) = h(j + 1, j) * a
        a = ONE / a
        do k = j, n1
          h(j, k + 1) = h(j, k + 1) * a
        end do
      end if
    end do
    nn = n
  end subroutine norhes

  !---------------------------------------------------------------------
  ! HESEIG: Eigenvalues of upper Hessenberg matrix.
  !---------------------------------------------------------------------
  subroutine heseig(a, n, nrt, x, xp, eig, eta, nzc, ierr)
    integer(int32), intent(in)    :: n, nrt, nzc
    real(real64),   intent(in)    :: a(n, n), eta
    real(real64),   intent(inout) :: x(n), xp(n)
    real(real64),   intent(out)   :: eig(nrt)
    integer(int32), intent(out)   :: ierr

    real(real64), parameter :: TOL = 2.0d0 * (16.0d0**(-8))
    real(real64), parameter :: ONE = 1.0d0
    integer(int32) :: nroot, m, nit, kk, ni, nri, ll, mm, nn, ipp, ii, jj
    real(real64)   :: r, y

    ierr = 0
    nroot = min(nrt, nzc)
    if (eta < 0.0d0) then
      r = a(1, 1) + ONE
      do nn = 2, n
        y = a(nn, nn)
        if (nn < n) y = y + ONE
        do ipp = 2, nn
          y = y + abs(a(ipp - 1, nn))
        end do
        if (r < y) r = y
      end do
    else
      r = a(1, 1) - ONE
      do ii = 2, n
        y = a(ii, ii)
        if (ii < n) y = y - ONE
        do jj = 2, ii
          y = y - abs(a(jj - 1, ii))
        end do
        if (y < r) r = y
      end do
    end if

    do m = 1, nroot
      xp(nzc) = ONE
      nit = 0
      do
        x(nzc) = r - a(nzc, nzc)
        do kk = 2, nzc
          ni = nzc - kk + 1
          x(ni) = (r - a(ni, ni)) * x(ni + 1) - a(ni, nzc)
          xp(ni) = x(ni + 1) + (r - a(ni, ni)) * xp(ni + 1)
          if (kk > 2) then
            nri = kk - 2
            do ll = 1, nri
              x(ni) = x(ni) - a(ni, ni + ll) * x(ni + ll + 1)
              xp(ni) = xp(ni) - a(ni, ni + ll) * xp(ni + ll + 1)
            end do
          end if
        end do

        if (m == 1) then
          y = x(ni) / xp(ni)
        else
          y = xp(ni) / x(ni)
          do mm = 2, m
            y = y - ONE / (r - eig(mm - 1))
          end do
          y = ONE / y
        end if

        r = r - y
        nit = nit + 1
        if (abs(y / r) <= TOL) then
          eig(m) = r
          exit
        end if
        if (nit > 25) then
          ierr = 1
          return
        end if
      end do
    end do
    r = r + eta
  end subroutine heseig

  !---------------------------------------------------------------------
  ! HESVEC: Eigenvectors of upper Hessenberg matrix by inverse iteration.
  !---------------------------------------------------------------------
  subroutine hesvec(h, n, nn, scr, root, eta, scrv, vec, ierr)
    integer(int32), intent(in)    :: n, nn
    real(real64),   intent(in)    :: h(n, n), root, eta
    real(real64),   intent(inout) :: scr(nn, nn), scrv(nn)
    real(real64),   intent(out)   :: vec(n)
    integer(int32), intent(out)   :: ierr

    real(real64), parameter :: UNZ = 15.0d0
    real(real64), parameter :: TOL = 16.0d0**(-12)
    real(real64), parameter :: ZERO = 0.0d0, ONE = 1.0d0
    integer(int32) :: nd, ii, jj, iii, ni, nit, iflag
    real(real64)   :: fac, avc

    ierr = 0
    fac = root + eta
    nd = nn - 1
    vec(nn:n) = ZERO

    do ii = 1, nn
      do jj = 1, nn
        scr(ii, jj) = h(ii, jj)
      end do
      scr(ii, ii) = scr(ii, ii) - fac
    end do

    do ii = 1, nd
      if (abs(scr(ii, ii)) < ONE) then
        do jj = ii, nn
          fac = scr(ii, jj)
          scr(ii, jj) = scr(ii + 1, jj)
          scr(ii + 1, jj) = fac
        end do
        fac = scr(ii + 1, ii)
      else
        scr(ii + 1, ii) = UNZ
        fac = ONE / scr(ii, ii)
      end if
      do jj = ii, nd
        scr(ii + 1, jj + 1) = scr(ii + 1, jj + 1) - fac * scr(ii, jj + 1)
      end do
    end do

    scrv(1:nn) = ONE
    nit = 0

    do
      vec(nn) = scrv(nn) / scr(nn, nn)
      avc = vec(nn)
      do iii = 1, nd
        ni = nn - iii
        fac = scrv(ni)
        do jj = ni, nd
          fac = fac - scr(ni, jj + 1) * vec(jj + 1)
        end do
        vec(ni) = fac / scr(ni, ni)
      end do
      if (abs(avc) < abs(vec(ni))) avc = vec(ni)

      iflag = 0
      do jj = 1, nn
        vec(jj) = vec(jj) / avc
        if (abs((vec(jj) - scrv(jj)) / vec(jj)) > TOL) iflag = 1
      end do
      if (iflag == 0) return

      nit = nit + 1
      if (nit > 8) then
        ierr = 1
        return
      end if

      scrv(1:nn) = vec(1:nn)
      do iii = 1, nd
        if (scr(iii + 1, iii) == UNZ) then
          scrv(iii + 1) = scrv(iii + 1) - scrv(iii) / scr(iii, iii)
        else
          fac = scrv(iii + 1)
          scrv(iii + 1) = scrv(iii) - fac * scr(iii + 1, iii)
          scrv(iii) = fac
        end if
      end do
    end do
  end subroutine hesvec

  !---------------------------------------------------------------------
  ! CDCT: Similarity transformation on diagonal matrix D: DD = C * D * C^T
  !---------------------------------------------------------------------
  subroutine cdct(c, d, dd, nvec, nbas, n2bas)
    integer(int32), intent(in)  :: nvec, nbas, n2bas
    real(real64),   intent(in)  :: c(nbas, nvec), d(nvec)
    real(real64),   intent(out) :: dd(n2bas)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k, ii, ij
    real(real64)   :: sum_val

    do i = 1, nbas
      ii = (i * (i - 1)) / 2
      do j = 1, i
        ij = ii + j
        sum_val = ZERO
        do k = 1, nvec
          sum_val = sum_val + c(i, k) * d(k) * c(j, k)
        end do
        dd(ij) = sum_val
      end do
    end do
  end subroutine cdct

  !---------------------------------------------------------------------
  ! CTSC: Similarity transformation of symmetric matrix S: SS = C^T * S * C
  !---------------------------------------------------------------------
  subroutine ctsc(c, s, scrat, ss, nbas, nvec)
    integer(int32), intent(in)    :: nbas, nvec
    real(real64),   intent(in)    :: c(nbas, nvec)
    real(real64),   intent(in)    :: s(*)
    real(real64),   intent(out)   :: scrat(nbas, nvec)
    real(real64),   intent(out)   :: ss(*)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k, ii, ij, ik
    real(real64)   :: sum_val

    do i = 1, nbas
      ii = (i * (i - 1)) / 2
      do j = 1, nvec
        sum_val = ZERO
        do k = 1, nbas
          ik = ii + k
          if (k > i) ik = (k * (k - 1)) / 2 + i
          sum_val = sum_val + s(ik) * c(k, j)
        end do
        scrat(i, j) = sum_val
      end do
    end do

    do i = 1, nvec
      ii = (i * (i - 1)) / 2
      do j = 1, i
        ij = ii + j
        sum_val = ZERO
        do k = 1, nbas
          sum_val = sum_val + c(k, i) * scrat(k, j)
        end do
        ss(ij) = sum_val
      end do
    end do
  end subroutine ctsc

  !---------------------------------------------------------------------
  ! CSCT: Transpose similarity transformation: SS = C * S * C^T
  !---------------------------------------------------------------------
  subroutine csct(c, s, scrat, ss, nbas, nvec)
    integer(int32), intent(in)    :: nbas, nvec
    real(real64),   intent(in)    :: c(nvec, nbas)
    real(real64),   intent(in)    :: s(*)
    real(real64),   intent(out)   :: scrat(nbas, nvec)
    real(real64),   intent(out)   :: ss(*)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k, ii, ij, ik
    real(real64)   :: sum_val

    do i = 1, nbas
      ii = (i * (i - 1)) / 2
      do j = 1, nvec
        sum_val = ZERO
        do k = 1, nbas
          ik = ii + k
          if (k > i) ik = (k * (k - 1)) / 2 + i
          sum_val = sum_val + s(ik) * c(j, k)
        end do
        scrat(i, j) = sum_val
      end do
    end do

    do i = 1, nvec
      ii = (i * (i - 1)) / 2
      do j = 1, i
        ij = ii + j
        sum_val = ZERO
        do k = 1, nbas
          sum_val = sum_val + c(i, k) * scrat(k, j)
        end do
        ss(ij) = sum_val
      end do
    end do
  end subroutine csct

  !---------------------------------------------------------------------
  ! PCCT: Generates PSINV = P * S^-1 = P * C * C^T
  !---------------------------------------------------------------------
  subroutine pcct(p, c, psinv, nvec, nbas, n2bas)
    integer(int32), intent(in)  :: nvec, nbas, n2bas
    real(real64),   intent(in)  :: p(n2bas), c(nbas, nvec)
    real(real64),   intent(out) :: psinv(nbas, nbas)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: k, l, m, i, kk, mm, km
    real(real64)   :: sum_val

    do k = 1, nbas
      kk = (k * (k - 1)) / 2
      do l = 1, nbas
        sum_val = ZERO
        do i = 1, nvec
          do m = 1, nbas
            mm = (m * (m - 1)) / 2
            km = kk + m
            if (m > k) km = mm + k
            sum_val = sum_val + p(km) * c(m, i) * c(l, i)
          end do
        end do
        psinv(k, l) = sum_val
      end do
    end do
  end subroutine pcct

  !---------------------------------------------------------------------
  ! DMPAB: Multiplies two rectangular matrices: C = A * B
  !---------------------------------------------------------------------
  subroutine dmpab(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
    integer(int32), intent(in)  :: na, ma, namx, mamx
    integer(int32), intent(in)  :: nb, mb, nbmx, mbmx
    integer(int32), intent(in)  :: ncmx, mcmx
    real(real64),   intent(in)  :: a(namx, mamx), b(nbmx, mbmx)
    real(real64),   intent(out) :: c(ncmx, mcmx)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k
    real(real64)   :: sum_val

    if (ma /= nb) then
      write(*,*) 'Incompatible dimensions in DMPAB...'
      return
    end if

    do i = 1, na
      do j = 1, mb
        sum_val = ZERO
        do k = 1, ma
          sum_val = sum_val + a(i, k) * b(k, j)
        end do
        c(i, j) = sum_val
      end do
    end do
  end subroutine dmpab

  !---------------------------------------------------------------------
  ! DMPATB: Multiplies rectangular matrices with left transpose: C = A^T * B
  !---------------------------------------------------------------------
  subroutine dmpatb(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
    integer(int32), intent(in)  :: na, ma, namx, mamx
    integer(int32), intent(in)  :: nb, mb, nbmx, mbmx
    integer(int32), intent(in)  :: ncmx, mcmx
    real(real64),   intent(in)  :: a(namx, mamx), b(nbmx, mbmx)
    real(real64),   intent(out) :: c(ncmx, mcmx)

    real(real64), parameter :: ZERO = 0.0d0
    integer(int32) :: i, j, k
    real(real64)   :: sum_val

    if (na /= nb) then
      write(*,*) 'Incompatible dimensions in DMPATB...'
      return
    end if

    do i = 1, ma
      do j = 1, mb
        sum_val = ZERO
        do k = 1, na
          sum_val = sum_val + a(k, i) * b(k, j)
        end do
        c(i, j) = sum_val
      end do
    end do
  end subroutine dmpatb

  !---------------------------------------------------------------------
  ! INSERT: Inserts a rectangular block into a larger lower triangular packed matrix
  !---------------------------------------------------------------------
  subroutine insert(block, supmat, i, j, nrow, ncol)
    integer(int32), intent(in)    :: i, j, nrow, ncol
    real(real64),   intent(in)    :: block(nrow, ncol)
    real(real64),   intent(inout) :: supmat(*)

    integer(int32) :: k, l, ij, ijskip

    ij = (i * (i - 1)) / 2 + j
    ijskip = i - ncol - 1
    do k = 1, nrow
      do l = 1, ncol
        supmat(ij) = block(k, l)
        ij = ij + 1
      end do
      ij = ij + ijskip + k
    end do
  end subroutine insert

  !---------------------------------------------------------------------
  ! INSRTD: Inserts diagonal symmetry-packed block into larger symmetry-packed matrix
  !---------------------------------------------------------------------
  subroutine insrtd(block, supmat, i, n)
    integer(int32), intent(in)    :: i, n
    real(real64),   intent(in)    :: block(*)
    real(real64),   intent(inout) :: supmat(*)

    integer(int32) :: k, l, kk, kl, ij

    ij = (i * (i + 1)) / 2
    do k = 1, n
      kk = (k * (k - 1)) / 2
      do l = 1, k
        kl = kk + l
        supmat(ij) = block(kl)
        ij = ij + 1
      end do
      ij = ij + i - 1
    end do
  end subroutine insrtd

end module linalg

!=======================================================================
! Global external subroutine wrappers for compatibility with legacy callers
!=======================================================================

subroutine derase(r8, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_derase => derase
  implicit none
  integer(int32), intent(in)  :: n
  real(real64),   intent(out) :: r8(n)
  call mod_derase(r8, n)
end subroutine derase

subroutine erase(r4, n)
  use iso_fortran_env, only: real32, int32
  use linalg,          only: mod_erase => erase
  implicit none
  integer(int32), intent(in)  :: n
  real(real32),   intent(out) :: r4(n)
  call mod_erase(r4, n)
end subroutine erase

subroutine ierase(i4, n)
  use iso_fortran_env, only: int32
  use linalg,          only: mod_ierase => ierase
  implicit none
  integer(int32), intent(in)  :: n
  integer(int32), intent(out) :: i4(n)
  call mod_ierase(i4, n)
end subroutine ierase

subroutine qerase(q1, n)
  use iso_fortran_env, only: int32
  use linalg,          only: mod_qerase => qerase
  implicit none
  integer(int32), intent(in)   :: n
  logical(kind=1), intent(out) :: q1(n)
  call mod_qerase(q1, n)
end subroutine qerase

subroutine dcopy(a, n, b)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_dcopy => dcopy
  implicit none
  integer(int32), intent(in)  :: n
  real(real64),   intent(in)  :: a(n)
  real(real64),   intent(out) :: b(n)
  call mod_dcopy(a, n, b)
end subroutine dcopy

subroutine addmat(a, b, c, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_addmat => addmat
  implicit none
  integer(int32), intent(in)  :: n
  real(real64),   intent(in)  :: a(n), b(n)
  real(real64),   intent(out) :: c(n)
  call mod_addmat(a, b, c, n)
end subroutine addmat

subroutine submat(a, b, c, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_submat => submat
  implicit none
  integer(int32), intent(in)  :: n
  real(real64),   intent(in)  :: a(n), b(n)
  real(real64),   intent(out) :: c(n)
  call mod_submat(a, b, c, n)
end subroutine submat

real(8) function asum(start, nterms, incr)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_asum => asum
  implicit none
  integer(int32), intent(in) :: nterms, incr
  real(real64),   intent(in) :: start(*)
  asum = mod_asum(start, nterms, incr)
end function asum

real(8) function dotprd(a, b, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_dotprd => dotprd
  implicit none
  integer(int32), intent(in) :: n
  real(real64),   intent(in) :: a(n), b(n)
  dotprd = mod_dotprd(a, b, n)
end function dotprd

subroutine scmult(array, scalar, ndim)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_scmult => scmult
  implicit none
  integer(int32), intent(in)    :: ndim
  real(real64),   intent(in)    :: scalar
  real(real64),   intent(inout) :: array(ndim)
  call mod_scmult(array, scalar, ndim)
end subroutine scmult

logical function qzero(array, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_qzero => qzero
  implicit none
  integer(int32), intent(in) :: n
  real(real64),   intent(in) :: array(n)
  qzero = mod_qzero(array, n)
end function qzero

subroutine flip(array1, icol1, ncol1, array2, icol2, ncol2, nrow)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_flip => flip
  implicit none
  integer(int32), intent(in)    :: icol1, ncol1, icol2, ncol2, nrow
  real(real64),   intent(inout) :: array1(nrow, ncol1), array2(nrow, ncol2)
  call mod_flip(array1, icol1, ncol1, array2, icol2, ncol2, nrow)
end subroutine flip

subroutine trisq(n, h, u, result)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_trisq => trisq
  implicit none
  integer(int32), intent(in)  :: n
  real(real64),   intent(in)  :: h((n * (n + 1)) / 2), u(n, n)
  real(real64),   intent(out) :: result(n, n)
  call mod_trisq(n, h, u, result)
end subroutine trisq

subroutine tritri(a, b, ab, n, n2)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_tritri => tritri
  implicit none
  integer(int32), intent(in)  :: n, n2
  real(real64),   intent(in)  :: a(n2), b(n2)
  real(real64),   intent(out) :: ab(n, n)
  call mod_tritri(a, b, ab, n, n2)
end subroutine tritri

subroutine uthu(n, h, u, scrat, result)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_uthu => uthu
  implicit none
  integer(int32), intent(in)    :: n
  real(real64),   intent(in)    :: h((n * (n + 1)) / 2), u(n, n)
  real(real64),   intent(out)   :: scrat(n, n)
  real(real64),   intent(out)   :: result((n * (n + 1)) / 2)
  call mod_uthu(n, h, u, scrat, result)
end subroutine uthu

subroutine invert(matrix, invers, nsize, ierror, nrow, ncolum)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_invert => invert
  implicit none
  integer(int32), intent(in)    :: nsize, nrow, ncolum
  real(real64),   intent(in)    :: matrix(nrow, ncolum)
  real(real64),   intent(inout) :: invers(nrow, ncolum)
  integer(int32), intent(out)   :: ierror
  call mod_invert(matrix, invers, nsize, ierror, nrow, ncolum)
end subroutine invert

subroutine matinv(a, n, b, m, determ)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_matinv => matinv
  implicit none
  integer(int32), intent(in)    :: n, m
  real(real64),   intent(inout) :: a(n, n)
  real(real64),   intent(inout) :: b(n, *)
  real(real64),   intent(out)   :: determ
  call mod_matinv(a, n, b, m, determ)
end subroutine matinv

subroutine schmid(s, nrows)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_schmid => schmid
  implicit none
  integer(int32), intent(in)    :: nrows
  real(real64),   intent(inout) :: s((nrows * (nrows + 1)) / 2)
  call mod_schmid(s, nrows)
end subroutine schmid

subroutine normlz(evec, nvec, ovrlap, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_normlz => normlz
  implicit none
  integer(int32), intent(in)    :: nvec, nbas, n2bas
  real(real64),   intent(inout) :: evec(nbas, nvec)
  real(real64),   intent(in)    :: ovrlap(n2bas)
  call mod_normlz(evec, nvec, ovrlap, nbas, n2bas)
end subroutine normlz

real(8) function round(x, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_round => round
  implicit none
  real(real64),   intent(in) :: x
  integer(int32), intent(in) :: n
  round = mod_round(x, n)
end function round

subroutine maxovl(iorder, evec1, evec2, ovlp, nbas, n2bas, nvec, qnword)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_maxovl => maxovl
  implicit none
  integer(int32),  intent(in)    :: nbas, n2bas, nvec
  real(real64),    intent(in)    :: evec1(nbas, nvec), evec2(nbas, nvec)
  real(real64),    intent(in)    :: ovlp(n2bas)
  integer(int32),  intent(out)   :: iorder(nvec)
  logical(kind=1), intent(out)   :: qnword
  call mod_maxovl(iorder, evec1, evec2, ovlp, nbas, n2bas, nvec, qnword)
end subroutine maxovl

subroutine hesbrg(a, n, mpr)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_hesbrg => hesbrg
  implicit none
  integer(int32), intent(in)    :: n
  real(real64),   intent(inout) :: a(n, n)
  integer(int32), intent(out)   :: mpr(n)
  call mod_hesbrg(a, n, mpr)
end subroutine hesbrg

subroutine norhes(h, n, fac, nn)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_norhes => norhes
  implicit none
  integer(int32), intent(in)    :: n
  real(real64),   intent(inout) :: h(n, n)
  real(real64),   intent(out)   :: fac(n)
  integer(int32), intent(out)   :: nn
  call mod_norhes(h, n, fac, nn)
end subroutine norhes

subroutine heseig(a, n, nrt, x, xp, eig, eta, nzc, *)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_heseig => heseig
  implicit none
  integer(int32), intent(in)    :: n, nrt, nzc
  real(real64),   intent(in)    :: a(n, n), eta
  real(real64),   intent(inout) :: x(n), xp(n)
  real(real64),   intent(out)   :: eig(nrt)
  integer(int32) :: ierr
  call mod_heseig(a, n, nrt, x, xp, eig, eta, nzc, ierr)
  if (ierr /= 0) return 1
end subroutine heseig

subroutine hesvec(h, n, nn, scr, root, eta, scrv, vec, *)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_hesvec => hesvec
  implicit none
  integer(int32), intent(in)    :: n, nn
  real(real64),   intent(in)    :: h(n, n), root, eta
  real(real64),   intent(inout) :: scr(nn, nn), scrv(nn)
  real(real64),   intent(out)   :: vec(n)
  integer(int32) :: ierr
  call mod_hesvec(h, n, nn, scr, root, eta, scrv, vec, ierr)
  if (ierr /= 0) return 1
end subroutine hesvec

subroutine cdct(c, d, dd, nvec, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_cdct => cdct
  implicit none
  integer(int32), intent(in)  :: nvec, nbas, n2bas
  real(real64),   intent(in)  :: c(nbas, nvec), d(nvec)
  real(real64),   intent(out) :: dd(n2bas)
  call mod_cdct(c, d, dd, nvec, nbas, n2bas)
end subroutine cdct

subroutine ctsc(c, s, scrat, ss, nbas, nvec)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_ctsc => ctsc
  implicit none
  integer(int32), intent(in)    :: nbas, nvec
  real(real64),   intent(in)    :: c(nbas, nvec)
  real(real64),   intent(in)    :: s(*)
  real(real64),   intent(out)   :: scrat(nbas, nvec)
  real(real64),   intent(out)   :: ss(*)
  call mod_ctsc(c, s, scrat, ss, nbas, nvec)
end subroutine ctsc

subroutine csct(c, s, scrat, ss, nbas, nvec)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_csct => csct
  implicit none
  integer(int32), intent(in)    :: nbas, nvec
  real(real64),   intent(in)    :: c(nvec, nbas)
  real(real64),   intent(in)    :: s(*)
  real(real64),   intent(out)   :: scrat(nbas, nvec)
  real(real64),   intent(out)   :: ss(*)
  call mod_csct(c, s, scrat, ss, nbas, nvec)
end subroutine csct

subroutine pcct(p, c, psinv, nvec, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_pcct => pcct
  implicit none
  integer(int32), intent(in)  :: nvec, nbas, n2bas
  real(real64),   intent(in)  :: p(n2bas), c(nbas, nvec)
  real(real64),   intent(out) :: psinv(nbas, nbas)
  call mod_pcct(p, c, psinv, nvec, nbas, n2bas)
end subroutine pcct

subroutine dmpab(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_dmpab => dmpab
  implicit none
  integer(int32), intent(in)  :: na, ma, namx, mamx
  integer(int32), intent(in)  :: nb, mb, nbmx, mbmx
  integer(int32), intent(in)  :: ncmx, mcmx
  real(real64),   intent(in)  :: a(namx, mamx), b(nbmx, mbmx)
  real(real64),   intent(out) :: c(ncmx, mcmx)
  call mod_dmpab(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
end subroutine dmpab

subroutine dmpatb(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_dmpatb => dmpatb
  implicit none
  integer(int32), intent(in)  :: na, ma, namx, mamx
  integer(int32), intent(in)  :: nb, mb, nbmx, mbmx
  integer(int32), intent(in)  :: ncmx, mcmx
  real(real64),   intent(in)  :: a(namx, mamx), b(nbmx, mbmx)
  real(real64),   intent(out) :: c(ncmx, mcmx)
  call mod_dmpatb(a, na, ma, namx, mamx, b, nb, mb, nbmx, mbmx, c, ncmx, mcmx)
end subroutine dmpatb

subroutine insert(block, supmat, i, j, nrow, ncol)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_insert => insert
  implicit none
  integer(int32), intent(in)    :: i, j, nrow, ncol
  real(real64),   intent(in)    :: block(nrow, ncol)
  real(real64),   intent(inout) :: supmat(*)
  call mod_insert(block, supmat, i, j, nrow, ncol)
end subroutine insert

subroutine insrtd(block, supmat, i, n)
  use iso_fortran_env, only: real64, int32
  use linalg,          only: mod_insrtd => insrtd
  implicit none
  integer(int32), intent(in)    :: i, n
  real(real64),   intent(in)    :: block(*)
  real(real64),   intent(inout) :: supmat(*)
  call mod_insrtd(block, supmat, i, n)
end subroutine insrtd
