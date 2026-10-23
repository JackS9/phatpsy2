!=======================================================================
! PHATPSY 2.0: Linear Algebra & Matrix Utilities Module (linalg.f90)
! Modern Fortran 2018 replacement for legacy addmat.f, asum.f, sum.f,
! dcopy.f, dotprd.f, erase.f, flip.f, scmult.f, and zero.f
!=======================================================================
module linalg
  use iso_fortran_env, only: real64, real32, int32
  implicit none

  private
  public :: addmat, submat, asum, dotprd, scmult, qzero, &
            erase, derase, ierase, qerase, dcopy, flip

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
    integer(int32), intent(in)  :: n
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
