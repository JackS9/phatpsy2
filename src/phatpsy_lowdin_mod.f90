! src/phatpsy_lowdin_mod.f90
!
! Modern Fortran 2018 Löwdin Orthonormal Transformation Module.
! Modernizes legacy lowdin.f using standard LAPACK DSPEV/DSPEVD eigensolvers
! and dynamic memory allocation, replacing legacy Jacobi iterations.
!
module phatpsy_lowdin_mod
  use iso_fortran_env, only: real64, int32
  implicit none
  private
  public :: run_lowdin

contains

  subroutine run_lowdin(ovrlap, ontran, scrat1, nbas, ndim, itype, qindef)
    integer(int32),  intent(in)    :: nbas, ndim, itype
    real(real64),    intent(inout) :: ovrlap((nbas * (nbas + 1)) / 2)
    real(real64),    intent(out)   :: ontran(ndim, nbas)
    real(real64),    intent(out)   :: scrat1(nbas, nbas)
    logical(kind=1), intent(out)   :: qindef

    real(real64), parameter :: ZERO = 0.0d0, ONE = 1.0d0, TOLER = 1.0d-10
    real(real64), allocatable :: w(:), work(:)
    integer(int32) :: info, i, j, k, kk, nn
    real(real64)   :: sum_val, diag

    qindef = .false.
    ontran = ZERO

    allocate(w(nbas), work(3 * nbas))

    ! Diagonalize packed symmetric overlap matrix S with LAPACK DSPEV
    call DSPEV('V', 'U', nbas, ovrlap, w, scrat1, nbas, work, info)
    deallocate(work)

    if (info /= 0) then
      write(6, *) '*** ERROR: DSPEV FAILED IN LOWDIN, INFO=', info
      qindef = .true.
      deallocate(w)
      return
    end if

    if (mod(itype, 2) == 0) then
      ! Symmetric powers:
      ! ITYPE = 0: S^-1    = U * D^-1   * U^T
      ! ITYPE = 2: S^-1/2  = U * D^-1/2 * U^T (Symmetric Löwdin)
      ! ITYPE = 4: S^+1/2  = U * D^+1/2 * U^T
      do j = 1, nbas
        do i = 1, nbas
          sum_val = ZERO
          do k = 1, nbas
            diag = w(k)
            if (abs(diag) > TOLER) then
              if (itype == 0) diag = ONE / diag
              if (itype == 2) then
                if (diag < ZERO) then
                  qindef = .true.
                  diag = ZERO
                else
                  diag = ONE / sqrt(diag)
                end if
              end if
              if (itype == 4) then
                if (diag < ZERO) then
                  qindef = .true.
                  diag = ZERO
                else
                  diag = sqrt(diag)
                end if
              end if
            else
              if (itype > 0 .and. diag < ZERO) qindef = .true.
              diag = ZERO
            end if
            sum_val = sum_val + scrat1(i, k) * diag * scrat1(j, k)
          end do
          ontran(i, j) = sum_val
        end do
      end do
    else
      ! Canonical transforms:
      ! ITYPE = 1: S^-1/2 = U * D^-1/2 (Canonical orthonormalization)
      ! ITYPE = 3: S^+1/2 = D^+1/2 * U^T
      do k = 1, nbas
        diag = w(k)
        if (diag < TOLER) then
          if (diag < -1.0d-8) then
            write(6, '(A, 1PE15.7)') '-*** WARNING - OVERLAP EIGENVALUE NEGATIVE IN LOWDIN: ', diag
            qindef = .true.
          end if
          diag = ZERO
        else
          diag = sqrt(diag)
        end if

        do i = 1, nbas
          if (diag > ZERO) then
            if (itype == 1) ontran(i, k) = scrat1(i, k) / diag
            if (itype == 3) ontran(k, i) = diag * scrat1(i, k)
          else
            ontran(i, k) = ZERO
          end if
        end do
      end do
    end if

    ! Set diagonal matrix D in OVRLAP
    nn = (nbas * (nbas + 1)) / 2
    ovrlap(1:nn) = ZERO
    do k = 1, nbas
      kk = (k * (k + 1)) / 2
      ovrlap(kk) = w(k)
    end do

    deallocate(w)
  end subroutine run_lowdin

end module phatpsy_lowdin_mod

! External legacy entry point callable from unconverted fixed-form Fortran
subroutine LOWDIN(ovrlap, ontran, scrat1, nbas, ndim, itype, qindef)
  use phatpsy_lowdin_mod, only: run_lowdin
  use iso_fortran_env, only: real64, int32
  implicit none
  integer(int32),  intent(in)    :: nbas, ndim, itype
  real(real64),    intent(inout) :: ovrlap((nbas * (nbas + 1)) / 2)
  real(real64),    intent(out)   :: ontran(ndim, nbas)
  real(real64),    intent(out)   :: scrat1(nbas, nbas)
  logical(kind=1), intent(out)   :: qindef

  call run_lowdin(ovrlap, ontran, scrat1, nbas, ndim, itype, qindef)
end subroutine LOWDIN
