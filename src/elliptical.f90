!=======================================================================
! PHATPSY 2.0: Elliptical Coordinate Expansion & Quadratures (elliptical.f90)
! Modern Fortran 2018 replacement for legacy absum.f, anmbnm.f, ascale.f,
! bscale.f, and ogen.f
!=======================================================================
module elliptical
  use iso_fortran_env, only: real64, int32
  implicit none

  private
  public :: absum, anmbnm, ascale, bscale, ogen

contains

  !---------------------------------------------------------------------
  ! ABSUM: Sum product of A- and B-functions with STO elliptical expansion
  ! coefficients for overlap (ITYPE=0) or nuclear attraction (ITYPE=1).
  !---------------------------------------------------------------------
  subroutine absum(sum_val, a, b, omegai, omegaj, limi, limj, itype)
    real(real64),   intent(out) :: sum_val
    real(real64),   intent(in)  :: a(:), b(:)
    real(real64),   intent(in)  :: omegai(:, :), omegaj(:, :)
    integer(int32), intent(in)  :: limi, limj, itype

    integer(int32) :: munulo(2), munuhi(2), mu(3), nu(3)
    integer(int32) :: munumn, munumx, imunu, mum1, num1, ip, iq, jp, jq
    integer(int32) :: ipjpmu, iqjqnu
    real(real64)   :: opqi, opqj, term

    data munulo /1, 2/
    data munuhi /1, 3/
    data mu     /0, 1, 0/
    data nu     /0, 0, 1/

    sum_val = 0.0d0
    munumn  = munulo(itype + 1)
    munumx  = munuhi(itype + 1)

    do imunu = munumn, munumx
      mum1 = mu(imunu) - 1
      num1 = nu(imunu) - 1

      do ip = 1, limi
        do iq = 1, limi
          opqi = omegai(ip, iq)
          if (opqi == 0.0d0) cycle

          do jp = 1, limj
            ipjpmu = ip + jp + mum1

            do jq = 1, limj
              opqj = omegaj(jp, jq)
              if (opqj == 0.0d0) cycle

              iqjqnu = iq + jq + num1

              if (itype == 0) then
                term = a(ipjpmu + 2) * b(iqjqnu) - a(ipjpmu) * b(iqjqnu + 2)
              else
                term = a(ipjpmu) * b(iqjqnu)
              end if

              sum_val = sum_val + term * opqi * opqj
            end do
          end do
        end do
      end do
    end do
  end subroutine absum

  !---------------------------------------------------------------------
  ! ASCALE: Evaluate scaled A(N, X) function via forward recursion.
  !---------------------------------------------------------------------
  subroutine ascale(n, x, a, nadim)
    integer(int32), intent(in)    :: n, nadim
    real(real64),   intent(in)    :: x
    real(real64),   intent(inout) :: a(nadim)

    integer(int32) :: np1, i
    real(real64)   :: fim1

    if (x <= 0.0d0) then
      write(*, '(A, 1PE23.15, A)') '*** ERROR *** Non-positive argument, X = ', x, ' in routine ASCALE.'
      stop 1
    end if

    if (n >= nadim) then
      write(*, '(A, I5, A)') '*** ERROR *** N = ', n, ' exceeds specified dimension in routine ASCALE.'
      stop 1
    end if

    a(1) = 1.0d0 / x
    if (n == 0) return

    np1  = n + 1
    fim1 = 1.0d0
    do i = 2, np1
      a(i) = a(1) + fim1 * a(i - 1) / x
      fim1 = fim1 + 1.0d0
    end do
  end subroutine ascale

  !---------------------------------------------------------------------
  ! BSCALE: Evaluate scaled B(N, X) function with piecewise stability.
  !---------------------------------------------------------------------
  subroutine bscale(n, x, b, nbdim)
    integer(int32), intent(in)    :: n, nbdim
    real(real64),   intent(in)    :: x
    real(real64),   intent(inout) :: b(nbdim)

    real(real64), parameter :: POINT6 = 0.6d0
    real(real64), parameter :: TINY   = 1.0d-15
    real(real64), parameter :: OVRFLO = 20.0d0

    integer(int32) :: np1, ieven, i, k, kpk, nflip, im1mod, mup1, nup1
    real(real64)   :: xabs, odd, even, fi, xx, u, tsum, tk, scale, t0, tnu, sum_val

    if (n >= nbdim) then
      write(*, '(A, I5, A)') '*** ERROR *** N = ', n, ' exceeds specified dimension in routine BSCALE.'
      stop 1
    end if

    np1  = n + 1
    xabs = abs(x)

    if (xabs > OVRFLO) then
      ! Large argument finite series representation
      t0 = 1.0d0 / xabs
      do mup1 = 1, np1
        sum_val = 0.0d0
        tnu     = t0
        do nup1 = 1, mup1
          sum_val = sum_val + tnu
          tnu     = -xabs * tnu / real(nup1, kind=real64)
        end do
        if (x < 0.0d0 .and. mod(mup1, 2) == 0) sum_val = -sum_val
        b(mup1) = sum_val
        t0 = real(mup1, kind=real64) * t0 / xabs
      end do
      return
    end if

    if (xabs > POINT6 * real(n, kind=real64)) then
      ! Recursion formula (sufficiently large argument avoiding subtraction loss)
      odd  = 2.0d0 / x
      even = odd * sinh(x)
      b(1) = even
      if (n == 0) then
        scale = exp(-xabs)
        b(1)  = scale * b(1)
        return
      end if

      odd = odd * cosh(x)
      fi  = 0.0d0
      do ieven = 2, np1, 2
        fi = fi + 2.0d0
        b(ieven) = ((fi - 1.0d0) / x) * b(ieven - 1) - odd
        if (ieven == np1) exit
        b(ieven + 1) = (fi / x) * b(ieven) + even
      end do

      scale = exp(-xabs)
      do i = 1, np1
        b(i) = scale * b(i)
      end do
      return
    end if

    ! Infinite series representation for small arguments [0, 0.6*N]
    xx    = x * x
    nflip = 1
    do i = 1, np1
      nflip  = -nflip
      im1mod = mod(i - 1, 2)
      u      = 2.0d0
      if (nflip == 1) u = -u * x
      tsum = u / real(i + im1mod, kind=real64)

      if (tsum /= 0.0d0) then
        k = 1
        do
          kpk = k + k
          u   = xx * u / real(kpk * (kpk + nflip), kind=real64)
          tk  = u / real(i + kpk + im1mod, kind=real64)
          if (abs(tk / tsum) < TINY) exit
          tsum = tsum + tk
          k    = k + 1
        end do
      end if
      b(i) = tsum
    end do

    scale = exp(-xabs)
    do i = 1, np1
      b(i) = scale * b(i)
    end do
  end subroutine bscale

  !---------------------------------------------------------------------
  ! ANMBNM: Generates scaled modified A and B functions via recurrence.
  !---------------------------------------------------------------------
  subroutine anmbnm(anm, alpha, bnm, beta, n, m, nabdim)
    integer(int32), intent(in)    :: n, m, nabdim
    real(real64),   intent(in)    :: alpha, beta
    real(real64),   intent(inout) :: anm(nabdim), bnm(nabdim)

    integer(int32) :: k, limit, i

    call ascale(n + 2 * m, alpha, anm, nabdim)
    call bscale(n + 2 * m, beta, bnm, nabdim)

    if (m == 0) return

    do k = 1, m
      limit = n + 2 * (m - k) + 1
      do i = 1, limit
        anm(i) = anm(i + 2) - anm(i)
        bnm(i) = bnm(i) - bnm(i + 2)
      end do
    end do
  end subroutine anmbnm

  !---------------------------------------------------------------------
  ! OGEN: Compute expansion coefficients of an STO in elliptical coordinates.
  !---------------------------------------------------------------------
  subroutine ogen(n, l, ml, rhalf, omega, nodim, binom, gamma, kp, plm)
    integer(int32), intent(in)    :: n, l, ml, nodim
    real(real64),   intent(in)    :: rhalf
    real(real64),   intent(out)   :: omega(nodim, nodim)
    real(real64),   intent(in)    :: binom(:), gamma(:)
    integer(int32), intent(in)    :: kp(:)
    real(real64),   intent(inout) :: plm(:)

    integer(int32) :: mlabs, nabs, lm, limit, lamp1, lambda, ll
    integer(int32) :: maxq, q, p, nll, index1, index2, index3
    real(real64)   :: term, rnm1, sum_val
    logical        :: flip_sign

    mlabs = abs(ml)
    nabs  = abs(n)

    omega = 0.0d0

    lm    = l - mlabs
    limit = 1 + lm / 2
    do lamp1 = 1, limit
      lambda = lamp1 - 1
      ll     = kp(l + 1) + lamp1
      term   = binom(ll) * gamma(2 * (l - lambda) + 1) / &
               (gamma(lm - 2 * lambda + 1) * gamma(l + 1) * (2.0d0**l))
      if (mod(mlabs + lambda, 2) /= 0) term = -term
      plm(lamp1) = term
    end do

    rnm1 = rhalf**(nabs - 1)
    maxq = nabs - mlabs
    do q = 1, maxq
      flip_sign = ((n < 0) .and. (mod(lm - q - 1, 2) /= 0))
      do p = 1, q
        sum_val = 0.0d0
        do lamp1 = 1, limit
          lambda = lamp1 - 1
          nll    = nabs + 2 * lambda - l - 1
          index1 = nll + q - p
          if (mod(index1, 2) /= 0) cycle

          index1 = index1 / 2
          if (index1 > nll) cycle
          index1 = kp(nll + 1) + index1 + 1

          index2 = p + q - nll - 2
          if (index2 < 0) cycle
          if (mod(index2, 2) /= 0) cycle

          index2 = index2 / 2 + 1
          index3 = lm - 2 * lambda + 1
          if (index2 > index3) cycle

          index2  = kp(index3) + index2
          sum_val = sum_val + plm(lamp1) * binom(index1) * binom(index2)
        end do

        sum_val = rnm1 * sum_val
        if (flip_sign) sum_val = -sum_val
        omega(p, q) = sum_val
        if (p == q) cycle

        if (n < 0 .and. mod(q - p, 2) /= 0) sum_val = -sum_val
        omega(q, p) = sum_val
      end do
    end do
  end subroutine ogen

end module elliptical

!=======================================================================
! Global external compatibility wrappers
!=======================================================================

subroutine absum(sum_val, a, b, omegai, omegaj, limi, limj, itype)
  use iso_fortran_env, only: real64, int32
  use elliptical,      only: mod_absum => absum
  implicit none
  real(real64),   intent(out) :: sum_val
  real(real64),   intent(in)  :: a(:), b(:)
  real(real64),   intent(in)  :: omegai(:, :), omegaj(:, :)
  integer(int32), intent(in)  :: limi, limj, itype
  call mod_absum(sum_val, a, b, omegai, omegaj, limi, limj, itype)
end subroutine absum

subroutine anmbnm(anm, alpha, bnm, beta, n, m, nabdim)
  use iso_fortran_env, only: real64, int32
  use elliptical,      only: mod_anmbnm => anmbnm
  implicit none
  integer(int32), intent(in)    :: n, m, nabdim
  real(real64),   intent(in)    :: alpha, beta
  real(real64),   intent(inout) :: anm(nabdim), bnm(nabdim)
  call mod_anmbnm(anm, alpha, bnm, beta, n, m, nabdim)
end subroutine anmbnm

subroutine ascale(n, x, a, nadim)
  use iso_fortran_env, only: real64, int32
  use elliptical,      only: mod_ascale => ascale
  implicit none
  integer(int32), intent(in)    :: n, nadim
  real(real64),   intent(in)    :: x
  real(real64),   intent(inout) :: a(nadim)
  call mod_ascale(n, x, a, nadim)
end subroutine ascale

subroutine bscale(n, x, b, nbdim)
  use iso_fortran_env, only: real64, int32
  use elliptical,      only: mod_bscale => bscale
  implicit none
  integer(int32), intent(in)    :: n, nbdim
  real(real64),   intent(in)    :: x
  real(real64),   intent(inout) :: b(nbdim)
  call mod_bscale(n, x, b, nbdim)
end subroutine bscale

subroutine ogen(n, l, ml, rhalf, omega, nodim, binom, gamma, kp, plm)
  use iso_fortran_env, only: real64, int32
  use elliptical,      only: mod_ogen => ogen
  implicit none
  integer(int32), intent(in)    :: n, l, ml, nodim
  real(real64),   intent(in)    :: rhalf
  real(real64),   intent(out)   :: omega(nodim, nodim)
  real(real64),   intent(in)    :: binom(:), gamma(:)
  integer(int32), intent(in)    :: kp(:)
  real(real64),   intent(inout) :: plm(:)
  call mod_ogen(n, l, ml, rhalf, omega, nodim, binom, gamma, kp, plm)
end subroutine ogen
