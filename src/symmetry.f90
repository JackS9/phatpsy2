!=======================================================================
! PHATPSY 2.0: Symmetry Transformations Module (symmetry.f90)
! Modern Fortran 2018 replacement for legacy symop.f, symxyz.f, and genrot.f
!=======================================================================
module symmetry
  use iso_fortran_env, only: real64, int32
  use state,           only: ALPHA_ANG, BETA_ANG
  use angular,         only: gendc, dcoef
  implicit none

  private
  public :: symop, symxyz, genrot

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
    real(real64),   intent(in)    :: coord(:, :)
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
    real(real64),   intent(inout) :: stevec(:, :, :)
    real(real64),   intent(inout) :: vcoef(:)
    integer(int32), intent(in)    :: l(:), lv(:)
    real(real64),   intent(inout) :: d(:, :)
    real(real64),   intent(inout) :: scrat(..)

    real(real64)   :: pi, angle, parity, alpha, beta, gamma, coef
    real(real64), allocatable :: work(:)
    integer(int32) :: nfold, isp, iorb, isto, jsto, lq, llqp1, imlq, mlq, isig, isigma, nv, mv
    logical        :: qinvrt
    external               :: bomb

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

end module symmetry
