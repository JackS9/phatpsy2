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
  use state,           only: MSTO, NORB, M2STO, ISPIN
  implicit none

  private

  public :: genfac, genbc, gennrm, gennrm_default, intnrm, pkepot, epkpot, epkpot_core, dampv

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
