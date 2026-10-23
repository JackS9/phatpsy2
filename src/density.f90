!=======================================================================
! PHATPSY 2.0: Modern Fortran 2018 Core
! Module: density
! Description: Density matrices, population analysis, wavefunctions,
!              expectation values, and orbital occupation metrics.
!
! Modernizes legacy routines:
!   - denmat.f (one-electron density matrix generation)
!   - chrgbo.f (charge bond-order matrix and gross populations, with mulpop)
!   - popout.f (orbital energy, occupation, charge and spin printout)
!   - wfct.f   (W-functions for one-center two-electron integrals)
!   - sfct.f   (S-functions for STO integral generation)
!   - averag.f (one-electron operator expectation values)
!   - depth.f  (perspective depth adjustment for mapping/plotting)
!   - occmet.f (occupation-weighted metric inverse)
!   - neword.f (vector reordering by index permutation)
!=======================================================================

module density
  use iso_fortran_env, only: real64, int32
  use state,           only: QOPEN, ISPIN, IW, FACT
  use symmetry,        only: sympak
  implicit none
  private

  public :: denmat
  public :: chrgbo
  public :: mulpop
  public :: popout
  public :: wfct
  public :: sfct
  public :: averag
  public :: depth
  public :: occmet
  public :: neword

contains

  !---------------------------------------------------------------------
  ! DENMAT: Constructs one-electron density matrix D from eigenvectors C
  !         and orbital occupation numbers:
  !           D_ij = sum_k C_ik * C_jk * occ_k
  !         Elements with |D_ij| < 10^-8 are zeroed to prevent symmetry mixing.
  !---------------------------------------------------------------------
  subroutine denmat(dens, evec, occ, nbas, n2bas, nvec)
    integer(int32), intent(in)  :: nbas, n2bas, nvec
    real(real64),   intent(in)  :: evec(nbas, nvec), occ(nvec)
    real(real64),   intent(out) :: dens(n2bas)

    real(real64), parameter :: ZERO = 0.0d0
    real(real64), parameter :: SIGNIF = 1.0d-8
    integer(int32) :: i, j, k, ij
    real(real64)   :: sumocc

    ij = 0
    do i = 1, nbas
      do j = 1, i
        ij = ij + 1
        sumocc = ZERO
        do k = 1, nvec
          sumocc = sumocc + evec(i, k) * evec(j, k) * occ(k)
        end do
        if (abs(sumocc) < SIGNIF) sumocc = ZERO
        dens(ij) = sumocc
      end do
    end do
  end subroutine denmat

  !---------------------------------------------------------------------
  ! CHRGBO: Constructs net-charge bond-order matrix and gross orbital
  !         populations from density and overlap matrices.
  !         Optional argument qmull_opt:
  !           .false. (default): Net-charge weighted overlap population
  !           .true.           : Mulliken 50/50 partition
  !---------------------------------------------------------------------
  subroutine chrgbo(dens, ovlp, popul, totpop, nbas, n2bas, qmull_opt)
    integer(int32), intent(in)    :: nbas, n2bas
    real(real64),   intent(inout) :: dens(n2bas)
    real(real64),   intent(in)    :: ovlp(n2bas)
    real(real64),   intent(out)   :: popul(nbas), totpop
    logical,        intent(in), optional :: qmull_opt

    real(real64), parameter :: ZERO = 0.0d0
    real(real64), parameter :: HALF = 0.5d0
    real(real64), parameter :: TWO  = 2.0d0
    logical :: qmull
    integer(int32) :: i, j, ii, jj, iii, jjj, ij, im1
    real(real64)   :: popi, popj, pop, wgt

    qmull = .false.
    if (present(qmull_opt)) qmull = qmull_opt

    if (nbas >= 2) then
      do i = 2, nbas
        ii = (i * (i - 1)) / 2
        im1 = i - 1
        do j = 1, im1
          ij = ii + j
          dens(ij) = dens(ij) * ovlp(ij) * TWO
        end do
      end do
    end if

    totpop = ZERO
    do i = 1, nbas
      ii = (i * (i - 1)) / 2
      iii = ii + i
      popi = dens(iii)
      pop = popi
      do j = 1, nbas
        if (j == i) cycle
        jj = (j * (j - 1)) / 2
        jjj = jj + j
        popj = dens(jjj)
        if (j > i) then
          ij = jj + i
        else
          ij = ii + j
        end if
        if (qmull) then
          wgt = HALF
        else
          wgt = popi / (popi + popj)
        end if
        pop = pop + dens(ij) * wgt
      end do
      popul(i) = pop
      totpop = totpop + pop
    end do
  end subroutine chrgbo

  !---------------------------------------------------------------------
  ! MULPOP: Mulliken population analysis entry point (50/50 weighting).
  !---------------------------------------------------------------------
  subroutine mulpop(dens, ovlp, popul, totpop, nbas, n2bas)
    integer(int32), intent(in)    :: nbas, n2bas
    real(real64),   intent(inout) :: dens(n2bas)
    real(real64),   intent(in)    :: ovlp(n2bas)
    real(real64),   intent(out)   :: popul(nbas), totpop

    call chrgbo(dens, ovlp, popul, totpop, nbas, n2bas, qmull_opt=.true.)
  end subroutine mulpop

  !---------------------------------------------------------------------
  ! POPOUT: Formatted orbital energy eigenvalues, occupations, charge
  !         analysis, and spin polarization printout.
  !---------------------------------------------------------------------
  subroutine popout(eval, occ, zcharg, alpha, beta, norb)
    integer(int32), intent(in)  :: norb
    real(real64),   intent(in)  :: eval(norb, ISPIN), occ(norb, ISPIN)
    real(real64),   intent(in)  :: zcharg
    real(real64),   intent(out) :: alpha, beta

    real(real64), parameter :: ZERO   = 0.0d0
    real(real64), parameter :: TWO    = 2.0d0
    real(real64), parameter :: EVSHAR = 27.21070d0
    real(real64), parameter :: CALSEV = 23.060d0
    real(real64), parameter :: DEGEN  = 5.0d-5
    character(len=8), parameter :: SPIN(2) = ['ALPHA...', 'BETA... ']

    integer(int32) :: isp, iorb, orbocc
    real(real64)   :: hart, olden, occup, evs, cals, echarg, tcharg, spdif
    character(len=1) :: qcc

    hart = ZERO
    do isp = 1, ISPIN
      if (QOPEN) write(IW, 1000) trim(SPIN(isp))
      write(IW, 2000)
      orbocc = 3 - ISPIN
      do iorb = 1, norb
        occup = occ(iorb, isp) * real(orbocc, kind=real64)
        olden = hart
        hart = eval(iorb, isp)
        qcc = ' '
        if (abs(hart - olden) < DEGEN) qcc = ' '
        evs = hart * EVSHAR
        cals = evs * CALSEV
        write(IW, 3000) qcc, iorb, hart, cals, evs, occup
      end do
    end do

    alpha = sum(occ(1:norb, 1))
    beta = sum(occ(1:norb, ISPIN))
    echarg = -(alpha + beta)
    tcharg = zcharg + echarg
    write(IW, 4000) zcharg, echarg, tcharg

    if (.not. QOPEN) return
    spdif = alpha - beta
    write(IW, 5000) alpha, beta, spdif

1000 format(//' ', A)
2000 format(/'   #', T12, 'HARTREES', T27, 'KCAL/MOLE', T45, 'EVS', T60, 'OCC'/ &
            '   _', T12, '________', T27, '_________', T45, '___', T60, '___'/)
3000 format(A1, I3, F15.4, F15.2, F15.4, F14.3)
4000 format(/' CHARGE ANALYSIS...'// &
            '    NUCLEAR CHARGE......', F9.4/ &
            '    ELECTRONIC CHARGE...', F9.4/ &
            '    NET CHARGE..........', F9.4)
5000 format(/' SPIN ANALYSIS...'// &
            '    ALPHA SPIN...', F9.4/ &
            '    BETA SPIN....', F9.4/ &
            '    NET SPIN.....', F9.4)
  end subroutine popout

  !---------------------------------------------------------------------
  ! WFCT: Generates W-functions for one-center two-electron integrals
  !       between STOs from S-functions:
  !         W(nu) = S(N1+N2, N2-nu-1; eta1/eta2) / eta2^(N1+N2+1)
  !               + S(N1+N2, N1-nu-1; eta2/eta1) / eta1^(N1+N2+1)
  !---------------------------------------------------------------------
  subroutine wfct(w, s, nmx, nmnm1, n1, eta1, n2, eta2)
    integer(int32), intent(in)  :: nmx, nmnm1, n1, n2
    real(real64),   intent(in)  :: eta1, eta2
    real(real64),   intent(out) :: w(nmnm1)
    real(real64),   intent(out) :: s(nmx)

    integer(int32) :: nn, nu
    real(real64)   :: eta2_pow, eta1_pow

    nn = n1 + n2
    eta2_pow = eta2**(nn + 1)
    eta1_pow = eta1**(nn + 1)

    call sfct(s, nn, n2, eta1 / eta2)
    do nu = 1, nmnm1
      w(nu) = s(n2 - nu + 1) / eta2_pow
    end do

    call sfct(s, nn, n1, eta2 / eta1)
    do nu = 1, nmnm1
      w(nu) = w(nu) + s(n1 - nu + 1) / eta1_pow
    end do
  end subroutine wfct

  !---------------------------------------------------------------------
  ! SFCT: Generates array of S-functions for STO two-electron integrals:
  !         S(nu, mu; alpha) = mu * S(nu-1, mu-1; alpha) + S(nu, 0; alpha)
  !         S(nu, 0; alpha)  = fact(nu) / (alpha + 1)^nu
  !---------------------------------------------------------------------
  subroutine sfct(s, nu, mumxp1, alpha)
    integer(int32), intent(in)  :: nu, mumxp1
    real(real64),   intent(in)  :: alpha
    real(real64),   intent(out) :: s(mumxp1)

    real(real64) :: alfp1
    integer(int32) :: nu0, i, j, inu, mu, mumx, jmu

    alfp1 = alpha + 1.0d0
    nu0 = nu - mumxp1

    do i = 1, mumxp1
      inu = nu0 + i
      mu = mumxp1 - i + 1
      s(mu) = FACT(inu) / (alfp1**inu)
    end do

    mumx = mumxp1 - 1
    do i = 1, mumx
      inu = mumxp1 - i
      do j = 1, i
        mu = mumxp1 - j + 1
        jmu = i - j + 1
        s(mu) = real(jmu, kind=real64) * s(mu) + s(inu)
      end do
    end do
  end subroutine sfct

  !---------------------------------------------------------------------
  ! AVERAG: Computes the expectation value (average) of a one-electron
  !         operator matrix over a one-electron density matrix:
  !           <O> = Tr(D * O) = sum_i D_ii O_ii + 2 * sum_{i>j} D_ij O_ij
  !---------------------------------------------------------------------
  real(real64) function averag(onemat, denmat, nbas, n2bas) result(val)
    integer(int32), intent(in) :: nbas, n2bas
    real(real64),   intent(in) :: onemat(n2bas), denmat(n2bas)

    real(real64), parameter :: TWO = 2.0d0
    integer(int32) :: i, j, ii, iii, ij, im1

    val = denmat(1) * onemat(1)
    if (nbas < 2) return

    do i = 2, nbas
      im1 = i - 1
      ii = (i * (i - 1)) / 2
      iii = ii + i
      val = val + denmat(iii) * onemat(iii)
      do j = 1, im1
        ij = ii + j
        val = val + TWO * denmat(ij) * onemat(ij)
      end do
    end do
  end function averag

  !---------------------------------------------------------------------
  ! DEPTH: Adjusts map array indices to allow for depth in the orthogonal
  !        (Z) direction for 3D perspective surface projection.
  !---------------------------------------------------------------------
  subroutine depth(x, y, z, jx, jy, irange)
    real(real64),   intent(in)  :: x, y, z
    integer(int32), intent(in)  :: irange
    integer(int32), intent(out) :: jx, jy

    real(real64), parameter :: TWO   = 2.0d0
    real(real64), parameter :: SKEW  = 10.0d0
    real(real64), parameter :: ZVIEW = 1.0d0

    real(real64) :: horiz, d, r, angle, dx, dy
    interface
      integer(int32) function mapndx(x_coord, mapdim)
        use iso_fortran_env, only: real64, int32
        real(real64),   intent(in) :: x_coord
        integer(int32), intent(in) :: mapdim
      end function mapndx
    end interface

    horiz = SKEW * real(irange, kind=real64)
    d = (x - horiz)**2 + (y - horiz)**2
    r = z * sqrt(d) / (TWO * horiz * ZVIEW)
    angle = atan2(y - horiz, x - horiz)
    dx = -r * cos(angle)
    dy = -r * sin(angle)
    jx = mapndx(dx, 0)
    jy = mapndx(dy, 0)
  end subroutine depth

  !---------------------------------------------------------------------
  ! OCCMET: Generates the occupation-weighted metric inverse in symmetric
  !         packed form:
  !           ONTRAN_ij = sqrt(occ_i * occ_j) * ONTRAN_ij
  !         followed by packing into STOVL via SYMPAK.
  !---------------------------------------------------------------------
  subroutine occmet(ontran, occ, stovl, norb, n2orb)
    integer(int32), intent(in)    :: norb, n2orb
    real(real64),   intent(inout) :: ontran(norb, norb)
    real(real64),   intent(in)    :: occ(norb)
    real(real64),   intent(out)   :: stovl(n2orb)

    integer(int32) :: iorb, jorb

    do jorb = 1, norb
      do iorb = 1, norb
        ontran(iorb, jorb) = sqrt(occ(iorb) * occ(jorb)) * ontran(iorb, jorb)
      end do
    end do

    call sympak(ontran, stovl, norb, n2orb)
  end subroutine occmet

  !---------------------------------------------------------------------
  ! NEWORD: Reorders elements of a vector according to an index permutation
  !         array IORDER: VECTOR(i) = SCRAT(IORDER(i)).
  !---------------------------------------------------------------------
  subroutine neword(vector, scrat, iorder, nbas)
    integer(int32), intent(in)    :: nbas
    real(real64),   intent(inout) :: vector(nbas)
    real(real64),   intent(out)   :: scrat(nbas)
    integer(int32), intent(in)    :: iorder(nbas)

    integer(int32) :: i, j

    scrat(1:nbas) = vector(1:nbas)
    do i = 1, nbas
      j = iorder(i)
      if (j >= 1 .and. j <= nbas) then
        vector(i) = scrat(j)
      end if
    end do
  end subroutine neword

end module density

!=======================================================================
! Global external subroutine wrappers for compatibility with legacy callers
!=======================================================================

subroutine denmat(dens, evec, occ, nbas, n2bas, nvec)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_denmat => denmat
  implicit none
  integer(int32), intent(in)  :: nbas, n2bas, nvec
  real(real64),   intent(in)  :: evec(nbas, nvec), occ(nvec)
  real(real64),   intent(out) :: dens(n2bas)
  call mod_denmat(dens, evec, occ, nbas, n2bas, nvec)
end subroutine denmat

subroutine chrgbo(dens, ovlp, popul, totpop, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_chrgbo => chrgbo
  implicit none
  integer(int32), intent(in)    :: nbas, n2bas
  real(real64),   intent(inout) :: dens(n2bas)
  real(real64),   intent(in)    :: ovlp(n2bas)
  real(real64),   intent(out)   :: popul(nbas), totpop
  call mod_chrgbo(dens, ovlp, popul, totpop, nbas, n2bas)
end subroutine chrgbo

subroutine mulpop(dens, ovlp, popul, totpop, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_mulpop => mulpop
  implicit none
  integer(int32), intent(in)    :: nbas, n2bas
  real(real64),   intent(inout) :: dens(n2bas)
  real(real64),   intent(in)    :: ovlp(n2bas)
  real(real64),   intent(out)   :: popul(nbas), totpop
  call mod_mulpop(dens, ovlp, popul, totpop, nbas, n2bas)
end subroutine mulpop

subroutine popout(eval, occ, zcharg, alpha, beta, norb)
  use iso_fortran_env, only: real64, int32
  use state,           only: ISPIN
  use density,         only: mod_popout => popout
  implicit none
  integer(int32), intent(in)  :: norb
  real(real64),   intent(in)  :: eval(norb, ISPIN), occ(norb, ISPIN)
  real(real64),   intent(in)  :: zcharg
  real(real64),   intent(out) :: alpha, beta
  call mod_popout(eval, occ, zcharg, alpha, beta, norb)
end subroutine popout

subroutine wfct(w, s, nmx, nmnm1, n1, eta1, n2, eta2)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_wfct => wfct
  implicit none
  integer(int32), intent(in)  :: nmx, nmnm1, n1, n2
  real(real64),   intent(in)  :: eta1, eta2
  real(real64),   intent(out) :: w(nmnm1), s(nmx)
  call mod_wfct(w, s, nmx, nmnm1, n1, eta1, n2, eta2)
end subroutine wfct

subroutine sfct(s, nu, mumxp1, alpha)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_sfct => sfct
  implicit none
  integer(int32), intent(in)  :: nu, mumxp1
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: s(mumxp1)
  call mod_sfct(s, nu, mumxp1, alpha)
end subroutine sfct

function averag(onemat, denmat, nbas, n2bas)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_averag => averag
  implicit none
  integer(int32), intent(in) :: nbas, n2bas
  real(real64),   intent(in) :: onemat(n2bas), denmat(n2bas)
  real(real64) :: averag
  averag = mod_averag(onemat, denmat, nbas, n2bas)
end function averag

subroutine depth(x, y, z, jx, jy, irange)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_depth => depth
  implicit none
  real(real64),   intent(in)  :: x, y, z
  integer(int32), intent(in)  :: irange
  integer(int32), intent(out) :: jx, jy
  call mod_depth(x, y, z, jx, jy, irange)
end subroutine depth

subroutine occmet(ontran, occ, stovl, norb, n2orb)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_occmet => occmet
  implicit none
  integer(int32), intent(in)    :: norb, n2orb
  real(real64),   intent(inout) :: ontran(norb, norb)
  real(real64),   intent(in)    :: occ(norb)
  real(real64),   intent(out)   :: stovl(n2orb)
  call mod_occmet(ontran, occ, stovl, norb, n2orb)
end subroutine occmet

subroutine neword(vector, scrat, iorder, nbas)
  use iso_fortran_env, only: real64, int32
  use density,         only: mod_neword => neword
  implicit none
  integer(int32), intent(in)    :: nbas
  real(real64),   intent(inout) :: vector(nbas)
  real(real64),   intent(out)   :: scrat(nbas)
  integer(int32), intent(in)    :: iorder(nbas)
  call mod_neword(vector, scrat, iorder, nbas)
end subroutine neword
