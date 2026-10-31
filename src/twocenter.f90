!=======================================================================
! PHATPSY 2.0: Two-Center Integral & Geometry Module (twocenter.f90)
! Modern Fortran 2018 replacement for legacy rbeta.f, gensab.f, and genvab.f
!=======================================================================
module twocenter
  use iso_fortran_env, only: real64, int32
  use state,           only: EXPMIN, NNMXM1, NABDIM, NSTO, NYLM, L3MX, NCGC, &
                             FACT, BINOM, HALF, ROOTPI, NVTRMB, LLMXP1,       &
                             mol_ws, init_block_data_and_defaults,           &
                             compute_derived_dimensions_and_allocate
  use elliptical,      only: absum, anmbnm, ogen
  use angular,         only: dcoef, cgcoef, gendc
  implicit none

  real(real64), parameter :: PI    = 3.141592653589793238462643383279502884d0
  real(real64), parameter :: TWOPI = 6.283185307179586476925286766559005768d0

  private
  public :: rbeta, rbeta2, gensab, genvab, d_gensab, d_genvab, &
            compute_gensab, compute_d_gensab, compute_genvab, compute_d_genvab

contains

  !---------------------------------------------------------------------
  ! RBETA: Compute internuclear distance and Euler angles (alpha, beta)
  ! between atom IATOM and atom JATOM from Cartesian coordinates.
  !---------------------------------------------------------------------
  subroutine rbeta(coord, natom, iatom, jatom, rab, alph, beta)
    integer(int32), intent(in)  :: natom, iatom, jatom
    real(real64),   intent(in)  :: coord(3, natom)
    real(real64),   intent(out) :: rab, alph, beta

    real(real64) :: x12, y12, z12

    x12 = coord(1, jatom) - coord(1, iatom)
    y12 = coord(2, jatom) - coord(2, iatom)
    z12 = coord(3, jatom) - coord(3, iatom)
    call rbeta2(x12, y12, z12, rab, alph, beta)
  end subroutine rbeta

  !---------------------------------------------------------------------
  ! RBETA2: Compute distance and Euler angles directly from displacement.
  !---------------------------------------------------------------------
  subroutine rbeta2(x, y, z, rab, alph, beta)
    real(real64), intent(in)  :: x, y, z
    real(real64), intent(out) :: rab, alph, beta

    real(real64) :: r

    r = sqrt(x * x + y * y)
    rab = sqrt(r * r + z * z)
    alph = 0.0d0
    beta = 0.0d0
    if (rab == 0.0d0) return
    beta = atan2(r, z)
    if (r == 0.0d0) return
    alph = atan2(y, x)
  end subroutine rbeta2

  !---------------------------------------------------------------------
  ! GENSAB: Generate diatomic two-center overlap integrals in elliptical
  ! coordinates using A- and B-functions.
  !---------------------------------------------------------------------
  subroutine gensab(ovrlap, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                    nstob, rab, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)
    integer(int32), intent(in)    :: nstoa, nstob
    integer(int32), intent(in)    :: na(:), la(:), nb(:), lb(:)
    real(real64),   intent(in)    :: etaa(:), anorm(:), etab(:), bnorm(:)
    real(real64),   intent(in)    :: rab
    real(real64),   intent(inout) :: a(:), b(:)
    real(real64),   intent(inout) :: omegai(:, :), omegaj(:, :)
    real(real64),   intent(inout) :: scrplm(:)
    integer(int32), intent(in)    :: index_arr(:)
    real(real64),   intent(in)    :: ylmnrm(:)
    real(real64),   intent(out)   :: ovrlap(:, :, :)

    real(real64)   :: rhalf, cnorm, etaai, anormi, retaai, etabj, bnormj, retabj
    real(real64)   :: retai, retaj, expon, dexpon, snorm, sum_val
    integer(int32) :: i, j, nai, lai, l2ai, nbj, lbj, l2bj, lmin, llmnp1
    integer(int32) :: iml, ml, iabsml, mlp1, limi, limj, limij

    rhalf = half * rab
    cnorm = twopi * rhalf * rhalf * rhalf

    do i = 1, nstoa
      etaai  = etaa(i)
      anormi = anorm(i)
      retaai = rhalf * etaai

      nai  = na(i)
      lai  = la(i)
      l2ai = index_arr(lai + 1)

      do j = 1, nstob
        etabj  = etab(j)
        bnormj = bnorm(j)
        retabj = rhalf * etabj
        retai  = retaai + retabj
        retaj  = retaai - retabj
        expon  = abs(retaj) - retai
        dexpon = exp(expon)
        snorm  = anormi * bnormj * cnorm * dexpon

        nbj  = nb(j)
        lbj  = lb(j)
        l2bj = index_arr(lbj + 1)

        lmin   = min(lai, lbj)
        llmnp1 = 2 * lmin + 1

        ovrlap(:, i, j) = 0.0d0

        do iml = 1, llmnp1
          if (expon <= expmin) cycle

          ml     = iml - lmin - 1
          iabsml = abs(ml)
          mlp1   = iabsml + 1

          limi  = nai - iabsml
          limj  = nbj - iabsml
          limij = limi + limj

          call ogen( nai, lai, ml, rhalf, omegai, nnmxm1, binom, fact, index_arr, scrplm)
          call ogen(-nbj, lbj, ml, rhalf, omegaj, nnmxm1, binom, fact, index_arr, scrplm)
          call anmbnm(a, retai, b, retaj, limij, iabsml, nabdim)
          call absum(sum_val, a, b, omegai, omegaj, limi, limj, 0)

          if (ml /= 0) sum_val = half * sum_val

          sum_val = sum_val * ylmnrm(l2ai + mlp1) * ylmnrm(l2bj + mlp1)

          ovrlap(iml, i, j) = snorm * sum_val
        end do
      end do
    end do
  end subroutine gensab

  !---------------------------------------------------------------------
  ! GENVAB: Generate 2-center model potential matrix elements between STOs
  ! on center A for a potential on center B.
  !---------------------------------------------------------------------
  subroutine genvab(stvab, n, l, eta, rab, lv, mv, vexp, &
                    omegai, omegaj, a, b, scrplm, index_arr, &
                    anorm, ylmnrm, d, cgc, qrealy)
    real(real64),   intent(out)   :: stvab(:, :)
    integer(int32), intent(in)    :: n(:), l(:)
    real(real64),   intent(in)    :: eta(:)
    real(real64),   intent(in)    :: rab
    integer(int32), intent(in)    :: lv(:), mv(:)
    real(real64),   intent(in)    :: vexp(:)
    real(real64),   intent(inout) :: omegai(:, :), omegaj(:, :)
    real(real64),   intent(inout) :: a(:), b(:)
    real(real64),   intent(inout) :: scrplm(:)
    integer(int32), intent(in)    :: index_arr(:)
    real(real64),   intent(in)    :: anorm(:)
    real(real64),   intent(in)    :: ylmnrm(:)
    real(real64),   intent(in)    :: d(:, :)
    real(real64),   intent(in)    :: cgc(:)
    logical(kind=1), intent(in)   :: qrealy

    real(real64)   :: rhalf, cnorm, etai, etaj, reta, vnorm, rvexpk, retai, retaj
    real(real64)   :: expon, vnormk, sumla, sum_val
    integer(int32) :: ij, i, ni, li, llip1, miplp1, mi, j, nj, lj, na
    integer(int32) :: ldifp1, lsump1, lljp1, mjhi, mjplp1, mj, mdif, msum, msign
    integer(int32) :: madif, masum, ilamn, ilamx, k, lb, nb, mb, l2b
    integer(int32) :: ila, la, l2a, lmin, imlmx, iml, ml, iabsml, mlp1
    integer(int32) :: limi, limj, limij, nvterm, nsto_val

    nvterm = size(vexp)
    if (nvterm <= 0) nvterm = NVTRMB
    nsto_val = size(n)
    if (nsto_val <= 0) nsto_val = NSTO
    rhalf  = half * rab
    cnorm  = pi * rootpi * rab * rab

    ij = 0

    do i = 1, nsto_val
      etai  = eta(i)
      ni    = n(i)
      li    = l(i)
      llip1 = 2 * li + 1

      do miplp1 = 1, llip1
        mi = miplp1 - li - 1

        do j = 1, i
          etaj  = eta(j)
          reta  = rhalf * (etai + etaj)
          vnorm = anorm(i) * anorm(j) * cnorm

          nj = n(j)
          lj = l(j)
          na = ni + nj - 1

          ldifp1 = abs(li - lj) + 1
          lsump1 = li + lj + 1
          lljp1  = 2 * lj + 1
          mjhi   = lljp1
          if (i == j) mjhi = miplp1

          do mjplp1 = 1, mjhi
            mj   = mjplp1 - lj - 1
            mdif = abs(mi - mj)
            msum = abs(mi + mj)

            msign = 1
            if (msum < mdif) then
              msign = -1
            else if (msum == mdif) then
              if (mi + mj < 0) msign = -1
            end if

            madif = msign * mdif
            masum = msign * msum

            ilamn = max(ldifp1, min(msum, mdif) + 1)
            ilamn = ilamn + mod(ilamn + lsump1, 2)
            ilamx = lsump1

            ij = ij + 1

            do k = 1, nvterm
              stvab(ij, k) = 0.0d0

              rvexpk = rhalf * vexp(k)
              retai  = reta + rvexpk
              retaj  = reta - rvexpk
              expon  = abs(retaj) - retai
              if (expon < expmin) cycle

              vnormk = vnorm * exp(expon)

              lb = lv(k)
              if (lb < 0) then
                lb = -lb
                nb = lb + 1
                mb = mv(k)
              else
                nb = lb
                lb = 0
                mb = 0
              end if

              nb  = nb + 1
              l2b = index_arr(lb + 1)
              sumla = 0.0d0

              do ila = ilamn, ilamx, 2
                la  = ila - 1
                l2a = index_arr(la + 1)
                lmin  = min(lb, la)
                imlmx = 2 * lmin + 1

                do iml = 1, imlmx
                  ml     = iml - lmin - 1
                  iabsml = abs(ml)
                  mlp1   = iabsml + 1

                  limi  = na - iabsml
                  limj  = nb - iabsml
                  limij = limi + limj

                  call ogen( na, la, ml, rhalf, omegai, nnmxm1, binom, fact, index_arr, scrplm)
                  call ogen(-nb, lb, ml, rhalf, omegaj, nnmxm1, binom, fact, index_arr, scrplm)
                  call anmbnm(a, retai, b, retaj, limij, iabsml, nabdim)
                  call absum(sum_val, a, b, omegai, omegaj, limi, limj, 1)

                  if (ml /= 0) sum_val = half * sum_val

                  sum_val = sum_val * dcoef(d, lb, mb, ml) * ylmnrm(l2a + mlp1) * ylmnrm(l2b + mlp1)

                  sumla = sumla + sum_val * dcoef(d, la, madif, ml) * &
                          cgcoef(cgc, madif, mi, mj, la, li, lj, qrealy)

                  if (masum /= madif) then
                    sumla = sumla + sum_val * dcoef(d, la, masum, ml) * &
                            cgcoef(cgc, masum, mi, mj, la, li, lj, qrealy)
                  end if
                end do
              end do

              stvab(ij, k) = vnormk * sumla
            end do
          end do
        end do
      end do
    end do
  end subroutine genvab

  !---------------------------------------------------------------------
  ! D_GENSAB: Compute analytical derivative of diatomic two-center overlap
  ! integrals with respect to internuclear distance R_AB (d S / d R_AB).
  ! Evaluated via 4th-order stencil on gensab.
  !---------------------------------------------------------------------
  subroutine d_gensab(d_ovrlap, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                      nstob, rab, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)
    integer(int32), intent(in)    :: nstoa, nstob
    integer(int32), intent(in)    :: na(:), la(:), nb(:), lb(:)
    real(real64),   intent(in)    :: etaa(:), anorm(:), etab(:), bnorm(:)
    real(real64),   intent(in)    :: rab
    real(real64),   intent(inout) :: a(:), b(:)
    real(real64),   intent(inout) :: omegai(:, :), omegaj(:, :)
    real(real64),   intent(inout) :: scrplm(:)
    integer(int32), intent(in)    :: index_arr(:)
    real(real64),   intent(in)    :: ylmnrm(:)
    real(real64),   intent(out)   :: d_ovrlap(:, :, :)

    real(real64), allocatable :: s_p2(:,:,:), s_p1(:,:,:), s_m1(:,:,:), s_m2(:,:,:)
    real(real64), parameter   :: h_step = 1.0d-5
    real(real64)              :: inv_12h
    integer(int32)            :: d1, d2, d3

    d1 = size(d_ovrlap, 1)
    d2 = size(d_ovrlap, 2)
    d3 = size(d_ovrlap, 3)
    allocate(s_p2(d1, d2, d3), s_p1(d1, d2, d3), s_m1(d1, d2, d3), s_m2(d1, d2, d3))

    call gensab(s_p2, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                nstob, rab + 2.0d0 * h_step, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)
    call gensab(s_p1, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                nstob, rab + h_step, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)
    call gensab(s_m1, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                nstob, rab - h_step, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)
    call gensab(s_m2, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                nstob, rab - 2.0d0 * h_step, a, b, omegai, omegaj, scrplm, index_arr, ylmnrm)

    inv_12h = 1.0d0 / (12.0d0 * h_step)
    d_ovrlap = (-s_p2 + 8.0d0 * s_p1 - 8.0d0 * s_m1 + s_m2) * inv_12h

    deallocate(s_p2, s_p1, s_m1, s_m2)
  end subroutine d_gensab

  !---------------------------------------------------------------------
  ! D_GENVAB: Compute analytical derivative of diatomic two-center model
  ! potential matrix elements with respect to internuclear distance R_AB.
  ! Evaluated via 4th-order stencil on genvab.
  !---------------------------------------------------------------------
  subroutine d_genvab(d_stvab, n, l, eta, rab, lv, mv, vexp, &
                      omegai, omegaj, a, b, scrplm, index_arr, &
                      anorm, ylmnrm, d, cgc, qrealy)
    real(real64),   intent(out)   :: d_stvab(:, :)
    integer(int32), intent(in)    :: n(:), l(:)
    real(real64),   intent(in)    :: eta(:)
    real(real64),   intent(in)    :: rab
    integer(int32), intent(in)    :: lv(:), mv(:)
    real(real64),   intent(in)    :: vexp(:)
    real(real64),   intent(inout) :: omegai(:, :), omegaj(:, :)
    real(real64),   intent(inout) :: a(:), b(:)
    real(real64),   intent(inout) :: scrplm(:)
    integer(int32), intent(in)    :: index_arr(:)
    real(real64),   intent(in)    :: anorm(:)
    real(real64),   intent(in)    :: ylmnrm(:)
    real(real64),   intent(in)    :: d(:, :)
    real(real64),   intent(in)    :: cgc(:)
    logical(kind=1), intent(in)   :: qrealy

    real(real64), allocatable :: v_p2(:,:), v_p1(:,:), v_m1(:,:), v_m2(:,:)
    real(real64), parameter   :: h_step = 1.0d-5
    real(real64)              :: inv_12h
    integer(int32)            :: d1, d2

    d1 = size(d_stvab, 1)
    d2 = size(d_stvab, 2)
    allocate(v_p2(d1, d2), v_p1(d1, d2), v_m1(d1, d2), v_m2(d1, d2))

    call genvab(v_p2, n, l, eta, rab + 2.0d0 * h_step, lv, mv, vexp, &
                omegai, omegaj, a, b, scrplm, index_arr, anorm, ylmnrm, d, cgc, qrealy)
    call genvab(v_p1, n, l, eta, rab + h_step, lv, mv, vexp, &
                omegai, omegaj, a, b, scrplm, index_arr, anorm, ylmnrm, d, cgc, qrealy)
    call genvab(v_m1, n, l, eta, rab - h_step, lv, mv, vexp, &
                omegai, omegaj, a, b, scrplm, index_arr, anorm, ylmnrm, d, cgc, qrealy)
    call genvab(v_m2, n, l, eta, rab - 2.0d0 * h_step, lv, mv, vexp, &
                omegai, omegaj, a, b, scrplm, index_arr, anorm, ylmnrm, d, cgc, qrealy)

    inv_12h = 1.0d0 / (12.0d0 * h_step)
    d_stvab = (-v_p2 + 8.0d0 * v_p1 - 8.0d0 * v_m1 + v_m2) * inv_12h

    deallocate(v_p2, v_p1, v_m1, v_m2)
  end subroutine d_genvab

  !---------------------------------------------------------------------
  ! High-level Python / Driver Wrappers with Automatic Buffer Management
  !---------------------------------------------------------------------

  subroutine compute_gensab(ovrlap, llmxp1, na, la, etaa, anorm, nstoa, &
                            nb, lb, etab, bnorm, nstob, rab)
    integer(int32), intent(in)  :: llmxp1, nstoa, nstob
    integer(int32), intent(in)  :: na(nstoa), la(nstoa), nb(nstob), lb(nstob)
    real(real64),   intent(in)  :: etaa(nstoa), anorm(nstoa), etab(nstob), bnorm(nstob)
    real(real64),   intent(in)  :: rab
    real(real64),   intent(out) :: ovrlap(llmxp1, nstoa, nstob)

    real(real64), allocatable :: a(:), b(:), omegai(:,:), omegaj(:,:), scrplm(:)

    if (.not. allocated(mol_ws%index_arr) .or. .not. allocated(mol_ws%ylmnrm)) then
      call init_block_data_and_defaults()
      call compute_derived_dimensions_and_allocate()
    end if

    allocate(a(NABDIM), b(NABDIM), omegai(NNMXM1, NNMXM1), omegaj(NNMXM1, NNMXM1), scrplm(NYLM))
    call gensab(ovrlap, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                nstob, rab, a, b, omegai, omegaj, scrplm, mol_ws%index_arr, mol_ws%ylmnrm)
    deallocate(a, b, omegai, omegaj, scrplm)
  end subroutine compute_gensab

  subroutine compute_d_gensab(d_ovrlap, llmxp1, na, la, etaa, anorm, nstoa, &
                              nb, lb, etab, bnorm, nstob, rab)
    integer(int32), intent(in)  :: llmxp1, nstoa, nstob
    integer(int32), intent(in)  :: na(nstoa), la(nstoa), nb(nstob), lb(nstob)
    real(real64),   intent(in)  :: etaa(nstoa), anorm(nstoa), etab(nstob), bnorm(nstob)
    real(real64),   intent(in)  :: rab
    real(real64),   intent(out) :: d_ovrlap(llmxp1, nstoa, nstob)

    real(real64), allocatable :: a(:), b(:), omegai(:,:), omegaj(:,:), scrplm(:)

    if (.not. allocated(mol_ws%index_arr) .or. .not. allocated(mol_ws%ylmnrm)) then
      call init_block_data_and_defaults()
      call compute_derived_dimensions_and_allocate()
    end if

    allocate(a(NABDIM), b(NABDIM), omegai(NNMXM1, NNMXM1), omegaj(NNMXM1, NNMXM1), scrplm(NYLM))
    call d_gensab(d_ovrlap, na, la, etaa, anorm, nstoa, nb, lb, etab, bnorm, &
                  nstob, rab, a, b, omegai, omegaj, scrplm, mol_ws%index_arr, mol_ws%ylmnrm)
    deallocate(a, b, omegai, omegaj, scrplm)
  end subroutine compute_d_gensab

  subroutine compute_genvab(stvab, m2sto, nvtrm, n, l, eta, nsto, rab, lv, mv, vexp, anorm, qrealy)
    integer(int32), intent(in)  :: m2sto, nvtrm, nsto
    integer(int32), intent(in)  :: n(nsto), l(nsto)
    real(real64),   intent(in)  :: eta(nsto), anorm(nsto)
    real(real64),   intent(in)  :: rab
    integer(int32), intent(in)  :: lv(nvtrm), mv(nvtrm)
    real(real64),   intent(in)  :: vexp(nvtrm)
    logical(kind=1), intent(in) :: qrealy
    real(real64),   intent(out) :: stvab(m2sto, nvtrm)

    real(real64), allocatable :: a(:), b(:), omegai(:,:), omegaj(:,:), scrplm(:)

    if (.not. allocated(mol_ws%index_arr) .or. .not. allocated(mol_ws%ylmnrm) .or. &
        .not. allocated(mol_ws%cgc) .or. .not. allocated(mol_ws%d)) then
      call init_block_data_and_defaults()
      call compute_derived_dimensions_and_allocate()
    end if
    call gendc(mol_ws%d, LLMXP1 - 1)

    allocate(a(NABDIM), b(NABDIM), omegai(NNMXM1, NNMXM1), omegaj(NNMXM1, NNMXM1), scrplm(NYLM))
    call genvab(stvab, n, l, eta, rab, lv, mv, vexp, &
                omegai, omegaj, a, b, scrplm, mol_ws%index_arr, anorm, &
                mol_ws%ylmnrm, mol_ws%d, mol_ws%cgc, qrealy)
    deallocate(a, b, omegai, omegaj, scrplm)
  end subroutine compute_genvab

  subroutine compute_d_genvab(d_stvab, m2sto, nvtrm, n, l, eta, nsto, rab, lv, mv, vexp, anorm, qrealy)
    integer(int32), intent(in)  :: m2sto, nvtrm, nsto
    integer(int32), intent(in)  :: n(nsto), l(nsto)
    real(real64),   intent(in)  :: eta(nsto), anorm(nsto)
    real(real64),   intent(in)  :: rab
    integer(int32), intent(in)  :: lv(nvtrm), mv(nvtrm)
    real(real64),   intent(in)  :: vexp(nvtrm)
    logical(kind=1), intent(in) :: qrealy
    real(real64),   intent(out) :: d_stvab(m2sto, nvtrm)

    real(real64), allocatable :: a(:), b(:), omegai(:,:), omegaj(:,:), scrplm(:)

    if (.not. allocated(mol_ws%index_arr) .or. .not. allocated(mol_ws%ylmnrm) .or. &
        .not. allocated(mol_ws%cgc) .or. .not. allocated(mol_ws%d)) then
      call init_block_data_and_defaults()
      call compute_derived_dimensions_and_allocate()
    end if
    call gendc(mol_ws%d, LLMXP1 - 1)

    allocate(a(NABDIM), b(NABDIM), omegai(NNMXM1, NNMXM1), omegaj(NNMXM1, NNMXM1), scrplm(NYLM))
    call d_genvab(d_stvab, n, l, eta, rab, lv, mv, vexp, &
                  omegai, omegaj, a, b, scrplm, mol_ws%index_arr, anorm, &
                  mol_ws%ylmnrm, mol_ws%d, mol_ws%cgc, qrealy)
    deallocate(a, b, omegai, omegaj, scrplm)
  end subroutine compute_d_genvab

end module twocenter

