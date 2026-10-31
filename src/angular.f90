!=======================================================================
! PHATPSY 2.0: Angular Momentum & Spherical Harmonic Transformations
! Module: angular (angular.f90)
! Modern Fortran 2018 replacement for legacy cgcoef.f, chkcgc.f, gencgc.f,
! ndxcgc.f, order.f, dcoef.f, gendc.f, ndxd.f, genplm.f, putcgc.f, putdc.f.
!=======================================================================
module angular
  use iso_fortran_env, only: real64, int32
  use state,           only: NCGC, ALPHA_ANG, BETA_ANG
  implicit none

  private

  public :: cgcoef, chkcgc, gencgc, ndxcgc, ndxcg, ndxmax
  public :: dcoef, gendc, ndxd, genplm, putcgc, putdc
  public :: order, reordr

  interface cgcoef
    module procedure cgcoef_default
    module procedure cgcoef_kind1
  end interface

  interface gencgc
    module procedure gencgc_default
    module procedure gencgc_kind1
  end interface

  interface putcgc
    module procedure putcgc_default
    module procedure putcgc_kind1
  end interface

  ! Module-level save variables for legacy order/reordr entry emulation
  integer(int32), save :: mmsav, m1sav, m2sav, llsav, l1sav, l2sav

contains

  !---------------------------------------------------------------------
  ! CHKCGC: Check if a Clebsch-Gordan coefficient is nonzero by triangle
  ! inequality and selection rules:
  !   1) LL + L1 + L2 must be even
  !   2) |L1 - L2| <= LL <= (L1 + L2)
  !   3) |MM| <= LL
  !   4) |M1| <= L1 and |M2| <= L2
  !   5) MM = M1 + M2
  !---------------------------------------------------------------------
  subroutine chkcgc(mm, m1, m2, ll, l1, l2, qzero)
    integer(int32), intent(in)  :: mm, m1, m2, ll, l1, l2
    logical,        intent(out) :: qzero

    qzero = .true.
    if (mod(ll + l1 + l2, 2) /= 0) return
    if (ll > l1 + l2) return
    if (ll < abs(l1 - l2)) return
    if (abs(mm) > ll) return
    if (abs(m1) > l1 .or. abs(m2) > l2) return
    if (mm /= (m1 + m2)) return
    qzero = .false.
  end subroutine chkcgc

  !---------------------------------------------------------------------
  ! ORDER: Permute the indices of a C-G coefficient to canonical order:
  !   LL >= L1 >= L2, MM >= M1 (for LL=L1), M1 >= M2 (for L1=L2), MM >= 0.
  !---------------------------------------------------------------------
  subroutine order(mm, m1, m2, ll, l1, l2, qreord)
    integer(int32), intent(inout) :: mm, m1, m2, ll, l1, l2
    logical,        intent(out)   :: qreord

    integer(int32) :: ltemp, mtemp

    mmsav = mm
    m1sav = m1
    m2sav = m2
    llsav = ll
    l1sav = l1
    l2sav = l2
    qreord = .false.

    if (l1 < l2) then
      ltemp = l1; l1 = l2; l2 = ltemp
      mtemp = m1; m1 = m2; m2 = mtemp
      qreord = .true.
    end if

    if (ll < l1) then
      ltemp = ll; ll = l1; l1 = ltemp
      mtemp = mm; mm = m1; m1 = mtemp
      m2 = -m2
      qreord = .true.
      if (l1 < l2) then
        ltemp = l1; l1 = l2; l2 = ltemp
        mtemp = m1; m1 = m2; m2 = mtemp
      end if
    end if

    if (mm < 0) then
      mm = -mm
      m1 = -m1
      m2 = -m2
      qreord = .true.
    end if

    if (l1 == l2 .and. m1 < m2) then
      mtemp = m1; m1 = m2; m2 = mtemp
      qreord = .true.
    end if

    if (ll == l1 .and. mm < m1) then
      mtemp = mm; mm = m1; m1 = mtemp
      m2 = -m2
      qreord = .true.
      if (l1 == l2 .and. m1 < m2) then
        mtemp = m1; m1 = m2; m2 = mtemp
      end if
    end if
  end subroutine order

  !---------------------------------------------------------------------
  ! REORDR: Restore original index order saved during ORDER call.
  !---------------------------------------------------------------------
  subroutine reordr(mm, m1, m2, ll, l1, l2)
    integer(int32), intent(out) :: mm, m1, m2, ll, l1, l2

    mm = mmsav
    m1 = m1sav
    m2 = m2sav
    ll = llsav
    l1 = l1sav
    l2 = l2sav
  end subroutine reordr

  !---------------------------------------------------------------------
  ! NDXCG: Single compact index without zero-checking.
  !---------------------------------------------------------------------
  function ndxcg(mm, m1, m2, ll, l1, l2) result(idx)
    integer(int32), intent(in) :: mm, m1, m2, ll, l1, l2
    integer(int32)             :: idx

    integer(int32) :: mm_loc, m1_loc, m2_loc, ll_loc, l1_loc, l2_loc
    integer(int32) :: iabm1, iabm2, n
    logical        :: qreord

    mm_loc = mm; m1_loc = m1; m2_loc = m2
    ll_loc = ll; l1_loc = l1; l2_loc = l2

    call order(mm_loc, m1_loc, m2_loc, ll_loc, l1_loc, l2_loc, qreord)

    if (m2_loc == 0 .and. l2_loc == 0) then
      idx = 2
      return
    end if

    iabm1 = abs(m1_loc)
    iabm2 = abs(m2_loc)
    n = l2_loc + 3
    if (mm_loc == (iabm1 + iabm2)) n = 2

    idx = l1_loc * (l1_loc + 1) * (2 * l1_loc + 1) * (l1_loc * l1_loc + l1_loc + 3) / 15 &
        + iabm1 * (l1_loc + 1) * (2 * l1_loc * l1_loc + l1_loc + iabm1 + 3) / 3 &
        + l2_loc * (l2_loc + 1) * (2 * l2_loc + 1) / 3 &
        + 2 * iabm2 * (l2_loc + 1) &
        + (ll_loc - l1_loc - l2_loc) / 2 &
        + n

    if (idx > NCGC .and. NCGC > 0) idx = 3
  end function ndxcg

  !---------------------------------------------------------------------
  ! NDXCGC: Single compact index for Clebsch-Gordan coefficients, returns:
  !   1 if zero by selection rules (CHKCGC)
  !   2 for special case C(M1, M1, 0; L1, L1, 0)
  !   3 if index > NCGC
  !---------------------------------------------------------------------
  function ndxcgc(mm, m1, m2, ll, l1, l2) result(idx)
    integer(int32), intent(in) :: mm, m1, m2, ll, l1, l2
    integer(int32)             :: idx

    logical :: qzero

    idx = 1
    call chkcgc(mm, m1, m2, ll, l1, l2, qzero)
    if (qzero) return

    idx = ndxcg(mm, m1, m2, ll, l1, l2)
  end function ndxcgc

  !---------------------------------------------------------------------
  ! NDXMAX: Maximum value of C-G index for a given LMAX.
  !---------------------------------------------------------------------
  function ndxmax(lmax) result(idx)
    integer(int32), intent(in) :: lmax
    integer(int32)             :: idx
    integer(int32)             :: n

    n = lmax + 3
    if (lmax == 0) n = 2
    idx = lmax * (lmax + 1) * (2 * lmax + 1) * (lmax * lmax + lmax + 3) / 15 &
        + lmax * (lmax + 1) * (2 * lmax * lmax + lmax + lmax + 3) / 3 &
        + lmax * (lmax + 1) * (2 * lmax + 1) / 3 &
        + 2 * lmax * (lmax + 1) &
        + n
  end function ndxmax

  !---------------------------------------------------------------------
  ! CGCOEF: Returns the Clebsch-Gordan coefficient C(MM, M1, M2; LL, L1, L2).
  !---------------------------------------------------------------------
  function cgcoef_default(cgc, mm, m1, m2, ll, l1, l2, qrealy) result(cg)
    real(real64),   intent(in) :: cgc(*)
    integer(int32), intent(in) :: mm, m1, m2, ll, l1, l2
    logical,        intent(in) :: qrealy
    real(real64)               :: cg

    real(real64), parameter :: ROOT2 = 1.41421356237309504880168872420970d0
    real(real64)   :: factor
    integer(int32) :: m1_loc, m2_loc, l1_loc, l2_loc, mm_loc
    integer(int32) :: mt1, mt2, ltemp, mtemp

    factor = 1.0d0
    m1_loc = m1
    m2_loc = m2
    l1_loc = l1
    l2_loc = l2
    mm_loc = mm

    if (.not. qrealy) then
      cg = cgc(ndxcgc(mm_loc, m1_loc, m2_loc, ll, l1_loc, l2_loc)) / factor
      return
    end if

    if (abs(m1_loc) < abs(m2_loc)) then
      mtemp = m1_loc; m1_loc = m2_loc; m2_loc = mtemp
      ltemp = l1_loc; l1_loc = l2_loc; l2_loc = ltemp
    end if

    mt1 = m1_loc
    mt2 = m2_loc

    if (m2_loc < 0) then
      if (m1_loc < 0) then
        ! Block 70
        if (mm_loc == -m1_loc - m2_loc) then
          ! Label 130 -> 140
          factor = -ROOT2
          mt1 = -m1_loc
          mt2 = -m2_loc
        else if (mm_loc /= m2_loc - m1_loc) then
          ! Label 100
          cg = 0.0d0
          return
        else if (mm_loc == 0) then
          ! Label 120
          factor = 1.0d0
          mt1 = -m1_loc
        else
          ! Label 110 -> 120
          factor = ROOT2
          mt1 = -m1_loc
        end if
      else if (m1_loc == 0) then
        ! GO TO 150: factor = 1.0d0, mt1 = m1_loc, mt2 = m2_loc
      else ! m1_loc > 0
        ! Block 60
        if (mm_loc == m2_loc - m1_loc) then
          ! Label 110 -> 120
          factor = ROOT2
          mt1 = -m1_loc
        else if (mm_loc /= -m1_loc - m2_loc) then
          ! Label 100
          cg = 0.0d0
          return
        else if (mm_loc == 0) then
          ! Label 100
          cg = 0.0d0
          return
        else
          ! Label 130 -> 140
          factor = -ROOT2
          mt1 = -m1_loc
          mt2 = -m2_loc
        end if
      end if
    else if (m2_loc == 0) then
      ! GO TO 150: factor = 1.0d0, mt1 = m1_loc, mt2 = m2_loc
    else ! m2_loc > 0
      if (m1_loc < 0) then
        ! Block 40
        if (mm_loc == m1_loc - m2_loc) then
          ! Label 90
          factor = ROOT2
          mt2 = -m2_loc
        else if (mm_loc /= m1_loc + m2_loc) then
          ! Label 100
          cg = 0.0d0
          return
        else if (mm_loc == 0) then
          ! Label 100
          cg = 0.0d0
          return
        else
          ! Label 80
          factor = ROOT2
        end if
      else if (m1_loc == 0) then
        ! GO TO 150: factor = 1.0d0, mt1 = m1_loc, mt2 = m2_loc
      else ! m1_loc > 0
        ! Block 30
        if (mm_loc == m1_loc + m2_loc) then
          ! Label 80
          factor = ROOT2
        else if (mm_loc /= m1_loc - m2_loc) then
          ! Label 100
          cg = 0.0d0
          return
        else if (mm_loc == 0) then
          ! Label 140
          factor = 1.0d0
          mt2 = -m2_loc
        else
          ! Label 90
          factor = ROOT2
          mt2 = -m2_loc
        end if
      end if
    end if

    cg = cgc(ndxcgc(mm_loc, mt1, mt2, ll, l1_loc, l2_loc)) / factor
  end function cgcoef_default

  function cgcoef_kind1(cgc, mm, m1, m2, ll, l1, l2, qrealy) result(cg)
    real(real64),    intent(in) :: cgc(*)
    integer(int32),  intent(in) :: mm, m1, m2, ll, l1, l2
    logical(kind=1), intent(in) :: qrealy
    real(real64)                :: cg
    cg = cgcoef_default(cgc, mm, m1, m2, ll, l1, l2, logical(qrealy))
  end function cgcoef_kind1

  !---------------------------------------------------------------------
  ! GENCGC: Generate Clebsch-Gordan coefficients via two-fold recurrence.
  !---------------------------------------------------------------------
  subroutine gencgc_default(cgc, ncgc_in, cgcscr, llmxp2, ylmnrm, nylm, fact, nfact, qrealy)
    real(real64),   intent(inout) :: cgc(*)
    integer(int32), intent(in)    :: ncgc_in, llmxp2, nylm, nfact
    real(real64),   intent(inout) :: cgcscr(llmxp2, *)
    real(real64),   intent(out)   :: ylmnrm(*)
    real(real64),   intent(in)    :: fact(*)
    logical,        intent(in)    :: qrealy

    real(real64), parameter :: PI     = 3.141592653589793238462643383279502884d0
    real(real64), parameter :: FOURPI = 4.0d0 * PI

    integer(int32) :: llmxp1, lmxp1, lmax, il, ll, im, m, lm
    integer(int32) :: imm, mm, m1mn, m1mx, im1, m1, m2, i
    integer(int32) :: l1mn, l1mx, il1, l1, llmn, llmx, ill
    integer(int32) :: l2mn, l2mx, il2, l2
    real(real64)   :: area, cg, term_prev

    if (ncgc_in < 0 .or. nylm < 0 .or. nfact < 0) return

    llmxp1 = llmxp2 - 1
    lmxp1  = llmxp2 / 2
    lmax   = lmxp1 - 1

    cgc(1) = 0.0d0
    cgc(2) = 1.0d0 / sqrt(FOURPI)

    ! Normalization constants for spherical harmonics Y(L, M)
    do il = 1, llmxp1
      ll = il * (il - 1) / 2
      do im = 1, il
        m = im - 1
        lm = ll + im
        area = FOURPI
        if (qrealy .and. m /= 0) area = area / 2.0d0
        ylmnrm(lm) = sqrt(real(2 * il - 1, kind=real64) * fact(il - m) / (area * fact(il + m)))
      end do
    end do

    do imm = 1, llmxp1
      mm   = imm - 1
      m1mn = (mm + 1) / 2 + 1
      m1mx = min(mm, lmax) + 1

      do im1 = m1mn, m1mx
        m1 = im1 - 1
        m2 = mm - m1

        do i = 1, llmxp2
          cgcscr(i, 1) = 0.0d0
        end do

        if (m2 == 0) then
          cg = cgc(2)
        else
          cg = fact(1 + mm) / (fact(1 + m1) * fact(1 + m2)) * &
               sqrt(fact(2 * m1 + 2) * fact(2 * m2 + 2) / (FOURPI * fact(2 * mm + 2)))
          cgc(ndxcg(mm, m1, m2, mm, m1, m2)) = cg
        end if
        cgcscr(imm, 1) = cg

        l1mn = im1
        l1mx = llmxp1
        if (mm > lmax .or. m2 == 0) l1mx = lmxp1

        do il1 = l1mn, l1mx
          l1 = il1 - 1
          do i = 1, llmxp2
            cgcscr(i, 2) = 0.0d0
          end do

          if (l1 /= m1) then
            llmn = max(mm, l1 - m2)
            llmn = llmn + mod(llmn + l1 + m2, 2) + 1
            llmx = min(il1 + m2, llmxp1)
            llmx = llmx - mod(llmn + llmx, 2)

            if (llmn <= llmx) then
              do ill = llmn, llmx, 2
                ll = ill - 1
                if (m2 == 0) then
                  cg = cgc(2)
                else
                  term_prev = 0.0d0
                  if (ill > 1) then
                    term_prev = sqrt(real((ll - mm) * (ll + mm), kind=real64) / real((2 * ll - 1) * (2 * ll + 1), kind=real64)) * &
                                cgcscr(ill - 1, 1)
                  end if
                  cg = (term_prev + &
                        sqrt(real((ll - mm + 1) * (ll + mm + 1), kind=real64) / real((2 * ll + 1) * (2 * ll + 3), kind=real64)) * &
                        cgcscr(ill + 1, 1) - &
                        sqrt(real((l1 - m1 - 1) * (l1 + m1 - 1), kind=real64) / real((2 * l1 - 3) * (2 * l1 - 1), kind=real64)) * &
                        cgcscr(ill, 1)) * &
                        sqrt(real((2 * l1 - 1) * (2 * l1 + 1), kind=real64) / real((l1 - m1) * (l1 + m1), kind=real64))
                  cgc(ndxcg(mm, m1, m2, ll, l1, m2)) = cg
                end if
                cgcscr(ill, 1) = cg
                cgcscr(ill, 2) = cg
              end do
            end if
          else
            cgcscr(imm, 2) = cgcscr(imm, 1)
          end if

          l2mn = m2 + 2
          l2mx = lmxp1 + l1
          if (mm > lmax .or. l1 > lmax) l2mx = lmxp1
          if (m2 == m1) l2mx = min(l2mx, il1)

          if (l2mn <= l2mx) then
            do il2 = l2mn, l2mx
              l2 = il2 - 1
              llmn = max(mm, abs(l2 - l1))
              llmn = llmn + mod(llmn + l1 + l2, 2) + 1
              llmx = min(il2 + l1, llmxp1, llmxp1 + l1 - l2)
              llmx = llmx - mod(llmn + llmx, 2)

              do ill = llmn, llmx, 2
                ll = ill - 1
                term_prev = 0.0d0
                if (ill > 1) then
                  term_prev = sqrt(real((ll - mm) * (ll + mm), kind=real64) / real((2 * ll - 1) * (2 * ll + 1), kind=real64)) * &
                              cgcscr(ill - 1, 2)
                end if
                cg = (term_prev + &
                      sqrt(real((ll - mm + 1) * (ll + mm + 1), kind=real64) / real((2 * ll + 1) * (2 * ll + 3), kind=real64)) * &
                      cgcscr(ill + 1, 2) - &
                      sqrt(real((l2 - m2 - 1) * (l2 + m2 - 1), kind=real64) / real((2 * l2 - 3) * (2 * l2 - 1), kind=real64)) * &
                      cgcscr(ill, 2)) * &
                      sqrt(real((2 * l2 - 1) * (2 * l2 + 1), kind=real64) / real((l2 - m2) * (l2 + m2), kind=real64))
                cgc(ndxcg(mm, m1, m2, ll, l1, l2)) = cg
                cgcscr(ill, 2) = cg
              end do
            end do
          end if
        end do
      end do
    end do
  end subroutine gencgc_default

  subroutine gencgc_kind1(cgc, ncgc_in, cgcscr, llmxp2, ylmnrm, nylm, fact, nfact, qrealy)
    real(real64),    intent(inout) :: cgc(*)
    integer(int32),  intent(in)    :: ncgc_in, llmxp2, nylm, nfact
    real(real64),    intent(inout) :: cgcscr(llmxp2, *)
    real(real64),    intent(out)   :: ylmnrm(*)
    real(real64),    intent(in)    :: fact(*)
    logical(kind=1), intent(in)    :: qrealy
    call gencgc_default(cgc, ncgc_in, cgcscr, llmxp2, ylmnrm, nylm, fact, nfact, logical(qrealy))
  end subroutine gencgc_kind1

  !---------------------------------------------------------------------
  ! PUTCGC: Write all valid Clebsch-Gordan coefficients generated by GENCGC.
  !---------------------------------------------------------------------
  subroutine putcgc_default(cgc, ncgc_in, lmax, qrealy, iw)
    real(real64),   intent(in) :: cgc(*)
    integer(int32), intent(in) :: ncgc_in, lmax, iw
    logical,        intent(in) :: qrealy

    integer(int32) :: lmxp1, l1p1, l1, ll1p1, l2p1, l2, ll2p1, llmn, llmx
    integer(int32) :: llp1, ll, m1plp1, m1, m2plp1, m2, mm
    logical        :: qfirst
    real(real64)   :: cg

    if (ncgc_in < 0) return

    write(iw, 1000)
    lmxp1 = lmax + 1
    do l1p1 = 1, lmxp1
      l1 = l1p1 - 1
      ll1p1 = 2 * l1 + 1
      do l2p1 = 1, lmxp1
        l2 = l2p1 - 1
        ll2p1 = 2 * l2 + 1
        llmn = abs(l1 - l2) + 1
        llmx = l1 + l2 + 1
        do llp1 = llmn, llmx
          ll = llp1 - 1
          do m1plp1 = 1, ll1p1
            m1 = m1plp1 - l1p1
            do m2plp1 = 1, ll2p1
              m2 = m2plp1 - l2p1
              mm = m1 + m2
              qfirst = .true.
              do
                if (m1 * m2 < 0) mm = -abs(mm)
                if (abs(mm) <= ll) then
                  cg = cgcoef(cgc, mm, m1, m2, ll, l1, l2, qrealy)
                  write(iw, 2000) mm, m1, m2, ll, l1, l2, cg
                end if
                if (.not. qfirst .or. .not. qrealy .or. (m1 * m2 == 0)) exit
                qfirst = .false.
                mm = abs(m1 - m2)
              end do
            end do
          end do
        end do
      end do
    end do
1000 format('- MM M1 M2 LL L1 L2  CLEBSCH-GORDON COEFFICIENT'/ &
            '+ __ __ __ __ __ __  __________________________'/)
2000 format(' ', 6i3, f17.8)
  end subroutine putcgc_default

  subroutine putcgc_kind1(cgc, ncgc_in, lmax, qrealy, iw)
    real(real64),    intent(in) :: cgc(*)
    integer(int32),  intent(in) :: ncgc_in, lmax, iw
    logical(kind=1), intent(in) :: qrealy
    call putcgc_default(cgc, ncgc_in, lmax, logical(qrealy), iw)
  end subroutine putcgc_kind1

  !---------------------------------------------------------------------
  ! NDXD: Compact index for D-coefficients (L >= M >= ISIGMA >= 0).
  !---------------------------------------------------------------------
  pure function ndxd(l, m, isigma) result(idx)
    integer(int32), intent(in) :: l, m, isigma
    integer(int32)             :: idx

    idx = (l * (l + 1) * (l + 2)) / 6 + (m * (m + 1)) / 2 + isigma + 1
  end function ndxd

  !---------------------------------------------------------------------
  ! GENDC: Generates D-coefficients for rotation by angle BETA about Y-axis.
  !---------------------------------------------------------------------
  subroutine gendc(d, lmax, beta_opt)
    real(real64),   intent(inout)        :: d(:, :)
    integer(int32), intent(in)           :: lmax
    real(real64),   intent(in), optional :: beta_opt

    real(real64), parameter :: SIGNIF = 1.0d-5
    real(real64) :: beta_val, t, csigt, cm, dm1, csigm, cl1, cl2, cl3, d1, d2
    integer(int32) :: lmxp1, im, m, is, isigma, il, l
    logical :: qtteq1

    if (present(beta_opt)) then
      beta_val = beta_opt
    else
      beta_val = BETA_ANG
    end if

    lmxp1 = lmax + 1
    t = cos(beta_val)
    qtteq1 = (abs(t * t - 1.0d0) < SIGNIF)

    d = 0.0d0
    d(1, 1) = 1.0d0
    d(1, 2) = 1.0d0
    if (lmax <= 0) return

    d(2, 1) = t
    d(2, 2) = -t
    csigt = 0.0d0
    if (.not. qtteq1) csigt = sqrt((1.0d0 + t) / (1.0d0 - t))

    do im = 1, lmxp1
      m = im - 1
      if (m /= 0 .and. .not. qtteq1) then
        cm = -sqrt(real(2 * m - 1, kind=real64) / real(2 * m, kind=real64) * (1.0d0 - t * t))
        dm1 = d(ndxd(m - 1, m - 1, 0), 1) * cm
        d(ndxd(m, m, 0), 1) = dm1
        d(ndxd(m, m, 0), 2) = dm1
      end if

      do is = 1, im
        isigma = is - 1
        if (isigma /= 0 .and. .not. qtteq1) then
          csigm = -sqrt(real(m - isigma + 1, kind=real64) / real(m + isigma, kind=real64))
          d(ndxd(m, m, isigma), 1) = csigm * csigt * d(ndxd(m, m, isigma - 1), 1)
          d(ndxd(m, m, isigma), 2) = csigm / csigt * d(ndxd(m, m, isigma - 1), 2)
        end if

        if (qtteq1 .and. t > 0.0d0 .and. isigma == m .and. m /= 0) d(ndxd(m, m, m), 1) = 1.0d0
        if (qtteq1 .and. t < 0.0d0 .and. isigma == m .and. m /= 0) d(ndxd(m, m, m), 2) = 1.0d0

        if (im <= lmax) then
          do il = im, lmax
            l = il - 1
            if (qtteq1) then
              if (m == isigma) then
                if (t > 0.0d0) then
                  d(ndxd(l + 1, m, m), 1) = 1.0d0
                  if (m == 0) d(ndxd(l + 1, 0, 0), 2) = (-1.0d0)**(l + 1)
                else if (t < 0.0d0) then
                  if (m == 0) d(ndxd(l + 1, 0, 0), 1) = (-1.0d0)**(l + 1)
                  d(ndxd(l + 1, m, m), 2) = 1.0d0
                end if
              end if
            else
              if (l /= 0) then
                cl1 = sqrt(real((l + isigma + 1) * (l - isigma + 1) * (l + m + 1) * (l - m + 1), kind=real64)) / &
                      real(l + 1, kind=real64)
                cl2 = real(m * isigma, kind=real64) / real(l * (l + 1), kind=real64)
                if (l /= m) then
                  cl3 = sqrt(real((l + isigma) * (l - isigma) * (l + m) * (l - m), kind=real64)) / real(l, kind=real64)
                end if

                d1 = real(2 * l + 1, kind=real64) * (t - cl2) * d(ndxd(l, m, isigma), 1)
                d2 = -real(2 * l + 1, kind=real64) * (t + cl2) * d(ndxd(l, m, isigma), 2)

                if (l /= m) then
                  d1 = d1 - cl3 * d(ndxd(l - 1, m, isigma), 1)
                  d2 = d2 - cl3 * d(ndxd(l - 1, m, isigma), 2)
                end if

                d(ndxd(l + 1, m, isigma), 1) = d1 / cl1
                d(ndxd(l + 1, m, isigma), 2) = d2 / cl1
              end if
            end if
          end do
        end if
      end do
    end do
  end subroutine gendc

  !---------------------------------------------------------------------
  ! DCOEF: Returns real spherical harmonic rotation D-matrix coefficient.
  !---------------------------------------------------------------------
  function dcoef(d, l, m, isigma, alpha_opt) result(dc)
    real(real64),   intent(in)           :: d(:, :)
    integer(int32), intent(in)           :: l, m, isigma
    real(real64),   intent(in), optional :: alpha_opt
    real(real64)                         :: dc

    real(real64), parameter :: SIGNIF = 1.0d-8
    real(real64), parameter :: ROOT2  = 1.41421356237309504880168872420970d0

    real(real64)   :: alpha_val, angle, sinma, cosma, sig, sign1, sign2, factor
    integer(int32) :: mm, isig, m_loc, isig_loc, it1, it2, mtemp, ndx

    if (present(alpha_opt)) then
      alpha_val = alpha_opt
    else
      alpha_val = ALPHA_ANG
    end if

    angle = real(m, kind=real64) * alpha_val
    sinma = sin(angle)
    cosma = cos(angle)

    mm       = m
    isig     = isigma
    m_loc    = m
    isig_loc = isigma

    sig = 1.0d0
    if (isigma < 0) sig = -1.0d0

    it1   = 1
    it2   = 2
    sign1 = 1.0d0
    sign2 = real(1 - 2 * mod(abs(l + m + isigma), 2), kind=real64)

    if (m_loc < 0) then
      m_loc    = -m_loc
      isig_loc = -isig_loc
    end if

    if (isig_loc < 0) then
      isig_loc = -isig_loc
      it1      = 2
      it2      = 1
      sign1    = sign2
      sign2    = 1.0d0
    end if

    if (m_loc < isig_loc) then
      mtemp    = m_loc
      m_loc    = isig_loc
      isig_loc = mtemp
      factor   = real(1 - 2 * mod(abs(m_loc + isig_loc), 2), kind=real64)
      sign1    = sign1 * factor
      sign2    = sign2 * factor
    end if

    ndx = ndxd(l, m_loc, isig_loc)
    dc  = d(ndx, it1) * sign1 + d(ndx, it2) * sign2 * sig

    if (mm < 0) then
      if (isig < 0) then
        factor = cosma
      else if (isig == 0) then
        factor = -sinma / ROOT2
      else
        factor = -sinma
      end if
    else if (mm == 0) then
      if (isig < 0) then
        factor = 0.0d0
      else if (isig == 0) then
        factor = 0.5d0
      else
        factor = 1.0d0 / ROOT2
      end if
    else ! mm > 0
      if (isig < 0) then
        factor = sinma
      else if (isig == 0) then
        factor = cosma / ROOT2
      else
        factor = cosma
      end if
    end if

    dc = dc * factor
    if (abs(dc) < SIGNIF) dc = 0.0d0
  end function dcoef

  !---------------------------------------------------------------------
  ! PUTDC: Prints the D-coefficients generated by GENDC / DCOEF.
  !---------------------------------------------------------------------
  subroutine putdc(d, l3mx_in, lmax, iw)
    real(real64),   intent(in) :: d(:, :)
    integer(int32), intent(in) :: l3mx_in, lmax, iw

    integer(int32) :: lmxp1, il, l, llp1, iml, ml, isig, isigma
    real(real64)   :: dc

    if (l3mx_in < 0) return

    write(iw, 1000)
    lmxp1 = lmax + 1
    do il = 1, lmxp1
      l = il - 1
      llp1 = 2 * l + 1
      do iml = 1, llp1
        ml = iml - l - 1
        do isig = 1, llp1
          isigma = isig - l - 1
          dc = dcoef(d, l, ml, isigma)
          write(iw, 2000) l, ml, isigma, dc
        end do
      end do
    end do
1000 format('- L  ML  SIGMA         D-COEFFICIENT'/ &
            '+ _  __  _____         _____________'/' ')
2000 format(' ', i3, i4, i5, f20.8)
  end subroutine putdc

  !---------------------------------------------------------------------
  ! GENPLM: Generates Associated Legendre Polynomials P(L, M; cos(beta)).
  !---------------------------------------------------------------------
  subroutine genplm(plm, beta, nplm, lmax)
    real(real64),   intent(out) :: plm(*)
    real(real64),   intent(in)  :: beta
    integer(int32), intent(in)  :: nplm, lmax

    real(real64), parameter :: SIGNIF = 1.0d-10
    real(real64)   :: cosb, factor, term, rl
    integer(int32) :: lmxp1, lp1, l, lp1m, lm, lm1m, mp1, m, lmp1, l0

    cosb = cos(beta)
    plm(1) = 1.0d0
    if (nplm < 2) return
    plm(2) = cosb

    if ((1.0d0 - cosb * cosb) >= SIGNIF) then
      do lp1 = 2, lmax
        l    = lp1 - 1
        rl   = real(l, kind=real64)
        lp1m = ((l + 2) * (l + 1)) / 2 + 1
        lm   = (lp1 * (lp1 - 1)) / 2 + 1
        lm1m = (l * (l - 1)) / 2 + 1
        plm(lp1m) = ((rl + rl + 1.0d0) * cosb * plm(lm) - rl * plm(lm1m)) / (rl + 1.0d0)
      end do

      factor = 1.0d0 / sqrt(1.0d0 - cosb * cosb)
      lmxp1 = lmax + 1
      do lp1 = 2, lmxp1
        l = lp1 - 1
        do mp1 = 1, l
          m    = mp1 - 1
          lm   = (lp1 * (lp1 - 1)) / 2 + m + 1
          lmp1 = lm + 1
          lm1m = (l * (l - 1)) / 2 + m + 1
          term = real(l - m, kind=real64) * cosb * plm(lm)
          if (mp1 <= l) term = term - real(l + m, kind=real64) * plm(lm1m)
          plm(lmp1) = factor * term
        end do
      end do
    else
      lmxp1 = lmax + 1
      lm = 1
      do lp1 = 1, lmxp1
        do mp1 = 1, lp1
          plm(lm) = 0.0d0
          lm = lm + 1
        end do
        l0 = (lp1 * (lp1 - 1)) / 2 + 1
        plm(l0) = 1.0d0
        if (cosb < 0.0d0 .and. mod(lp1, 2) == 0) plm(l0) = -1.0d0
      end do
    end if
  end subroutine genplm

end module angular

!=======================================================================
! Global external compatibility wrappers for legacy code
!=======================================================================

subroutine chkcgc(mm, m1, m2, ll, l1, l2, qzero)
  use iso_fortran_env, only: int32
  use angular,         only: mod_chkcgc => chkcgc
  implicit none
  integer(int32), intent(in)  :: mm, m1, m2, ll, l1, l2
  logical,        intent(out) :: qzero
  call mod_chkcgc(mm, m1, m2, ll, l1, l2, qzero)
end subroutine chkcgc

subroutine order(mm, m1, m2, ll, l1, l2, qreord)
  use iso_fortran_env, only: int32
  use angular,         only: mod_order => order
  implicit none
  integer(int32), intent(inout) :: mm, m1, m2, ll, l1, l2
  logical,        intent(out)   :: qreord
  call mod_order(mm, m1, m2, ll, l1, l2, qreord)
end subroutine order

subroutine reordr(mm, m1, m2, ll, l1, l2)
  use iso_fortran_env, only: int32
  use angular,         only: mod_reordr => reordr
  implicit none
  integer(int32), intent(out) :: mm, m1, m2, ll, l1, l2
  call mod_reordr(mm, m1, m2, ll, l1, l2)
end subroutine reordr

function ndxcg(mm, m1, m2, ll, l1, l2) result(idx)
  use iso_fortran_env, only: int32
  use angular,         only: mod_ndxcg => ndxcg
  implicit none
  integer(int32), intent(in) :: mm, m1, m2, ll, l1, l2
  integer(int32)             :: idx
  idx = mod_ndxcg(mm, m1, m2, ll, l1, l2)
end function ndxcg

function ndxcgc(mm, m1, m2, ll, l1, l2) result(idx)
  use iso_fortran_env, only: int32
  use angular,         only: mod_ndxcgc => ndxcgc
  implicit none
  integer(int32), intent(in) :: mm, m1, m2, ll, l1, l2
  integer(int32)             :: idx
  idx = mod_ndxcgc(mm, m1, m2, ll, l1, l2)
end function ndxcgc

function ndxmax(lmax) result(idx)
  use iso_fortran_env, only: int32
  use angular,         only: mod_ndxmax => ndxmax
  implicit none
  integer(int32), intent(in) :: lmax
  integer(int32)             :: idx
  idx = mod_ndxmax(lmax)
end function ndxmax

function cgcoef(cgc, mm, m1, m2, ll, l1, l2, qrealy) result(cg)
  use iso_fortran_env, only: real64, int32
  use angular,         only: mod_cgcoef => cgcoef
  implicit none
  real(real64),    intent(in) :: cgc(*)
  integer(int32),  intent(in) :: mm, m1, m2, ll, l1, l2
  logical(kind=1), intent(in) :: qrealy
  real(real64)                :: cg
  cg = mod_cgcoef(cgc, mm, m1, m2, ll, l1, l2, qrealy)
end function cgcoef

subroutine gencgc(cgc, ncgc, cgcscr, llmxp2, ylmnrm, nylm, fact, nfact, qrealy)
  use iso_fortran_env, only: real64, int32
  use angular,         only: mod_gencgc => gencgc
  implicit none
  real(real64),    intent(inout) :: cgc(*)
  integer(int32),  intent(in)    :: ncgc, llmxp2, nylm, nfact
  real(real64),    intent(inout) :: cgcscr(llmxp2, *)
  real(real64),    intent(out)   :: ylmnrm(*)
  real(real64),    intent(in)    :: fact(*)
  logical(kind=1), intent(in)    :: qrealy
  call mod_gencgc(cgc, ncgc, cgcscr, llmxp2, ylmnrm, nylm, fact, nfact, qrealy)
end subroutine gencgc

subroutine putcgc(cgc, ncgc, lmax, qrealy, iw)
  use iso_fortran_env, only: real64, int32
  use angular,         only: mod_putcgc => putcgc
  implicit none
  real(real64),    intent(in) :: cgc(*)
  integer(int32),  intent(in) :: ncgc, lmax, iw
  logical(kind=1), intent(in) :: qrealy
  call mod_putcgc(cgc, ncgc, lmax, qrealy, iw)
end subroutine putcgc

function ndxd(l, m, isigma) result(idx)
  use iso_fortran_env, only: int32
  use angular,         only: mod_ndxd => ndxd
  implicit none
  integer(int32), intent(in) :: l, m, isigma
  integer(int32)             :: idx
  idx = mod_ndxd(l, m, isigma)
end function ndxd

subroutine gendc(d, lmax)
  use iso_fortran_env, only: real64, int32
  use state,           only: L3MX
  use angular,         only: mod_gendc => gendc
  implicit none
  real(real64),   intent(inout) :: d(L3MX, 2)
  integer(int32), intent(in)    :: lmax
  call mod_gendc(d, lmax)
end subroutine gendc

function dcoef(d, l, m, isigma) result(dc)
  use iso_fortran_env, only: real64, int32
  use state,           only: L3MX
  use angular,         only: mod_dcoef => dcoef
  implicit none
  real(real64),   intent(in) :: d(L3MX, 2)
  integer(int32), intent(in) :: l, m, isigma
  real(real64)               :: dc
  dc = mod_dcoef(d, l, m, isigma)
end function dcoef

subroutine putdc(d, l3mx_in, lmax, iw)
  use iso_fortran_env, only: real64, int32
  use state,           only: L3MX
  use angular,         only: mod_putdc => putdc
  implicit none
  real(real64),   intent(in) :: d(L3MX, 2)
  integer(int32), intent(in) :: l3mx_in, lmax, iw
  call mod_putdc(d, l3mx_in, lmax, iw)
end subroutine putdc

subroutine genplm(plm, beta, nplm, lmax)
  use iso_fortran_env, only: real64, int32
  use angular,         only: mod_genplm => genplm
  implicit none
  real(real64),   intent(out) :: plm(*)
  real(real64),   intent(in)  :: beta
  integer(int32), intent(in)  :: nplm, lmax
  call mod_genplm(plm, beta, nplm, lmax)
end subroutine genplm
