module integrals
  use iso_fortran_env, only: real64, int32
  use state
  use angular, only: cgcoef
  implicit none
  private
  public :: run_oneint, run_twoint, run_fokmat, USE_IN_MEMORY_TEI, clear_tei_cache
  !f2py private :: run_oneint, run_twoint, run_fokmat

  ! Toggle to cache 1-center 2-electron integrals in RAM per element (avoids disk I/O)
  logical, save :: USE_IN_MEMORY_TEI = .true.

  type :: ElementTEICache
    logical :: populated = .false.
    integer(int32) :: ntei = 0
    real(real64),   allocatable :: val(:)
    integer(int32), allocatable :: idx(:,:) ! (6, ntei): IAC, IBD, IAD, IBC, IAB, ICD
    integer(int32), allocatable :: wgt(:,:) ! (6, ntei): KAC, KBD, KAD, KBC, KAB, KCD
  end type ElementTEICache

  type(ElementTEICache), save :: tei_cache(10)

contains

  subroutine clear_tei_cache()
    integer(int32) :: i
    do i = 1, size(tei_cache)
      tei_cache(i)%populated = .false.
      tei_cache(i)%ntei      = 0
      if (allocated(tei_cache(i)%val)) deallocate(tei_cache(i)%val, tei_cache(i)%idx, tei_cache(i)%wgt)
    end do
  end subroutine clear_tei_cache

  pure real(real64) function ovlx(n1, n2, eta1, eta2)
    integer(int32), intent(in) :: n1, n2
    real(real64),   intent(in) :: eta1, eta2
    ovlx = FACT(n1 + n2 + 1) / (eta1 + eta2)**(n1 + n2 + 1)
  end function ovlx

  !---------------------------------------------------------------------
  ! 1. Single-Center One-Electron Integrals (Replaces oneint.f)
  !---------------------------------------------------------------------
  subroutine run_oneint(stovl, stkin, stpot, stpotx, lv, mv, vexp, &
                        n, l, eta, anorm, index_arr, cgc, qrealy)
    real(real64),    intent(out) :: stovl(M2STO), stkin(M2STO), stpot(M2STO, NVTERM), &
                                    stpotx(NVTERM)
    integer(int32),  intent(in)  :: lv(NVTERM), mv(NVTERM), n(NSTO), l(NSTO),         &
                                    index_arr(MSTO)
    real(real64),    intent(in)  :: vexp(NVTERM), eta(NSTO), anorm(NSTO), cgc(NCGC)
    logical(kind=1), intent(in)  :: qrealy

    integer(int32) :: iskip, jskip, i, j, k, ni, li, llip1, nj, lj, lljp1
    integer(int32) :: nk, mi, mli, ij, mjmx, mj, mlj, lk, mk
    real(real64)   :: etai, anormi, etaj, stnorm_val, vnorm, stovlx, stkinx, etak

    stovl = 0.0d0
    stkin = 0.0d0
    stpot = 0.0d0
    stovlx = 0.0d0
    stkinx = 0.0d0

    iskip = 0
    do i = 1, NSTO
      ni     = n(i)
      li     = l(i)
      llip1  = 2 * li + 1
      etai   = eta(i)
      anormi = anorm(i)
      jskip  = 0

      do j = 1, i
        nj         = n(j)
        lj         = l(j)
        lljp1      = 2 * lj + 1
        etaj       = eta(j)
        stnorm_val = anormi * anorm(j)
        vnorm      = stnorm_val * 2.0d0 * ROOTPI

        if (lj == li) then
          stovlx = stnorm_val * ovlx(ni, nj, etai, etaj)
          stkinx = stovlx * (nj * etai**2 + ni * etaj**2 - (nj * etai - ni * etaj)**2 &
                 + li * (li + 1) * (etai + etaj)**2) / (2.0d0 * (ni + nj) * (ni + nj - 1))
        end if

        do k = 1, NVTERM
          etak = etaj + vexp(k)
          nk   = merge(nj - lv(k) + 1, nj + lv(k), lv(k) < 0)
          stpotx(k) = vnorm * ovlx(ni, nk, etai, etak) * (etai + etak) / (ni + nk)
        end do

        do mi = 1, llip1
          mli = mi - li - 1
          if (lj == li) then
            ij = index_arr(mi + iskip) + mi + jskip
            stovl(ij) = stovlx
            stkin(ij) = stkinx
          end if

          mjmx = merge(mi, lljp1, i == j)
          do mj = 1, mjmx
            mlj = mj - lj - 1
            ij  = index_arr(mi + iskip) + mj + jskip
            do k = 1, NVTERM
              if (lv(k) < 0) then
                lk = -lv(k)
                mk = mv(k)
              else
                lk = 0
                mk = 0
              end if
              stpot(ij, k) = stpotx(k) * CGCOEF(cgc, mli, mk, mlj, li, lk, lj, qrealy)
            end do
          end do
        end do
        jskip = jskip + lljp1
      end do
      iskip = iskip + llip1
    end do
  end subroutine run_oneint

  !---------------------------------------------------------------------
  ! 2. Single-Center Two-Electron Integrals (Replaces twoint.f)
  !---------------------------------------------------------------------
  subroutine run_twoint(ntei, n, l, eta, s, w, index_arr, anorm, cgc, qrealy, teibuf)
    integer(int32),  intent(out)   :: ntei
    integer(int32),  intent(in)    :: n(NSTO), l(NSTO), index_arr(MSTO)
    real(real64),    intent(in)    :: eta(NSTO), anorm(NSTO), cgc(NCGC)
    real(real64),    intent(inout) :: s(NNMX), w(NNMXM1), teibuf(LENBUF)
    logical(kind=1), intent(in)    :: qrealy

    integer(int32) :: ja, jc, jb, jd, ia, ic, ib, id
    integer(int32) :: na, la, llap1, nc, lc, llcp1, n1, nb, lb, llbp1, nd, ld, lldp1, n2
    integer(int32) :: lsummn, ldifmx, imumn, imumx, nmx, nmnm1
    integer(int32) :: masign, ima, ma, iabsma, ija, iia
    integer(int32) :: imcmx, mcsign, imc, mc, iabsmc, msign, ijc, iic
    integer(int32) :: imbmx, mbsign, imb, mb, iabsmb, ijb, iib
    integer(int32) :: imdmx, mdsign, imd, md, iabsmd, ijd, iid
    integer(int32) :: msummn, mdifmx, mskip, isigmn, isigmx, imu, mu, tumup1, isig, isigma
    integer(int32) :: idac, idbd, idad, idbc, idab, idcd, cap
    real(real64)   :: etaa, snorma, etac, eta1, snormc, etab, snormb, etad, eta2, snormd
    real(real64)   :: stnorm_val, summu, tei, fourpi
    real(real64),   allocatable :: tmp_val(:)
    integer(int32), allocatable :: tmp_idx(:,:), tmp_wgt(:,:)

    fourpi = HALFPI(8)
    ntei   = 0

    if (USE_IN_MEMORY_TEI .and. IELEM >= 1 .and. IELEM <= size(tei_cache)) then
      cap = max(1024, (M2STO * (M2STO + 1)) / 4)
      if (allocated(tei_cache(IELEM)%val)) then
        deallocate(tei_cache(IELEM)%val, tei_cache(IELEM)%idx, tei_cache(IELEM)%wgt)
      end if
      allocate(tei_cache(IELEM)%val(cap))
      allocate(tei_cache(IELEM)%idx(6, cap))
      allocate(tei_cache(IELEM)%wgt(6, cap))
    end if

    ja = 0
    do ia = 1, NSTO
      na     = n(ia); la = l(ia); llap1 = 2 * la + 1
      etaa   = eta(ia); snorma = anorm(ia)
      jc     = 0
      do ic = 1, ia
        nc     = n(ic); lc = l(ic); llcp1 = 2 * lc + 1
        etac   = eta(ic); n1 = na + nc; eta1 = etaa + etac
        snormc = anorm(ic)
        jb     = 0
        do ib = 1, ia
          nb     = n(ib); lb = l(ib); llbp1 = 2 * lb + 1
          etab   = eta(ib); snormb = anorm(ib)
          jd     = 0
          do id = 1, ib
            nd     = n(id); ld = l(id); lldp1 = 2 * ld + 1
            etad   = eta(id); n2 = nb + nd; eta2 = etab + etad
            snormd = anorm(id)

            if (mod(la + lb + lc + ld, 2) == 0) then
              lsummn = min(la + lc, lb + ld)
              ldifmx = max(abs(la - lc), abs(lb - ld))
              imumn  = ldifmx + 1
              imumx  = lsummn + 1

              if (imumn <= imumx) then
                nmx        = max(n1, n2)
                nmnm1      = min(n1, n2) - 1
                call WFCT(w, s, nmx, nmnm1, n1, eta1, n2, eta2)
                stnorm_val = fourpi * snorma * snormb * snormc * snormd

                masign = -1
                do ima = 1, llap1
                  ma     = ima - la - 1
                  iabsma = abs(ma)
                  if (ma >= 0) masign = 1
                  ija = ja + ima
                  iia = index_arr(ija)

                  imcmx  = merge(ima, llcp1, ic == ia)
                  mcsign = -1
                  do imc = 1, imcmx
                    mc     = imc - lc - 1
                    iabsmc = abs(mc)
                    if (mc >= 0) mcsign = 1
                    msign = masign * mcsign
                    ijc   = jc + imc
                    iic   = index_arr(ijc)

                    imbmx  = merge(ima, llbp1, ib == ia)
                    mbsign = -1
                    do imb = 1, imbmx
                      mb     = imb - lb - 1
                      iabsmb = abs(mb)
                      if (mb >= 0) mbsign = 1
                      ijb = jb + imb
                      iib = index_arr(ijb)

                      if (.not. ((ija == ijb) .and. (id > ic))) then
                        imdmx = merge(imb, lldp1, id == ib)
                        if ((ija == ijb) .and. (ic == id)) imdmx = min(imdmx, imc)
                        mdsign = -1

                        do imd = 1, imdmx
                          md     = imd - ld - 1
                          iabsmd = abs(md)
                          if (md >= 0) mdsign = 1
                          ijd = jd + imd
                          iid = index_arr(ijd)

                          msummn = min(iabsma + iabsmc, iabsmb + iabsmd)
                          mdifmx = max(abs(iabsma - iabsmc), abs(iabsmb - iabsmd))
                          mskip  = msummn - mdifmx
                          if (mskip < 0) cycle
                          if (mskip == 0) mskip = 1
                          isigmn = mdifmx + 1
                          if ((msign /= mbsign * mdsign) .and. (iabsmb /= iabsmd)) cycle

                          IAC = iia + ijc
                          IBD = iib + ijd
                          IAD = iia + ijd
                          IBC = merge(iic + ijb, iib + ijc, ijc > ijb)
                          IAB = iia + ijb
                          ICD = merge(iid + ijc, iic + ijd, ijd > ijc)

                          idac = merge(1, 0, ija == ijc)
                          idbd = merge(1, 0, ijb == ijd)
                          idad = merge(1, 0, ija == ijd)
                          idbc = merge(1, 0, ijb == ijc)
                          idab = merge(1, 0, ija == ijb)
                          idcd = merge(1, 0, ijc == ijd)

                          KAC = 2 - idbd
                          KBD = merge(0, 2 - idac, IAC == IBD)
                          KAD = 1 + idad - idad * idbc
                          KBC = merge(0, 1 + idbc - idbc * idad, IAD == IBC)
                          KAB = 1 + idab - idab * idcd - idac - idbd + idac * idbd &
                              - idab * idac - idab * idbd + 2 * idab * idbc * idcd
                          KCD = merge(0, 1 + idcd - idcd * idab - idac - idbd + idac * idbd &
                              - idcd * idac - idcd * idbd + 2 * idcd * idbc * idab, IAB == ICD)

                          summu = 0.0d0
                          do imu = imumn, imumx, 2
                            mu     = imu - 1
                            tumup1 = 2 * mu + 1
                            isigmx = merge(isigmn, msummn + 1, mu < msummn)
                            do isig = isigmn, isigmx, mskip
                              isigma = msign * (isig - 1)
                              summu  = summu + CGCOEF(cgc, isigma, ma, mc, mu, la, lc, qrealy) &
                                             * CGCOEF(cgc, isigma, mb, md, mu, lb, ld, qrealy) &
                                             * w(imu) / tumup1
                            end do
                          end do

                          if (summu /= 0.0d0) then
                            ntei = ntei + 1
                            tei  = summu * stnorm_val

                            if (USE_IN_MEMORY_TEI .and. IELEM >= 1 .and. IELEM <= size(tei_cache)) then
                              if (ntei > size(tei_cache(IELEM)%val)) then
                                cap = 2 * size(tei_cache(IELEM)%val)
                                allocate(tmp_val(cap), tmp_idx(6, cap), tmp_wgt(6, cap))
                                tmp_val(1:ntei-1)    = tei_cache(IELEM)%val(1:ntei-1)
                                tmp_idx(:, 1:ntei-1) = tei_cache(IELEM)%idx(:, 1:ntei-1)
                                tmp_wgt(:, 1:ntei-1) = tei_cache(IELEM)%wgt(:, 1:ntei-1)
                                call move_alloc(tmp_val, tei_cache(IELEM)%val)
                                call move_alloc(tmp_idx, tei_cache(IELEM)%idx)
                                call move_alloc(tmp_wgt, tei_cache(IELEM)%wgt)
                              end if
                              tei_cache(IELEM)%val(ntei)    = tei
                              tei_cache(IELEM)%idx(:, ntei) = [IAC, IBD, IAD, IBC, IAB, ICD]
                              tei_cache(IELEM)%wgt(:, ntei) = [KAC, KBD, KAD, KBC, KAB, KCD]
                            end if
                          end if
                        end do
                      end if
                    end do
                  end do
                end do
              end if
            end if
            jd = jd + lldp1
          end do
          jb = jb + llbp1
        end do
        jc = jc + llcp1
      end do
      ja = ja + llap1
    end do

    IAC = 0
    if (USE_IN_MEMORY_TEI .and. IELEM >= 1 .and. IELEM <= size(tei_cache)) then
      tei_cache(IELEM)%ntei      = ntei
      tei_cache(IELEM)%populated = .true.
    end if
  end subroutine run_twoint

  !---------------------------------------------------------------------
  ! 3. Two-Electron Fock Matrix Builder (Replaces fokmat.f)
  !---------------------------------------------------------------------
  subroutine run_fokmat(fc, fx, d, teibuf)
    real(real64), intent(out)   :: fc(M2STO, ISPIN), fx(M2STO, ISPIN)
    real(real64), intent(in)    :: d(M2STO, ISPIN)
    real(real64), intent(inout) :: teibuf(LENBUF)

    integer(int32) :: n, isp, iac_l, ibd_l, iad_l, ibc_l, iab_l, icd_l
    integer(int32) :: kac_l, kbd_l, kad_l, kbc_l, kab_l, kcd_l
    real(real64)   :: tei, fac, fbd

    fc = 0.0d0
    fx = 0.0d0

    ! Fast in-memory contraction path (zero disk I/O during SCF iterations)
    if (USE_IN_MEMORY_TEI .and. IELEM >= 1 .and. IELEM <= size(tei_cache)) then
      if (tei_cache(IELEM)%populated) then
        do n = 1, tei_cache(IELEM)%ntei
          tei   = tei_cache(IELEM)%val(n)
          iac_l = tei_cache(IELEM)%idx(1, n); kac_l = tei_cache(IELEM)%wgt(1, n)
          ibd_l = tei_cache(IELEM)%idx(2, n); kbd_l = tei_cache(IELEM)%wgt(2, n)
          iad_l = tei_cache(IELEM)%idx(3, n); kad_l = tei_cache(IELEM)%wgt(3, n)
          ibc_l = tei_cache(IELEM)%idx(4, n); kbc_l = tei_cache(IELEM)%wgt(4, n)
          iab_l = tei_cache(IELEM)%idx(5, n); kab_l = tei_cache(IELEM)%wgt(5, n)
          icd_l = tei_cache(IELEM)%idx(6, n); kcd_l = tei_cache(IELEM)%wgt(6, n)

          fac = kac_l * (d(ibd_l, 1) + d(ibd_l, ISPIN)) * tei
          fbd = kbd_l * (d(iac_l, 1) + d(iac_l, ISPIN)) * tei
          do isp = 1, ISPIN
            fc(iac_l, isp) = fc(iac_l, isp) + fac
            fc(ibd_l, isp) = fc(ibd_l, isp) + fbd
            fx(iad_l, isp) = fx(iad_l, isp) - kad_l * d(ibc_l, isp) * tei
            fx(ibc_l, isp) = fx(ibc_l, isp) - kbc_l * d(iad_l, isp) * tei
            fx(iab_l, isp) = fx(iab_l, isp) - kab_l * d(icd_l, isp) * tei
            fx(icd_l, isp) = fx(icd_l, isp) - kcd_l * d(iab_l, isp) * tei
          end do
        end do
      end if
    end if
  end subroutine run_fokmat

end module integrals