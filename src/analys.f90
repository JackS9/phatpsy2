module analys
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin, only: run_lowdin
  implicit none
  private
  public :: run_analys, run_chgan, USE_LAPACK_FIT
  !f2py private :: run_analys, run_chgan

  logical, save :: USE_LAPACK_FIT = .false. ! Set .true. to use LAPACK DGELSD SVD solver

contains

  !---------------------------------------------------------------------
  ! SOLVE_HELMERT_NULLSPACE_FIT:
  ! Solves min ||G0 * c - g_rhs|| subject to exact linear cusp sum(c) = d1
  ! using an analytical Helmert orthonormal basis Z for the null space (1^T Z = 0).
  ! Guarantees the cusp condition is satisfied to exact machine precision
  ! without condition-number blowup from penalty weights.
  !---------------------------------------------------------------------
  subroutine solve_helmert_nullspace_fit(g0_packed, g0_rhs, d1, nc, c_out, qindef)
    integer(int32), intent(in)  :: nc
    real(real64),   intent(in)  :: g0_packed(210), g0_rhs(20), d1
    real(real64),   intent(out) :: c_out(nc)
    logical(kind=1), intent(out) :: qindef

    real(real64), allocatable :: g0(:,:), z(:,:), gz(:,:), g_tilde(:,:), g_sub_rhs(:)
    real(real64), allocatable :: x0(:), y(:), s_svd(:), work(:)
    integer(int32), allocatable :: iwork(:)
    real(real64) :: norm, inv_norm
    integer(int32) :: i, j, ij, k, info, rank, lwork, liwork

    qindef = .false.
    if (nc <= 1) then
      c_out(1) = d1
      return
    end if

    allocate(g0(nc, nc), z(nc, nc - 1), x0(nc))
    x0 = d1 / real(nc, real64)

    ! Construct Helmert orthonormal basis Z for 1^T z = 0
    z = 0.0d0
    do k = 1, nc - 1
      norm = sqrt(real(k * (k + 1), real64))
      inv_norm = 1.0d0 / norm
      z(1:k, k) = inv_norm
      z(k + 1, k) = -real(k, real64) * inv_norm
    end do

    ! Unpack symmetric G0
    ij = 0
    do i = 1, nc
      do j = 1, i
        ij = ij + 1
        g0(i, j) = g0_packed(ij)
        g0(j, i) = g0_packed(ij)
      end do
    end do

    ! Form G_tilde = Z^T * G0 * Z and g_sub_rhs = Z^T * (g0_rhs - G0 * x0)
    allocate(gz(nc, nc - 1), g_tilde(nc - 1, nc - 1), g_sub_rhs(nc - 1))
    gz = matmul(g0, z)
    g_tilde = matmul(transpose(z), gz)
    g_sub_rhs = matmul(transpose(z), g0_rhs(1:nc) - matmul(g0, x0))

    ! Solve G_tilde * y = g_sub_rhs using DGELSD (guaranteed robust even if rank-deficient)
    allocate(y(nc - 1), s_svd(nc - 1))
    y = g_sub_rhs
    lwork  = 12 * (nc - 1) + 2 * (nc - 1) * 25 + 8 * (nc - 1)**2 + 100
    liwork = max(1, 3 * (nc - 1) * 10 + 11 * (nc - 1))
    allocate(work(lwork), iwork(liwork))

    call DGELSD(nc - 1, nc - 1, 1, g_tilde, nc - 1, y, nc - 1, s_svd, 1.0d-12, &
                rank, work, lwork, iwork, info)

    if (info /= 0) then
      qindef = .true.
      c_out = x0
    else
      c_out = x0 + matmul(z, y)
    end if

    deallocate(g0, z, gz, g_tilde, g_sub_rhs, x0, y, s_svd, work, iwork)
  end subroutine solve_helmert_nullspace_fit

  !---------------------------------------------------------------------
  ! SOLVE_CONSTRAINED_MODEL_POTENTIAL_FIT:
  ! Equality-constrained linear fitting with KKT system and Helmert null-space fallback.
  ! Constraint 1: Exact nuclear cusp sum(c_2..c_nvterm) = -sumocc
  ! Constraint 2: External field matching (when WGT2 > 0 and tevz2 is active)
  !---------------------------------------------------------------------
  subroutine solve_constrained_model_potential_fit(g0_packed, g0_rhs, vfit2, tevz2, &
                                                   vcoef, sumocc, nvterm, wgt2, qindef)
    integer(int32), intent(in)    :: nvterm
    real(real64),   intent(in)    :: g0_packed(210), g0_rhs(20), vfit2(nvterm)
    real(real64),   intent(in)    :: tevz2, sumocc, wgt2
    real(real64),   intent(inout) :: vcoef(nvterm)
    logical(kind=1), intent(out)   :: qindef

    integer(int32) :: nc, p_const, n_kkt, i, j, ij, info, lwork
    real(real64), allocatable :: kkt(:,:), rhs(:), work(:)
    integer(int32), allocatable :: ipiv(:)
    real(real64) :: d1, d2
    logical :: use_wgt2

    nc = nvterm - 1
    qindef = .false.
    if (nc <= 0) return

    ! Constraint 1: Nuclear cusp sum(c_2..c_nvterm) = -sumocc
    d1 = -sumocc

    ! Constraint 2: External field matching
    use_wgt2 = (wgt2 > 1.0d-4 .and. abs(tevz2 - 0.5d0) > 1.0d-3)
    if (use_wgt2) then
      p_const = 2
      d2 = tevz2 - sumocc * vfit2(1)
    else
      p_const = 1
    end if

    n_kkt = nc + p_const
    allocate(kkt(n_kkt, n_kkt), rhs(n_kkt), ipiv(n_kkt))
    kkt = 0.0d0
    rhs = 0.0d0

    ! 1. Unpack G0 into upper-left block of KKT
    ij = 0
    do i = 1, nc
      do j = 1, i
        ij = ij + 1
        kkt(i, j) = g0_packed(ij)
        kkt(j, i) = g0_packed(ij)
      end do
      rhs(i) = g0_rhs(i)
    end do

    ! 2. Constraint rows/columns B and B^T
    ! Row 1: Cusp (all 1's)
    do j = 1, nc
      kkt(nc + 1, j) = 1.0d0
      kkt(j, nc + 1) = 1.0d0
    end do
    rhs(nc + 1) = d1

    ! Row 2: External field (vfit2(2..nvterm))
    if (use_wgt2) then
      do j = 1, nc
        kkt(nc + 2, j) = vfit2(j + 1)
        kkt(j, nc + 2) = vfit2(j + 1)
      end do
      rhs(nc + 2) = d2
    end if

    ! 3. Query optimal workspace and solve KKT system via LAPACK DSYSV
    lwork = -1
    allocate(work(1))
    call DSYSV('U', n_kkt, 1, kkt, n_kkt, ipiv, rhs, n_kkt, work, lwork, info)
    lwork = max(1, int(work(1)))
    deallocate(work)
    allocate(work(lwork))

    call DSYSV('U', n_kkt, 1, kkt, n_kkt, ipiv, rhs, n_kkt, work, lwork, info)

    if (info == 0) then
      vcoef(1) = sumocc
      vcoef(2:nvterm) = rhs(1:nc)
    else
      ! Null-space Helmert fallback for constraint 1
      call solve_helmert_nullspace_fit(g0_packed, g0_rhs, d1, nc, vcoef(2:nvterm), qindef)
      vcoef(1) = sumocc
    end if

    deallocate(kkt, rhs, ipiv, work)
  end subroutine solve_constrained_model_potential_fit

  subroutine solve_model_potential_fit(vmat_packed, vsum, vcoef, sumocc, nvterm, qindef)
    integer(int32),  intent(in)    :: nvterm
    real(real64),    intent(inout) :: vmat_packed(210), vsum(20)
    real(real64),    intent(inout) :: vcoef(nvterm)
    real(real64),    intent(in)    :: sumocc
    logical(kind=1), intent(out)   :: qindef

    real(real64)   :: vinv(20, 20), scrat1(20, 20)
    integer(int32) :: nfit, i, j, ij, info, rank, lwork, liwork
    real(real64),   allocatable :: a_full(:,:), rhs(:), s_svd(:), work(:)
    integer(int32), allocatable :: iwork(:)

    nfit   = nvterm - 1
    qindef = .false.

    if (.not. USE_LAPACK_FIT) then
      vinv   = 0.0d0
      scrat1 = 0.0d0
      call run_lowdin(vmat_packed, vinv, scrat1, nfit, 20, 0, qindef)
      if (qindef) return
      vcoef(1) = sumocc
      do i = 2, nvterm
        vcoef(i) = dot_product(vinv(i - 1, 1:nfit), vsum(1:nfit))
      end do
    else
      allocate(a_full(nfit, nfit), rhs(nfit), s_svd(nfit))
      ij = 0
      do i = 1, nfit
        do j = 1, i
          ij = ij + 1
          a_full(i, j) = vmat_packed(ij)
          a_full(j, i) = vmat_packed(ij)
        end do
        rhs(i) = vsum(i)
      end do
      lwork  = 12 * nfit + 2 * nfit * 25 + 8 * nfit * nfit + 100
      liwork = max(1, 3 * nfit * 10 + 11 * nfit)
      allocate(work(lwork), iwork(liwork))
      call DGELSD(nfit, nfit, 1, a_full, nfit, rhs, nfit, s_svd, 1.0d-10, &
                  rank, work, lwork, iwork, info)
      if (info /= 0) then
        qindef = .true.
      else
        vcoef(1)        = sumocc
        vcoef(2:nvterm) = rhs(1:nfit)
      end if
      deallocate(a_full, rhs, s_svd, work, iwork)
    end if
  end subroutine solve_model_potential_fit

  subroutine run_analys(stkin, stpot, stcoul, stexch, cpot2, zpot2, vfit2,    &
                        vcoef, vexp, lv, pkpot, xpot2, stevec, eval, occ,     &
                        gamma, index_arr, toten, tke, zzrep, stovl, charge, tcoul0)
    real(real64),   intent(in)    :: stkin(M2STO), stpot(M2STO, NVTERM),      &
                                     stcoul(M2STO, ISPIN), stexch(M2STO, ISPIN), &
                                     vexp(NVTERM), xpot2(M2STO, ISPIN),       &
                                     cpot2(M2STO), stevec(MSTO, NORB, ISPIN), &
                                     eval(NORB, ISPIN), occ(NORB, ISPIN),     &
                                     zpot2(M2STO), vfit2(NVTERM),             &
                                     stovl(M2STO), pkpot(M2STO, 6, ISPIN),    &
                                     tcoul0
    integer(int32), intent(in)    :: lv(NVTERM), index_arr(MSTO)
    real(real64),   intent(inout) :: vcoef(NVTERM), gamma(NORB, ISPIN),       &
                                     toten, tke, zzrep, charge(3, NATOM)

    real(real64)    :: oe(9), te(9), charg(6), epot(20), vsum(20), vmat(210)
    real(real64)    :: eorb, ekin, enuc, echf, exhf, evz2, evc2, evx2, pkv2
    real(real64)    :: teorB, tekin, tenuc, techf, texhf, tevz2, tevc2, tevx2, tpkv2
    real(real64)    :: aspin, bspin, spdiff, znuc, echarg, totchg
    real(real64)    :: sumocc, oldchg, nelec, vdevv, vdevc, vcalct, teval, tpkerr
    real(real64)    :: wgt2sv, c1c1, proj, ckck, ckcl2, occup, vdev, wgt, vcalc
    real(real64)    :: smax, hole, screen, schole, el2, pkerr, vcalc0, vcalc2
    real(real64)    :: vdev2, tetot, tpe, virial
    integer(int32)  :: ncore, isp, iorb, nv, k, indxk, kk, km1, l, kl, i, j, ij
    logical(kind=1) :: qpkinc, qnofit, qindef, qretry

    real(real64),   external :: ASUM
    integer(int32), external :: NOBLE

    qpkinc = .true.
    ORBOCC = 3 - ISPIN
    aspin  = ASUM(occ(1, 1), NORB, 1)
    bspin  = ASUM(occ(1, ISPIN), NORB, 1)
    sumocc = aspin + bspin
    znuc   = charge(1, IATOM)
    oldchg = charge(2, IATOM)
    nelec  = min(oldchg, znuc)
    ncore  = NOBLE(nelec)

    te     = 0.0d0
    vsum   = 0.0d0
    vmat   = 0.0d0
    vdevv  = 0.0d0; vdevc  = 0.0d0; vcalct = 0.0d0
    teval  = 0.0d0; tpkerr = 0.0d0

    wgt2sv = WGT2
    if (QATOM .or. QSEPAT) WGT2 = 0.0d0
    qnofit = (QFIXED .or. (.not. QC2FIT) .or. (WGTC + WGTV + WGT2 <= 0.1d0) .or. (NVTERM < 3))

    if (QPRINT) write(IW, 1000)

    do isp = 1, ISPIN
      if (QPRINT .and. QOPEN) write(IW, 2000) SPIN(isp)
      if (QPRINT) write(IW, 3000)

      do iorb = 1, NORB
        c1c1 = stevec(1, iorb, isp)**2
        ekin = stkin(1) * c1c1
        echf = stcoul(1, isp) * c1c1
        exhf = stexch(1, isp) * c1c1
        evz2 = zpot2(1) * c1c1
        evc2 = cpot2(1) * c1c1
        evx2 = xpot2(1, isp) * c1c1
        proj = pkpot(1, 1, isp) * c1c1
        pkv2 = pkpot(1, 2, isp) * c1c1
        epot(1:NVTERM) = stpot(1, 1:NVTERM) * c1c1

        if (MSTO >= 2) then
          do k = 2, MSTO
            indxk = index_arr(k)
            kk    = indxk + k
            km1   = k - 1
            ckck  = stevec(k, iorb, isp)**2
            ekin  = ekin + stkin(kk) * ckck
            echf  = echf + stcoul(kk, isp) * ckck
            exhf  = exhf + stexch(kk, isp) * ckck
            evz2  = evz2 + zpot2(kk) * ckck
            evc2  = evc2 + cpot2(kk) * ckck
            evx2  = evx2 + xpot2(kk, isp) * ckck
            proj  = proj + pkpot(kk, 1, isp) * ckck
            pkv2  = pkv2 + pkpot(kk, 2, isp) * ckck
            epot(1:NVTERM) = epot(1:NVTERM) + stpot(kk, 1:NVTERM) * ckck

            do l = 1, km1
              kl    = indxk + l
              ckcl2 = 2.0d0 * stevec(k, iorb, isp) * stevec(l, iorb, isp)
              ekin  = ekin + stkin(kl) * ckcl2
              echf  = echf + stcoul(kl, isp) * ckcl2
              exhf  = exhf + stexch(kl, isp) * ckcl2
              evz2  = evz2 + zpot2(kl) * ckcl2
              evc2  = evc2 + cpot2(kl) * ckcl2
              evx2  = evx2 + xpot2(kl, isp) * ckcl2
              proj  = proj + pkpot(kl, 1, isp) * ckcl2
              pkv2  = pkv2 + pkpot(kl, 2, isp) * ckcl2
              epot(1:NVTERM) = epot(1:NVTERM) + stpot(kl, 1:NVTERM) * ckcl2
            end do
          end do
        end if

        enuc  = -znuc * epot(1)
        occup = occ(iorb, isp) * ORBOCC

        vdev = ORBOCC * (echf - dot_product(vcoef(1:NVTERM), epot(1:NVTERM)))
        if (iorb <= ncore) then
          vdevc = vdevc + vdev**2
          wgt   = WGTC / (ORBOCC * ncore)
        else
          vdevv = vdevv + vdev**2
          wgt   = WGTV / (ORBOCC * (NORB - ncore))
        end if

        vcalc = vcoef(1) * epot(1)
        ij    = 0
        do i = 2, NVTERM
          do j = 2, i
            ij = ij + 1
            vmat(ij) = vmat(ij) + epot(i) * wgt * epot(j)
          end do
          vsum(i - 1) = vsum(i - 1) + epot(i) * wgt * (echf - sumocc * epot(1))
          vcalc       = vcalc + vcoef(i) * epot(i)
        end do
        vcalct = vcalct + occup * vcalc

        ! N-1 virtual orbital potential modification (if QMODVO=T)
        if (QMODVO) then
          smax   = real(NATOM - 1, real64)
          hole   = 1.0d0 - occ(iorb, isp)
          screen = merge(log(ELSUM / sumocc) / smax, 0.0d0, NATOM > 1)
          schole = hole * exp(-screen * proj)
          echf   = echf * (sumocc - schole) / sumocc
          el2    = ELSUM - sumocc
          if (el2 /= 0.0d0) evc2 = evc2 * (el2 - hole + schole) / el2

          charg(isp) = ASUM(occ(1, isp), NORB, 1)
          screen = 0.0d0
          if ((NATOM > 1) .and. (charg(isp) /= 0.0d0)) then
            screen = log(ELECS(isp) / charg(isp)) / smax
          end if
          schole = hole * exp(-screen * proj)
          if (charg(isp) /= 0.0d0) exhf = exhf * (charg(isp) - schole) / charg(isp)
          el2 = ELECS(isp) - charg(isp)
          if (el2 /= 0.0d0) evx2 = evx2 * (el2 - hole + schole) / el2
        end if

        oe(2:8) = [ekin, enuc, echf, exhf, evz2, evc2, evx2]
        eorb    = sum(oe(2:8))
        pkv2    = eval(iorb, isp) * proj - pkv2
        if (qpkinc) eorb = eorb + pkv2
        oe(1)   = eorb
        oe(9)   = pkv2
        pkerr   = eval(iorb, isp) - eorb

        te     = te + occup * oe
        tpkerr = tpkerr + pkerr * occup
        teval  = teval + eval(iorb, isp) * occup

        if (QPRINT) then
          write(IW, 4000) iorb, oe, occup
          write(IW, 4500) eval(iorb, isp), vcalc, pkerr
        end if
        gamma(iorb, isp) = 0.5d0 * (eval(iorb, isp) + oe(1) + oe(2))
      end do
      charg(isp) = ASUM(occ(1, isp), NORB, 1)
    end do

    tekin = te(2); tenuc = te(3); techf = te(4); texhf = te(5)
    tevz2 = te(6); tevc2 = te(7); tevx2 = te(8); tpkv2 = te(9)

    !-------------------------------------------------------------------
    ! Solve for new local model potential coefficients VCOEF(1..NVTERM)
    !-------------------------------------------------------------------
    if (QCSTFT .and. (.not. qnofit)) then
      call solve_constrained_model_potential_fit(vmat, vsum, vfit2, tevz2, &
                                                 vcoef, sumocc, NVTERM, WGT2, qindef)
      vcalc0 = sum(vcoef(1:NVTERM))
      vcalc2 = dot_product(vcoef(1:NVTERM), vfit2(1:NVTERM))
      vdev2  = tevz2 - vcalc2
      if (abs(tevz2) < 0.1d0) vdev2 = -zzrep - vcalc2
      if (abs(zzrep) < 0.1d0) vdev2 = 0.0d0
    else
      vcalc0 = vcoef(1)
      vcalc2 = vcoef(1) * vfit2(1)
      ij     = 0
      do i = 2, NVTERM
        do j = 2, i
          ij = ij + 1
          vmat(ij) = vmat(ij) + WGT0 / sumocc &
                   + vfit2(i) * WGT2 * vfit2(j) / (tevz2 - 0.5d0)
        end do
        vsum(i - 1) = vsum(i - 1) - WGT0 &
                    + vfit2(i) * WGT2 * (tevz2 - sumocc * vfit2(1)) / (tevz2 - 0.5d0)
        vcalc0 = vcalc0 + vcoef(i)
        vcalc2 = vcalc2 + vcoef(i) * vfit2(i)
      end do

      vdev2 = tevz2 - vcalc2
      if (abs(tevz2) < 0.1d0) vdev2 = -zzrep - vcalc2
      if (abs(zzrep) < 0.1d0) vdev2 = 0.0d0

      if (.not. qnofit) then
        call solve_model_potential_fit(vmat, vsum, vcoef, sumocc, NVTERM, qindef)
        if (qindef) then
          call WARN(18, qretry)
          if (NVTERM >= 4) then
            vcoef(1)        = sumocc
            vcoef(2:NVTERM) = 0.0d0
            vcoef(3)        = -sumocc
          end if
        end if
      end if
    end if

    teorB = sum(te(2:8))
    if (qpkinc) teorB = teorB + tpkv2
    te(1) = teorB

    if (QPRINT) then
      write(IW, 5000) te, sumocc
      write(IW, 5500) teval, vcalct, vcalc2, tpkerr, sumocc - vcalc0
    end if

    tke    = tekin
    tetot  = teorB - 0.5d0 * (techf + texhf)
    tpe    = tetot - tke
    virial = tpe / tke
    toten  = tetot - 0.5d0 * (tevc2 + tevx2 - zzrep)
    if (qpkinc) toten = toten - 0.5d0 * tpkv2 + 0.5d0 * tpkerr

    if (QPRINT) write(IW, 6000) tetot, tke, tpe, virial, zzrep, toten

    aspin  = charg(1)
    bspin  = merge(charg(2), aspin, logical(QOPEN))
    echarg = aspin + bspin
    spdiff = aspin - bspin
    charge(2, IATOM) = echarg
    charge(3, IATOM) = spdiff
    totchg = znuc - echarg
    call run_chgan(charge)

    vdevc = sqrt(vdevc)
    vdevv = sqrt(vdevv)
    mol_ws%last_pk_penalty  = mol_ws%last_pk_penalty + tpkerr
    mol_ws%last_rms_coulomb = max(mol_ws%last_rms_coulomb, max(vdevc, vdevv))
    if (QDFTEX) then
      mol_ws%last_rms_exchange = max(mol_ws%last_rms_exchange, abs(vdev2))
    end if

    if (QPRINT) then
      if (NORB * ISPIN > 20) write(IW, 8000)
      write(IW, 9000) vdevc, vdevv, vdev2
      if (qnofit) then
        write(IW, 9500)
      else
        write(IW, 10000) (nv, vcoef(nv), nv = 1, NVTERM)
      end if
    end if

    WGT2 = wgt2sv

1000 format(/' ENERGY ANALYSIS...')
2000 format(/' ',a8)
3000 format(/'  #',4x,'ORBITAL',3x,'KINETIC',4x,'INT',7x,'INT',7x,'INT',7x,'EXT',7x,'EXT',7x,'EXT',5x,'PSEUDO-',6x,'ORB'/ &
            ' ',6x,'ENERGY',4x,'ENERGY',3x,'NUC ATTR',2x,'COULOMB',3x,'EXCHANGE',2x, &
            'NUC ATTR',2x,'COULOMB',3x,'EXCHANGE',2x,'POTENTIAL',3x,'OCCUP'/)
4000 format(' ',i2,10f10.3)
4500 format(' ',3x,'(',f8.3,')',20x,'(',f8.3,')',42x,'(',f6.3,')')
5000 format(/' =>',10f10.3)
5500 format(' ',3x,'(',f8.3,')',20x,'(',f8.3,')',9x,'(',f9.3,')',22x,'(',f6.3,')','(',f8.3,')')
6000 format(/'    TOTAL ELECTRONIC ENERGY...',f11.4/ &
            '    TOTAL KINETIC ENERGY......',f11.4/ &
            '    TOTAL POTENTIAL ENERGY....',f11.4/ &
            '    VIRIAL THEOREM (V/T)......',f11.4// &
            '    NUCLEAR REPULSION.........',f11.4/ &
            '    CONTR. TO MOLEC. ENERGY...',f11.4)
8000 format(///' ')
9000 format(/' STD DEVS FOR THE LAST MODEL POTENTIAL FIT:'// &
            '    CORE HF COULOMB POTENTIAL........',f9.4/ &
            '    VALENCE HF COULOMB POTENTIAL.....',f9.4/ &
            '    EXTERNAL HF NUCLEAR ATTRACTION...',f9.4)
9500 format(/' MODEL POTENTIAL HELD FIXED...')
10000 format(/' NEW LOCAL MODEL POTENTIAL COEFFICIENTS ...'//(' ',i2,2x,f10.4))
  end subroutine run_analys

  subroutine run_chgan(charge)
    real(real64), intent(in) :: charge(3, NATOM)
    real(real64) :: znuc, echarg, spdiff, aspin, bspin, totchg

    znuc   = charge(1, IATOM)
    echarg = charge(2, IATOM)
    spdiff = charge(3, IATOM)
    aspin  = 0.5d0 * (echarg + spdiff)
    bspin  = 0.5d0 * (echarg - spdiff)
    totchg = znuc - echarg
    if (QPRINT) write(IW, 1000) aspin, bspin, spdiff, znuc, echarg, totchg

1000 format(/' SPIN ANALYSIS...'// &
            '    ALPHA SPIN...',f9.4/ &
            '    BETA SPIN....',f9.4/ &
            '    NET SPIN.....',f9.4// &
            ' CHARGE ANALYSIS...'// &
            '    NUCLEAR CHARGE......',f9.4/ &
            '    ELECTRONIC CHARGE...',f9.4/ &
            '    NET CHARGE..........',f9.4)
  end subroutine run_chgan

end module analys