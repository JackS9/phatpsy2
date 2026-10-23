module atomic
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin, only: run_lowdin
  use symmetry, only: symop, symxyz
  implicit none
  private
  public :: run_atomic_element

contains

  pure real(real64) function stnorm(nxx, etaxx)
    integer(int32), intent(in) :: nxx
    real(real64),   intent(in) :: etaxx
    stnorm = (2.0d0 * etaxx)**nxx * sqrt(2.0d0 * etaxx / FACT(2 * nxx + 1))
  end function stnorm

  subroutine run_atomic_element(nratom, qrealy, eof_elem)
    integer(int32),  intent(inout) :: nratom
    logical(kind=1), intent(in)    :: qrealy
    logical,         intent(out)   :: eof_elem

    ! Local allocatable arrays (formerly CORE(IC(1))..CORE(IC(26)) in contrl.f)
    integer(int32), allocatable :: n(:), l(:), m0(:), lv(:), mv(:)
    real(real64),   allocatable :: eta(:), anorm(:), stovl(:), stkin(:),     &
      stpot(:,:), stpotx(:), vexp(:), vcoef(:), sthmat(:), ontran(:,:),      &
      stevec(:,:,:), eval(:,:), occ(:,:), fock(:,:), gamma(:,:), zpot2(:),   &
      scrat1(:,:), scrat2(:,:), vfit2(:), cpot2(:), pkpot(:,:,:),            &
      stovl_bak(:), stevec_bak(:,:,:), eval_bak(:,:), occ_bak(:,:)

    ! Local scalars and small fixed arrays from atomic.f
    character(len=1), parameter :: LTYPE(6) = ['S', 'P', 'D', 'F', 'G', 'H']
    real(real64), parameter :: ESCALE = 1.55d0, ZSCALE = 1.12956d0,          &
      RMAX0 = 3.0d0, ZPOWER = 0.33333d0, BPERA = 1.889727d0, APERB = 0.529177d0
    character(len=2)  :: latom
    character(len=40) :: descr
    integer(int32)    :: nucz, isym, ixyz, neqat, iuatom, ncoef, iurst, ios
    integer(int32)    :: isto, ni, li, lli, mli, iml, jsto, nv, lvn, mvn
    integer(int32)    :: ntyp, ltyp, mtyp, ij, ntei, isp, jdifat, iorb, iiorb
    integer(int32)    :: last, icoef, mi, matom, korbt, kstot, ieqat, jsym
    real(real64)      :: rscale, rmAx, te, zzrep, znuc, etai, anormi, coef, expn
    real(real64)      :: toten, remain, coef1, coef2, alpha, beta, scale, r2
    real(real64)      :: xyz(3), vcoef0(10), eigval(2), occno(2)
    logical(kind=1)   :: qmnbas, qtfpot, qdfvec, qindef, qfirst_atom

    real(real64), external :: DOTPRD

    namelist /ELEM/  descr, latom, nucz, MDIFAT, qtfpot, rscale, qmnbas
    namelist /ATOM/  QFIXED, QVIRTL, QNWVEC, qdfvec, QFLIP, QMOVE,           &
                     QPNCHV, QPLOT_AT, QNWPOT, vcoef0, rmAx, iuatom, te,     &
                     neqat, isym, ixyz, xyz
    namelist /ATORB/ ncoef, eigval, occno

    eof_elem = .false.

    ! Deallocate any previous workspace arrays
    if (allocated(n)) deallocate(n, l, m0, eta, anorm, stovl, stkin, stpot, stpotx, &
                                 lv, mv, vexp, vcoef, sthmat, ontran, stevec, eval,  &
                                 occ, fock, gamma, zpot2, scrat1, scrat2, vfit2, cpot2, pkpot)
    if (allocated(stovl_bak)) deallocate(stovl_bak, stevec_bak, eval_bak, occ_bak)

    ! Allocate element-level workspace arrays
    allocate(n(NSTO), l(NSTO), m0(NSTO), eta(NSTO), anorm(NSTO))
    allocate(stovl(M2STO), stkin(M2STO), stpot(M2STO, NVTERM), stpotx(NVTERM))
    allocate(lv(NVTERM), mv(NVTERM), vexp(NVTERM), vcoef(NVTERM))
    allocate(sthmat(M2STO), ontran(MSTO, MSTO))
    allocate(stevec(MSTO, NORB, ISPIN), eval(NORB, ISPIN), occ(NORB, ISPIN))
    allocate(fock(N2ORB, ISPIN), gamma(NORB, ISPIN), zpot2(M2STO))
    allocate(scrat1(MSTO, MSTO), scrat2(MSTO, 9), vfit2(NVTERM))
    allocate(cpot2(M2STO), pkpot(M2STO, 6, ISPIN))
    allocate(stovl_bak(M2STO))
    allocate(stevec_bak(MSTO, NORB, ISPIN), eval_bak(NORB, ISPIN), occ_bak(NORB, ISPIN))

    ! Initialize defaults (&ELEM)
    qmnbas = .false.
    MDIFAT = 1
    qtfpot = .true.
    rscale = 0.0d0
    descr  = 'STO BASIS...'
    iurst  = merge(IUNIT3, IUNIT4, logical(QBAKUP))
    if (QRSTRT) call read_atom_stream_mdifat(iurst, MDIFAT)

    stevec = 0.0d0; cpot2 = 0.0d0; fock = 0.0d0
    gamma  = 0.0d0; pkpot = 0.0d0; vfit2 = 0.0d0; zpot2 = 0.0d0
    zzrep  = 0.0d0

    if (.not. QNWBAS) then
      call read_elem_basis(IELEM, latom, nucz, n, l, eta, anorm)
      call read_elem_potentials(IELEM, stovl, stkin, stpot, lv, mv, vexp)
      call read_elem_core_matrices(IELEM, sthmat, ontran)
    end if

    if (QTRACE) write(9, *) 'Reading Element Description (&ELEM)...'
    read(IR, nml=ELEM, iostat=ios)
    if (ios < 0) then
      eof_elem = .true.
      if (allocated(n)) deallocate(n, l, m0, eta, anorm, stovl, stkin, stpot, stpotx, &
                                   lv, mv, vexp, vcoef, sthmat, ontran, stevec, eval,  &
                                   occ, fock, gamma, zpot2, scrat1, scrat2, vfit2, cpot2, pkpot)
      return
    end if

    write(IW, 1000) latom, nucz
    if (.not. QNWBAS) write(IW, 2000) latom, IUNIT0
    znuc = real(nucz, real64)
    if (rscale <= 0.0d0) rscale = ZSCALE * znuc**(1.0d0 / 3.0d0)
    if ((rscale <= 0.0d0) .and. (nucz == 1)) rscale = 1.0d0
    if (QDEBUG) write(IUDUMP, nml=ELEM)
    if (.not. QBAKUP) call write_atom_stream_mdifat(IUNIT3, MDIFAT)

    !-------------------------------------------------------------------
    ! Read and write STO basis
    !-------------------------------------------------------------------
    write(IW, 3000) descr
    jsto = 0
    do isto = 1, NSTO
      if (QNWBAS) then
        read(IR, 4000) ni, li, etai
        anormi = stnorm(ni, etai)
        n(isto) = ni; l(isto) = li; eta(isto) = etai; anorm(isto) = anormi
      else
        ni = n(isto); li = l(isto); etai = eta(isto); anormi = anorm(isto)
      end if
      etai = abs(etai)
      jsto = jsto + 1
      lli  = 2 * li
      mli  = -li
      JSTOT = JSTOT + 1
      mol_ws%iatomx(JSTOT) = IATOM + 1
      mol_ws%nx(JSTOT)     = ni
      mol_ws%lx(JSTOT)     = li
      mol_ws%mlx(JSTOT)    = mli
      mol_ws%etax(JSTOT)   = etai
      mol_ws%anormx(JSTOT) = anormi
      write(IW, 5000) jsto, ni, LTYPE(li + 1), mli, etai, anormi

      if (li /= 0) then
        do iml = 1, lli
          mli   = iml - li
          jsto  = jsto + 1
          JSTOT = JSTOT + 1
          mol_ws%iatomx(JSTOT) = IATOM + 1
          mol_ws%nx(JSTOT)     = ni
          mol_ws%lx(JSTOT)     = li
          mol_ws%mlx(JSTOT)    = mli
          mol_ws%etax(JSTOT)   = etai
          mol_ws%anormx(JSTOT) = anormi
          write(IW, 5500) jsto, mli
        end do
      end if

      if (ni > NMAX)   call BOMB(1)
      if (li > LMAX)   call BOMB(2)
      if (jsto > MSTO) call BOMB(3)
      if (JSTOT > MSTOT) call BOMB(0)
      m0(isto) = jsto - li
    end do

    !-------------------------------------------------------------------
    ! Read and write model potential parameters
    !-------------------------------------------------------------------
    lv(1) = 0; mv(1) = 0; vexp(1) = 0.0d0; vcoef(1) = znuc
    if (NVTERM >= 2 .and. QNWBAS) then
      write(IW, 8000)
      do nv = 1, NVTERM
        coef = 0.0d0; expn = 1.0d0; lvn = 0; mvn = 0
        if (qtfpot .and. nv <= MXTF) then
          coef = TFCOEF(nv); expn = TFEXP(nv); lvn = LTF(nv)
        end if
        if ((.not. qtfpot) .or. (nv > MXTF)) then
          read(IR, 14000) lvn, mvn, expn, coef
        end if
        lv(nv)   = lvn
        mv(nv)   = mvn
        vexp(nv) = expn * rscale
        if ((znuc == 0.0d0) .or. (expn < 0.0d0)) vexp(nv) = abs(expn)
        if (lv(nv) >= 0) then
          ntyp = lv(nv); ltyp = 0; mtyp = 0
        else
          ntyp = -lv(nv) + 1; ltyp = -lv(nv); mtyp = mv(nv)
        end if
        vcoef(nv) = merge(coef, znuc * coef, znuc == 0.0d0)
        write(IW, 10000) nv, ntyp, LTYPE(ltyp + 1), mtyp, &
                         vexp(nv), expn, vcoef(nv), coef
      end do
    end if

    !-------------------------------------------------------------------
    ! Compute 1-Center Integrals and write on IUNIT0
    !-------------------------------------------------------------------
    if (QNWBAS) then
      call TIMOUT(0)
      call ONEINT(stovl, stkin, stpot, stpotx, lv, mv, vexp, n, l, eta, &
                  anorm, mol_ws%index_arr, mol_ws%cgc, qrealy)
      call write_elem_basis(IELEM, latom, nucz, n, l, eta, anorm)
      call write_elem_potentials(IELEM, stovl, stkin, stpot, lv, mv, vexp)
      stovl_bak = stovl
      call run_lowdin(stovl, ontran, scrat1, MSTO, MSTO, 0, qindef)
      if (qindef) call BOMB(0)
      stovl = stovl_bak
      do ij = 1, M2STO
        sthmat(ij) = stkin(ij) - znuc * stpot(ij, 1)
      end do
      call write_elem_core_matrices(IELEM, sthmat, ontran)

      call TWOINT(ntei, n, l, eta, mol_ws%s, mol_ws%w, mol_ws%index_arr, &
                  anorm, mol_ws%cgc, qrealy, mol_ws%buffer)
      write(IW, 11000) latom, IUNIT0
      call TIMOUT(4)
    end if

    !-------------------------------------------------------------------
    ! Loop over non-equivalent atoms (JDIFAT = 1..MDIFAT)
    !-------------------------------------------------------------------
    qfirst_atom = .true.
    occno(2) = 1.0d0
    ORBOCC   = 3 - ISPIN
    do isp = 1, ISPIN
      occ(1, isp)  = 1.0d0
      eval(1, isp) = 0.0d0
      if (qmnbas) stevec(1, 1, isp) = 1.0d0
    end do
    te     = 0.0d0
    iuatom = IR
    JATOM  = IATOM

    write(IW, 12000) MDIFAT, latom
    do jdifat = 1, MDIFAT
      IATOM = IATOM + 1
      if (IATOM > NATOM) call BOMB(8)

      if (QRSTRT) then
        read(iurst) QFLAGS, neqat, xyz, stevec, eval, occ, fock, gamma, &
                    cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx
        if (QANGST) then
          xyz  = APERB * xyz
          rmAx = APERB * rmAx
        end if
      end if

      isym = 0; ixyz = 0
      if (.not. QRSTRT) then
        xyz  = 0.0d0
        rmAx = RMAX0 * znuc**ZPOWER
      else
        mol_ws%coord(:, IATOM) = xyz
      end if
      vcoef0(1:NVTERM) = vcoef(1:NVTERM)

      QNWVEC   = .false.; qdfvec   = QRSTRT;  QMOVE  = (.not. QRSTRT)
      QFLIP    = .false.; QFIXED   = .false.; QVIRTL = .false.
      QPNCHV   = .false.; QPLOT_AT = .false.; QNWPOT = .false.
      neqat    = 0

      if (.not. QATOM) read(IR, nml=ATOM)
      if (QRSTRT) qdfvec = .true.
      if (.not. QOPEN) QFLIP = .false.
      if (QFLIP .and. .not. QRSTRT) qdfvec = .true.
      if (QBAKUP .and. (QNWVEC .or. QMOVE)) call WARN(12, QNWVEC)
      if (QVIRTL) then
        write(IW, 13000)
        rmAx = 0.0d0
      end if
      if (QANGST) then
        xyz  = BPERA * xyz
        rmAx = BPERA * rmAx
      end if
      if (QDEBUG) write(IUDUMP, nml=ATOM)

      if (qfirst_atom .or. qdfvec) then
        if (.not. (QRSTRT .and. .not. QNWVEC)) then
          if (QFLIP .and. .not. QRSTRT) then
            call FLIP(stevec, 1, ISPIN, stevec, ISPIN, ISPIN, MSTO * NORB)
            call FLIP(eval,   1, ISPIN, eval,   ISPIN, ISPIN, NORB)
            call FLIP(fock,   1, ISPIN, fock,   ISPIN, ISPIN, N2ORB)
            call FLIP(gamma,  1, ISPIN, gamma,  ISPIN, ISPIN, NORB)
            call FLIP(occ,    1, ISPIN, occ,    ISPIN, ISPIN, NORB)
            call FLIP(pkpot,  1, ISPIN, pkpot,  ISPIN, ISPIN, 6 * M2STO)
            qdfvec = .false.
          else
            remain = znuc
            do iorb = 1, NORB
              iiorb = (iorb * (iorb + 1)) / 2
              last  = merge(iorb, max(iorb - 1, 1), logical(qdfvec))
              do isp = 1, ISPIN
                occno(isp)  = min(remain / ORBOCC, occ(last, isp))
                eigval(isp) = eval(last, isp)
              end do
              ncoef = 0
              if (.not. QOPEN) occno(1) = 2.0d0 * occno(1) - occno(2)
              read(iuatom, nml=ATORB)
              if (QDEBUG) write(IUDUMP, nml=ATORB)
              if (.not. QOPEN) occno(1) = 0.5d0 * (occno(1) + occno(2))
              do isp = 1, ISPIN
                occ(iorb, isp)  = occno(isp)
                remain          = remain - ORBOCC * occno(isp)
                eval(iorb, isp) = eigval(isp)
                if (.not. QRSTRT) then
                  fock(iiorb, isp) = eigval(isp)
                  gamma(iorb, isp) = merge(0.0d0, eigval(isp), eigval(isp) > 0.0d0)
                end if
              end do

              if ((iorb /= last) .and. (.not. qdfvec) .and. (ncoef <= 0)) then
                stevec(2:MSTO, iorb, 1) = stevec(1:MSTO-1, last, 1)
                if (QOPEN) stevec(2:MSTO, iorb, 2) = stevec(1:MSTO-1, last, 2)
              else if (ncoef /= 0) then
                ncoef = abs(ncoef)
                do icoef = 1, ncoef
                  read(iuatom, 14000) isto, mli, coef1, coef2
                  if ((isto > NSTO) .or. (isto <= 0)) call BOMB(11)
                  if (abs(mli) > l(isto)) call BOMB(10)
                  mi = m0(isto) + mli
                  stevec(mi, iorb, 1) = coef1
                  if (QOPEN) then
                    if ((coef2 == 0.0d0) .and. (eigval(1) == eigval(2))) coef2 = coef1
                    stevec(mi, iorb, 2) = coef2
                  end if
                end do
              end if
            end do
            if (iuatom /= IR) write(IW, 15000) latom, jdifat, iuatom
            call NORMLZ(stevec(1,1,1), NORB, stovl, MSTO, M2STO)
            if (QOPEN) call NORMLZ(stevec(1,1,2), NORB, stovl, MSTO, M2STO)
          end if
        end if

        if (.not. (QFLIP .and. .not. QRSTRT)) then
          write(IW, 16000) latom
          if (qdfvec) write(IW, 17000) jdifat
          write(IW, 18000)
        end if

        do isp = 1, ISPIN
          if (.not. (QFLIP .and. .not. QRSTRT)) then
            if (QOPEN) write(IW, 19000) SPIN(isp)
            call OUTVEC(stevec(1,1,isp), eval(1,isp), MSTO, NORB, IW)
          end if
          if (.not. QRSTRT) then
            call DENMAT(pkpot(1,4,isp), stevec(1,1,isp), occ(1,isp), MSTO, M2STO, NORB)
            pkpot(:,4,isp) = PKSCAL * pkpot(:,4,isp)
            call DENMAT(pkpot(1,5,isp), stevec(1,1,isp), eval(1,isp), MSTO, M2STO, NORB)
            pkpot(:,5,isp) = PKSCAL * pkpot(:,5,isp)
            pkpot(:,6,isp) = 0.0d0
          end if
        end do

        if (.not. (QFLIP .and. .not. QRSTRT)) then
          write(IW, 16000) latom
          if (qdfvec) write(IW, 17000) jdifat
          write(IW, 20000)
          call POPOUT(eval, occ, znuc, alpha, beta, NORB)
          if (.not. QRSTRT) toten = DOTPRD(eval, occ, NORB * ISPIN) * ESCALE * (3 - ISPIN)
          if (te /= 0.0d0) toten = te
          if (QRSTRT .and. QNWVEC) write(IW, 21000)
        end if
      end if

      if (qfirst_atom .or. qdfvec .or. QNWPOT) then
        if (NVTERM >= 2) then
          if (QNWPOT .and. .not. qfirst_atom) then
            write(IW, 16000) latom
            write(IW, 17000) jdifat
          end if
          scale = 1.0d0
          if ((.not. QRSTRT) .and. (nucz /= 0)) scale = (alpha + beta) / znuc
          write(IW, 22000)
          do nv = 1, NVTERM
            if (qfirst_atom .and. .not. QNWPOT) vcoef(nv) = scale * vcoef(nv)
            if (QNWPOT) vcoef(nv) = vcoef0(nv)
            write(IW, 23000) nv, vcoef(nv)
          end do
          if (QDEBUG) call PLOTV(lv, mv, vexp, vcoef, NVTERM, IUDUMP)
        end if
        if (.not. QATOM) write(IW, 24000)
      end if

      if (.not. QATOM) then
        if (.not. QMOVE) then
          if (any(xyz /= mol_ws%coord(:, IATOM))) QMOVE = .true.
        end if
        mol_ws%coord(:, IATOM) = xyz
        mol_ws%label(IATOM)    = latom
        write(IW, 25000) latom, jdifat, xyz
        if (QFLIP) write(IW, 32000)
        if (QVIRTL) write(IW, 27000)
        if (QMOVE .and. QRSTRT) write(IW, 28000)
        if (QFIXED) write(IW, 29000)
        write(IW, 30000) rmAx
      end if

      mol_ws%charge(1, IATOM) = znuc
      mol_ws%charge(2, IATOM) = alpha + beta
      mol_ws%charge(3, IATOM) = alpha - beta

      if (.not. QBAKUP) then
        call write_atom_record_ref(IUNIT3, QFLAGS, neqat, xyz(1), xyz(2), xyz(3), &
                                   stevec, eval, occ, fock, gamma, &
                                   cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)
      end if
      TMOLE = TMOLE + toten

      if (.not. QVIRTL) then
        nratom = nratom + 1
        do isp = 1, ISPIN
          korbt = IORBT
          do iorb = 1, NORB
            korbt = korbt + 1
            mol_ws%evalx(korbt, isp) = eval(iorb, isp)
            mol_ws%occx(korbt, isp)  = occ(iorb, isp)
            kstot = merge(JSTOT - MSTO, JSTOT, jdifat == 1)
            do jsto = 1, MSTO
              kstot = kstot + 1
              if (isp == 1 .and. iorb == 1 .and. jdifat > 1) then
                mol_ws%iatomx(kstot) = IATOM
                mol_ws%nx(kstot)     = mol_ws%nx(kstot - MSTO)
                mol_ws%lx(kstot)     = mol_ws%lx(kstot - MSTO)
                mol_ws%mlx(kstot)    = mol_ws%mlx(kstot - MSTO)
                mol_ws%etax(kstot)   = mol_ws%etax(kstot - MSTO)
                mol_ws%anormx(kstot) = mol_ws%anormx(kstot - MSTO)
              end if
              mol_ws%evecx(kstot, korbt, isp) = stevec(jsto, iorb, isp)
            end do
          end do
        end do
        if (jdifat > 1) JSTOT = JSTOT + MSTO
        IORBT = IORBT + NORB
        if (IORBT > NORBT) call BOMB(9)
      end if
      qfirst_atom = .false.

      !-----------------------------------------------------------------
      ! Loop over symmetry-equivalent atoms (IEQAT = 1..NEQAT)
      !-----------------------------------------------------------------
      if (neqat > 0) then
        if (QRSTRT .and. QNWVEC) then
          stevec_bak = stevec
          eval_bak   = eval
          occ_bak    = occ
        end if
        do ieqat = 1, neqat
          IATOM = IATOM + 1
          if (IATOM > NATOM) call BOMB(8)
          mol_ws%charge(1, IATOM) = znuc
          mol_ws%charge(2, IATOM) = mol_ws%charge(2, IATOM - 1)
          mol_ws%charge(3, IATOM) = mol_ws%charge(3, IATOM - 1)

          ! FIXED BUG: Restored QFLAGS instead of implicit REAL*8 FLAGS
          if (QRSTRT) then
            read(iurst) QFLAGS, isym, ixyz, xyz, stevec, eval, occ, &
                        fock, gamma, cpot2, vfit2, vcoef, pkpot,    &
                        zpot2, toten, zzrep, rmAx
          end if
          if (QMOVE) xyz = 0.0d0
          QFLIP = .false.; QPNCHV = .false.; QPLOT_AT = .false.
          read(IR, nml=ATOM)
          if (.not. QOPEN) QFLIP = .false.
          if (QVIRTL) rmAx = 0.0d0
          if (QDEBUG) write(IUDUMP, nml=ATOM)

          if (.not. (QRSTRT .and. .not. QNWVEC)) then
            if (QRSTRT) then
              stevec = stevec_bak
              eval   = eval_bak
              occ    = occ_bak
            end if
            if (isym /= 0) then
              call SYMOP(isym, ixyz, stevec, vcoef, l, lv, mol_ws%d, &
                         mol_ws%scrat, NSTO, MSTO, NORB, ISPIN,      &
                         NVTERM, LMAX, LLMXP1)
            end if
            if (QFLIP) then
              call FLIP(stevec, 1, 2, stevec, 2, 2, MSTO * NORB)
              call FLIP(eval,   1, 2, eval,   2, 2, NORB)
              call FLIP(occ,    1, 2, occ,    2, 2, NORB)
              call FLIP(fock,   1, 2, fock,   2, 2, N2ORB)
              call FLIP(gamma,  1, 2, gamma,  2, 2, NORB)
              call FLIP(pkpot,  1, ISPIN, pkpot, ISPIN, ISPIN, 6 * M2STO)
            end if
          end if

          if (QFLIP) mol_ws%charge(3, IATOM) = -mol_ws%charge(3, IATOM)
          r2 = sum(xyz**2)
          if (QMOVE .and. r2 < 1.0d-4) then
            call SYMXYZ(isym, ixyz, xyz, mol_ws%coord, NATOM, IATOM)
          end if
          write(IW, 26000) xyz
          if (QFLIP) write(IW, 32000)
          if (QVIRTL) write(IW, 27000)
          if (QMOVE .and. QRSTRT) write(IW, 28000)
          if (QFIXED) write(IW, 29000)
          jsym = isym
          if (isym < 0) jsym = -isym
          if (jsym > 1) write(IW, 33000) jsym
          if (isym < 0) write(IW, 34000)
          mol_ws%coord(:, IATOM) = xyz
          mol_ws%label(IATOM)    = latom
          if (.not. QBAKUP) then
            call write_atom_record_equiv(IUNIT3, QFLAGS, isym, ixyz, xyz(1), xyz(2), xyz(3), &
                                         stevec, eval, occ, fock, gamma, &
                                         cpot2, vfit2, vcoef, pkpot, zpot2, toten, zzrep, rmAx)
          end if
          TMOLE = TMOLE + toten

          if (.not. QVIRTL) then
            nratom = nratom + 1
            do isp = 1, ISPIN
              korbt = IORBT
              do iorb = 1, NORB
                korbt = korbt + 1
                mol_ws%evalx(korbt, isp) = eval(iorb, isp)
                mol_ws%occx(korbt, isp)  = occ(iorb, isp)
                kstot = JSTOT
                do jsto = 1, MSTO
                  kstot = kstot + 1
                  if (isp == 1 .and. iorb == 1) then
                    mol_ws%iatomx(kstot) = IATOM
                    mol_ws%nx(kstot)     = mol_ws%nx(kstot - MSTO)
                    mol_ws%lx(kstot)     = mol_ws%lx(kstot - MSTO)
                    mol_ws%mlx(kstot)    = mol_ws%mlx(kstot - MSTO)
                    mol_ws%etax(kstot)   = mol_ws%etax(kstot - MSTO)
                    mol_ws%anormx(kstot) = mol_ws%anormx(kstot - MSTO)
                  end if
                  mol_ws%evecx(kstot, korbt, isp) = stevec(jsto, iorb, isp)
                end do
              end do
            end do
            JSTOT = JSTOT + MSTO
            IORBT = IORBT + NORB
            if (IORBT > NORBT) call BOMB(9)
          end if

          if (QRSTRT .and. QNWVEC .and. ieqat < neqat) then
            stevec_bak = stevec
            eval_bak   = eval
            occ_bak    = occ
          end if
        end do
      end if
    end do

    matom = IATOM - JATOM
    write(IW, 35000) matom, latom

1000 format(/' -------------------'/' ATOMIC SYMBOL...',1x,a2/' ATOMIC NUMBER...',i3/' -------------------')
2000 format(/' ... BASIS AND ONE-CENTER INTEGRALS FOR ',a2,' READ FROM UNIT',i3,' ...')
3000 format(/' ',a40//'  #',5x,'TYPE',5x,'EXPONENT',5x,'NORMALIZATION'/'  _',5x,'____',5x,'________',5x,'_____________'/)
4000 format(2i3,f10.6)
5000 format(' ',i2,i6,a1,i2,f13.6,1pd18.6)
5500 format(' ',i2,7x,i2)
8000 format(/' MODEL POTENTIAL ...'//'  #',5x,'TYPE',5x,'EXPONENT',17x,'COEF'/'  _',5x,'____',5x,'________',17x,'____'/)
10000 format(' ',i2,i6,a1,i2,f13.6,2x,'(',f6.3,') ',f10.4,2x,'(',f6.3,')')
11000 format(/' ... THE BASIS AND ONE-CENTER INTEGRALS FOR ',a2,' WERE WRITTEN ON UNIT',i3,' ...')
12000 format(/' ',i2,' NON-EQUIVALENT ',a2,' ATOMS WILL BE INPUT.')
13000 format(/' *** WARNING - THE TOTAL ENERGY WILL BE IN ERROR UNLESS THIS ATOM IS MADE ''REAL'' ***')
14000 format(2i3,2f10.6)
15000 format(/' ... ATOMIC ORBITALS FOR ',a2,i2,' READ FROM UNIT',i3,' ...')
16000 format(///' ',a2/' __')
17000 format(' ',i4/'   __')
18000 format(/' STARTING VECTORS AND ENERGIES...')
19000 format(/' ',a8)
20000 format(/' STARTING ORBITAL ENERGIES AND OCCUPATIONS...')
21000 format(/' ... RESTART DATA HAS BEEN PARTIALLY OVERRIDDEN ...')
22000 format(/' STARTING COEFFICIENTS FOR THE MODEL POTENTIAL...'/)
23000 format(' ',i2,2x,f10.4)
24000 format(/' NUCLEAR COORDINATES...'//' ',10x,'X',14x,'Y',14x,'Z'/' ',10x,'_',14x,'_',14x,'_')
25000 format(/' ',a2,i2,3f15.8/' ____')
26000 format(' ',4x,3f15.8)
27000 format(' ',t55,'* VIRTUAL        *')
28000 format(' ',t55,'* MOVED          *')
29000 format(' ',t55,'* FIXED          *')
30000 format(' ',t55,'* RMAX =',f8.3,' *')
32000 format(' ',t55,'* SPIN FLIPPED   *')
33000 format(' ',t55,'* ROTATED',i3,'-FOLD *')
34000 format(' ',t55,'* REFLECTED      *')
35000 format(/' THE INPUT FOR THE',i3,1x,a2,' ATOMS IS COMPLETE.')
    if (allocated(n)) deallocate(n, l, m0, eta, anorm, stovl, stkin, stpot, stpotx, &
                                 lv, mv, vexp, vcoef, sthmat, ontran, stevec, eval,  &
                                 occ, fock, gamma, zpot2, scrat1, scrat2, vfit2, cpot2, pkpot)
    if (allocated(stovl_bak)) deallocate(stovl_bak, stevec_bak, eval_bak, occ_bak)
  end subroutine run_atomic_element

end module atomic