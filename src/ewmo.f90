module ewmo
  use iso_fortran_env, only: real64, int32
  use state
  use lowdin, only: run_lowdin
  use density, only: denmat, lowdin_pop
  implicit none
  private
  public :: run_ewmo, QNOPOP_EWMO, USE_LOWDIN_EWMO, solve_fermi_level

  logical(kind=1), save :: QNOPOP_EWMO     = .true.  ! Legacy default (.true. uses bloop Fermi-Dirac POPUL)
  logical,         save :: USE_LOWDIN_EWMO = .false. ! .true. uses F = S*eps*S with Lowdin S^(-1/2)

contains

  subroutine run_ewmo()
    ! Scoped allocatable workspaces (formerly CORE(IC(1))..CORE(IC(6)) in contrl.f)
    real(real64), allocatable :: evalmo(:,:), occmo(:,:), scrat1(:,:), &
                                 scrat2(:,:), evecmo(:,:,:), ontran(:,:)
    real(real64), allocatable :: ovlp_bak(:,:), ewmat_bak(:,:)

    real(real64), parameter :: SMALL = 5.0d-5
    integer(int32)  :: isp, iorbt, jorbt, korbt, iiorbt, ii, jj, kk, ik, kj, ij
    integer(int32)  :: ierr, idegen, id
    real(real64)    :: remain, split, occup, sum_pop, oldpop, ct
    logical(kind=1) :: qindef

    call TIMOUT(0)

    if (allocated(evalmo)) then
      deallocate(evalmo, occmo, scrat1, scrat2, evecmo, ontran, ovlp_bak, ewmat_bak)
    end if

    allocate(evalmo(NORBT, ISPIN), occmo(NORBT, ISPIN))
    allocate(scrat1(NORBT, NORBT), scrat2(NORBT, 9))
    allocate(evecmo(NORBT, NORBT, ISPIN), ontran(NORBT, NORBT))
    allocate(ovlp_bak(N2ORBT, ISPIN), ewmat_bak(N2ORBT, ISPIN))

    occmo  = 0.0d0
    evecmo = 0.0d0
    evalmo = 0.0d0

    if (QEWMO) then
      if (QDEBUG) write(IUDUMP, 1000)
      if (QPRINT) then
        write(IW, 2000)
        write(IW, 3000) NCYCL
      end if
      ! Backup OVLP and EWMAT in memory instead of writing/rewinding IUTEMP
      ovlp_bak  = mol_ws%ovlp
      ewmat_bak = mol_ws%ewmat
    end if

    do isp = 1, ISPIN
      do iorbt = 1, NORBT
        iiorbt = (iorbt * (iorbt + 1)) / 2
        evecmo(iorbt, iorbt, isp) = 1.0d0
        if (USE_LOWDIN_EWMO) then
          evalmo(iorbt, isp) = mol_ws%ewmat(iiorbt, isp)
        else
          evalmo(iorbt, isp) = merge(0.0d0, mol_ws%ewmat(iiorbt, isp), &
                                     mol_ws%ewmat(iiorbt, isp) >= 0.0d0)
        end if
      end do
    end do

    do isp = 1, ISPIN
      if (QEWMO) then
        if (.not. USE_LOWDIN_EWMO) then
          ! Standard EWMO Matrix: F'_{ij} = -sqrt(eps_i * eps_j) * S_{ij} (Eq. VI-18)
          ij = 0
          do iorbt = 1, NORBT
            do jorbt = 1, iorbt
              ij = ij + 1
              mol_ws%ewmat(ij, isp) = -sqrt(evalmo(iorbt, isp) * evalmo(jorbt, isp)) &
                                    * mol_ws%ovlp(ij, isp)
            end do
          end do
        else
          ! Full Model Fock Matrix: F = S * eps * S in Lowdin S^(-1/2) basis (Eq. VI-12)
          ij = 0
          do iorbt = 1, NORBT
            ii = (iorbt * (iorbt - 1)) / 2
            do jorbt = 1, iorbt
              jj = (jorbt * (jorbt - 1)) / 2
              ij = ij + 1
              mol_ws%ewmat(ij, isp) = 0.0d0
              do korbt = 1, NORBT
                kk = (korbt * (korbt - 1)) / 2
                ik = merge(kk + iorbt, ii + korbt, iorbt < korbt)
                kj = merge(kk + jorbt, jj + korbt, jorbt < korbt)
                mol_ws%ewmat(ij, isp) = mol_ws%ewmat(ij, isp) &
                  + evalmo(korbt, isp) * mol_ws%ovlp(ik, isp) * mol_ws%ovlp(kj, isp)
              end do
            end do
          end do
          call run_lowdin(mol_ws%ovlp(1, isp), ontran, scrat1, NORBT, NORBT, 4, qindef)
          if (qindef) call BOMB(17)
          call CTSC(ontran, mol_ws%ewmat(1, isp), scrat1, mol_ws%ewmat(1, isp), NORBT, NORBT)
        end if

        if (QDEBUG) then
          if (QOPEN) write(IUDUMP, 4000) SPIN(isp)
          call PUTONE(mol_ws%ewmat(1, isp), NORBT, IUDUMP)
        end if

        block
          integer(int32) :: lwork, liwork
          real(real64), allocatable :: work(:)
          integer(int32), allocatable :: iwork(:)
          lwork  = 1 + 6 * NORBT + 2 * NORBT * NORBT
          liwork = 3 + 5 * NORBT
          allocate(work(lwork), iwork(liwork))
          call DSPEVD('V', 'U', NORBT, mol_ws%ewmat(1, isp), evalmo(1, isp), &
                      evecmo(1, 1, isp), NORBT, work, lwork, iwork, liwork, ierr)
          deallocate(work, iwork)
          if (ierr /= 0) call BOMB(17)
        end block

        mol_ws%ovlp  = ovlp_bak
        mol_ws%ewmat = ewmat_bak

        if (QPRINT .and. QOPEN) write(IW, 4000) SPIN(isp)

        if (.not. USE_LOWDIN_EWMO) then
          do iorbt = 1, NORBT
            iiorbt = (iorbt * (iorbt + 1)) / 2
            do jorbt = 1, NORBT
              if (mol_ws%ewmat(iiorbt, isp) >= 0.0d0) then
                evecmo(iorbt, jorbt, isp) = 0.0d0
              else
                evecmo(iorbt, jorbt, isp) = evecmo(iorbt, jorbt, isp) &
                                          / sqrt(abs(mol_ws%ewmat(iiorbt, isp)))
              end if
            end do
          end do
        else
          call run_lowdin(mol_ws%ovlp(1, isp), ontran, scrat1, NORBT, NORBT, 2, qindef)
          if (.not. qindef) then
            call DMPAB(ontran, NORBT, NORBT, NORBT, NORBT, evecmo(1, 1, isp), &
                       NORBT, NORBT, NORBT, NORBT, scrat1, NORBT, NORBT)
            evecmo(:, :, isp) = scrat1
          end if
          mol_ws%ovlp = ovlp_bak
        end if

        call NORMLZ(evecmo(1, 1, isp), NORBT, mol_ws%ovlp(1, isp), NORBT, N2ORBT)
        if (QPRINT) call OUTVEC(evecmo(1, 1, isp), evalmo(1, isp), NORBT, NORBT, IW)
      end if

      !-----------------------------------------------------------------
      ! Ground-state MO occupation assignment (FIXED out-of-bounds check)
      !-----------------------------------------------------------------
      if (QGFERM) then
        call solve_fermi_level(evalmo(1, isp), occmo(1, isp), NORBT, ELECS(isp) / ORBOCC, THETA, FERMI_MU(isp))
      else
        remain = ELECS(isp) / ORBOCC
        iorbt  = 1
        occ_loop: do while ((iorbt <= NORBT) .and. (remain > 0.0d0))
          idegen = 1
          degen_loop: do while (iorbt + idegen <= NORBT)
            split = abs(evalmo(iorbt, isp) - evalmo(iorbt + idegen, isp))
            if (split > SMALL) exit degen_loop
            idegen = idegen + 1
          end do degen_loop

          occup = min(1.0d0, remain / real(idegen, real64))
          do id = 1, idegen
            occmo(iorbt, isp) = occup
            iorbt  = iorbt + 1
            remain = remain - occup
          end do
        end do occ_loop
      end if
    end do

    !-------------------------------------------------------------------
    ! Excited-state occupations and optional EWMO population analysis
    !-------------------------------------------------------------------
    CHTRAN = 0.0d0
    DELTAE = 0.0d0
    if (.not. QNOPOP_EWMO .or. QGFERM) then
      do isp = 1, ISPIN
        if (.not. QGSTAT) call EXCITE(occmo, NORBT, ISPIN)
        if (QGFERM) then
          block
            real(real64) :: totpop_l
            real(real64), allocatable :: dens(:), lpop(:)
            allocate(dens(N2ORBT), lpop(NORBT))
            call denmat(dens, evecmo(1, 1, isp), occmo(1, isp), NORBT, N2ORBT, NORBT)
            call lowdin_pop(dens, mol_ws%ovlp(1, isp), lpop, totpop_l, NORBT, N2ORBT)
            do iorbt = 1, NORBT
              oldpop = mol_ws%popul(iorbt, isp)
              mol_ws%popul(iorbt, isp) = lpop(iorbt)
              ct = ORBOCC * (lpop(iorbt) - oldpop)
              CHTRAN = CHTRAN + 0.5d0 * abs(ct)
              DELTAE = DELTAE + evalmo(iorbt, isp) * ct
            end do
            deallocate(dens, lpop)
          end block
        else
          do iorbt = 1, NORBT
            sum_pop = 0.0d0
            oldpop  = mol_ws%popul(iorbt, isp)
            do jorbt = 1, NORBT
              sum_pop = sum_pop + occmo(jorbt, isp) * evecmo(iorbt, jorbt, isp)**2
            end do
            mol_ws%popul(iorbt, isp) = sum_pop
            ct     = ORBOCC * (sum_pop - oldpop)
            CHTRAN = CHTRAN + 0.5d0 * abs(ct)
            DELTAE = DELTAE + evalmo(iorbt, isp) * ct
          end do
        end if
      end do
    end if

    if (.not. QPRINT) then
      deallocate(evalmo, occmo, scrat1, scrat2, evecmo, ontran, ovlp_bak, ewmat_bak)
      return
    end if

    write(IW, 5000)
    write(IW, 3000) NCYCL
    call POPOUT(evalmo, occmo, ZNSUM, ELECS(1), ELECS(2), NORBT)

    ETOT   = TMOLE - ZZPOT
    TOTPE  = TMOLE - TOTKE
    VIRTHM = TOTPE / TOTKE
    write(IW, 6000) TOTPE, TOTKE, ETOT, ZZPOT, TMOLE, VIRTHM

    if (.not. QEWMO) then
      deallocate(evalmo, occmo, scrat1, scrat2, evecmo, ontran, ovlp_bak, ewmat_bak)
      return
    end if
    write(IW, 7000)
    write(IW, 3000) NCYCL
    if (QOPEN) write(IW, 8000)
    do iorbt = 1, NORBT
      write(IW, 9000) iorbt, (mol_ws%popul(iorbt, isp), isp = 1, ISPIN)
    end do
    ELECS(1) = ELECS(1) * ORBOCC
    write(IW, 10000) (ELECS(isp), isp = 1, ISPIN)
    call TIMOUT(6)

    deallocate(evalmo, occmo, scrat1, scrat2, evecmo, ontran, ovlp_bak, ewmat_bak)


1000 format(/' EWMO MATRIX...')
2000 format(///' MOLECULAR ORBITALS AND ENERGIES...')
3000 format(' ',t100,'... CYCLE',i3,' ...')
4000 format(/' ',a8)
5000 format(///' MOLECULAR ORBITAL ENERGIES AND OCCUPATIONS...')
6000 format(///' TOTAL ENERGY ANALYSIS...'//'    TOTAL POTENTIAL ENERGY.....',f12.4/'    TOTAL KINETIC ENERGY.......',f12.4/'    TOTAL ELECTRONIC ENERGY....',f12.4/'    NUCLEAR REPULSION ENERGY...',f12.4//'    TOTAL MOLECULAR ENERGY.....',f12.4//'    VIRIAL THEOREM (V/T).......',f12.6)
7000 format(///' GROSS ATOMIC ORBITAL POPULATIONS...'//)
8000 format(/'              ALPHA       BETA'/'              _____       ____'/)
9000 format(' ',6x,i3,f9.4,f12.4)
10000 format(/'     TOTAL',f9.4,f12.4)
  end subroutine run_ewmo

  subroutine EXCITE(occno, norb, ispin)
    integer(int32), intent(in)    :: norb, ispin
    real(real64),   intent(inout) :: occno(norb, ispin)

    integer(int32) :: ncr, icr, iorb, isp, nan, ian
    real(real64)   :: orbocc_local

    orbocc_local = 0.5d0 * real(ispin, real64)
    if (QTRANS) orbocc_local = 0.5d0 * orbocc_local

    if (NCREAT > 0) then
      do ncr = 1, NCREAT
        icr = ICREAT(ncr)
        iorb = abs(icr)
        if (iorb >= 1 .and. iorb <= norb) then
          isp = 1
          if (icr < 0) isp = ispin
          occno(iorb, isp) = occno(iorb, isp) + orbocc_local
        end if
      end do
    end if

    if (NANNIH > 0) then
      do nan = 1, NANNIH
        ian = IANNIH(nan)
        iorb = abs(ian)
        if (iorb >= 1 .and. iorb <= norb) then
          isp = 1
          if (ian < 0) isp = ispin
          occno(iorb, isp) = occno(iorb, isp) - orbocc_local
        end if
      end do
    end if
  end subroutine EXCITE

  !---------------------------------------------------------------------
  ! SOLVE_FERMI_LEVEL: Determines the chemical potential mu (Fermi level)
  ! and fractional orbital occupations n_i according to the Fermi-Dirac
  ! distribution:
  !   n_i = 1 / (1 + exp((eps_i - mu) / theta))
  ! such that sum_i n_i = nelec to machine precision.
  ! Monotonic root-finding via hybrid Newton-Raphson / bisection.
  !---------------------------------------------------------------------
  subroutine solve_fermi_level(evals, occs, norb, nelec, theta, fermi_mu)
    integer(int32), intent(in)            :: norb
    real(real64),   intent(in)            :: evals(norb)
    real(real64),   intent(out)           :: occs(norb)
    real(real64),   intent(in)            :: nelec
    real(real64),   intent(in)            :: theta
    real(real64),   intent(out), optional :: fermi_mu

    real(real64), parameter :: TOL_F = 1.0d-15
    real(real64), parameter :: TOL_MU = 1.0d-14
    real(real64), parameter :: MAX_EXP = 60.0d0
    integer(int32), parameter :: MAX_ITER = 100

    integer(int32) :: i, iter, khomo
    real(real64)   :: emin, emax, mu, mu_low, mu_high, f, df, arg, occ, dmu
    real(real64)   :: mu_new, th

    th = theta
    if (th < 1.0d-12) th = 1.0d-12

    if (nelec <= 0.0d0) then
      occs = 0.0d0
      if (present(fermi_mu)) fermi_mu = minval(evals) - 1.0d0
      return
    end if

    if (nelec >= real(norb, real64)) then
      occs = 1.0d0
      if (present(fermi_mu)) fermi_mu = maxval(evals) + 1.0d0
      return
    end if

    emin = minval(evals)
    emax = maxval(evals)

    ! Safe wide bracket ensuring f(mu_low) < 0 and f(mu_high) > 0
    mu_low  = emin - MAX_EXP * th
    mu_high = emax + MAX_EXP * th

    ! Initial guess near HOMO
    khomo = max(1, min(norb, nint(nelec)))
    mu = evals(khomo)
    if (mu <= mu_low .or. mu >= mu_high) mu = 0.5d0 * (mu_low + mu_high)

    do iter = 1, MAX_ITER
      f = -nelec
      df = 0.0d0
      do i = 1, norb
        arg = (evals(i) - mu) / th
        if (arg > MAX_EXP) then
          occ = 0.0d0
        else if (arg < -MAX_EXP) then
          occ = 1.0d0
        else
          occ = 1.0d0 / (1.0d0 + exp(arg))
        end if
        f = f + occ
        df = df + (occ * (1.0d0 - occ)) / th
      end do

      if (abs(f) < TOL_F .or. (mu_high - mu_low) < TOL_MU) exit

      if (f < 0.0d0) then
        mu_low = mu
      else
        mu_high = mu
      end if

      ! Hybrid Newton-Raphson with bisection fallback
      if (df > 1.0d-15) then
        dmu = -f / df
        mu_new = mu + dmu
      else
        mu_new = mu_low - 1.0d0
      end if

      if (mu_new <= mu_low .or. mu_new >= mu_high) then
        mu_new = 0.5d0 * (mu_low + mu_high)
      end if
      mu = mu_new
    end do

    ! Final occupations
    do i = 1, norb
      arg = (evals(i) - mu) / th
      if (arg > MAX_EXP) then
        occs(i) = 0.0d0
      else if (arg < -MAX_EXP) then
        occs(i) = 1.0d0
      else
        occs(i) = 1.0d0 / (1.0d0 + exp(arg))
      end if
    end do

    if (present(fermi_mu)) fermi_mu = mu
  end subroutine solve_fermi_level

end module ewmo