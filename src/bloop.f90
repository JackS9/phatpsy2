module bloop
  use iso_fortran_env, only: real64, int32
  use state
  use twocenter, only: rbeta, gensab, genvab
  use angular,   only: gendc, putdc, genplm
  use symmetry,  only: rotsab
  implicit none
  private
  public :: run_bloop
  !f2py private :: run_bloop

contains

  subroutine run_bloop(na, la, etaa, anorm, zpot2a, cpot2a, stovla, stpota, &
                       lva, mva, vexpa, vcoefa, gammaa, vfit2a, pkpota,     &
                       projec, evala, occa, eveca, zzrepa, rmaxa, qrealy)
    ! Active Atom A arrays passed from ALOOP
    integer(int32),  intent(in)    :: na(NSTO), la(NSTO), lva(NVTERM), mva(NVTERM)
    real(real64),    intent(in)    :: etaa(NSTO), anorm(NSTO), stovla(M2STO),          &
                                      stpota(M2STO, NVTERM), vexpa(NVTERM),            &
                                      vcoefa(NVTERM), gammaa(NORB, ISPIN),             &
                                      evala(NORB, ISPIN), occa(NORB, ISPIN),           &
                                      eveca(MSTO, NORB, ISPIN), rmaxa
    real(real64),    intent(inout) :: zpot2a(M2STO), cpot2a(M2STO), vfit2a(NVTERM),    &
                                      pkpota(M2STO, 6, ISPIN),                         &
                                      projec(MSTO, MSTO, ISPIN), zzrepa
    logical(kind=1), intent(in)    :: qrealy

    ! Scoped allocatable arrays for Environment Atom B (formerly CORE(IC(1)..IC(29)))
    integer(int32), allocatable :: nb(:), lb(:), lvb(:), mvb(:)
    real(real64),   allocatable :: etab(:), bnorm(:), zpot2b(:), cpot2b(:),            &
      vfit2b(:), pkpotb(:,:,:), evecb(:,:,:), evalb(:,:), occb(:,:), fockb(:,:),       &
      gammab(:,:), stovlb(:), stkinb(:), stpotb(:,:), vexpb(:), vcoefb(:),             &
      stvab(:,:), ovrlap(:,:,:), stovlp(:,:), ovlap(:,:,:), block_ab(:,:,:),           &
      scrat1(:), scrat2(:), scrat3(:,:), scrat4(:)

    real(real64), parameter :: DEGRAD = 57.29577951d0
    character(len=2) :: latomb
    logical(kind=1)  :: qflagb(8), qfixdb, qvirtb, qtrans_ab
    equivalence (qflagb(1), qfixdb), (qflagb(2), qvirtb)

    integer(int32) :: mstox, m2stox, nuczb, jdfat2, jdfatb, neqatb, ieqatb
    integer(int32) :: ijatom, jrec, nva, lv_val, nv_val, mv_val, lmv, nvb
    integer(int32) :: ia, jb, isp, nv, jstob, matomb, istob, llbp1, imlb, mlb
    integer(int32) :: iorbb, istoa, indxi, jstoa, ijstoa, iorba, isymb, ixyzb
    real(real64)   :: fractn, occmx, znuca, znucb, xb, yb, zb, totenb, zzrepb
    real(real64)   :: rmaxb, rab, rabmax, alphd, betad, vsuma, phi, phifac, va
    real(real64)   :: vsumb, vabeff, cibka, cibkb, zpotij, cpotij, popsum
    real(real64)   :: ediff, esum, fermia, fermib, ct, plm(66)

    ! Allocate B-atom and 2-center workspaces using safe max dimensions
    mstox  = max(MSTO, MSTOB)
    m2stox = mol_ws%index_arr(mstox) + mstox

    allocate(nb(NSTOB), lb(NSTOB), etab(NSTOB), bnorm(NSTOB))
    allocate(zpot2b(M2STOB), cpot2b(M2STOB), vfit2b(NVTRMB))
    allocate(pkpotb(M2STOB, 6, ISPIN), evecb(MSTOB, NORBB, ISPIN))
    allocate(evalb(NORBB, ISPIN), occb(NORBB, ISPIN), fockb(N2ORBB, ISPIN))
    allocate(gammab(NORBB, ISPIN), stovlb(M2STOB), stkinb(M2STOB))
    allocate(stpotb(M2STOB, NVTRMB), lvb(NVTRMB), mvb(NVTRMB))
    allocate(vexpb(NVTRMB), vcoefb(NVTRMB), stvab(M2STO, NVTRMB))
    allocate(ovrlap(LLMXP1, NSTO, NSTOB), stovlp(MSTO, MSTOB))
    allocate(ovlap(MSTO, NORBB, ISPIN), block_ab(NORB, NORBB, ISPIN))
    allocate(scrat1(m2stox), scrat2(M2STOB), scrat3(mstox, MSTOB), scrat4(LMXP1))

    fractn = 1.0d0 - VDAMP
    occmx  = real(3 - ISPIN, real64)

    call read_elem_basis(IUNITB - 10, latomb, nuczb, nb, lb, etab, bnorm)
    call read_elem_potentials(IUNITB - 10, stovlb, stkinb, stpotb, lvb, mvb, vexpb)

    znuca  = real(NUCZA, real64)
    znucb  = real(nuczb, real64)
    jdfat2 = merge(JDFATA, MDFATB, IELEMB == NELEM + 1)

    jdfat_loop: do jdfatb = JDFAT1, jdfat2
      if (MEQATB == 0) then
        call read_atom_record_ref(IUNIT3, qflagb, neqatb, xb, yb, zb, evecb, evalb, occb, fockb,   &
                                  gammab, cpot2b, vfit2b, vcoefb, pkpotb, zpot2b, totenb,  &
                                  zzrepb, rmaxb)
        MEQATB = neqatb
        if ((IELEMB == NELEM + 1) .and. (jdfatb == JDFATA)) MEQATB = IEQATA
      else
        call read_atom_record_equiv(IUNIT3, qflagb, isymb, ixyzb, xb, yb, zb, evecb, evalb, occb,    &
                                    fockb, gammab, cpot2b, vfit2b, vcoefb, pkpotb, zpot2b,   &
                                    totenb, zzrepb, rmaxb)
        MEQATB = MEQATB - 1
      end if

      eqat_loop: do
        ieqatb = neqatb - MEQATB
        JATOM  = mod(JATOM, NATOM) + 1
        if (JATOM == IATOM) cycle jdfat_loop

        ijatom    = mol_ws%index_arr(max(IATOM, JATOM)) + min(IATOM, JATOM)
        qtrans_ab = ((JATOM < IATOM) .and. (.not. qvirtb))
        call RBETA(mol_ws%coord, NATOM, IATOM, JATOM, rab, ALPHA_ANG, BETA_ANG)
        rabmax = rmaxa + rmaxb

        if (.not. (QFIRST .and. QSEPAT) .and. (rab <= rabmax)) then
          zzrepa = zzrepa + fractn * znuca * znucb / rab
        end if

        !-----------------------------------------------------------------
        ! Two-center overlap matrix STOVLP (compute or read from DAUnit1)
        !-----------------------------------------------------------------
        if (.not. QCMPTD) then
          call GENDC(mol_ws%d, LLMXP1 - 1)
          if (QDEBUG) then
            alphd = DEGRAD * ALPHA_ANG
            betad = DEGRAD * BETA_ANG
            write(IUDUMP, 1000) IATOM, JATOM, rab, alphd, betad
            write(IUDUMP, 2000)
            call PUTDC(mol_ws%d, L3MX, LLMXP1 - 1, IUDUMP)
          end if
        end if

        if (rab <= rabmax) then
          if (.not. QCMPTD .and. .not. qtrans_ab) then
            call GENSAB(ovrlap, na, la, etaa, anorm, NSTO, nb, lb, etab,      &
                        bnorm, NSTOB, rab, mol_ws%a, mol_ws%b, mol_ws%omegai, &
                        mol_ws%omegaj, scrat4, mol_ws%index_arr, mol_ws%ylmnrm)
            call ROTSAB(stovlp, ovrlap, la, NSTO, MSTO, lb, NSTOB, MSTOB, mol_ws%d)
            mol_ws%pair_stovlp(ijatom)%populated = .true.
            if (.not. allocated(mol_ws%pair_stovlp(ijatom)%mat)) then
              allocate(mol_ws%pair_stovlp(ijatom)%mat(MSTO, MSTOB))
            end if
            mol_ws%pair_stovlp(ijatom)%mat = stovlp
            mol_ws%krec(ijatom) = ijatom
            NBYTES = NBYTES + 8 * MSTO * MSTOB
            NLRECS = NLRECS + 1
            if (QDEBUG) then
              write(IUDUMP, 3000) IATOM, JATOM
              call PUTMAT(stovlp, MSTO, MSTOB, IUDUMP)
            end if
          else
            stovlp = mol_ws%pair_stovlp(ijatom)%mat
          end if

          !---------------------------------------------------------------
          ! Point-charge / multipole nuclear attraction fitting terms
          !---------------------------------------------------------------
          call GENPLM(plm, BETA_ANG, NYLM, LMAX)
          vsuma = 0.0d0
          do nva = 1, NVTERM
            lv_val = lva(nva)
            if (lv_val >= 0) then
              nv_val = lv_val; lv_val = 0; mv_val = 0
            else
              nv_val = -lv_val + 1; lv_val = -lv_val; mv_val = mva(nva)
            end if
            lmv    = mol_ws%index_arr(lv_val + 1) + abs(mv_val) + 1
            phi    = abs(mv_val) * ALPHA_ANG
            phifac = merge(sin(phi), cos(phi), mv_val < 0)
            va     = exp(-vexpa(nva) * rab) * rab**(nv_val - 1) * plm(lmv)    &
                   * mol_ws%ylmnrm(lmv) * phifac * 2.0d0 * ROOTPI
            vfit2a(nva) = vfit2a(nva) - fractn * znucb * va
            vsuma       = vsuma + vcoefa(nva) * va
          end do

          call RBETA(mol_ws%coord, NATOM, JATOM, IATOM, rab, ALPHA_ANG, BETA_ANG)
          call GENPLM(plm, BETA_ANG, NYLM, LMAX)
          vsumb = 0.0d0
          do nvb = 1, NVTRMB
            lv_val = lvb(nvb)
            if (lv_val >= 0) then
              nv_val = lv_val; lv_val = 0; mv_val = 0
            else
              nv_val = -lv_val + 1; lv_val = -lv_val; mv_val = mvb(nvb)
            end if
            lmv    = mol_ws%index_arr(lv_val + 1) + abs(mv_val) + 1
            phi    = abs(mv_val) * ALPHA_ANG
            phifac = merge(sin(phi), cos(phi), mv_val < 0)
            vsumb  = vsumb + vcoefb(nvb) * exp(-vexpb(nvb) * rab)             &
                   * rab**(nv_val - 1) * plm(lmv) * mol_ws%ylmnrm(lmv)        &
                   * phifac * 2.0d0 * ROOTPI
          end do

          vabeff = vsuma * vsumb * rab / (vcoefa(1) * vcoefb(1))
          if (QDEBUG) write(IUDUMP, 3500) IATOM, JATOM, vfit2a

          ia = IORBT + 1
          jb = mod(IORBTB, NORBT) + 1

          do isp = 1, ISPIN
            if (.not. qtrans_ab) then
              call DMPAB(stovlp, MSTO, MSTOB, MSTO, MSTOB, evecb(1,1,isp),    &
                         MSTOB, NORBB, MSTOB, NORBB, ovlap(1,1,isp), MSTO, NORBB)
            else
              call DMPATB(stovlp, MSTOB, MSTO, MSTOB, MSTO, evecb(1,1,isp),   &
                          MSTOB, NORBB, MSTOB, NORBB, ovlap(1,1,isp), MSTO, NORBB)
            end if
            call DMPATB(eveca(1,1,isp), MSTO, NORB, MSTO, NORB,               &
                        ovlap(1,1,isp), MSTO, NORBB, MSTO, NORBB,             &
                        block_ab(1,1,isp), NORB, NORBB)
            if (qtrans_ab) then
              call INSERT(block_ab(1,1,isp), mol_ws%ovlp(1,isp), ia, jb, NORB, NORBB)
            end if
          end do

          !---------------------------------------------------------------
          ! Two-center model potential integrals STVAB
          !---------------------------------------------------------------
          if (QCMPTD .or. MXCYCL > 0) then
            if (.not. QCMPTD) then
              call GENVAB(stvab, na, la, etaa, rab, lvb, mvb, vexpb,          &
                          mol_ws%omegai, mol_ws%omegaj, mol_ws%a, mol_ws%b,   &
                          scrat4, mol_ws%index_arr, anorm, mol_ws%ylmnrm,     &
                          mol_ws%d, mol_ws%cgc, qrealy)
              mol_ws%stvab_count = mol_ws%stvab_count + 1
              if (mol_ws%stvab_count > size(mol_ws%pair_stvab)) then
                block
                  type(PairMat), allocatable :: tmp_stvab(:)
                  integer(int32) :: old_sz
                  old_sz = size(mol_ws%pair_stvab)
                  allocate(tmp_stvab(old_sz * 2))
                  tmp_stvab(1:old_sz) = mol_ws%pair_stvab
                  call move_alloc(tmp_stvab, mol_ws%pair_stvab)
                end block
              end if
              mol_ws%pair_stvab(mol_ws%stvab_count)%populated = .true.
              if (.not. allocated(mol_ws%pair_stvab(mol_ws%stvab_count)%mat)) then
                allocate(mol_ws%pair_stvab(mol_ws%stvab_count)%mat(M2STO, NVTRMB))
              end if
              mol_ws%pair_stvab(mol_ws%stvab_count)%mat = stvab
              if (QDEBUG) then
                do nv = 1, NVTRMB
                  write(IUDUMP, 4000) IATOM, JATOM, nv
                  call PUTONE(stvab(1, nv), MSTO, IUDUMP)
                end do
              end if
            else
              mol_ws%stvab_pos = mol_ws%stvab_pos + 1
              stvab = mol_ws%pair_stvab(mol_ws%stvab_pos)%mat
            end if

            ! Punch multicenter projection tail if requested
            if (QPNCHK .and. (KATOM < 0) .and. (KORB /= 0) .and. &
                (abs(KORB) <= NORB) .and. (.not. qvirtb)) then
              jstob  = 0
              matomb = ieqatb * MDFATB + jdfatb
              do istob = 1, NSTOB
                llbp1 = 2 * lb(istob) + 1
                do imlb = 1, llbp1
                  mlb   = imlb - lb(istob) - 1
                  jstob = jstob + 1
                  cibka = 0.0d0
                  cibkb = 0.0d0
                  do iorbb = 1, NORBB
                    cibka = cibka - evecb(jstob, iorbb, KSP)   * block_ab(abs(KORB), iorbb, KSP)
                    if (jstob == 1) RENRMA = RENRMA - block_ab(abs(KORB), iorbb, KSP)**2
                    cibkb = cibkb - evecb(jstob, iorbb, ISPIN) * block_ab(abs(KORB), iorbb, ISPIN)
                    if (jstob == 1) RENRMB = RENRMB - block_ab(abs(KORB), iorbb, ISPIN)**2
                  end do
                  write(IP, 5000) nb(istob), lb(istob), mlb, etab(istob),     &
                                  bnorm(istob), latomb, matomb, xb, yb, zb, 1, 2
                  write(IP, 5100) 1.0d0, 1.0d0
                  write(IP, 5100) 1.0d0, 1.0d0
                  write(IP, 5100) cibka, cibkb
                end do
              end do
            end if

            !-------------------------------------------------------------
            ! Accumulate Interatomic Pseudopotential, Coulomb, Exchange,
            ! and Fermi-Dirac Charge Transfer
            !-------------------------------------------------------------
            if (.not. ((QSEPAT .and. QFIRST) .or. (fractn == 0.0d0))) then
              do isp = 1, ISPIN
                do nv = 1, 3
                  if (.not. qtrans_ab) then
                    call CSCT(stovlp, pkpotb(1, nv+3, isp), scrat3, scrat1, MSTOB, MSTO)
                  else
                    call CTSC(stovlp, pkpotb(1, nv+3, isp), scrat3, scrat1, MSTOB, MSTO)
                  end if
                  pkpota(:, nv, isp) = pkpota(:, nv, isp) + fractn * scrat1(1:M2STO)
                end do
              end do

              do istoa = 1, MSTO
                indxi = mol_ws%index_arr(istoa)
                do jstoa = 1, istoa
                  ijstoa = indxi + jstoa
                  zpotij = -znucb * stvab(ijstoa, 1)
                  zpot2a(ijstoa) = zpot2a(ijstoa) + fractn * zpotij
                  cpotij = dot_product(vcoefb(1:NVTRMB), stvab(ijstoa, 1:NVTRMB))
                  cpot2a(ijstoa) = cpot2a(ijstoa) + fractn * cpotij
                end do
              end do

              do isp = 1, ISPIN
                do iorba = 1, NORB
                  if (JATOM == mod(IATOM, NATOM) + 1) then
                    mol_ws%popul(IORBT + iorba, isp) = occa(iorba, isp)
                  end if
                  popsum = 0.0d0
                  do iorbb = 1, NORBB
                    ediff  = evalb(iorbb, isp) - evala(iorba, isp)
                    esum   = 0.5d0 * (evala(iorba, isp) + evalb(iorbb, isp))
                    fermia = 1.0d0 / (1.0d0 + exp((evala(iorba, isp) - esum) / THETA))
                    fermib = 1.0d0 / (1.0d0 + exp((evalb(iorbb, isp) - esum) / THETA))
                    ct     = (occa(iorba, isp) + occb(iorbb, isp)) * fermia - occa(iorba, isp)
                    if (ct > 0.0d0) then
                      ct = min(ct, occb(iorbb, isp), occmx - occa(iorba, isp))
                    else if (ct < 0.0d0) then
                      ct = -min(-ct, occa(iorba, isp), occmx - occb(iorbb, isp))
                    end if
                    popsum = popsum + ct * block_ab(iorba, iorbb, isp)**2
                  end do
                  mol_ws%popul(IORBT + iorba, isp) = mol_ws%popul(IORBT + iorba, isp) + popsum
                end do
              end do
            end if
          end if
        end if

        if (.not. qvirtb) IORBTB = mod(IORBTB + NORBB - 1, NORBT) + 1
        if (MEQATB == 0) exit eqat_loop

        call read_atom_record_equiv(IUNIT3, qflagb, isymb, ixyzb, xb, yb, zb, evecb, evalb, occb,    &
                                    fockb, gammab, cpot2b, vfit2b, vcoefb, pkpotb, zpot2b,   &
                                    totenb, zzrepb, rmaxb)
        MEQATB = MEQATB - 1
      end do eqat_loop
    end do jdfat_loop

    JDFAT1 = 1

    deallocate(nb, lb, etab, bnorm, zpot2b, cpot2b, vfit2b, pkpotb, evecb, &
               evalb, occb, fockb, gammab, stovlb, stkinb, stpotb, lvb, mvb, &
               vexpb, vcoefb, stvab, ovrlap, stovlp, ovlap, block_ab,        &
               scrat1, scrat2, scrat3, scrat4)

1000 format(//' R(',i2,'-',i2,') =',f9.4,', ALPHA =',f9.4,', BETA =',f9.4)
2000 format(//' D-COEFFICIENTS...')
3000 format(//' ',i2,'-',i2,' OVERLAP MATRIX...')
3500 format(//' ',i2,'-',i2,' NUCLEAR ATTRACTION TERMS...'//' ',5f10.5/(' ',5f10.5))
4000 format(//' ',i2,'-',i2,' MODEL POTENTIAL MATRIX, TERM',i2,'...')
5000 format(/,3i3,2f12.5,1x,a2,i3,3f10.5,i3,i2)
5100 format(f10.5,$)
  end subroutine run_bloop

end module bloop