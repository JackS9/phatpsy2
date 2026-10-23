module phatpsy_aloop_mod
  use iso_fortran_env, only: real64, int32
  use phatpsy_state_mod
  use phatpsy_bloop_mod, only: run_bloop
  implicit none
  private
  public :: run_aloop_element

contains

  subroutine process_active_atom_a(latoma, matoma, neqata, is_equiv_atom,     &
                                   xa, ya, za, totena, zzrepa, rmaxa,         &
                                   na, la, etaa, anorm, eveca, evala, occa,   &
                                   focka, gammaa, stovla, stpota, lva, mva,   &
                                   vexpa, zpot2a, cpot2a, pkpota, projec,     &
                                   vfit2a, vcoefa, scrat, qrealy)
    character(len=2), intent(in)    :: latoma
    integer(int32),   intent(in)    :: matoma, neqata
    logical,          intent(in)    :: is_equiv_atom
    real(real64),     intent(in)    :: xa, ya, za, totena, rmaxa
    real(real64),     intent(inout) :: zzrepa
    integer(int32),   intent(in)    :: na(NSTO), la(NSTO), lva(NVTERM), mva(NVTERM)
    real(real64),     intent(in)    :: etaa(NSTO), anorm(NSTO), stovla(M2STO),          &
                                       stpota(M2STO, NVTERM), vexpa(NVTERM),            &
                                       vcoefa(NVTERM), evala(NORB, ISPIN),              &
                                       occa(NORB, ISPIN), eveca(MSTO, NORB, ISPIN),     &
                                       focka(N2ORB, ISPIN), gammaa(NORB, ISPIN)
    real(real64),     intent(inout) :: zpot2a(M2STO), cpot2a(M2STO), vfit2a(NVTERM),    &
                                       pkpota(M2STO, 6, ISPIN),                         &
                                       projec(MSTO, MSTO, ISPIN),                       &
                                       scrat(MSTO, MSTO, 2)
    logical(kind=1),  intent(in)    :: qrealy

    integer(int32)  :: isp, korbta, iorba, kstota, jstoa, istoa, llap1, imla, mla
    integer(int32)  :: nv, nva, incr, nelemb, korb_abs, i, j
    real(real64)    :: vnorm
    logical(kind=1) :: qskip

    if (.not. QVIRTL) then
      do isp = 1, ISPIN
        call CTSC(eveca(1,1,isp), stovla, scrat(1,1,1), scrat(1,1,2), MSTO, NORB)
        call INSRTD(scrat(1,1,2), mol_ws%ovlp(1,isp),  IORBT + 1, NORB)
        call INSRTD(focka(1,isp), mol_ws%ewmat(1,isp), IORBT + 1, NORB)
        korbta = IORBT
        do iorba = 1, NORB
          korbta = korbta + 1
          mol_ws%evalx(korbta, isp) = evala(iorba, isp)
          mol_ws%occx(korbta, isp)  = occa(iorba, isp)
          kstota = JSTOT
          do jstoa = 1, MSTO
            kstota = kstota + 1
            mol_ws%evecx(kstota, korbta, isp) = eveca(jstoa, iorba, isp)
          end do
        end do
      end do

      ZNSUM = ZNSUM + mol_ws%charge(1, IATOM + 1)
      ELSUM = ELSUM + mol_ws%charge(2, IATOM + 1)
      ELDIF = ELDIF + mol_ws%charge(3, IATOM + 1)
    end if

    IATOM  = IATOM + 1
    QPNCHK = .false.
    if (.not. is_equiv_atom .and. QLAST .and. (KATOM /= 0) .and. (IATOM == 1)) then
      write(IP, 1200) NORBT, 'Phatpsy.car'
    end if
    if (QLAST .and. (IATOM == abs(KATOM))) QPNCHK = .true.
    if (QLAST .and. (abs(KATOM) > NATOM) .and. (.not. QVIRTL)) QPNCHK = .true.
    if ((KATOM < 0) .and. (abs(KORB) > NORB)) QPNCHK = .false.

    if (QPNCHK) then
      RENRMA = 1.0d0
      RENRMB = 1.0d0
      if (KORB /= 0) then
        jstoa    = 0
        korb_abs = abs(KORB)
        do istoa = 1, NSTO
          llap1 = 2 * la(istoa) + 1
          do imla = 1, llap1
            mla   = imla - la(istoa) - 1
            jstoa = jstoa + 1
            if (korb_abs <= NORB) then
              if ((KSP > 0) .and. (KSP <= ISPIN)) then
                write(IP, 1000) na(istoa), la(istoa), mla, etaa(istoa),       &
                                anorm(istoa), latoma, matoma, xa, ya, za, 1, 1
                ! FIXED BUG: Used korb_abs = abs(KORB) for OCCA and EVALA
                write(IP, 1100) occa(korb_abs, KSP)
                write(IP, 1100) evala(korb_abs, KSP)
                write(IP, 1100) eveca(jstoa, korb_abs, KSP)
              end if
              if ((KSP == 0) .or. (KSP > ISPIN)) then
                write(IP, 1000) na(istoa), la(istoa), mla, etaa(istoa),       &
                                anorm(istoa), latoma, matoma, xa, ya, za, 1, 2
                write(IP, 1100) occa(korb_abs, 1), occa(korb_abs, ISPIN)
                write(IP, 1100) evala(korb_abs, 1), evala(korb_abs, ISPIN)
                write(IP, 1100) eveca(jstoa, korb_abs, 1), eveca(jstoa, korb_abs, ISPIN)
              end if
            else
              write(IP, 1000) na(istoa), la(istoa), mla, etaa(istoa),         &
                              anorm(istoa), latoma, matoma, xa, ya, za, NORB, ISPIN
              write(IP, 1100) ((occa(i, j), i=1,NORB), j=1,ISPIN)
              write(IP, 1100) ((evala(i, j), i=1,NORB), j=1,ISPIN)
              write(IP, 1100) ((eveca(jstoa, i, j), i=1,NORB), j=1,ISPIN)
            end if
          end do
        end do
      else
        do nv = 1, NVTERM
          nva   = 0
          vnorm = 1.0d0
          write(IP, 1000) nva, lva(nv), mva(nv), vexpa(nv), vnorm,            &
                          latoma, matoma, xa, ya, za, 1, 1
          write(IP, 1100) 1.0d0
          write(IP, 1100) 1.0d0
          write(IP, 1100) vcoefa(nv)
        end do
      end if
    end if

    ! Apply external potential damping factor VDAMP
    cpot2a = VDAMP * cpot2a
    zpot2a = VDAMP * zpot2a
    vfit2a = VDAMP * vfit2a
    pkpota(:, 1:3, 1) = VDAMP * pkpota(:, 1:3, 1)
    if (QOPEN) pkpota(:, 1:3, 2) = VDAMP * pkpota(:, 1:3, 2)
    zzrepa = VDAMP * zzrepa

    if (.not. QVIRTL) then
      IORBTB = IORBT + NORB
      JATOM  = IATOM
      JDFAT1 = JDFATA
      if (.not. is_equiv_atom) then
        if (neqata == 0) JDFAT1 = mod(JDFATA, MDIFAT) + 1
        MEQATB = neqata
        incr   = merge(0, 1, (JDFATA == MDIFAT) .and. (neqata == 0))
      else
        if (IEQATA == neqata) JDFAT1 = mod(JDFATA, MDIFAT) + 1
        MEQATB = neqata - IEQATA
        incr   = merge(0, 1, (IEQATA == neqata) .and. (JDFATA == MDIFAT))
      end if
      qskip  = (incr == 1)
      nelemb = NELEM + incr
      MDFATB = MDIFAT
      NORBB  = NORB

      do IELEMB = 1, nelemb
        IUNITB = IUNIT(11 + mod(IELEM + IELEMB - 1 - incr, NELEM))
        call read_elem_dims(IUNITB - 10, NSTOB, MSTOB, NORBB, NVTRMB)
        M2STOB = mol_ws%index_arr(MSTOB) + MSTOB
        if (JATOM == NATOM) call rewind_atom_stream(IUNIT3)
        if (.not. qskip) call read_atom_stream_mdifat(IUNIT3, MDFATB)
        N2ORBB = mol_ws%index_arr(NORBB) + NORBB

        call run_bloop(na, la, etaa, anorm, zpot2a, cpot2a, stovla, stpota,   &
                       lva, mva, vexpa, vcoefa, gammaa, vfit2a, pkpota,       &
                       projec, evala, occa, eveca, zzrepa, rmaxa, qrealy)
        JDFAT1 = 1
        qskip  = .false.
      end do

      if (QPNCHK .and. (KATOM < 0) .and. (KORB /= 0)) write(IP, 2000) RENRMA, RENRMB
      if (QPNCHK .and. (KSP > 0) .and. (KSP <= ISPIN)) then
        write(IW, 3000) KORB, SPIN(KSP), latoma, matoma, IP
      end if
      IORBT = IORBT + NORB
      JSTOT = JSTOT + MSTO
    end if

1000 format(/,3i3,2f12.5,1x,a2,i3,3f10.5,i3,i2)
1100 format(f10.5,$)
1200 format(i4,a20,$)
2000 format(/64x,f10.5)
3000 format(//' ... THE',i3,1x,a8,' ORBITAL ON ATOM ',a2,i2,' HAS BEEN PUNCHED ON UNIT',i3,' ...')
  end subroutine process_active_atom_a

  subroutine run_aloop_element(qrealy)
    logical(kind=1), intent(in) :: qrealy

    ! Scoped allocatable workspaces for Atom A (formerly CORE(IC(1)..IC(22)))
    integer(int32), allocatable :: na(:), la(:), lva(:), mva(:)
    real(real64),   allocatable :: etaa(:), anorm(:), eveca(:,:,:), evala(:,:), &
      occa(:,:), focka(:,:), gammaa(:,:), stovla(:), stkina(:), stpota(:,:),    &
      vexpa(:), zpot2a(:), cpot2a(:), pkpota(:,:,:), projec(:,:,:),             &
      vfit2a(:), vcoefa(:), scrat(:,:,:)

    character(len=2) :: latoma
    integer(int32)   :: neqata, isyma, ixyza, matoma
    real(real64)     :: xa, ya, za, totena, zzrepa, rmaxa

    allocate(na(NSTO), la(NSTO), etaa(NSTO), anorm(NSTO))
    allocate(eveca(MSTO, NORB, ISPIN), evala(NORB, ISPIN), occa(NORB, ISPIN))
    allocate(focka(N2ORB, ISPIN), gammaa(NORB, ISPIN))
    allocate(stovla(M2STO), stkina(M2STO), stpota(M2STO, NVTERM))
    allocate(lva(NVTERM), mva(NVTERM), vexpa(NVTERM))
    allocate(zpot2a(M2STO), cpot2a(M2STO), pkpota(M2STO, 6, ISPIN))
    allocate(projec(MSTO, MSTO, ISPIN), vfit2a(NVTERM), vcoefa(NVTERM))
    allocate(scrat(MSTO, MSTO, 2))

    call read_elem_basis(IELEM, latoma, NUCZA, na, la, etaa, anorm)
    call read_elem_potentials(IELEM, stovla, stkina, stpota, lva, mva, vexpa)

    do JDFATA = 1, MDIFAT
      call read_atom_record_ref(IUNIT3, QFLAGS, neqata, xa, ya, za, eveca, evala, occa, focka,     &
                                gammaa, cpot2a, vfit2a, vcoefa, pkpota, zpot2a, totena,    &
                                zzrepa, rmaxa)

      call process_active_atom_a(latoma, JDFATA, neqata, .false.,             &
                                 xa, ya, za, totena, zzrepa, rmaxa,           &
                                 na, la, etaa, anorm, eveca, evala, occa,     &
                                 focka, gammaa, stovla, stpota, lva, mva,     &
                                 vexpa, zpot2a, cpot2a, pkpota, projec,       &
                                 vfit2a, vcoefa, scrat, qrealy)

      call write_atom_record_ref(IUNIT4, QFLAGS, neqata, xa, ya, za, eveca, evala, occa, focka,    &
                                 gammaa, cpot2a, vfit2a, vcoefa, pkpota, zpot2a, totena,   &
                                 zzrepa, rmaxa)

      if (neqata > 0) then
        do IEQATA = 1, neqata
          call read_atom_record_equiv(IUNIT3, QFLAGS, isyma, ixyza, xa, ya, za, eveca, evala, occa,  &
                                      focka, gammaa, cpot2a, vfit2a, vcoefa, pkpota, zpot2a, &
                                      totena, zzrepa, rmaxa)
          matoma = IEQATA * MDIFAT + JDFATA

          call process_active_atom_a(latoma, matoma, neqata, .true.,          &
                                     xa, ya, za, totena, zzrepa, rmaxa,       &
                                     na, la, etaa, anorm, eveca, evala, occa, &
                                     focka, gammaa, stovla, stpota, lva, mva, &
                                     vexpa, zpot2a, cpot2a, pkpota, projec,   &
                                     vfit2a, vcoefa, scrat, qrealy)

          call write_atom_record_equiv(IUNIT4, QFLAGS, isyma, ixyza, xa, ya, za, eveca, evala, occa, &
                                       focka, gammaa, cpot2a, vfit2a, vcoefa, pkpota,        &
                                       zpot2a, totena, zzrepa, rmaxa)
        end do
      end if
    end do

    deallocate(na, la, etaa, anorm, eveca, evala, occa, focka, gammaa, &
               stovla, stkina, stpota, lva, mva, vexpa, zpot2a, cpot2a, &
               pkpota, projec, vfit2a, vcoefa, scrat)
  end subroutine run_aloop_element

end module phatpsy_aloop_mod