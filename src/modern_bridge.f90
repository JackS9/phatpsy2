! src/modern_bridge.f90
! Routes legacy external subroutine calls to refactored Modern Fortran modules.

subroutine ONEINT(stovl, stkin, stpot, stpotx, lv, mv, vexp, &
                  n, l, eta, anorm, index_arr, cgc, qrealy)
  use iso_fortran_env, only: real64, int32
  use phatpsy_state_mod, only: MSTO, M2STO, NVTERM, NSTO, NCGC
  use phatpsy_integrals_mod, only: run_oneint
  implicit none
  real(real64),    intent(out) :: stovl(M2STO), stkin(M2STO), stpot(M2STO, NVTERM), stpotx(NVTERM)
  integer(int32),  intent(in)  :: lv(NVTERM), mv(NVTERM), n(NSTO), l(NSTO), index_arr(MSTO)
  real(real64),    intent(in)  :: vexp(NVTERM), eta(NSTO), anorm(NSTO), cgc(NCGC)
  logical(kind=1), intent(in)  :: qrealy

  call run_oneint(stovl, stkin, stpot, stpotx, lv, mv, vexp, &
                  n, l, eta, anorm, index_arr, cgc, qrealy)
end subroutine ONEINT

subroutine TWOINT(ntei, n, l, eta, s, w, index_arr, anorm, cgc, qrealy, teibuf)
  use iso_fortran_env, only: real64, int32
  use phatpsy_state_mod, only: MSTO, NSTO, NNMX, NNMXM1, NCGC, LENBUF
  use phatpsy_integrals_mod, only: run_twoint
  implicit none
  integer(int32),  intent(out)   :: ntei
  integer(int32),  intent(in)    :: n(NSTO), l(NSTO), index_arr(MSTO)
  real(real64),    intent(in)    :: eta(NSTO), anorm(NSTO), cgc(NCGC)
  real(real64),    intent(inout) :: s(NNMX), w(NNMXM1), teibuf(LENBUF)
  logical(kind=1), intent(in)    :: qrealy

  call run_twoint(ntei, n, l, eta, s, w, index_arr, anorm, cgc, qrealy, teibuf)
end subroutine TWOINT

subroutine FOKMAT(fc, fx, d, teibuf)
  use iso_fortran_env, only: real64
  use phatpsy_state_mod, only: M2STO, ISPIN, LENBUF
  use phatpsy_integrals_mod, only: run_fokmat
  implicit none
  real(real64), intent(out)   :: fc(M2STO, ISPIN), fx(M2STO, ISPIN)
  real(real64), intent(in)    :: d(M2STO, ISPIN)
  real(real64), intent(inout) :: teibuf(LENBUF)

  call run_fokmat(fc, fx, d, teibuf)
end subroutine FOKMAT

subroutine ANALYS(stkin, stpot, stcoul, stexch, cpot2, zpot2, vfit2,    &
                  vcoef, vexp, lv, pkpot, xpot2, stevec, eval, occ,     &
                  gamma, index_arr, toten, tke, zzrep, stovl, charge, tcoul0)
  use iso_fortran_env, only: real64, int32
  use phatpsy_state_mod, only: NATOM, MSTO, NORB, M2STO, NVTERM, ISPIN
  use phatpsy_analys_mod, only: run_analys
  implicit none
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

  call run_analys(stkin, stpot, stcoul, stexch, cpot2, zpot2, vfit2,    &
                  vcoef, vexp, lv, pkpot, xpot2, stevec, eval, occ,     &
                  gamma, index_arr, toten, tke, zzrep, stovl, charge, tcoul0)
end subroutine ANALYS

subroutine CHGAN(charge)
  use iso_fortran_env, only: real64
  use phatpsy_state_mod, only: NATOM
  use phatpsy_analys_mod, only: run_chgan
  implicit none
  real(real64), intent(in) :: charge(3, NATOM)

  call run_chgan(charge)
end subroutine CHGAN