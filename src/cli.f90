! src/cli.f90
program phatpsy_cli
  use iso_fortran_env, only: real64, int32
  use state
  use contrl
  implicit none

  integer(int32) :: isp
  logical        :: eof_reached, converged, done_scf

  call init_block_data_and_defaults()

  deck_loop: do
    call contrl_read_control_input(eof_reached)
    if (eof_reached) exit deck_loop
    if (QCHECK) stop

    call compute_derived_dimensions_and_allocate()
    call contrl_init_tables()
    call contrl_init_elements(eof_reached)
    if (eof_reached) exit deck_loop

    scf_cycle_loop: do
      if (.not. QATOM) then
        call contrl_step_aloop_and_ewmo(converged)
        if (converged) exit scf_cycle_loop
      end if
      call contrl_step_scf(done_scf)
      if (done_scf) exit scf_cycle_loop
    end do scf_cycle_loop

    do isp = 1, ISPIN
      if (QOPEN) write(IW, '(/1X,A8)') SPIN(isp)
      call OUTVEC(mol_ws%evecx(1,1,isp), mol_ws%evalx(1,isp), MSTOT, NORBT, IW)
    end do
    call TIMOUT(-7)
  end do deck_loop

  if (IW /= 6) close(IW)
  if (IUDUMP /= IW) close(IUDUMP)
  close(IP)

end program phatpsy_cli