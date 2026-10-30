!=======================================================================
! PHATPSY 2.0: Modern Fortran 2018 Core
! Module: diagnostics
! Description: Formatted matrix and vector output, 3D coordinate mapping,
!              potential plotting, timing, and diagnostic error handling.
!
! Modernizes legacy routines:
!   - output.f  (formatted rectangular matrix output)
!   - outpak.f  (formatted packed symmetric matrix output)
!   - outvec.f  (formatted eigenvector/eigenvalue output, with putmat)
!   - putone.f  (formatted 10-column lower-triangular packed matrix output)
!   - twoout.f  (formatted two-particle operator matrix output)
!   - timout.f  (CPU timing and elapsed execution reporting, with getset)
!   - xyzmap.f  (perspective printer plots of atomic coordinates)
!   - mapndx.f  (coordinate grid index calculation)
!   - plotv.f   (radial potential printer plot)
!   - punchv.f  (punches out atomic eigenvectors and namelists)
!   - bomb.f    (diagnostic error handler and state dump, with warn, setbom, dump)
!   - noble.f   (noble gas core orbital count)
!   - err243.f  (0^0 floating point fixup)
!   - arrmap.f  (array memory mapping diagnostic)
!   - esca.f    (ESCA photoelectron intensity evaluation)
!=======================================================================

module diagnostics
  use iso_fortran_env, only: real64, int32
  use state,           only: QCORE, IW
  use density,         only: depth
  implicit none
  private

  public :: output
  public :: outpak
  public :: outvec
  public :: putmat
  public :: putone
  public :: twoout
  public :: timout
  public :: getset
  public :: xyzmap
  public :: mapndx
  public :: plotv
  public :: punchv
  public :: bomb
  public :: warn
  public :: setbom
  public :: dump
  public :: noble
  public :: err243
  public :: arrmap
  public :: esca

  real(real64), save :: g_begin_time = 0.0d0
  real(real64), save :: g_start_time = 0.0d0

contains

  !---------------------------------------------------------------------
  ! OUTPUT: Prints a real*8 rectangular matrix with numbered rows & cols
  !---------------------------------------------------------------------
  subroutine output(matrix, rowlow, rowhi, collow, colhi, rowdim, coldim, nctl)
    integer(int32), intent(in) :: rowlow, rowhi, collow, colhi, rowdim, coldim, nctl
    real(real64),   intent(in) :: matrix(rowdim, coldim)

    character(len=1), parameter :: ASA(3) = [' ', '0', '-']
    character(len=6), parameter :: COLUMN = 'COLUMN'
    integer(int32),   parameter :: KCOL   = 5
    character(len=1) :: ctl
    integer(int32)   :: begin_col, last_col, k, i
    logical          :: qallzero

    ctl = ' '
    if (nctl >= 1 .and. nctl <= 3) ctl = ASA(nctl)
    if (rowhi < rowlow .or. colhi < collow) return

    do begin_col = collow, colhi, KCOL
      last_col = min(colhi, begin_col + KCOL - 1)
      write(6, 1000) (COLUMN, i, i = begin_col, last_col)
      do k = rowlow, rowhi
        qallzero = .true.
        do i = begin_col, last_col
          if (matrix(k, i) /= 0.0d0) then
            qallzero = .false.
            exit
          end if
        end do
        if (.not. qallzero) then
          write(6, 2000) ctl, k, (matrix(k, i), i = begin_col, last_col)
        end if
      end do
    end do

1000 format(/' ', 8X, 5(A6, I4, 14X))
2000 format(A1, 'ROW ', I4, 2X, 1P5D24.15)
  end subroutine output

  !---------------------------------------------------------------------
  ! OUTPAK: Prints row-packed lower-triangular symmetric matrix
  !---------------------------------------------------------------------
  subroutine outpak(matrix, nrow, nctl)
    integer(int32), intent(in) :: nrow, nctl
    real(real64),   intent(in) :: matrix(*)

    character(len=1), parameter :: ASA(3) = [' ', '0', '-']
    character(len=6), parameter :: COLUMN = 'COLUMN'
    integer(int32),   parameter :: KCOL   = 6
    character(len=1) :: ctl
    integer(int32)   :: begin_col, last_col, j, i, k, ii, ij
    real(real64)     :: row_buf(6)
    logical          :: qallzero

    ctl = ' '
    if (nctl >= 1 .and. nctl <= 3) ctl = ASA(nctl)

    do begin_col = 1, nrow, KCOL
      last_col = min(nrow, begin_col + KCOL - 1)
      write(6, 1000) (COLUMN, i, i = begin_col, last_col)
      do j = begin_col, nrow
        ii = (j * (j - 1)) / 2
        qallzero = .true.
        do i = begin_col, min(j, last_col)
          k = i - begin_col + 1
          ij = ii + i
          row_buf(k) = matrix(ij)
          if (row_buf(k) /= 0.0d0) qallzero = .false.
        end do
        if (.not. qallzero) then
          write(6, 2000) ctl, j, (row_buf(k), k = 1, min(j, last_col) - begin_col + 1)
        end if
      end do
    end do

1000 format(/' ', 8X, 6(A6, I3, 11X))
2000 format(A1, 'ROW ', I4, 2X, 6F20.14)
  end subroutine outpak

  !---------------------------------------------------------------------
  ! OUTVEC: Writes out eigenvectors and eigenvalues in column format
  !---------------------------------------------------------------------
  subroutine outvec(evec, eval, nbas, nvec, iw)
    integer(int32), intent(in) :: nbas, nvec, iw
    real(real64),   intent(in) :: evec(nbas, nvec), eval(nvec)

    integer(int32),   parameter :: MAXCOL = 6
    real(real64),     parameter :: ZERO   = 0.0d0
    real(real64),     parameter :: TOLER  = 5.0d-5
    character(len=8), parameter :: UNSCOR = '________'

    integer(int32) :: ibegin, last_col, i, j, ncol
    real(real64)   :: row_buf(10), val
    logical        :: qdelet

    last_col = 0
    do ibegin = 1, nvec, MAXCOL
      last_col = min(last_col + MAXCOL, nvec)
      write(iw, 1000) ' ', (i, i = ibegin, last_col)
      write(iw, 2000) (eval(i), i = ibegin, last_col)
      write(iw, 3000) (UNSCOR, i = ibegin, last_col)
      write(iw, 4000)

      do j = 1, nbas
        ncol = 0
        qdelet = .true.
        do i = ibegin, last_col
          ncol = ncol + 1
          val = evec(j, i)
          if (abs(val) <= TOLER) val = ZERO
          if (val /= ZERO) qdelet = .false.
          row_buf(ncol) = val
        end do
        if (.not. qdelet) then
          write(iw, 4000) j, (row_buf(i), i = 1, ncol)
        end if
      end do
    end do

1000 format(A1, 9X, 6(I4, 8X))
2000 format(' ', 4X, 6F12.4)
3000 format(' ', 8X, 6(A8, 4X))
4000 format(' ', I4, 6F12.6)
  end subroutine outvec

  !---------------------------------------------------------------------
  ! PUTMAT: Outputs a general matrix in column packed form without eigenvalues
  !---------------------------------------------------------------------
  subroutine putmat(evec, nbas, nvec, iw)
    integer(int32), intent(in) :: nbas, nvec, iw
    real(real64),   intent(in) :: evec(nbas, nvec)

    integer(int32), parameter :: MAXCOL = 6
    real(real64),   parameter :: ZERO   = 0.0d0
    real(real64),   parameter :: TOLER  = 5.0d-5
    integer(int32) :: ibegin, last_col, i, j, ncol
    real(real64)   :: row_buf(10), val
    logical        :: qdelet

    last_col = 0
    do ibegin = 1, nvec, MAXCOL
      last_col = min(last_col + MAXCOL, nvec)
      write(iw, 1000) ' ', (i, i = ibegin, last_col)
      write(iw, 4000)

      do j = 1, nbas
        ncol = 0
        qdelet = .true.
        do i = ibegin, last_col
          ncol = ncol + 1
          val = evec(j, i)
          if (abs(val) <= TOLER) val = ZERO
          if (val /= ZERO) qdelet = .false.
          row_buf(ncol) = val
        end do
        if (.not. qdelet) then
          write(iw, 4000) j, (row_buf(i), i = 1, ncol)
        end if
      end do
    end do

1000 format(A1, 9X, 6(I4, 8X))
4000 format(' ', I4, 6F12.6)
  end subroutine putmat

  !---------------------------------------------------------------------
  ! PUTONE: Writes out a symmetry packed matrix in lower-triangular form
  !---------------------------------------------------------------------
  subroutine putone(onemat, nbas, iw)
    integer(int32), intent(in) :: nbas, iw
    real(real64),   intent(in) :: onemat(*)

    real(real64),   parameter :: ZERO   = 0.0d0
    real(real64),   parameter :: TOLER  = 5.0d-5
    integer(int32), parameter :: MAXCOL = 10
    integer(int32) :: ibegin, last_col, ncol, iend, j, k, ij
    real(real64)   :: row_buf(10), rowk

    last_col = 0
    do ibegin = 1, nbas, MAXCOL
      last_col = min(last_col + MAXCOL, nbas)
      ncol = last_col - ibegin + 1
      ij = (ibegin * (ibegin - 1)) / 2
      write(iw, 1000) ' ', (k, k = ibegin, last_col)
      write(iw, 2000)

      iend = 0
      do j = ibegin, nbas
        iend = min(iend + 1, ncol)
        ij = ij + j - 1
        do k = 1, iend
          rowk = onemat(ij + k)
          if (abs(rowk) < TOLER) rowk = ZERO
          row_buf(k) = rowk
        end do
        write(iw, 2000) j, (row_buf(k), k = 1, iend)
      end do
    end do

1000 format(A1, I9, 9I12)
2000 format(' ', I4, 10F12.6)
  end subroutine putone

  !---------------------------------------------------------------------
  ! TWOOUT: Formatted printout for two-particle operator matrix
  !---------------------------------------------------------------------
  subroutine twoout(matrix, nsize, index_arr, nctl)
    integer(int32), intent(in) :: nsize, nctl
    integer(int32), intent(in) :: index_arr(*)
    real(real64),   intent(in) :: matrix(*)

    character(len=1), parameter :: ASA(3) = [' ', '0', '-']
    integer(int32),   parameter :: KCOL   = 6
    character(len=1) :: ctl
    integer(int32)   :: n2, begin_col, last_col, j, i, k, ii, ij
    real(real64)     :: row_buf(6)
    logical          :: qallzero

    ctl = ' '
    if (nctl >= 1 .and. nctl <= 3) ctl = ASA(nctl)
    n2 = (nsize * (nsize + 1)) / 2

    do begin_col = 1, n2, KCOL
      last_col = min(n2, begin_col + KCOL - 1)
      write(6, 1000) (i, i = begin_col, last_col)
      do j = begin_col, n2
        ii = index_arr(j)
        qallzero = .true.
        do i = begin_col, min(j, last_col)
          k = i - begin_col + 1
          ij = ii + i
          row_buf(k) = matrix(ij)
          if (row_buf(k) /= 0.0d0) qallzero = .false.
        end do
        if (.not. qallzero) then
          write(6, 2000) ctl, j, (row_buf(k), k = 1, min(j, last_col) - begin_col + 1)
        end if
      end do
    end do

1000 format(/' ', 8X, 6(I5, 15X))
2000 format(A1, I5, 2X, 6F20.14)
  end subroutine twoout

  !---------------------------------------------------------------------
  ! GETSET: Initializes execution timing
  !---------------------------------------------------------------------
  subroutine getset()
    call cpu_time(g_begin_time)
    g_start_time = g_begin_time
  end subroutine getset

  !---------------------------------------------------------------------
  ! TIMOUT: Handles timing measurements and step-completion logging
  !---------------------------------------------------------------------
  subroutine timout(msgtyp)
    integer(int32), intent(in) :: msgtyp

    character(len=32), parameter :: TIMMSG(8) = [ &
      'PROCESS INPUT ...               ', &
      'RECONSTRUCT 2-CENTER INTEGRALS .', &
      'COMPUTE 1-CENTER INTEGRALS ...  ', &
      'COMPLETE SCF FOR THIS ATOM ...  ', &
      'PERFORM EWMO CALCULATION ...    ', &
      'COMPLETE THIS CALCULATION ...   ', &
      'COMPUTE 2-CENTER INTEGRALS ...  ', &
      'COMPLETE THE ANALYSIS ...       '  &
    ]
    real(real64) :: t_now, finish, secs
    integer(int32) :: msg

    call cpu_time(t_now)
    finish = t_now - g_begin_time

    if (msgtyp < 0) then
      secs = finish
      if (msgtyp < -1) then
        msg = abs(msgtyp)
        if (msg >= 1 .and. msg <= 8) then
          write(IW, 2000) secs, trim(TIMMSG(msg))
        else
          write(IW, 3000) secs
        end if
      else
        write(IW, 1000) secs
      end if
    else if (msgtyp == 0) then
      g_start_time = t_now
    else
      secs = t_now - g_start_time
      if (msgtyp >= 1 .and. msgtyp <= 8) then
        write(IW, 2000) secs, trim(TIMMSG(msgtyp))
      else
        write(IW, 3000) secs
      end if
      g_start_time = t_now
    end if

1000 format(' ... ELAPSED TIME IS', F6.2, ' SECONDS ...')
2000 format(/' ... IT TOOK ', F7.2, ' SECONDS TO ', A)
3000 format(/' ... IT TOOK ', F7.2, ' SECONDS TO COMPLETE THIS STEP ...')
  end subroutine timout

  !---------------------------------------------------------------------
  ! MAPNDX: Returns map array index for a projected cartesian coordinate
  !---------------------------------------------------------------------
  integer(int32) function mapndx(x, mapdim) result(idx)
    real(real64),   intent(in) :: x
    integer(int32), intent(in) :: mapdim

    real(real64), parameter :: HALF = 0.5d0
    real(real64) :: xmod
    integer(int32) :: ixmod

    xmod = x
    if (x < 0.0d0) then
      xmod = xmod - HALF
    else
      xmod = xmod + HALF
    end if
    ixmod = int(xmod, kind=int32)
    idx = ixmod + (mapdim + 1) / 2
  end function mapndx

  !---------------------------------------------------------------------
  ! XYZMAP: Generates perspective maps of atomic coordinates in planes
  !---------------------------------------------------------------------
  subroutine xyzmap(coord, latom, natom, iw, range_val, x0, y0, z0)
    integer(int32),   intent(in) :: natom, iw
    real(real64),     intent(in) :: coord(3, natom)
    character(len=2), intent(in) :: latom(natom)
    real(real64),     intent(in) :: range_val, x0, y0, z0

    integer(int32),   parameter  :: MAPDIM = 21
    character(len=2) :: map_arr(MAPDIM, 3, MAPDIM)
    integer(int32)   :: irange, iatom, ix, iy, iz, jx, jy, jz, i, j, k
    real(real64)     :: scale, x, y, z

    irange = (MAPDIM - 1) / 2
    scale = real(irange, kind=real64) / range_val
    map_arr = '  '

    do iatom = 1, natom
      x = (coord(1, iatom) - x0) * scale
      y = (coord(2, iatom) - y0) * scale
      z = (coord(3, iatom) - z0) * scale

      ix = mapndx(x, MAPDIM)
      iy = mapndx(y, MAPDIM)
      iz = mapndx(z, MAPDIM)

      call depth(z, y, x, jz, jy, irange)
      if (iy + jy >= 1 .and. iy + jy <= MAPDIM .and. &
          iz + jz >= 1 .and. iz + jz <= MAPDIM) then
        map_arr(iy + jy, 1, iz + jz) = latom(iatom)
      end if

      call depth(x, z, y, jx, jz, irange)
      if (iz + jz >= 1 .and. iz + jz <= MAPDIM .and. &
          ix + jx >= 1 .and. ix + jx <= MAPDIM) then
        map_arr(iz + jz, 2, ix + jx) = latom(iatom)
      end if

      call depth(y, x, z, jy, jx, irange)
      if (ix + jx >= 1 .and. ix + jx <= MAPDIM .and. &
          iy + jy >= 1 .and. iy + jy <= MAPDIM) then
        map_arr(ix + jx, 3, iy + jy) = latom(iatom)
      end if
    end do

    write(iw, '(A)') ' --- ATOMIC CARTESIAN PERSPECTIVE MAPS ---'
    do j = 1, 3
      write(iw, 1000) j
      do k = MAPDIM, 1, -1
        write(iw, 2000) (map_arr(i, j, k), i = 1, MAPDIM)
      end do
    end do

1000 format(/' PLANE ', I2, ':')
2000 format(' ', 21A2)
  end subroutine xyzmap

  !---------------------------------------------------------------------
  ! PLOTV: Printer plot of radial potential
  !---------------------------------------------------------------------
  subroutine plotv(lv, mv, vexp, vcoef, nvterm, iw)
    integer(int32), intent(in) :: nvterm, iw
    integer(int32), intent(in) :: lv(nvterm), mv(nvterm)
    real(real64),   intent(in) :: vexp(nvterm), vcoef(nvterm)

    character(len=1) :: line(50)
    real(real64)     :: scale_val, r, rr, vn, ve, vt, v, sgn
    integer(int32)   :: i, j, k, nv

    scale_val = -5.0d0 / vcoef(1)
    r = -4.0d0
    write(iw, 1000)

    do i = 1, 41
      vn = 0.0d0
      ve = 0.0d0
      rr = abs(r)
      line = ' '
      if (rr < 0.1d0) then
        do j = 1, 50
          line(j) = '-'
          if (mod(j, 5) == 0) line(j) = '+'
        end do
      end if
      line(25) = '|'
      if (rr < 0.1d0) line(25) = '+'

      if (rr >= 0.1d0) then
        vn = scale_val * vcoef(1) / rr
        k = int(vn + 0.5d0, kind=int32)
        if (vn < 0.0d0) k = int(vn - 0.5d0, kind=int32)
        k = max(-24, min(25, k))
        line(25 + k) = '+'

        do nv = 2, nvterm
          if (lv(nv) >= 0) then
            v = vcoef(nv) * (rr**(lv(nv) - 1)) * exp(-vexp(nv) * rr)
          else
            if (mv(nv) == 0) then
              sgn = real((-1)**mv(nv), kind=real64) * r / rr
              v = sgn * vcoef(nv) * (rr**(-lv(nv))) * exp(-vexp(nv) * rr)
            else
              v = 0.0d0
            end if
          end if
          v = scale_val * v
          k = int(v + 0.5d0, kind=int32)
          if (v < 0.0d0) k = int(v - 0.5d0, kind=int32)
          k = max(-24, min(25, k))
          line(25 + k) = '.'
          ve = ve + v
        end do

        vt = vn + ve
        k = int(ve + 0.5d0, kind=int32)
        if (ve < 0.0d0) k = int(ve - 0.5d0, kind=int32)
        k = max(-24, min(25, k))
        line(25 + k) = '-'

        k = int(vt + 0.5d0, kind=int32)
        if (vt < 0.0d0) k = int(vt - 0.5d0, kind=int32)
        k = max(-24, min(25, k))
        line(25 + k) = '*'
      end if

      write(iw, 2000) r, line
      r = r + 0.2d0
    end do

1000 format(/' POTENTIAL PLOT:'/)
2000 format(F6.1, ' ', 50A1)
  end subroutine plotv

  !---------------------------------------------------------------------
  ! PUNCHV: Punches out atomic eigenvectors and namelists to unit IP
  !---------------------------------------------------------------------
  subroutine punchv(stevec, eval, occ, l, nsto, msto, norb, ispin, ip)
    integer(int32), intent(in) :: nsto, msto, norb, ispin, ip
    real(real64),   intent(in) :: stevec(msto, norb, ispin)
    real(real64),   intent(in) :: eval(norb, ispin), occ(norb, ispin)
    integer(int32), intent(in) :: l(nsto)

    integer(int32) :: iorb, isto, li, llip1, mli, ml, jsto
    real(real64)   :: eigval(2), occno(2), coef(2)

    do iorb = 1, norb
      eigval(1) = eval(iorb, 1)
      eigval(2) = eval(iorb, ispin)
      occno(1)  = occ(iorb, 1)
      occno(2)  = occ(iorb, ispin)
      write(ip, 1000) msto, eigval(1), eigval(2), occno(1), occno(2)

      jsto = 0
      do isto = 1, nsto
        li = l(isto)
        llip1 = 2 * li + 1
        do mli = 1, llip1
          ml = mli - li - 1
          jsto = jsto + 1
          coef(1) = stevec(jsto, iorb, 1)
          coef(2) = stevec(jsto, iorb, ispin)
          write(ip, 2000) isto, ml, coef(1), coef(2)
        end do
      end do
    end do

1000 format(' &ATORB NCOEF=', I4, ', EIGVAL=', 2F12.6, ', OCCNO=', 2F10.4, ' &END')
2000 format(2I3, 2F10.6)
  end subroutine punchv

  !---------------------------------------------------------------------
  ! BOMB: Diagnostic error handler and program exit
  !---------------------------------------------------------------------
  subroutine bomb(ierror)
    use iso_fortran_env, only: error_unit
    integer(int32), intent(in) :: ierror

    write(error_unit, 1000) ierror
    flush(error_unit)
    write(IW, 1000) ierror
    flush(IW)
    call dump(0)
    error stop 'PHATPSY TERMINATED VIA BOMB'

1000 format(/' *** FATAL ERROR ENCOUNTERED (ERROR CODE =', I4, ') ***')
  end subroutine bomb

  !---------------------------------------------------------------------
  ! WARN: Prints non-fatal warning message and resets flag
  !---------------------------------------------------------------------
  subroutine warn(ierror, qreset)
    integer(int32), intent(in)  :: ierror
    logical,        intent(out) :: qreset

    write(IW, 1000) ierror
    qreset = .true.

1000 format(/' *** WARNING ENCOUNTERED (WARNING CODE =', I4, ') ***')
  end subroutine warn

  !---------------------------------------------------------------------
  ! SETBOM: Error handler table reset (portable no-op)
  !---------------------------------------------------------------------
  subroutine setbom()
    ! Portable no-op in Fortran 2018
  end subroutine setbom

  !---------------------------------------------------------------------
  ! DUMP: State dump diagnostic
  !---------------------------------------------------------------------
  subroutine dump(nrtns)
    integer(int32), intent(in) :: nrtns
    if (nrtns > 0) then
      write(IW, '(A)') ' --- DIAGNOSTIC STATE DUMP ---'
    end if
  end subroutine dump

  !---------------------------------------------------------------------
  ! NOBLE: Returns the number of core orbitals for noble gas core
  !---------------------------------------------------------------------
  integer(int32) function noble(nucz) result(ncore)
    integer(int32), intent(in) :: nucz

    ncore = 0
    if (QCORE) return
    if (nucz <= 2) return
    ncore = 1
    if (nucz <= 10) return
    ncore = 5
    if (nucz <= 18) return
    ncore = 9
    if (nucz <= 36) return
    ncore = 18
    if (nucz <= 54) return
    ncore = 27
    if (nucz <= 86) return
    ncore = 43
  end function noble

  !---------------------------------------------------------------------
  ! ERR243: 0.0^0 floating point fixup
  !---------------------------------------------------------------------
  subroutine err243(iertcd, ierno, d, i)
    integer(int32), intent(out)   :: iertcd, ierno, i
    real(real64),   intent(inout) :: d

    iertcd = 1
    ierno  = 243
    d      = 1.0d0
    i      = 0
  end subroutine err243

  !---------------------------------------------------------------------
  ! ARRMAP: Array memory mapping diagnostic (portable no-op)
  !---------------------------------------------------------------------
  subroutine arrmap(nr)
    integer(int32), intent(in) :: nr
    ! In modern in-memory architecture, scratch disks and fixed core
    ! buffers are eliminated; keep as portable no-op.
  end subroutine arrmap

  !---------------------------------------------------------------------
  ! ESCA: Photoelectron intensity estimation under plane wave approximation
  !---------------------------------------------------------------------
  subroutine esca(evec, eval, n, l, eta, norb, msto, nsto, iw)
    integer(int32), intent(in) :: norb, msto, nsto, iw
    real(real64),   intent(in) :: evec(msto, norb), eval(norb)
    integer(int32), intent(in) :: n(nsto), l(nsto)
    real(real64),   intent(in) :: eta(nsto)

    real(real64), parameter :: ZERO   = 0.0d0
    real(real64), parameter :: TWO    = 2.0d0
    real(real64), parameter :: THREE  = 3.0d0
    real(real64), parameter :: FOUR   = 4.0d0
    real(real64), parameter :: FIVE   = 5.0d0
    real(real64), parameter :: SIX    = 6.0d0
    real(real64), parameter :: SEVEN  = 7.0d0
    real(real64), parameter :: EIGHT  = 8.0d0
    real(real64), parameter :: ESOURC = 46.0d0
    real(real64), parameter :: EVPERH = 27.2107d0

    integer(int32) :: korb, isto, ni, li, llip1, nli, imsto, mli
    real(real64)   :: evsrc, amp, ekappa, etai, ampi

    evsrc = ESOURC * EVPERH
    write(iw, 1000) evsrc

    do korb = 1, norb
      amp = ZERO
      if (eval(korb) >= -ESOURC) then
        ekappa = sqrt(TWO * (eval(korb) + ESOURC))
        imsto = 0
        do isto = 1, nsto
          ni = n(isto)
          li = l(isto)
          llip1 = 2 * li + 1
          nli = (ni * (ni - 1)) / 2 + li + 1
          if (nli <= 10) then
            etai = eta(isto)
            if (nli == 1) then
              ampi = (EIGHT / sqrt(TWO)) * (etai**(FIVE / TWO)) * ekappa / &
                     ((etai*etai + ekappa*ekappa)**2)
            else if (nli == 2) then
              ampi = (EIGHT / sqrt(SIX)) * (etai**(FIVE / TWO)) * ekappa * &
                     (THREE * etai * etai - ekappa * ekappa) / &
                     ((etai*etai + ekappa*ekappa)**3)
            else if (nli == 3) then
              ampi = (FOUR * EIGHT / sqrt(SIX)) * (etai**(SEVEN / TWO)) * (ekappa**2) / &
                     ((etai*etai + ekappa*ekappa)**3)
            else
              ampi = ZERO
            end if

            do mli = 1, llip1
              imsto = imsto + 1
              amp = amp + abs(evec(imsto, korb) * ampi)
            end do
          else
            imsto = imsto + llip1
          end if
        end do
      end if
      write(iw, 2000) eval(korb) * EVPERH, amp * 1000.0d0
    end do

1000 format(//'      EV        INTENSITY      (SOURCE =', F8.2, ')'/ &
              '      __        _________'/)
2000 format(F10.2, F15.4)
  end subroutine esca

end module diagnostics

!=======================================================================
! Global external subroutine wrappers for compatibility with legacy callers
!=======================================================================

subroutine output(matrix, rowlow, rowhi, collow, colhi, rowdim, coldim, nctl)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_output => output
  implicit none
  integer(int32), intent(in) :: rowlow, rowhi, collow, colhi, rowdim, coldim, nctl
  real(real64),   intent(in) :: matrix(rowdim, coldim)
  call mod_output(matrix, rowlow, rowhi, collow, colhi, rowdim, coldim, nctl)
end subroutine output

subroutine outpak(matrix, nrow, nctl)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_outpak => outpak
  implicit none
  integer(int32), intent(in) :: nrow, nctl
  real(real64),   intent(in) :: matrix(*)
  call mod_outpak(matrix, nrow, nctl)
end subroutine outpak

subroutine outvec(evec, eval, nbas, nvec, iw)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_outvec => outvec
  implicit none
  integer(int32), intent(in) :: nbas, nvec, iw
  real(real64),   intent(in) :: evec(nbas, nvec), eval(nvec)
  call mod_outvec(evec, eval, nbas, nvec, iw)
end subroutine outvec

subroutine putmat(evec, nbas, nvec, iw)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_putmat => putmat
  implicit none
  integer(int32), intent(in) :: nbas, nvec, iw
  real(real64),   intent(in) :: evec(nbas, nvec)
  call mod_putmat(evec, nbas, nvec, iw)
end subroutine putmat

subroutine putone(onemat, nbas, iw)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_putone => putone
  implicit none
  integer(int32), intent(in) :: nbas, iw
  real(real64),   intent(in) :: onemat(*)
  call mod_putone(onemat, nbas, iw)
end subroutine putone

subroutine twoout(matrix, nsize, index_arr, nctl)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_twoout => twoout
  implicit none
  integer(int32), intent(in) :: nsize, nctl
  integer(int32), intent(in) :: index_arr(*)
  real(real64),   intent(in) :: matrix(*)
  call mod_twoout(matrix, nsize, index_arr, nctl)
end subroutine twoout

subroutine getset()
  use diagnostics, only: mod_getset => getset
  implicit none
  call mod_getset()
end subroutine getset

subroutine timout(msgtyp)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_timout => timout
  implicit none
  integer(int32), intent(in) :: msgtyp
  call mod_timout(msgtyp)
end subroutine timout

function mapndx(x, mapdim)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_mapndx => mapndx
  implicit none
  real(real64),   intent(in) :: x
  integer(int32), intent(in) :: mapdim
  integer(int32) :: mapndx
  mapndx = mod_mapndx(x, mapdim)
end function mapndx

subroutine xyzmap(coord, latom, natom, iw, range_val, x0, y0, z0)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_xyzmap => xyzmap
  implicit none
  integer(int32),   intent(in) :: natom, iw
  real(real64),     intent(in) :: coord(3, natom)
  character(len=2), intent(in) :: latom(natom)
  real(real64),     intent(in) :: range_val, x0, y0, z0
  call mod_xyzmap(coord, latom, natom, iw, range_val, x0, y0, z0)
end subroutine xyzmap

subroutine plotv(lv, mv, vexp, vcoef, nvterm, iw)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_plotv => plotv
  implicit none
  integer(int32), intent(in) :: nvterm, iw
  integer(int32), intent(in) :: lv(nvterm), mv(nvterm)
  real(real64),   intent(in) :: vexp(nvterm), vcoef(nvterm)
  call mod_plotv(lv, mv, vexp, vcoef, nvterm, iw)
end subroutine plotv

subroutine punchv(stevec, eval, occ, l, nsto, msto, norb, ispin, ip)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_punchv => punchv
  implicit none
  integer(int32), intent(in) :: nsto, msto, norb, ispin, ip
  real(real64),   intent(in) :: stevec(msto, norb, ispin)
  real(real64),   intent(in) :: eval(norb, ispin), occ(norb, ispin)
  integer(int32), intent(in) :: l(nsto)
  call mod_punchv(stevec, eval, occ, l, nsto, msto, norb, ispin, ip)
end subroutine punchv

subroutine bomb(ierror)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_bomb => bomb
  implicit none
  integer(int32), intent(in) :: ierror
  call mod_bomb(ierror)
end subroutine bomb

subroutine warn(ierror, qreset)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_warn => warn
  implicit none
  integer(int32), intent(in)  :: ierror
  logical,        intent(out) :: qreset
  call mod_warn(ierror, qreset)
end subroutine warn

subroutine setbom()
  use diagnostics, only: mod_setbom => setbom
  implicit none
  call mod_setbom()
end subroutine setbom

subroutine dump(nrtns)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_dump => dump
  implicit none
  integer(int32), intent(in) :: nrtns
  call mod_dump(nrtns)
end subroutine dump

function noble(nucz)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_noble => noble
  implicit none
  integer(int32), intent(in) :: nucz
  integer(int32) :: noble
  noble = mod_noble(nucz)
end function noble

subroutine err243(iertcd, ierno, d, i)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_err243 => err243
  implicit none
  integer(int32), intent(out)   :: iertcd, ierno, i
  real(real64),   intent(inout) :: d
  call mod_err243(iertcd, ierno, d, i)
end subroutine err243

subroutine arrmap(nr)
  use iso_fortran_env, only: int32
  use diagnostics,     only: mod_arrmap => arrmap
  implicit none
  integer(int32), intent(in) :: nr
  call mod_arrmap(nr)
end subroutine arrmap

subroutine esca(evec, eval, n, l, eta, norb, msto, nsto, iw)
  use iso_fortran_env, only: real64, int32
  use diagnostics,     only: mod_esca => esca
  implicit none
  integer(int32), intent(in) :: norb, msto, nsto, iw
  real(real64),   intent(in) :: evec(msto, norb), eval(norb)
  integer(int32), intent(in) :: n(nsto), l(nsto)
  real(real64),   intent(in) :: eta(nsto)
  call mod_esca(evec, eval, n, l, eta, norb, msto, nsto, iw)
end subroutine esca
