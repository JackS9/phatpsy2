      SUBROUTINE XYZMAP(COORD,LATOM,NATOM,IW,RANGE,X0,Y0,Z0)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      CHARACTER*2 LATOM,MAP,ISYMBL,IBLANK
C-----------------------------------------------------------------------
C
C     XYZMAP...
C
C        THIS ROUTINE GENERATES PERSPECTIVE MAPS OF ATOMIC CARTESIAN
C     COORDINATES IN THE YZ- ZX- AND XY-PLANES WITH A PRINTER PLOT.
C
C     VARIABLE DEFINITIONS:
C
C        COORD(1,*)... ATOMIC X-COORDINATES.
C             (2,*)...   "    Y-     "
C             (3,*)...   "    Z-     "
C        LATOM(*)..... ATOMIC SYMBOLS (LABELS).
C        NATOM........ NUMBER OF ATOMS.
C        IW........... FORTRAN I/O UNIT FOR PRINTING.
C        RANGE........ RANGE OF COORDINATE VALUES.
C        X0,Y0,Z0..... COORDINATES OF ORIGIN.
C
C     ROUTINES CALLED:  MAPNDX, DEPTH
C
C-----------------------------------------------------------------------
      DIMENSION COORD(3,NATOM),LATOM(NATOM),MAP(21,3,21)
      DATA MAPDIM/21/
      DATA IBLANK/'  '/
C-----------------------------------------------------------------------
C
C     DEFINE VALID RANGE LOGICAL FUNCTION.
C
C-----------------------------------------------------------------------
      QVALID(IARG)=((IARG.GT.0).AND.(IARG.LE.MAPDIM))
      IRANGE=(MAPDIM-1)/2
      SCALE=IRANGE/RANGE
C-----------------------------------------------------------------------
C
C     BLANK OUT MAP ARRAY.
C
C-----------------------------------------------------------------------
      DO 30 K=1,MAPDIM
      DO 20 J=1,3
      DO 10 I=1,MAPDIM
      MAP(I,J,K)=IBLANK
   10 CONTINUE
   20 CONTINUE
   30 CONTINUE
C-----------------------------------------------------------------------
C
C     LOOP OVER ATOMS.
C
C-----------------------------------------------------------------------
      DO 100 IATOM=1,NATOM
      X=(COORD(1,IATOM)-X0)*SCALE
      Y=(COORD(2,IATOM)-Y0)*SCALE
      Z=(COORD(3,IATOM)-Z0)*SCALE
      IX=MAPNDX(X,MAPDIM)
      IY=MAPNDX(Y,MAPDIM)
      IZ=MAPNDX(Z,MAPDIM)
      ISYMBL=LATOM(IATOM)
      CALL DEPTH(Z,Y,X,JZ,JY,IRANGE)
      KY=IY+JY
      KZ=IZ+JZ
      IF (QVALID(KY).AND.QVALID(KZ)) MAP(KZ,1,MAPDIM-KY+1)=ISYMBL
      CALL DEPTH(X,Z,Y,JX,JZ,IRANGE)
      KX=IX+JX
      KZ=IZ+JZ
      IF (QVALID(KX).AND.QVALID(KZ)) MAP(KX,2,MAPDIM-KZ+1)=ISYMBL
      CALL DEPTH(Y,X,Z,JY,JX,IRANGE)
      KX=IX+JX
      KY=IY+JY
      IF (QVALID(KX).AND.QVALID(KY)) MAP(KY,3,MAPDIM-KX+1)=ISYMBL
  100 CONTINUE
C-----------------------------------------------------------------------
C
C     PLOT PROJECTION MAPS.
C
C-----------------------------------------------------------------------
      WRITE(IW,1000)
      DO 200 K=1,MAPDIM
      WRITE(IW,2000) (MAP(I,1,K),I=1,MAPDIM)
  200 CONTINUE
      WRITE(IW,1000)
      DO 300 K=1,MAPDIM
      WRITE(IW,2000) (MAP(I,2,K),I=1,MAPDIM)
  300 CONTINUE
      WRITE(IW,1000)
      DO 400 K=1,MAPDIM
      WRITE(IW,2000) (MAP(I,3,K),I=1,MAPDIM)
  400 CONTINUE
      WRITE(IW,1000)
      WRITE(IW,3000)
      RETURN
C-----------------------------------------------------------------------
 1000 FORMAT(' +',42('-'),'+')
 2000 FORMAT(' |',21A2,'|')
 3000 FORMAT(/'  YZ, XZ, XY Projections'//)
      END
