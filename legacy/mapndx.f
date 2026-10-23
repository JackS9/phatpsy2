      FUNCTION MAPNDX(X,MAPDIM)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     MAPNDX...
C
C        THIS ROUTINE RETURNS THE INDEX VALUE IN THE MAP ARRAY TO BE
C     PLOTTED CORRESPONDING TO A PROJECTED CARTESIAN COORDINATE.
C
C     VARIABLE DEFINITIONS:
C
C        X........ ACTUAL (OR SCALED) COORDINATE.
C        MAPDIM... DIMENSION OF MAP ARRAY.
C
C-----------------------------------------------------------------------
      DATA HALF/0.5D0/
      XMOD=X
C     REPLACE ARITHMETIC IF WITH MODERN IF-ELSE IF STRUCTURE
      IF (X .LT. 0.0D0) THEN
         GO TO 30
      ELSE IF (X .EQ. 0.0D0) THEN
         GO TO 20
      ELSE
         GO TO 10
      END IF
   10 XMOD=XMOD+HALF
   20 XMOD=XMOD+HALF
   30 XMOD=XMOD-HALF
      IXMOD=XMOD
      MAPNDX=IXMOD+(MAPDIM+1)/2
      RETURN
      END
