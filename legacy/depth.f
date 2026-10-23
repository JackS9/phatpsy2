      SUBROUTINE DEPTH(X,Y,Z,JX,JY,IRANGE)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     DEPTH...
C
C        THIS ROUTINE ADJUSTS THE MAP ARRAY INDICES TO ALLOW FOR
C     DEPTH IN THE ORTHOGONAL (Z) DIRECTION. THIS GIVES THE MAP A
C     PERSPECTIVE VIEW TOWARD THE UPPER RIGHTHAND CORNER.
C
C     VARIABLE DEFINITIONS:
C
C        X,Y,Z.... ACTUAL (OR SCALED) CARTESIAN COORDINATES.
C        JX,JY.... ADJUSTMENTS TO THE MAP ARRAY INDICES FOR DEPTH
C                  IN THE SURFACE (XY) PLANE.
C        IRANGE... RANGE OF CARTESIAN VALUES PLOTTED.
C
C     ROUTINES CALLED:  MAPNDX; DATAN2, DSQRT, DCOS, DSIN
C
C----------------------------------------------------------------------
      DATA TWO/2.0D0/,SKEW/10.0D0/,ZVIEW/1.0D0/
C-----------------------------------------------------------------------
C
C     SKEW..... MULTIPLE OF RANGE USED FOR POINT ON HORIZON.  A VALUE
C               OF 1.0 USES UPPER RIGHTHAND CORNER OF PLOT FRAME.  A
C               LARGER VALUE CORRESPONDS TO OUTSIDE THE FRAME. /10.0/
C     ZVIEW.... MINIMUM DEPTH IN THE Z-DIRECTION VISIBLE AT THE
C               ORIGIN /1.0/
C
C-----------------------------------------------------------------------
      HORIZ=SKEW*IRANGE
      D=(X-HORIZ)*(X-HORIZ)+(Y-HORIZ)*(Y-HORIZ)
      R=Z*DSQRT(D)/(TWO*HORIZ*ZVIEW)
      ANGLE=DATAN2(Y-HORIZ,X-HORIZ)
      DX=-R*DCOS(ANGLE)
      DY=-R*DSIN(ANGLE)
      JX=MAPNDX(DX,0)
      JY=MAPNDX(DY,0)
      RETURN
      END
