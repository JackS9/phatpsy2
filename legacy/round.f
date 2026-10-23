      FUNCTION ROUND(X,N)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     ROUND...
C
C        THIS FUNCTION RETURNS A REAL NUMBER ROUNDED OFF TO A SPECIFIED
C     NUMBER OF DIGITS BEFORE OR AFTER THE DECIMAL POINT.
C
C        EXAMPLE:  ROUND(12.3456789,-2) = 12.3500000
C
C-----------------------------------------------------------------------
      DATA HALF/0.5D0/,TEN/1.D1/
      SIGNIF=TEN**N
      NX=X/SIGNIF+HALF
      ROUND=NX*SIGNIF
      RETURN
      END
