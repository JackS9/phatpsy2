      SUBROUTINE BSCALE (N,X,B,NBDIM)
C...........VERSION = 11/07/73/00
C.......................................................................
C     THIS ROUTINE COMPUTES THE SCALED B(N,X) FUNCTION DEFINED BY
C
C     B(N,X) = EXP(-ABS(X)) INTEGRAL (T = -1,+1) T**N EXP(-XT) DT
C
C N AND X ARE RECEIVED AS ARGUMENTS, AND ALL THE FUNCTIONS FOR
C INDICES 0,1,...,N ARE RETURNED IN B(1),B(2),...,B(N+1).  EXECUTION
C WILL TERMINATE IF THE SPECIFIED DIMENSION, NBDIM, OF B(*) IS
C EXCEEDED.
C     THE ALGORITHM IS DUE TO C.E. REID, QUANTUM THEORY PROJECT,
C UNIVERSITY OF FLORIDA, GAINESVILLE FOR THE PRODUCTION OF THE ORDINARY
C UNSCALED B FUNCTION.  FOR THE SCALED B FUNCTION, THE FINITE SERIES
C SECTION HAS BEEN ADDED BY THE CURRENT WRITER.  THE VALUES COMPUTED
C ARE CORRECT TO 14 FIGURES OR MORE, AND 13 FIGURES IN THE WORST CASE
C FOR DOUBLE PRECISION (16 FIGURES) ON THE IBM 360/370 MACHINES.
C
C AUTHOR%  NELSON H.F. BEEBE, QUANTUM THEORY PROJECT, UNIVERSITY OF
C          FLORIDA, GAINESVILLE, FLORIDA, AND DIVISION OF THEORETICAL
C          CHEMISTRY, DEPARTMENT OF CHEMISTRY, AARHUS UNIVERSITY,
C          AARHUS, DENMARK
C
C.......................................................................
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION B(NBDIM)
C.......................................................................
C THE FOLLOWING ARITHMETIC STATEMENT FUNCTION(S) DEFINE THE LIBRARY
C FUNCTION(S) REQUIRED BY THIS SUBROUTINE.
C.......................................................................
      ZABS(ARG) = DABS(ARG)
      ZCOSH(ARG) = DCOSH(ARG)
      ZEXP(ARG) = DEXP(ARG)
      ZFLOAT(IARG) = DFLOAT(IARG)
      ZSINH(ARG) = DSINH(ARG)
C.......................................................................
      DATA ZERO/0.D+00/, POINT6/0.6D+00/, TWO/2.D+00/, TINY/1.D-15/,
     X     ONE/1.D+00/, OVRFLO/20.D+00/
      IF (N .GE. NBDIM) GO TO 110
      NP1 = N + 1
      XABS = ZABS(X)
      IF (XABS .GT. OVRFLO) GO TO 70
      IF (XABS .LE. (POINT6*ZFLOAT(N))) GO TO 20
C.......................................................................
C USE THE RECURSION FORMULA SINCE THE ARGUMENT X IS IN THE RANGE
C (.6*N,OVRFLO) IN ABSOLUTE VALUE WHICH IS SUFFICIENTLY LARGE TO AVOID
C SUBTRACTION ACCURACY LOSSES.
C.......................................................................
      ODD = TWO/X
      EVEN = ODD*ZSINH(X)
      B(1) = EVEN
      IF (N .EQ. 0) GO TO 50
      ODD = ODD*ZCOSH(X)
      FI = ZERO
      DO 10 IEVEN = 2,NP1,2
      FI = FI + TWO
      B(IEVEN) = ((FI - ONE)/X)*B(IEVEN-1) - ODD
      IF (IEVEN .EQ. NP1) GO TO 10
      B(IEVEN+1) = (FI/X)*B(IEVEN) + EVEN
   10 CONTINUE
      GO TO 50
C.......................................................................
C USE THE INFINITE SERIES REPRESENTATION SINCE THE ARGUMENT IS IN
C THE RANGE (0,.6*N) IN ABSOLUTE VALUE.  THE SERIES IS SUMMED UNTIL
C THE REMAINDER IS SUFFICIENTLY SMALL.  THE SERIES IS OBTAINED BY
C A STRAIGHTFORWARD SUBSTITUTION OF THE TAYLOR SERIES EXPANSION OF
C THE EXPONENTIAL INTO THE INTEGRAL.
C.......................................................................
   20 XX = X*X
      NFLIP = +1
      DO 40 I = 1,NP1
         NFLIP = -NFLIP
         IM1MOD = MOD(I-1,2)
         U = TWO
         IF (NFLIP .EQ. 1) U = -U*X
         TSUM = U/ZFLOAT(I+IM1MOD)
         IF (TSUM .EQ. ZERO) GO TO 45
         K = 1
   30    KPK = K + K
         U = XX*U/ZFLOAT(KPK*(KPK+NFLIP))
         TK = U/ZFLOAT(I+KPK+IM1MOD)
         IF (ZABS(TK/TSUM) .LT. TINY) GO TO 45
         TSUM = TSUM + TK
         K = K + 1
         GO TO 30
   45    CONTINUE
         B(I) = TSUM
   40 CONTINUE
C.......................................................................
C AT THIS POINT, THE TWO PRECEDING SECTIONS HAVE COMPUTED THE ORDINARY
C UNSCALED B FUNCTIONS.  INCLUDE THE SCALE FACTOR NOW, THEN RETURN.
C.......................................................................
   50 SCALE = ZEXP(-XABS)
      DO 60 I = 1,NP1
         B(I) = SCALE*B(I)
   60 CONTINUE
      GO TO 100
C.......................................................................
C USE THE FINITE SERIES REPRESENTATION OF THE B FUNCTION.  THIS IS USED
C ONLY WHEN THE ARGUMENT IS SUFFICIENTLY LARGE THAT 1 - EXP(-2*ABS(X))
C MAY BE TRUNCATED TO 1.
C.......................................................................
   70 T0 = ONE/XABS
      DO 90 MUP1 = 1,NP1
      SUM = ZERO
      TNU = T0
      DO 80 NUP1 = 1,MUP1
         SUM = SUM + TNU
         TNU = -XABS*TNU/ZFLOAT(NUP1)
   80 CONTINUE
      IF ((X .LT. ZERO) .AND. (MOD(MUP1,2) .EQ. 0)) SUM = -SUM
      B(MUP1) = SUM
      T0 = ZFLOAT(MUP1)*T0/XABS
   90 CONTINUE
  100 RETURN
  110 WRITE (6,1000) N
      STOP
 1000 FORMAT (18H0*** ERROR *** N =,I5,47H EXCEEDS SPECIFIED DIMENSION I
     XN ROUTINE BSCALE.)
      END
