      SUBROUTINE GENFAC(FACT,NFACT)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     GENFAC...
C
C        THIS ROUTINE GENERATES ALL THE FACTORIALS UP TO A SPECIFIED
C     MAXIMUM.
C
C        FACT(1)     = 1  (0-FACTORIAL)
C            (2)     = 1
C            (3)     = 1*2
C             :         :
C            (NFACT) = 1*2*3*...*(NFACT-1)
C
C     ROUTINES CALLED:  DFLOAT
C
C-----------------------------------------------------------------------
      DIMENSION FACT(NFACT)
      DATA ONE/1.D0/
      FACT(1)=ONE
      NMAX=NFACT-1
      IF (NMAX.LT.1) RETURN
      DO 10 N=1,NMAX
      FACT(N+1)=FACT(N)*DFLOAT(N)
   10 CONTINUE
      RETURN
      END
