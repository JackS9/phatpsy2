      SUBROUTINE GENBC(BINOM,NBINOM,NROWS,INDEX)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     GENBC...
C
C        THIS ROUTINE GENERATES ALL THE BINOMIAL COEFFICIENTS UP TO A
C     SPECIFIED ORDER.
C
C        (X+Y)**NORDER = X**NORDER + ... + Y**NORDER
C
C        PASCAL'S TRIANGLE (OF BINOMIAL COEFFICIENTS):
C
C             1
C            1 1
C           1 2 1
C          1 3 3 1
C         1 4 6 4 1
C        . . . . . .
C
C     VARIABLE DEFINITIONS:
C
C        BINOM(INDEX(N)+M)... BINOMIAL COEFFICIENT OF X**(N-1) Y**(M-1).
C        NBINOM.............. =INDEX(NROWS+1). DIMENSION OF BINOM(*).
C        NROWS............... =NORDER+1, NUMBER OF ROWS IN PASCAL'S
C                             TRIANGLE.
C        INDEX(N)............ =N*(N-1)/2.
C
C-----------------------------------------------------------------------
      DIMENSION BINOM(NBINOM),INDEX(NROWS)
      DATA ONE/1.D0/
      BINOM(1)=ONE
      IF (NROWS.LT.2) RETURN
      BINOM(2)=ONE
      BINOM(3)=ONE
      IF (NROWS.LT.3) RETURN
      DO 20 N=3,NROWS
      NN=INDEX(N)
      NM1=N-1
      BINOM(NN+1)=ONE
      DO 10 M=2,NM1
      NM=NN+M
      BINOM(NM)=BINOM(NM-N)+BINOM(NM-N+1)
   10 CONTINUE
      BINOM(NN+N)=ONE
   20 CONTINUE
      RETURN
      END
