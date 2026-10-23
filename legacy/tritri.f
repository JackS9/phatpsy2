      SUBROUTINE TRITRI(A,B,AB,N,N2)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     TRITRI...
C
C        THIS ROUTINE PERFORMS A MATRIX PRODUCT OF TWO TRIANGULAR-
C     PACKED MATRICES.
C
C     VARIABLE DEFINITIONS:
C
C        A(*)........ FIRST TRIANGULAR MATRIX.
C        B(*)........ SECOND TRIANGULAR MATRIX.
C        AB(*,*)..... SQUARE PRODUCT MATRIX.
C        N........... ORDER OF MATRICES, DIMENSION OF AB(*,*).
C        N2.......... =N*(N+1)/2, DIMENSION OF A(*) AND B(*).
C
C-----------------------------------------------------------------------
      DIMENSION A(N2),B(N2),AB(N,N)
      DATA ZERO/0.0D0/
      DO 30 I=1,N
      II=(I*(I-1))/2
      DO 20 J=1,N
      JJ=(J*(J-1))/2
      SUM=ZERO
      DO 10 K=1,N
      KK=(K*(K-1))/2
      IK=II+K
      IF (K.GT.I) IK=KK+I
      JK=JJ+K
      IF (K.GT.J) JK=KK+J
      SUM=SUM+A(IK)*B(JK)
   10 CONTINUE
      AB(I,J)=SUM
   20 CONTINUE
   30 CONTINUE
      RETURN
      END
