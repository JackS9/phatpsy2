      SUBROUTINE TRISQ(N,H,U,RESULT) 
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     TRISQ...
C
C        THIS ROUTINE MULTIPLIES A SYMMETRIC PACKED TRIANGULAR
C     MATRIX TIMES A SQUARE MATRIX
C
C           H U
C
C     VARIABLE DEFINITIONS:
C
C        N............ ORDER OF U AND H.
C        H(*)......... SYMMETRIC (PACKED) MATRIX.
C        U(*,*)....... SQUARE MATRIX.
C        RESULT(*,*).. RESULTANT SQUARE MATRIX.
C   
C-----------------------------------------------------------------------
      DIMENSION U(N,N),H(1),RESULT(N,N)
      DATA ZERO/0.0D0/

      DO 30 I=1,N
         II=(I*(I-1))/2
         DO 20 J=1,N
            SUM=ZERO
            DO 10 K=1,N
               IK=II+K
               IF (K.GT.I) IK=(K*(K-1))/2+I
               SUM=SUM+H(IK)*U(K,J)
   10       CONTINUE
            RESULT(I,J)=SUM
   20    CONTINUE
   30 CONTINUE

      RETURN
      END
