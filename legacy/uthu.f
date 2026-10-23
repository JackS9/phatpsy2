      SUBROUTINE UTHU(N,H,U,SCRAT,RESULT) 
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     UTHU...
C
C        THIS ROUTINE PERFORMS A SIMILARITY TRANSFORMATION OF A 
C     SYMMETRIC MATRIX.
C
C            T
C           U H U
C
C     VARIABLE DEFINITIONS:
C
C        N............ ORDER OF U AND H.
C        H(*)......... SYMMETRIC (PACKED) MATRIX TO BE TRANSFORMED.
C        U(*,*)....... TRANSFORMATION MATRIX.
C        SCRAT(*,*)... A SCRATCH MATRIX AT LEAST ORDER N.
C        RESULT(*).... SYMMETRIC (PACKED) TRANSFORMED MATRIX.
C   
C-----------------------------------------------------------------------
      DIMENSION U(N,N),SCRAT(N,N),H(1),RESULT(1)
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
            SCRAT(I,J)=SUM
   20    CONTINUE
   30 CONTINUE
   
      DO 60 I=1,N
         II=(I*(I-1))/2
         DO 50 J=1,I
            IJ=II+J
            SUM=ZERO
            DO 40 K=1,N
               SUM=SUM+U(K,I)*SCRAT(K,J)
   40       CONTINUE
            RESULT(IJ)=SUM
   50    CONTINUE
   60 CONTINUE
   
      RETURN
      END
