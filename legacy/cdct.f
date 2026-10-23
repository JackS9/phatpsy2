      SUBROUTINE CDCT(C,D,DD,NVEC,NBAS,N2BAS)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     CDCT...
C
C        THIS ROUTINE PERFORMS A (REVERSE) SIMILARITY TRANSFORMATION ON
C     A DIAGONAL MATRIX.
C
C                  T
C        DD = C D C
C
C     VARIABLE DEFINITIONS:
C
C        C(*)........ TRANSFORMATION MATRIX.
C        D(*)........ DIAGONAL MATRIX (VECTOR) TO BE TRANSFORMED.
C        DD(*)....... SYMMETRIC (PACKED) MATRIX RESULTING FROM THE
C                     TRANSFORMATION (SEE ABOVE).
C        NVEC........ ORDER OF DIAGONAL MATRIX.
C        NBAS........ ORDER OF TRANSFORMED MATRIX.
C        N2BAS....... =(NBAS*(NBAS+1))/2, PACKED DIMENSION.
C
C-----------------------------------------------------------------------
      DIMENSION C(NBAS,NVEC),D(NVEC),DD(N2BAS)
      DATA ZERO/0.0D0/
      DO 30 I=1,NBAS
      II=(I*(I-1))/2
      DO 20 J=1,I
      IJ=II+J
      SUM=ZERO
      DO 10 K=1,NVEC
      SUM=SUM+C(I,K)*D(K)*C(J,K)
   10 CONTINUE
      DD(IJ)=SUM
   20 CONTINUE
   30 CONTINUE
      RETURN
      END
