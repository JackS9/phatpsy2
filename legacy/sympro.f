      SUBROUTINE SYMPRO(A,B,SCRAT,NBAS,N2BAS,ICODE)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SYMPRO...
C
C        THIS ROUTINE MODIFIES A SYMMETRIC MATRIX BY ADDING A SYMMETRIC
C     PRODUCT OF THIS MATRIX WITH ANOTHER NONSYMMETRIC MATRIX ACCORDING
C     TO THE VALUE OF ICODE AS FOLLOWS:
C
C        ICODE             RESULT
C
C           4        A + BA
C
C           3        A + AB
C                                   T
C           2        A + 1/2(BA + AB )
C                                  T
C           1        A + 1/2(AB + B A)
C
C           0        A
C                                  T
C          -1        A - 1/2(AB + B A )
C                                   T
C          -2        A - 1/2(BA + AB )
C
C          -3        A - AB
C
C          -4        A - BA
C
C     VARIABLE DEFINITIONS:
C
C        A(*)........ SYMMETRIC MATRIX (PACKED) TO BE MODIFIED.
C        B(*)........ NONSYMMETRIC MATRIX (PROJECTION MATRIX).
C        SCRAT(*).... SYMMETRIC SCRATCH MATRIX (PACKED).
C        NBAS........ ORDER OF MATRICES.
C        N2BAS....... =(NBAS*(NBAS+1))/2, DIMENSION OF PACKED MATRICES.
C        ICODE....... TYPE OF PRODUCT TO BE TAKEN (SEE ABOVE).
C
C     ROUTINES CALLED:  IABS
C
C-----------------------------------------------------------------------
      DIMENSION A(N2BAS),B(NBAS,NBAS),SCRAT(N2BAS)
      DATA ZERO/0.0D0/,TWO/2.0D0/
      IGO=IABS(ICODE)+1
      DO 30 I=1,NBAS
      II=(I*(I-1))/2
      DO 20 J=1,I
      JJ=(J*(J-1))/2
      IJ=II+J
      SUM=ZERO
      DO 10 K=1,NBAS
      KK=(K*(K-1))/2
      IK=II+K
      IF (K.GT.I) IK=KK+I
      JK=JJ+K
      IF (K.GT.J) JK=KK+J
      GO TO (1,2,3,4,5),IGO
    1 GO TO 10
    2 SUM=SUM+A(IK)*B(K,J)+A(JK)*B(K,I)
      GO TO 10
    3 SUM=SUM+A(IK)*B(J,K)+A(JK)*B(I,K)
      GO TO 10
    4 SUM=SUM+TWO*A(IK)*B(K,J)
      GO TO 10
    5 SUM=SUM+TWO*A(JK)*B(I,K)
   10 CONTINUE
      SCRAT(IJ)=SUM/TWO
   20 CONTINUE
   30 CONTINUE
      DO 40 IJ=1,N2BAS
      IF (ICODE) 32,40,31
   31 A(IJ)=A(IJ)+SCRAT(IJ)
      GO TO 40
   32 A(IJ)=A(IJ)-SCRAT(IJ)
   40 CONTINUE
      RETURN
      END
