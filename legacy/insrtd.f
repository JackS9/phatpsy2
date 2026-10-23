      SUBROUTINE INSRTD(BLOCK,SUPMAT,I,N)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     INSRTD...
C
C        THIS ROUTINE IS A MODIFIED VERSION OF INSERT FOR INSERTING
C     DIAGONAL SYMMETRY-PACKED BLOCKS INTO A LARGER MATRIX.
C
C     VARIABLE DEFINITIONS:
C
C        BLOCK(*).... SYMMETRY-PACKED BLOCK TO BE INSERTED.
C        SUPMAT(*)... LARGER MATRIX (ALSO SYMMETRY-PACKED).
C        I........... DIAGONAL ELEMENT WHERE INSERTION BEGINS.
C        N........... DIMENSION (NUMBER OF ROWS) OF BLOCK(*).
C
C-----------------------------------------------------------------------
      DIMENSION BLOCK(1),SUPMAT(1)
      IJ=(I*(I+1))/2
      DO 20 K=1,N
      KK=(K*(K-1))/2
      DO 10 L=1,K
      KL=KK+L
      SUPMAT(IJ)=BLOCK(KL)
      IJ=IJ+1
   10 CONTINUE
      IJ=IJ+I-1
   20 CONTINUE
      RETURN
      END
