      SUBROUTINE SYMPAK(A,ASYM,N,N2)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SYMPAK...
C
C        THIS ROUTINE PACKS A SYMMETRIC MATRIX WHICH IS IN SQUARE
C     FORM INTO A MATRIX WHICH IS LOWER TRIANGULAR PACKED.
C        IF NOT SYMMETRIC THEN AN ARITHMETIC AVERAGE OF THE
C     TRANSPOSE ELEMENTS IS USED.
C
C     VARIABLE DEFINITIONS:
C
C        A(*,*)...... SQUARE INPUT MATRIX.
C        ASYM(*)..... SYMMETRY-PACKED OUTPUT MATRIX.
C        N........... DIMENSION OF A(*,*).
C        N2.......... =N(N+1)/2, DIMENSION OF ASYM(*).
C
C-----------------------------------------------------------------------
      DIMENSION A(N,N),ASYM(N2)
      DATA TWO/2.0D0/
      IJ=0
      DO 20 I=1,N
      DO 10 J=1,I
      IJ=IJ+1
      ASYM(IJ)=(A(I,J)+A(J,I))/TWO
   10 CONTINUE
   20 CONTINUE
      RETURN
      END
