      SUBROUTINE SCMULT(ARRAY,SCALAR,NDIM)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SCMULT...
C
C        THIS ROUTINE PERFORMS A SCALAR MULTIPLICATION OF AN ARRAY.
C
C     VARIABLE DEFINITIONS:
C
C        ARRAY(*)... ARRAY TO BE MULTIPLIED.
C        SCALAR..... SCALAR USED TO SCALE ARRAY.
C        NDIM....... NUMBER OF ELEMENTS IN ARRAY.
C
C-----------------------------------------------------------------------
      DIMENSION ARRAY(NDIM)
      DO 10 I=1,NDIM
      ARRAY(I)=SCALAR*ARRAY(I)
   10 CONTINUE
      RETURN
      END
