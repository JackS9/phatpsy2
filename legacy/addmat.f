      SUBROUTINE ADDMAT(A,B,C,N)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     ADDMAT...
C
C        THIS ROUTINE ADDS TWO ARRAYS WHICH ARE STORED CONTIGUOUSLY.
C     THE RESULTANT MATRIX IS ALSO ASSUMED CONTIGUOUS.
C
C        C = A + B
C
C     VARIABLE DEFINITIONS:
C
C        A(*)... FIRST ARRAY TO BE ADDED.
C        B(*)... SECOND ARRAY.
C        C(*)... RESULTANT MATRIX.
C        N...... NUMBER OF CONTIGUOUS STORAGE ELEMENTS TO BE ADDED.
C
C     ENTRY:  SUBMAT...
C
C        THIS ENTRY POINT SUBRACTS THE TWO ARRAYS.
C
C     NOTE:  NONE OF THE THREE ARRAYS NEED BE UNIQUE.
C
C-----------------------------------------------------------------------
      DIMENSION A(N),B(N),C(N)
      DO 10 I=1,N
      C(I)=A(I)+B(I)
   10 CONTINUE
      RETURN
      END
C
C...
      SUBROUTINE SUBMAT(A,B,C,N)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION A(N),B(N),C(N)
      DO 10 I=1,N
      C(I)=A(I)-B(I)
   10 CONTINUE
      RETURN
      END
