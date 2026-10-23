      SUBROUTINE DCOPY(A,N,B)
C-----------------------------------------------------------------------
C
C     DCOPY...
C
C        THIS ROUTINE COPIES AN ARRAY OR MATRIX WHICH IS STORED 
C     CONTIGUOUSLY TO ANOTHER ARRAY/MATRIX
C
C        B = A
C
C     VARIABLE DEFINITIONS:
C
C        A(*).... ARRAY/MATRIX TO BE COPIED FROM.
C        N....... NUMBER OF CONTIGUOUS STORAGE ELEMENTS TO BE COPIED.
C        B(*).... ARRAY/MATRIX TO BE COPIED TO.
C
C-----------------------------------------------------------------------
      REAL*8    A(N),B(N)
	  
      DO 10 I=1,N
      B(I)=A(I)
   10 CONTINUE
      RETURN
      END
