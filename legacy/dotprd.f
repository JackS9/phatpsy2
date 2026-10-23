      FUNCTION DOTPRD(A,B,N)
      REAL*8 DOTPRD
C-----------------------------------------------------------------------
C
C     DOTPRD...
C
C        THIS FUNCTION RETURNS A DOT PRODUCT OF TWO VECTORS.
C
C        F = A(dot)B
C
C     VARIABLE DEFINITIONS:
C
C        A(*).... ROW VECTOR.
C        B(*).... COLUMN VECTOR.
C        N....... NUMBER OF ELEMENTS.
C
C-----------------------------------------------------------------------
      REAL*8    A(N),B(N)
	  
      DOTPRD = 0.0D0
	  
      DO 10 I=1,N
         DOTPRD = DOTPRD + A(I)*B(I)
   10 CONTINUE
   
      RETURN
      END
