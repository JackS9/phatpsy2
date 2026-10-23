      SUBROUTINE DMPATB(A,NA,MA,NAMX,MAMX,B,NB,MB,NBMX,MBMX,C,NCMX,MCMX)
C-----------------------------------------------------------------------
C
C     DMPATB...
C
C        THIS ROUTINE MULTIPLIES TWO RECTANGULAR MATRICES, WHERE THE IS
C     THE TRANSPOSE.
C
C             T
C        C = A B
C
C     VARIABLE DEFINITIONS:
C
C        A(*).... LEFT MATRIX (TRANSPOSE).
C        NA...... NUMBER OF ROWS IN A TO MULTIPLY.
C        MA...... NUMBER OF COLUMNS IN A TO MULTIPLY.
C        NAMX.... MAXIMUM NUMBER OF ROWS IN A (DIMENSION).
C        MAMX.... MAXIMUM NUMBER OF COLUMNS IN A (DIMENSION).
C        B(*).... RIGHT MATRIX.
C        NB...... NUMBER OF ROWS IN B TO MULTIPLY.
C        MB...... NUMBER OF COLUMNS IN B TO MULTIPLY.
C        NBMX.... MAXIMUM NUMBER OF ROWS IN B (DIMENSION).
C        MBMX.... MAXIMUM NUMBER OF COLUMNS IN B (DIMENSION).
C        C(*).... RESULTANT MATRIX.
C        NCMX.... MAXIMUM NUMBER OF ROWS IN C (DIMENSION).
C        MCMX.... MAXIMUM NUMBER OF COLUMNS IN C (DIMENSION).
C
C-----------------------------------------------------------------------
      REAL*8 A(NAMX,MAMX),B(NBMX,MBMX),C(NCMX,MCMX)
	  DATA ZERO /0.0D0/
	  
      DO 30 I=1,MA
	     DO 20 J=1,MB
		 	SUM = ZERO
		    DO 10 K=1,NA
               SUM = SUM + A(K,I)*B(K,J)
   10       CONTINUE
   			C(I,J) = SUM
   20    CONTINUE
   30 CONTINUE
   
      RETURN
      END
