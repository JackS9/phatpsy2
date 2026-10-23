      SUBROUTINE JACOBI (NSIZE,ARRAY,VECTOR,NDIM)
C.......................................................................
C
C THIS ROUTINE USES THE JACOBI METHOD TO FIND ALL THE EIGENVALUES AND
C EIGENVECTORS OF A  SYMMETRIC MATRIX.
C THE INPUT IS:
C
C NSIZE........SIZE OF THE EIGENVALUE PROBLEM.
C
C NDIM.........ROW DIMENSION OF VECTOR(*,*).
C
C ARRAY(*)....SYMMETRIC MATRIX PACKED IN LOWER TRIANGULAR FORM.
C
C VECTOR(*,*)......EIGENVECTOR ARRAY.  THE EIGENVECTORS WILL BE
C                  ORTHOGONAL AND NORMALIZED TO UNITY.
C
C THIS ROUTINE WAS ORIGINALLY WRITTEN BY GORDON GALLUP AT THE QUANTUM
C THEORY PROJECT, UNIVERSITY OF FLORIDA (NOW AT UNIVERSITY OF NEBRASKA).
C IT WAS RECODED FOR PACKED ARRAYS BY NELSON H.F. BEEBE AT THE QUANTUM
C THEORY PROJECT.
C
C 2/7/90 - Language Systems Fortran 2.0: Use opt=1 (jas)
C.......................................................................
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION ARRAY(1),VECTOR(NDIM,1)
      DATA TINY/1.D-14/, ZERO/0.D0/, ONE/1.D0/, HALF/5.D-1/,
     X     FOURTH/2.5D-1/
	 
      N = IABS(NSIZE)
      VECTOR(1,1) = ONE
      IF (N .EQ. 1) RETURN
C.......................................................................
C SET A UNIT MATRIX INTO THE EIGENVECTOR ARRAY.
C.......................................................................
      DO 20 I = 2,N
      IM1 = I - 1
      DO 10 J = 1,IM1
      VECTOR(J,I) = ZERO
   10 VECTOR(I,J) = ZERO
   20 VECTOR(I,I) = ONE
   
   30 P = ZERO
      II = 1
	  
      DO 50 I = 2,N
      IM1 = I - 1
      DO 40 J = 1,IM1
      Q = ARRAY(II+J)
      IF (P .GE. DABS(Q)) GO TO 40
      P = DABS(Q)
      INOW = I
      JNOW = J
   40 CONTINUE
   50 II = II + I
   
      IF (P .EQ. ZERO) GO TO 190
	  
      II = (INOW*(INOW+1))/2
      IJ = II - INOW + JNOW
      JJ = (JNOW*(JNOW+1))/2
      P = ARRAY(II)
      Q = ARRAY(IJ)
      R = ARRAY(JJ)
      DIFF = HALF*(P - R)
      IF (DABS(Q).LT.TINY*TINY) GO TO 60
	  
      IF (DABS(DIFF) .LT. DABS(Q)) GO TO 70
	  
      IF (DABS(Q/DIFF) .GT. TINY) GO TO 70
	  
   60 ARRAY(IJ)=ZERO
      GO TO 30
	  
   70 S = DSQRT(FOURTH*(P-R)*(P-R) + Q*Q)
      SUM = HALF*(P + R)
      D = P*R - Q*Q
      IF (SUM .GT. ZERO) GO TO 80
      ALN = SUM - S
      ALP = D/ALN
      GO TO 90
	  
   80 ALP = SUM + S
      ALN = D/ALP
	  
   90 IF (DIFF .GT. ZERO) GO TO 100
      T = Q/(DIFF - S)
      ARRAY(II) = ALN
      ARRAY(JJ) = ALP
      GO TO 110
	  
  100 T = Q/(DIFF + S)
      ARRAY(II) = ALP
      ARRAY(JJ) = ALN
	  
  110 C = ONE/DSQRT(ONE + T*T)
      S = T*C
      ARRAY(IJ) = ZERO
      DO 180 I = 1,N
      P = VECTOR(I,INOW)
      Q = VECTOR(I,JNOW)
      VECTOR(I,INOW) = C*P + S*Q
      VECTOR(I,JNOW) = C*Q - S*P
	  
      IF (I - INOW) 120,180,130
  120 II = (INOW*(INOW-1))/2 + I
      GO TO 140
	  
  130 II = (I*(I-1))/2 + INOW
  
  140 IF (I - JNOW) 150,180,160
  150 IJ = (JNOW*(JNOW-1))/2 + I
      GO TO 170
	  
  160 IJ = (I*(I-1))/2 + JNOW
  
  170 P = ARRAY(II)
      Q = ARRAY(IJ)
      ARRAY(II) = C*P + S*Q
      ARRAY(IJ) = Q*C - P*S
	  
  180 CONTINUE
      GO TO 30
	  
  190 CONTINUE
      RETURN
      END
