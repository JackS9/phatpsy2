      SUBROUTINE EIG(A,B,N,NDIM)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION A(NDIM,NDIM),B(NDIM,NDIM)
      DATA TOL /1.0D-14/
        B(1,1)=1.
      IF(N.EQ.1)   RETURN
       DO1I=2,N
         IM1=I-1
         DO2J=1,IM1
            B(I,J)=0.
            B(J,I)=0.
    2    CONTINUE
         B(I,I)=1.
    1 CONTINUE
  200 P=0.
      DO3I=2,N
         IM1=I-1
         DO4J=1,IM1
            Q=A(I,J)
            IF(P.GE.DABS(Q)) GO TO 4
            P=DABS(Q)
            II=I
            JJ=J
    4       CONTINUE
    3    CONTINUE
      IF (P.EQ.0.) GO TO 205
      P=A(II,II)
      Q=A(II,JJ)
      R=A(JJ,JJ)
      DIFF=0.5*(P-R)
      IF(DABS(DIFF).LT.DABS(Q)) GO TO 201
      IF(DABS(Q/DIFF).GT.TOL) GO TO 201
      A(II,JJ)=0.
      A(JJ,II)=0.
      GO TO 200
  201 S=DSQRT(0.25D0*(P-R)**2+Q**2)
      SUM=0.5*(P+R)
      D=R*P-Q**2
      IF(SUM.GT.0.) GO TO 5
      ALN=SUM-S
      ALP=D/ALN
      GO TO 6
    5 ALP=SUM+S
      ALN=D/ALP
    6 IF(DIFF.GT.0.) GO TO 7
      T=Q/(DIFF-S)
      A(II,II)=ALN
      A(JJ,JJ)=ALP
      GO TO 8
    7 T=Q/(DIFF+S)
      A(II,II)=ALP
      A(JJ,JJ)=ALN
    8 C=1.0D0/DSQRT(1.0D0+T**2)
      S=T*C
      A(II,JJ)=0.
      A(JJ,II)=0.
      DO9 I=1,N
      P=B(I,II)
      Q=B(I,JJ)
      B(I,II)=C*P+S*Q
      B(I,JJ)=C*Q-S*P
      IF(I.EQ.II.OR.I.EQ.JJ) GO TO 9
      P=A(I,II)
      Q=A(I,JJ)
      R=C*P+S*Q
      A(I,II)=R
      A(II,I)=R
      R=Q*C-P*S
      A(I,JJ)=R
      A(JJ,I)=R
    9 CONTINUE
      GO TO 200
  205 CONTINUE
      RETURN
      END
