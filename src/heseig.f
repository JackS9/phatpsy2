      SUBROUTINE HESEIG(A,N,NRT,X,XP,EIG,ETA,NZC,*)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION A(N,N),X(N),XP(N),EIG(NRT)
      REAL*8 TOL
C	  PARAMETER (TOL = $3920000000000000)
	  PARAMETER (TOL = 2.0D0*16.0D0**(-8))
      NROOT=MIN0(NRT,NZC)
      IF (ETA.LT.0.) GO TO 100
      R=A(1,1)-1.D0
      DO 10 II=2,N
      Y=A(II,II)
      IF (II.LT.N) Y=Y-1.D0
      DO 5 JJ=2,II
      Y=Y-DABS(A(JJ-1,II))
    5 CONTINUE
      IF (Y.LT.R) R=Y
   10 CONTINUE
   11 N1=N-1
      DO 50 M=1,NROOT
      XP(NZC)=1.D0
      NIT=0
   12 X(NZC)=R-A(NZC,NZC)
      DO 32 KK=2,NZC
      NI=NZC-KK+1
      X(NI)=(R-A(NI,NI))*X(NI+1)-A(NI,NZC)
      XP(NI)=X(NI+1)+(R-A(NI,NI))*XP(NI+1)
      IF (KK.EQ.2) GO TO 32
      NRI=KK-2
      DO 30 LL=1,NRI
      X(NI)=X(NI)-A(NI,NI+LL)*X(NI+LL+1)
      XP(NI)=XP(NI)-A(NI,NI+LL)*XP(NI+LL+1)
   30 CONTINUE
   32 CONTINUE
      IF (M.EQ.1) GO TO 40
      Y=XP(NI)/X(NI)
      DO 35 MM=2,M
      Y=Y-1.D0/(R-EIG(MM-1))
   35 CONTINUE
      Y=1.D0/Y
      GO TO 42
   40 Y=X(NI)/XP(NI)
   42 R=R-Y
   44 NIT=NIT+1
      IF (DABS(Y/R).LE.TOL) GO TO 45
      IF (NIT.GT.25) RETURN 1
      GO TO 12
   45 EIG(M)=R
   50 CONTINUE
      R=R+ETA
      RETURN
  100 R=A(1,1)+1.D0
      DO 110 NN=2,N
      Y=A(NN,NN)
      IF (NN.LT.N) Y=Y+1.D0
      DO 105 IPP=2,NN
      Y=Y+DABS(A(IPP-1,NN))
  105 CONTINUE
      IF (R.LT.Y) R=Y
  110 CONTINUE
      GO TO 11
      END
