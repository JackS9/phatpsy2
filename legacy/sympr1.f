      SUBROUTINE SYMPR1(P,R,S,F,SCRAT,N,N2)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SYMPR1...
C
C        THIS ROUTINE RETURNS THE "ON-SITE" CONTRIBUTION OF THE PSEUDO-
C     POTENTIAL TO THE FOCK MATRIX.
C
C        F = PRS + SRP - PRP
C
C     VARIABLE DEFINITIONS:
C
C        P(*)......... PROJECTION OPERATOR MATRIX.
C        R(*)......... "ON-SITE" PROJECTION OF FOCK MATRIX.
C        S(*)......... OVERLAP MATRIX.
C        F(*)......... "ON-SITE" CONTRIBUTION TO PSEUDOPOTENTIAL.
C        SCRAT(*,*)... SCRATCH MATRIX.
C        N............ DIMENSION OF BASIS.
C        N2........... N*(N+1)/2, PACKED DIMENSION.
C
C     ROUTINES CALLED:  TRITRI;  MAX0, MIN0
C
C-----------------------------------------------------------------------
      DIMENSION P(N2),R(N2),S(N2),F(N2),SCRAT(N,N)
      DATA ZERO/0.0D0/
      IINDEX(IARG)=(IARG*(IARG-1))/2
      CALL TRITRI(R,P,SCRAT,N,N2)
      IJ=0
      DO 30 I=1,N
      DO 20 J=1,I
      IJ=IJ+1
      SUM=ZERO
      DO 10 K=1,N
      IKMN=MIN0(I,K)
      IKMX=MAX0(I,K)
      IK=IINDEX(IKMX)+IKMN
      KJMN=MIN0(K,J)
      KJMX=MAX0(K,J)
      KJ=IINDEX(KJMX)+KJMN
      SUM=SUM+(S(IK)-P(IK))*SCRAT(K,J)+SCRAT(K,I)*S(KJ)
   10 CONTINUE
      F(IJ)=SUM
   20 CONTINUE
   30 CONTINUE
      RETURN
      END
