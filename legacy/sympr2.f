      SUBROUTINE SYMPR2(P,R,S,SCRAT1,SCRAT2,N,N2)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SYMPR2...
C
C        THIS ROUTINE GENERATES THE "OFF-SITE" PSEUDOPOTENTIAL FROM
C     THE "ON-SITE" PROJECTION OF THE FOCK MATRIX.
C
C        R = RSP + PSR - PSRSP
C
C     VARIABLE DEFINITIONS:
C
C        P(*).......... PROJECTION OPERATOR MATRIX.
C        R(*).......... "ON-SITE" PROJECTION OF FOCK MATRIX ON INPUT,
C                       "OFF-SITE" PSEUDOPOTENTIAL MATRIX ON OUTPUT.
C        S(*).......... OVERLAP MATRIX.
C        SCRAT1(*,*)... SCRATCH MATRIX.
C        SCRAT2(*,*)... SCRATCH MATRIX.
C        N............. DIMENSION OF BASIS.
C        N2............ N*(N+1)/2, PACKED DIMENSION.
C
C     ROUTINES CALLED:  TRITRI, TRISQ
C
C-----------------------------------------------------------------------
      DIMENSION P(N2),R(N2),S(N2),SCRAT1(N,N),SCRAT2(N,N)
      CALL TRITRI(S,P,SCRAT1,N,N2)
      CALL TRISQ(N,R,SCRAT1,SCRAT2)
      IJ=0
      DO 30 I=1,N
      DO 20 J=1,I
      IJ=IJ+1
      SUM=SCRAT2(I,J)+SCRAT2(J,I)
      DO 10 K=1,N
      SUM=SUM-SCRAT1(K,I)*SCRAT2(K,J)
   10 CONTINUE
      R(IJ)=SUM
   20 CONTINUE
   30 CONTINUE
      RETURN
      END
