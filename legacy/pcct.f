      SUBROUTINE PCCT(P,C,PSINV,NVEC,NBAS,N2BAS)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     PCCT...
C
C        THIS ROUTINE GENERATES THE MATRIX
C
C                   -1
C        PSINV = P S
C                     T
C              = P C C
C
C     VARIABLE DEFINITIONS:
C
C        P(*)......... SYMMETRIC (PACKED) MATRIX P.
C        C(*,*)....... TRANSFORMATION MATRIX.
C        PSINV(*,*)... RESULTANT NON-SYMMETRIC MATRIX.
C        NVEC......... COLUMN DIMENSION OF C(*,*).
C        NBAS......... ORDER P AND RESULTANT MATRIX AND ROW
C                      DIMENSION OF C(*,*).
C        N2BAS....... =(NBAS*(NBAS+1))/2, PACKED DIMENSION.
C
C     NOTE:  THIS ROUTINE COULD BE MADE FASTER BY PREMULTIPLYING
C               T         -1
C            C C  TO GET S   BUT IS WOULD REQUIRE A SCRATCH ARRAY
C            OF DIMENSION N2BAS (PACKED).
C
C-----------------------------------------------------------------------
      DIMENSION P(N2BAS),C(NBAS,NVEC),PSINV(NBAS,NBAS)
      DATA ZERO/0.0D0/
      DO 40 K=1,NBAS
      KK=(K*(K-1))/2
      DO 30 L=1,NBAS
      SUM=ZERO
      DO 20 I=1,NVEC
      DO 10 M=1,NBAS
      MM=(M*(M-1))/2
      KM=KK+M
      IF (M.GT.K) KM=MM+K
      SUM=SUM+P(KM)*C(M,I)*C(L,I)
   10 CONTINUE
   20 CONTINUE
      PSINV(K,L)=SUM
   30 CONTINUE
   40 CONTINUE
      RETURN
      END
