      SUBROUTINE CHKCGC(MM,M1,M2,LL,L1,L2,QZERO)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     CHKCGC...
C
C        THIS ROUTINE CHECKS TO SEE IF A CLEBSCH-GORDON COEFFICIENT IS
C     NONZERO BY ALL OF THE FOLLOWING CONDITIONS:
C
C        1)  LL+L1+L2 MUST BE EVEN
C        2)  !L1-L2! LE LL LE (L1+L2)   (TRIANGULAR INEQUALITY)
C        3)  !MM! LE LL
C        4)  !M1! LE L1  AND  !M2! LE L2
C        5)  MM=M1+M2
C
C     ROUTINES CALLED:  MOD, IABS
C
C-----------------------------------------------------------------------
      QZERO=.TRUE.
      IF (MOD(LL+L1+L2,2).NE.0) RETURN
      IF (LL.GT.L1+L2) RETURN
      IF (LL.LT.IABS(L1-L2)) RETURN
      IF (IABS(MM).GT.LL) RETURN
      IF (IABS(M1).GT.L1.OR.IABS(M2).GT.L2) RETURN
      IF (MM.NE.(M1+M2)) RETURN
      QZERO=.FALSE.
      RETURN
      END
