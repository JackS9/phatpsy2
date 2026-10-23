      SUBROUTINE ORDER(MM,M1,M2,LL,L1,L2,QREORD)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     ORDER...
C
C        THIS ROUTINE PERMUTES THE INDICES OF A C-G COEFFICIENT (WITH
C     POSSIBLE CHANGE IN SIGN) TO AN EQUIVALENT ORDER SUCH THAT
C     LL GE L1 GE L2, THAT MM GE M1 FOR LL=L1 AND THAT M1 GE M2 FOR
C     L1=L2.  FOR CONSISTENCY MM IS ALWAYS CHOSEN TO BE POSITIVE.
C     THE FOLLOWING EQUALITIES MAKE THIS POSSIBLE:
C
C        C(MM,M1,M2;LL,L1,L2) = C(MM,M2,M1;LL,L2,L1)
C                             = C(M1,MM,-M2;L1,LL,L2)
C                             = C(-MM,-M1,-M2;LL,L1,L2)
C
C     ENTRY:  REORDR...
C
C        THIS ENTRY POINT RESTORES THE ORIGINAL ORDER OF THE INDICES
C     IF THEY HAVE BEEN REORDERED PREVIOUSLY (QREORD=.TRUE.).
C
C-----------------------------------------------------------------------
      SAVE MMSAV,M1SAV,M2SAV,LLSAV,L1SAV,L2SAV
	  
      MMSAV=MM
      M1SAV=M1
      M2SAV=M2
      LLSAV=LL
      L1SAV=L1
      L2SAV=L2
      QREORD=.FALSE.
      IF (L1.LT.L2) GO TO 10
      GO TO 20
   10 L1TEMP=L1
      L1=L2
      L2=L1TEMP
      M1TEMP=M1
      M1=M2
      M2=M1TEMP
      QREORD=.TRUE.
   20 IF (LL.LT.L1) GO TO 30
      GO TO 40
   30 LLTEMP=LL
      LL=L1
      L1=LLTEMP
      MMTEMP=MM
      MM=M1
      M1=MMTEMP
      M2=-M2
      QREORD=.TRUE.
      IF (L1.GE.L2) GO TO 40
      L1TEMP=L1
      L1=L2
      L2=L1TEMP
      M1TEMP=M1
      M1=M2
      M2=M1TEMP
   40 IF (MM.LT.0) GO TO 50
      GO TO 60
   50 MM=-MM
      M1=-M1
      M2=-M2
      QREORD=.TRUE.
   60 IF ((L1.EQ.L2).AND.(M1.LT.M2)) GO TO 70
      GO TO 80
   70 M1TEMP=M1
      M1=M2
      M2=M1TEMP
      QREORD=.TRUE.
   80 IF ((LL.EQ.L1).AND.(MM.LT.M1)) GO TO 90
      RETURN
   90 MMTEMP=MM
      MM=M1
      M1=MMTEMP
      M2=-M2
      QREORD=.TRUE.
      IF ((L1.EQ.L2).AND.(M1.LT.M2)) GO TO 100
      RETURN
  100 M1TEMP=M1
      M1=M2
      M2=M1TEMP
      QREORD=.TRUE.
      RETURN
C
C -->
      ENTRY REORDR(MM,M1,M2,LL,L1,L2)
      MM=MMSAV
      M1=M1SAV
      M2=M2SAV
      LL=LLSAV
      L1=L1SAV
      L2=L2SAV
      RETURN
      END
