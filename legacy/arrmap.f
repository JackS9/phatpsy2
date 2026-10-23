      SUBROUTINE ARRMAP(NR)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     ARRMAP...
C
C        THIS ROUTINE WRITES (IUDUMP) THE ADDRESSES AND LENGTHS (IN HEX)
C     OF THE DYNAMICALLY ALLOCATED ARRAYS FOR THE CALL TO A GIVEN
C     ROUTINE THROUGH ASSARD(K).
C
C        NR     SUBROUTINE
C
C         1 ... CONTRL
C         2 ... ATOMIC
C         3 ... ALOOP
C         4 ... EWMO
C         5 ... SCF
C         6 ... BLOOP
C
C     COMMON USAGE:
C
C        /DYNAMC/  USES - LISTAR(*),IPTR(*),NNIA(*),ROUTIN(*)
C
C        /IODATA/  USES - IUNIT(8)(=IUDUMP)
C
C-----------------------------------------------------------------------
      CHARACTER*8 ROUTIN
      COMMON /DYNAMC/ LISTAR(500),IPTR(10),NNIA(10),NID(10),NXARG(10),
     X                ROUTIN(10),NROUT
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(8),IUDUMP)
CD----------------------------------------------------------------------
CD  FIXED CORE ALLOCATION...
CD
CD    IPT2=IPTR(NR)-1
CD    NARR=NNIA(NR)
CD    IPT1=IPT2+NARR
CD    WRITE (IUDUMP,1000) ROUTIN(NR),(LISTAR(IPT1+IPT),LISTAR(IPT2+IPT),
CD   X     IPT=1,NARR)
      RETURN
C1000 FORMAT('-DYNAMICALLY ALLOCATED ARRAYS FOR CALL TO ',A8,'...'/
CD   X       '0ADDRESS     LENGTH'/
CD   X       ' -------     ------'/
CD   X      (' ',Z8,4X,Z8))
      END
