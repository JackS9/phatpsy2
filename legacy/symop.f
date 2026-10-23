      SUBROUTINE SYMOP(ISYM,IXYZ,STEVEC,VCOEF,L,LV,D,SCRAT,
     X                 NSTO,MSTO,NORB,ISPIN,NVTERM,
     X                 LMAX,LLMXP1)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     SYMOP...
C
C        THIS ROUTINE PERFORMS THE SYMMETRY OPERATIONS NECESSARY TO
C     BRING EQUIVALENT ATOMS INTO EACH OTHER.
C
C     VARIABLE DEFINITIONS:
C
C        ISYM........ = N --> N-FOLD ROTATION.
C                     = 1 --> IDENTITY.
C                     = 0 --> IDENTITY.
C                     =-1 --> PURE REFLECTION.
C                     =-2 --> INVERSION.
C                     =-N --> N-FOLD ROTO-REFLECTION.
C        IXYZ........ A PNEUMONIC VECTOR REPRESENTING THE AXIS OF
C                     ROTATION OR THE PERPENDICULAR PLANE OF REFLECTION.
C        STEVEC(*)... ATOMIC ORBITAL COEFFICIENTS.
C        VCOEF(*).... MODEL POTENTIAL COEFFICIENTS.
C        D(*)........ ROTATION COEFFICIENTS.
C        SCRAT(*).... SCRATCH ARRAY.
C        MSTO........ NUMBER OF STO'S IN BASIS (INCLUDING ML-VALUES).
C        NORB........ NUMBER OF ATOMIC ORBITALS.
C        ISPIN....... =1 FOR RESTRICTED, =2 FOR UNRESTRICTED CASE.
C        NVTERM...... NUMBER OF TERMS IN MODEL POTENTIAL.
C        L(*)........ L-QUANTUM NUMBERS IN BASIS.
C        LV(*)....... L-TYPE OF MODEL POTENTIAL TERMS.
C        NSTO........ NUMBER OF ST0'S (NOT INCLUDING ML-VALUES).
C        LMAX........ MAXIMUM L-VALUE.
C        LLMXP1...... =2*LMAX+1.
C        L3MX........ DIMENSION OF D(*).
C
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM(28)(=L3MX)
C                 SETS - APARM(9)(=ALPHA),  IPARM(10)(=BETA)
C
C     ROUTINES CALLED:  GENDC, DCOEF, DCOPY, BOMB, QZERO; IABS, DFLOAT,
C                       DATAN2, DSIN, DCOS, MOD
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (IPARM(28),L3MX)
      EQUIVALENCE (APARM(9),ALPHA),   (APARM(10),BETA)
      DIMENSION STEVEC(MSTO,NORB,ISPIN),L(NSTO),
     X          D(L3MX,2),SCRAT(LLMXP1)
      DIMENSION VCOEF(NVTERM),LV(NVTERM)
      DATA ZERO/0.0D0/,ONE/1.0D0/,TWO/2.0D0/
      IF ((ISYM.EQ.0).OR.(ISYM.EQ.1)) RETURN
      PI=DATAN2(ONE,ZERO)*TWO
      NFOLD=ISYM
      ANGLE=TWO*PI/DFLOAT(IABS(NFOLD))
      QINVRT=.FALSE.
C     REPLACE ARITHMETIC IF WITH MODERN IF-ELSE IF STRUCTURE
      IF (ISYM .LT. 0) THEN
         GO TO 1
      ELSE IF (ISYM .EQ. 0) THEN
         GO TO 50
      ELSE
         GO TO 2
      END IF
C-----------------------------------------------------------------------
C
C     REFLECTION.
C
C-----------------------------------------------------------------------
    1 ANGLE=ANGLE+PI
      QINVRT=.TRUE.
C-----------------------------------------------------------------------
C
C     ROTATION.
C
C-----------------------------------------------------------------------
    2 CONTINUE
      IF (IXYZ.EQ.1) GO TO 3
      IF (IXYZ.EQ.10) GO TO 4
      IF (IXYZ.EQ.100) GO TO 5
      IF (IXYZ.NE.0) CALL BOMB(4)
    3 ALPHA=ANGLE
      BETA=ZERO
      GAMMA=ZERO
      GO TO 6
    4 ALPHA=PI/TWO
      BETA=ANGLE
      GAMMA=-PI/TWO
C     CALL BOMB(4)
      GO TO 6
    5 ALPHA=ZERO
      BETA=ANGLE
      GAMMA=ZERO
    6 CALL GENDC(D,LMAX)
      PARITY=ONE
      DO 40 ISP=1,ISPIN
      DO 30 IORB=1,NORB
      JSTO=1
      DO 20 ISTO=1,NSTO
      LQ=L(ISTO)
      IF (QINVRT) PARITY=1-2*MOD(LQ,2)
      LLQP1=2*LQ+1
      CALL DCOPY(STEVEC(JSTO,IORB,ISP),LLQP1,SCRAT)
      DO 10 IMLQ=1,LLQP1
      MLQ=IMLQ-LQ-1
      COEF=ZERO
      DO 9 ISIG=1,LLQP1
      ISIGMA=ISIG-LQ-1
    9 COEF=COEF+DCOEF(D,LQ,MLQ,ISIGMA)*SCRAT(ISIG)
      STEVEC(JSTO,IORB,ISP)=PARITY*COEF
      JSTO=JSTO+1
   10 CONTINUE
   20 CONTINUE
   30 CONTINUE
   40 CONTINUE
      IF (NVTERM.LE.2) RETURN
      MV=1
      DO 48 NV=1,NVTERM
      IF (MV.GT.NVTERM) GO TO 48
      LQ=0
      IF (LV(MV).LT.0) LQ=-LV(MV)
      IF (QINVRT) PARITY=1-2*MOD(LQ,2)
      LLQP1=2*LQ+1
      CALL DCOPY(VCOEF(MV),LLQP1,SCRAT)
      DO 45 IMLQ=1,LLQP1
      MLQ=IMLQ-LQ-1
      COEF=ZERO
      DO 43 ISIG=1,LLQP1
      ISIGMA=ISIG-LQ-1
   43 COEF=COEF+DCOEF(D,LQ,MLQ,ISIGMA)*SCRAT(ISIG)
      VCOEF(MV)=PARITY*COEF
      MV=MV+1
   45 CONTINUE
   48 CONTINUE
   50 RETURN
      END
