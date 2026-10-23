      FUNCTION NOBLE(NUCZ)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     NOBLE...
C
C        THIS FUNCTION RETURNS THE NUMBER OF ORBITALS WHICH MAKE UP THE
C     LARGEST NOBLE GAS CONFIGURATION WITHIN AN ATOM WITH THE ATOMIC
C     NUMBER NUCZ (NUCLEAR CHARGE, Z).
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(21),QCORE)
      NOBLE=0
      IF (QCORE) RETURN
      IF (NUCZ.LE.2) RETURN
      NOBLE=1
      IF (NUCZ.LE.10) RETURN
      NOBLE=5
      IF (NUCZ.LE.18) RETURN
      NOBLE=9
      IF (NUCZ.LE.36) RETURN
      NOBLE=18
      IF (NUCZ.LE.54) RETURN
      NOBLE=27
      IF (NUCZ.LE.86) RETURN
      NOBLE=43
      RETURN
      END
