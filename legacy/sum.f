      FUNCTION SUM(START,NTERMS,INCR)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SUM...
C
C        THIS FUNCTION RETURNS THE SUM OF EVENLY SPACED ELEMENTS IN AN
C     ARRAY.
C
C     DEFINITIONS:
C
C        START(*)... AN ARRAY OR SOME ARBITRARY STARTING PLACE IN CORE.
C        NTERMS..... THE NUMBER OF TERMS IN THE ARRAY TO BE ADDED.
C        INCR....... THE SPACE BETWEEN ELEMENTS TO BE ADDED, I.E., =1 IF
C                    CONTINUOUS ELEMENTS, =2 IF EVERY OTHER ONE, ETC.
C
C-----------------------------------------------------------------------
      DIMENSION START(1)
      DATA ZERO/0.D0/
      SUM=ZERO
      IEND=1+INCR*(NTERMS-1)
      DO 10 I=1,IEND,INCR
      SUM=SUM+START(I)
   10 CONTINUE
      RETURN
      END
