      SUBROUTINE ERR243(IERTCD,IERNO,D,I)
C-----------------------------------------------------------------------
C
C     ERR243...
C
C        THIS ROUTINE IS A MODIFICATION TO THE STANDARD FIX-UP FOR THE
C     ERROR 243.  THE FIX-UP HERE SETS 0.0**0 TO 1.0 INSTEAD OF 0.0.
C
C          DA = D**I
C
C-----------------------------------------------------------------------
      REAL*8 D,ONE
      DATA ONE/1.0D0/
      IERTCD=1
      IERNO=243
      D=ONE
      I=0
      RETURN
      END
