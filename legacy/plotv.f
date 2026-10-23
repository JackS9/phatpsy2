      SUBROUTINE PLOTV(LV,MV,VEXP,VCOEF,NVTERM,IW)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION LV(NVTERM),MV(NVTERM),VEXP(NVTERM),VCOEF(NVTERM)
      CHARACTER*1 LINE(50)

      SCALE = -5.0/VCOEF(1)
      R = -4.0
      WRITE(IW,1000)

      DO 500 I=1,41
      VN = 0.0
      VE = 0.0
      VT = 0.0
      RR = DABS(R)

      DO 100 J=1,50
      LINE(J) = ' '
      IF (RR.LT.0.1) LINE(J) = '-'     
      IF (RR.LT.0.1 .AND. MOD(J,5).EQ.0) LINE(J) = '+'     
  100 CONTINUE

      LINE(25) = '|'
      IF (RR.LT.0.1) LINE(25) = '+'
      IF (RR.LT.0.1) GOTO 400

      VN = SCALE*VCOEF(1)/RR
      K = INT(VN+.5)
      IF (VN.LT.0.0) K = INT(VN-.5)
      IF (K.LE.-25) K = -24
      IF (K.GT.25) K = 25
      LINE(25+K) = '+'
      
      DO 200 NV=2,NVTERM

      IF (LV(NV).GE.0) THEN
         V = VCOEF(NV)*RR**(LV(NV)-1)*DEXP(-VEXP(NV)*RR)
      ELSE
         IF (MV(NV).EQ.0) THEN
            SGN = (-1)**MV(NV)*R/RR
            V = SGN*VCOEF(NV)*RR**(-LV(NV))*DEXP(-VEXP(NV)*RR)
         ENDIF
      ENDIF

      V = SCALE*V
      K = INT(V+.5)
      IF (V.LT.0.0) K = INT(V-.5)
      IF (K.LE.-25) K = -24
      IF (K.GT.25) K = 25
      LINE(25+K) = '.'

      VE = VE + V
  200 CONTINUE

      VT = VN + VE

      K = INT(VE+.5)
      IF (VE.LT.0.0) K = INT(VE-.5)
      IF (K.LE.-25) K = -24
      IF (K.GT.25) K = 25
      LINE(25+K) = '-'

      K = INT(VT+.5)
      IF (VT.LT.0.0) K = INT(VT-.5)
      IF (K.LE.-25) K = -24
      IF (K.GT.25) K = 25
      LINE(25+K) = 'o'

  400 CONTINUE
      VT = VT/SCALE
      WRITE (IW,2000) R,LINE,VT
      R = R + 0.2
  500 CONTINUE

      RETURN

 1000 FORMAT(/' Model Potential:  V(o) = N/R(+) + TF_Screening(-) ...'/)
 2000 FORMAT(' ',F4.1,2X,50A1,E10.2)
      END
