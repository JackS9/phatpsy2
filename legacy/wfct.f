      SUBROUTINE WFCT(W,S,NMX,NMNM1,N1,ETA1,N2,ETA2)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     WFCT...
C
C        THIS ROUTINE RETURNS AN ARRAY OF W-FUNCTIONS
C
C             W(I-1;N1,N2,ETA1,ETA2)  I=1,...,NMNM1
C             NMNM1 = MIN(N1,N2) - 1, NMX = MAX(N1,N2)
C
C     USED IN THE COMPUTATION OF ONE-CENTER TWO-ELECTRON INTEGRALS
C     BETWEEN STO'S.  THEY ARE GENERATED DIRECTLY FROM THE S-FUNCTIONS
C     AS FOLLOWS:
C
C             W(NU;N1,N2,ETA1,ETA2) = S(N1+N2,N2-NU-1;ETA1/ETA2)
C                                    /ETA2**(N1+N2+1)
C                                   + S(N1+N2,N1-NU-1;ETA2/ETA1)
C                                    /ETA1**(N1+N2+1)
C
C     ROUTINES CALLED:  SFCT
C
C-----------------------------------------------------------------------
      DIMENSION W(NMNM1),S(NMX)
      NN=N1+N2
      CALL SFCT(S,NN,N2,ETA1/ETA2)
      DO 10 NU=1,NMNM1
      W(NU)=S(N2-NU+1)/ETA2**(NN+1)
   10 CONTINUE
      CALL SFCT(S,NN,N1,ETA2/ETA1)
      DO 20 NU=1,NMNM1
      W(NU)=W(NU)+S(N1-NU+1)/ETA1**(NN+1)
   20 CONTINUE
      RETURN
      END
