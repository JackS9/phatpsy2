      SUBROUTINE GENCGC(CGC,NCGC,CGCSCR,LLMXP2,YLMNRM,NYLM,FACT,NFACT,
     X                  QREALY)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     GENCGC...
C
C        THIS ROUTINE GENERATES ALL THE NECESSARY CLEBSCH-GORDON COEF-
C     FICIENTS FOR A MAXIMUM VALUE OF L BY MEANS OF A TWO-FOLD RECUR-
C     SION.  THEIR STORAGE AND RETRIEVAL ARE DETERMINED BY 'NDXCGC'.
C     THE COEFFICIENTS ARE DEFINED BY FRANK HARRIS AS FOLLOWS:
C
C        Y(L1,M1)*Y(L2,M2) = SUM(LL,MM) C(MM,M1,M2;LL,L1,L2)*Y(LL,MM)
C
C     VARIABLE DEFINITIONS:
C
C        CGC(*)........ ARRAY CONTAINING C-G COEFFICIENTS CORRESPONDING
C                       TO COMPLEX Y(L,M)'S.  RETRIEVED AND CONVERTED
C                       (IF DESIRED) TO REAL FORM BY 'CGCOEF'.
C        NCGC.......... =NDXCG(0,LMAX,-LMAX,2*LMAX,LMAX,LMAX), DIMENSION
C                       OF CGC(*).
C        CGCSCR(*,1)... A SCRATCH ARRAY USED FOR TEMPORARY STORAGE OF
C                       THE LAST LLMXP2 COEFFICIENTS IN THE FIRST PART
C                       OF THE RECURSION.
C        CGCSCR(*,2)... SIMILAR TO CGCSCR(*,1) BUT FOR THE SECOND PART
C                       OF THE RECURSION.
C        LLMXP2........ =2*LMAX+2, DIMENSION OF CGCSCR(*,*).
C        LMAX.......... MAXIMUM L-VALUE.
C        YLMNRM(*)..... NORMALIZATION CONSTANTS FOR THE Y(L,M)'S.
C        NYLM.......... =(LLMXP2*(LLMXP2-1))/2, DIMENSION OF YLMNRM(*).
C        FACT(N)....... =(N-1)-FACTORIAL.
C        NFACT......... =4*LMAX+2, DIMENSION OF FACT(*).
C        QREALY........ =T --> REAL Y(L,M)'S WILL BE ASSUMED.
C
C     ROUTINES CALLED:  NDXCGC(NDXCG); DSQRT, DFLOAT, MIN0,
C                       MAX0, MOD, IABS
C
C.......................................................................
C
C     WRITTEN:     AUGUST 12, 1975
C
C          BY:     JACK A. SMITH
C                  QUANTUM THEORY PROJECT
C                  UNIVERSITY OF FLORIDA
C                  GAINESVILLE, FLORIDA
C
C     REFERENCE:   COMPUTATION METHODS OF QUANTUM CHEMISTRY. PT I.
C                  BY FRANK HARRIS (UNIV. OF UTAH).
C
C     SUBORDINATE ROUTINES:  NDXCGC(NDXCG,NDXMAX), CHKCGC, ORDER(REORDR)
C                            CGCOEF, PUTCGC, GENFAC
C
C-----------------------------------------------------------------------
      DIMENSION CGC(NCGC),CGCSCR(LLMXP2,2),YLMNRM(NYLM),FACT(NFACT)
      DATA ZERO/0.D0/,ONE/1.D0/,TWO/2.D0/
      PI=TWO*DATAN2(ONE,ZERO)
      FOURPI=TWO*TWO*PI
      LLMXP1=LLMXP2-1
      LMXP1=LLMXP2/2
      LMAX=LMXP1-1
C-----------------------------------------------------------------------
C
C     RESERVE CGC(1)=0.D0 FOR COEFFICIENTS DETERMINED TO BE ZERO BY
C     'CHKCGC' (CALLED BY 'NDXCGC').
C
C     RESERVE CGC(2)=SQRT(1/(4*PI)) FOR THE SPECIAL CASE OF C(M1,M1,0;
C     L1,L1,0). DETERMINED BY 'NDXCGC'.
C
C     RESERVE CGC(3) FOR THOSE COEFFICIENTS WHOSE INDICES ARE GREATER
C     THAN NCGC (NOT STORED).  CHECKED BY 'NDXCGC'.
C
C-----------------------------------------------------------------------
      CGC(1)=ZERO
      CGC(2)=ONE/DSQRT(FOURPI)
C-----------------------------------------------------------------------
C
C     GENERATE THE NORMALIZATION CONSTANTS FOR THE Y(L,M)'S.
C     WHERE YLMNRM(LM) = NORMALIZATION CONSTANT OF Y(L,M) AND
C     LM = L*(L+1)/2+IABS(M)+1.
C     (NOT USED HERE BUT SOMETIMES NEEDED ALONG WITH CG-COEFFICIENTS).
C
C-----------------------------------------------------------------------
      DO 20 IL=1,LLMXP1
      LL=IL*(IL-1)/2
      DO 10 IM=1,IL
      M=IM-1
      LM=LL+IM
      AREA=FOURPI
      IF (QREALY.AND.(M.NE.0)) AREA=AREA/TWO
      YLMNRM(LM)=DSQRT((2*IL-1)*FACT(IL-M)/(AREA*FACT(IL+M)))
   10 CONTINUE
   20 CONTINUE
C-----------------------------------------------------------------------
C
C     LOOP OVER MM,M1,M2 SUCH THAT MM GE M1 GE M2 GE 0. THIS RESULTS
C     IN THE FOLLOWING RESTRICTIONS FOR A MAXIMUM VALUE OF L (LMAX):
C
C             1)  0 LE MM LE 2*LMAX
C             2)  MM/2 LE M1 LE MIN(MM,LMAX)
C             3)  M2=MM-M1
C
C     INITIALIZE CGCSCR(*,1) TO 'ZERO'.
C
C     THE COEFFICIENTS C(MM,M1,M2;MM,M1,M2) ARE COMPUTED EXPLICITLY.
C
C-----------------------------------------------------------------------
      DO 160 IMM=1,LLMXP1
      MM=IMM-1
      M1MN=(MM+1)/2+1
      M1MX=MIN0(MM,LMAX)+1
      DO 150 IM1=M1MN,M1MX
      M1=IM1-1
      M2=MM-M1
      DO 30 I=1,LLMXP2
         CGCSCR(I,1)=ZERO
   30 CONTINUE
      IF (M2.NE.0) GO TO 40
      CG=CGC(2)
      GO TO 50
   40 CG=FACT(1+MM)/(FACT(1+M1)*FACT(1+M2))* DSQRT(FACT(2*M1+2)*FACT(2*
     X     M2+2)/(FOURPI*FACT(2*MM+2)))
      CGC(NDXCG(MM,M1,M2,MM,M1,M2))=CG
   50 CGCSCR(IMM,1)=CG
C-----------------------------------------------------------------------
C
C     LOOP THROUGH L1 WITH THE FOLLOWING RESTRICTIONS:
C
C             1)  M1 LE L1 LE 2*LMAX
C             2)  IF MM GT LMAX OR M2=0 THEN L1 LE LMAX
C
C     INITIALIZE CGCSCR(*,2) TO 'ZERO'.
C
C-----------------------------------------------------------------------
      L1MN=IM1
      L1MX=LLMXP1
      IF ((MM.GT.LMAX).OR.(M2.EQ.0)) L1MX=LMXP1
      DO 140 IL1=L1MN,L1MX
      L1=IL1-1
      DO 60 I=1,LLMXP2
         CGCSCR(I,2)=ZERO
   60 CONTINUE
C-----------------------------------------------------------------------
C
C     LOOP THROUGH LL WITH THE FOLLOWING RESTRICTIONS:
C
C             1)  L1 NE M1
C             2)  L1+L2+LL  MUST BE EVEN
C             3)  LL GE MM
C             4)  L1-M2 LE LL LE MIN(L1+M2,2*LMAX)
C             5)  L2=M2
C             6)  M2 NE 0
C
C     THE COEFFICIENTS C(MM,M1,M2;LL,L1,M2) ARE COMPUTED FROM THE
C     COEFFICIENTS C(MM,M1,M2;LL-1,L1-1,M2), C(MM,M1,M2;LL+1,L1-1,M2),
C     AND C(MM,M1,M2;LL,L1-2,M2).
C
C     SAVE COEFFICIENTS IN CGCSCR(*,1) AND CGCSCR(*,2).
C
C-----------------------------------------------------------------------
      IF (L1.EQ.M1) GO TO 100
      LLMN=MAX0(MM,L1-M2)
      LLMN=LLMN+MOD(LLMN+L1+M2,2)+1
      LLMX=MIN0(IL1+M2,LLMXP1)
      LLMX=LLMX-MOD(LLMN+LLMX,2)
      IF (LLMN.GT.LLMX) GO TO 110
      DO 90 ILL=LLMN,LLMX,2
      LL=ILL-1
      IF (M2.NE.0) GO TO 70
      CG=CGC(2)
      GO TO 80
   70 CG=(DSQRT(DFLOAT((LL-MM)*(LL+MM))/DFLOAT((2*LL-1)*(2*LL+1)))*
     X     CGCSCR(ILL-1,1)+ DSQRT(DFLOAT((LL-MM+1)*(LL+MM+1))/DFLOAT((2*
     X     LL+1)*(2*LL+3)))* CGCSCR(ILL+1,1)- DSQRT(DFLOAT((L1-M1-1)*
     X     (L1+M1-1))/DFLOAT((2*L1-3)*(2*L1-1)))* CGCSCR(ILL,1))*
     X     DSQRT(DFLOAT((2*L1-1)*(2*L1+1))/DFLOAT((L1-M1)*(L1+M1)))
C     NOTE: Use L2TMP to avoid pass-by-reference argument aliasing with M2
C     when ORDER is called within NDXCG.
      L2TMP=M2
      CGC(NDXCG(MM,M1,M2,LL,L1,L2TMP))=CG
   80 CGCSCR(ILL,1)=CG
      CGCSCR(ILL,2)=CG
   90 CONTINUE
C-----------------------------------------------------------------------
C
C     LOOP THROUGH L2 WITH THE FOLLOWING RESTRICTIONS:
C
C             1)  M2 LT L2 LE LMAX+L1
C             2)  IF MM OR L1 GT LMAX THEN L2 LE LMAX
C             3)  IF M1=M2 THEN L2 LE L1
C
C-----------------------------------------------------------------------
      GO TO 110
  100 CGCSCR(IMM,2)=CGCSCR(IMM,1)
  110 L2MN=M2+2
      L2MX=LMXP1+L1
      IF ((MM.GT.LMAX).OR.(L1.GT.LMAX)) L2MX=LMXP1
      IF (M2.EQ.M1) L2MX=MIN0(L2MX,IL1)
      IF (L2MN.GT.L2MX) GO TO 140
      DO 130 IL2=L2MN,L2MX
      L2=IL2-1
C-----------------------------------------------------------------------
C
C     LOOP THROUGH LL WITH THE FOLLOWING RESTRICTIONS:
C
C             1)  L1+L2+LL  MUST BE EVEN
C             2)  LL GE MM
C             3)  !L1-L2! LE LL LE MIN(L1+L2,2*LMAX,2*LMAX+L1-L2)
C
C     THE COEFFICIENTS C(MM,M1,M2;LL,L1,L2) ARE COMPUTED FROM THE
C     COEFFICIENTS C(MM,M1,M2;LL-1,L1,L2-1), C(MM,M1,M2;LL+1,L1,L2-1)
C     AND C(MM,M1,M2;LL,L1,L2-2).
C
C     SAVE COEFFICIENTS IN CGCSCR(*,2).
C
C-----------------------------------------------------------------------
      LLMN=MAX0(MM,IABS(L2-L1))
      LLMN=LLMN+MOD(LLMN+L1+L2,2)+1
      LLMX=MIN0(IL2+L1,LLMXP1,LLMXP1+L1-L2)
      LLMX=LLMX-MOD(LLMN+LLMX,2)
      DO 120 ILL=LLMN,LLMX,2
      LL=ILL-1
      CG=(DSQRT(DFLOAT((LL-MM)*(LL+MM))/DFLOAT((2*LL-1)*(2*LL+1)))*
     X     CGCSCR(ILL-1,2)+ DSQRT(DFLOAT((LL-MM+1)*(LL+MM+1))/DFLOAT((2*
     X     LL+1)*(2*LL+3)))* CGCSCR(ILL+1,2)- DSQRT(DFLOAT((L2-M2-1)*
     X     (L2+M2-1))/DFLOAT((2*L2-3)*(2*L2-1)))* CGCSCR(ILL,2))*
     X     DSQRT(DFLOAT((2*L2-1)*(2*L2+1))/DFLOAT((L2-M2)*(L2+M2)))
      CGC(NDXCG(MM,M1,M2,LL,L1,L2))=CG
      CGCSCR(ILL,2)=CG
  120 CONTINUE
  130 CONTINUE
  140 CONTINUE
  150 CONTINUE
  160 CONTINUE
      RETURN
      END
