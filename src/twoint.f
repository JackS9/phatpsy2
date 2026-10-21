      SUBROUTINE TWOINT(NTEI,N,L,ETA,S,W,INDEX,ANORM,CGC,QREALY,TEIBUF)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     TWOINT...
C
C        THIS ROUTINE GENERATES ALL THE NON-ZERO TWO-ELECTRON INTEGRAL
C     OVER STO'S ON A SINGLE CENTER ALONG WITH POINTERS TO THE PROPER
C     FOCK AND DENSITY MATRIX ELEMENTS AND INDEX-SYMMETRY WEIGHT FACTORS
C     FOR EACH POINTER.  EACH INTEGRAL REPRESENTS EIGHT EQUIVALENT
C     INTEGRALS.  THE INTEGRAL IS PASSED TO 'PUTTEI' VIA A CALL AND
C     THE POINTERS AND WEIGHTS ARE PASSED THROUGH COMMON /NDXTEI/.
C
C     THE INTEGRAL:
C
C        TEI = (AC!BD) = (CA!BD) = (AC!DB) = (CA!DB)     MULLIKEN
C            = (BD!AC) = (DB!AC) = (BD!CA) = (DB!CA)     NOTATION
C
C           (= <AB!CD> = <CB!AD> = <AD!CB> = <CD!AB>     DIRAC
C            = <BA!DC> = <DA!BC> = <BC!DA> = <DC!BA>)    NOTATION
C
C        WHERE
C
C        !C> = NORM(NC,ETAC)*R**(NC-1)*EXP(-ETAC*R)*Y(LC,MC)
C        NC = N(IC)
C        LC = L(IC)
C        -LC LE MC LE LC
C        ETAC = ETA(IC)
C        NORM(NC,ETAC) = ANORM(IC), STO NORMALIZATION CONSTANT.
C        Y(LC,MC) = NORMALIZED REAL SPHERICAL HARMONIC (REF. HARRIS).
C
C     THE POINTERS:
C
C        IAC = INDEX(MAX(IJA,IJC)) + MIN(IJA,IJC)
C              (ANALOGOUSLY FOR IBD,IAD,IBD,IAB,ICD)
C
C     THE WEIGHTS:
C
C        KAC = 2-D(A,C)
C        KBD = (2-D(B,D))*D'(AC,BD)
C        KAD = 1+D(A,D)*D'(B,C)
C        KBC = (1+D(B,C)*D'(A,D))*D'(AD,BC)
C        KAB = (1+D(A,B)*D'(C,D))*D'(A,C)*D'(B,D)
C        KCD = ((1+D(C,D)*D'(A,B))*D'(A,C)*D'(B,D))*D'(AB,CD)
C
C        WHERE D(A,C) = 1 IF IJA=IJC, = 0 OTHERWISE
C              D'(A,C) = 1 - D(A,C)
C              D(AC,BD) = 1 IF IAC=IBD, = 0 OTHERWISE
C              D'(AC,BD) = 1 - D(AC,BD)
C
C        (SEE 'FOKMAT' FOR THEIR USE.)
C
C
C     NOTE:  'IC' DOES NOT LOOP OVER ML-VALUES, BUT 'IJC' DOES.
C
C
C     OTHER DEFINITIONS:
C
C        NTEI...... NUMBER OF NON-ZERO INTEGRALS GENERATED.
C        S(*)...... S-FUNCTIONS (SEE 'SFCT').
C        W(*)...... W-FUNCTIONS (SEE 'WFCT').
C        NNMX...... =2*NMAX, DIMENSION OF S(*).
C        NNMXM1.... =NNMX-1, DIMENSION OF W(*).
C        NSTO...... NUMBER OF STO'S (NOT INCLUDING DIFFERENT ML-VALUES).
C        MSTO...... THE NUMBER OF STO'S (INCLUDING ML-VALUES).
C        INDEX(N).. =N*(N-1)/2, INDEXING ARRAY FOR SYMMETRY PACKING.
C        CGC(*).... CLEBSCH-GORDON COEFFICIENTS.
C        TEIBUF(*). TWO-ELECTRON INTEGRAL BUFFER.
C
C     ROUTINES CALLED:  PUTTEI, WFCT, CGCOEF; DSQRT, IABS, MAX0, MIN0,
C                       MOD
C
C     COMMON USAGE:
C
C        /PARMS/   USES - IPARM(5)(=MSTO),    IPARM(18)(=NNMX),
C                         IPARM(19)(=NNMXM1), IPARM(32)(=NSTO)
C
C        /TABLES/  USES - ZERO,HALFPI(8)(=FOURPI)
C
C        /NDXTEI/  SETS - IAC,KAC,IBD,KBD,IAD,KAD,IBC,KBC,
C                         IAB,KAB,ICD,KCD
C
C
C     RESTRICTION:  FACT(*) HAS BEEN DIMENSIONED TO HANDLE UP TO A
C                   MAXIMUM L-VALUE OF 5 AND A MAXIMUM N-VALUE OF 10.
C                   DIMENSION SHOULD BE MAX(4*LMAX+2,2*NMAX+2).
C
C.......................................................................
C
C     WRITTEN:     AUGUST 22, 1977
C
C          BY:     JACK A. SMITH
C                  QUANTUM THEORY PROJECT
C                  UNIVERSITY OF FLORIDA
C                  GAINESVILLE, FLORIDA
C
C     REFERENCE:   COMPUTATION METHODS OF QUANTUM CHEMISTRY. PT 1.
C                  BY FRANK HARRIS (UNIV. OF UTAH).
C
C     SUBORDINATE ROUTINES:  TEINDX(PUTTEI,GETTEI,DMPBUF), WFCT, SFCT,
C                            GENCGC, NDXCGC(NDXCG,NDXMAX), CHKCGC,
C                            ORDER(REORDR), ATOMIC, FOKMAT
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM( 5)(=MSTO),  IPARM(18)(=NNMX),
C                        IPARM(19)(=NNMXM1),IPARM(24)(=NCGC),
C                        IPARM(32)(=NSTO)
C
C        /IODATA/ USES - LENBUF
C
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (IPARM(5),MSTO),    (IPARM(18),NNMX),
     X            (IPARM(19),NNMXM1), (IPARM(24),NCGC),
     X            (IPARM(32),NSTO)
      COMMON /TABLES/ REALS(10),FACT(22),FFAC(19),BINOM(91),HALFPI(8),
     X                ZERO,HALF,ROOT(10),ROOTPI,CONST(10),CONVRT(10)
      EQUIVALENCE (HALFPI(8),FOURPI)
	  COMMON /IODATA/ IUNIT(20),LENBUF
      DIMENSION N(NSTO),       L(NSTO),
     X          ETA(NSTO),     S(NNMX),
     X          W(NNMXM1),     INDEX(MSTO),
     X          ANORM(NSTO),   CGC(NCGC),
     X          TEIBUF(LENBUF)
      COMMON /NDXTEI/ IAC,KAC,IBD,KBD,IAD,KAD,IBC,KBC,IAB,KAB,ICD,KCD
      NTEI=0
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER NA AND LA (IA).
C
C-----------------------------------------------------------------------
      JA=0
      DO 130 IA=1,NSTO
      NA=N(IA)
      LA=L(IA)
      LLAP1=2*LA+1
      ETAA=ETA(IA)
      SNORMA=ANORM(IA)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER NC AND LC (IC).
C
C-----------------------------------------------------------------------
      JC=0
      DO 120 IC=1,IA
      NC=N(IC)
      LC=L(IC)
      LLCP1=2*LC+1
      ETAC=ETA(IC)
      N1=NA+NC
      ETA1=ETAA+ETAC
      SNORMC=ANORM(IC)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER NB AND LB (IB).
C
C-----------------------------------------------------------------------
      JB=0
      DO 110 IB=1,IA
      NB=N(IB)
      LB=L(IB)
      LLBP1=2*LB+1
      ETAB=ETA(IB)
      SNORMB=ANORM(IB)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER ND AND LD (ID).
C
C-----------------------------------------------------------------------
      JD=0
      DO 100 ID=1,IB
      ND=N(ID)
      LD=L(ID)
      LLDP1=2*LD+1
      ETAD=ETA(ID)
      N2=NB+ND
      ETA2=ETAB+ETAD
      SNORMD=ANORM(ID)
C-----------------------------------------------------------------------
C
C     IF LA+LB+LC+LD IS NOT EVEN SKIP THE M-LOOPS SINCE ALL SUCH INTE-
C     GRALS ARE ZERO.
C
C-----------------------------------------------------------------------
      IF (MOD(LA+LB+LC+LD,2).NE.0) GO TO 90
C-----------------------------------------------------------------------
C
C     DETERMINE THE LIMITS ON THE MU-LOOP OVER THE CLEBSCH-GORDON COEF-
C     FICIENTS AND RADIAL FUNCTIONS LATER.  IF THE LOWER LIMIT IS
C     HIGHER THAT THE UPPER LIMIT SKIP THE REMAINING LOOPS.
C
C-----------------------------------------------------------------------
      LSUMMN=MIN0(LA+LC,LB+LD)
      LDIFMX=MAX0(IABS(LA-LC),IABS(LB-LD))
      IMUMN=LDIFMX+1
      IMUMX=LSUMMN+1
      IF (IMUMN.GT.IMUMX) GO TO 90
C-----------------------------------------------------------------------
C
C     COMPUTE THE RADIAL PARTS (W-FUNCTIONS) AND NORMALIZATION CONSTANT
C     FOR ALL THE INTEGRALS TO BE COMPUTED IN THE FOLLOWING M-LOOPS.
C
C-----------------------------------------------------------------------
      NMX=MAX0(N1,N2)
      NMNM1=MIN0(N1,N2)-1
      CALL WFCT(W,S,NMX,NMNM1,N1,ETA1,N2,ETA2)
      STNORM=FOURPI*SNORMA*SNORMB*SNORMC*SNORMD
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER MA (IMA).
C
C-----------------------------------------------------------------------
      MASIGN=-1
      DO 80 IMA=1,LLAP1
      MA=IMA-LA-1
      IABSMA=IABS(MA)
      IF (MA.GE.0) MASIGN=1
      IJA=JA+IMA
      IIA=INDEX(IJA)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER MC (IMC).
C
C-----------------------------------------------------------------------
      IMCMX=LLCP1
      IF (IC.EQ.IA) IMCMX=IMA
      MCSIGN=-1
      DO 70 IMC=1,IMCMX
      MC=IMC-LC-1
      IABSMC=IABS(MC)
      IF (MC.GE.0) MCSIGN=1
      MSIGN=MASIGN*MCSIGN
      IJC=JC+IMC
      IIC=INDEX(IJC)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER MB (IMB).
C
C-----------------------------------------------------------------------
      IMBMX=LLBP1
      IF (IB.EQ.IA) IMBMX=IMA
      MBSIGN=-1
      DO 60 IMB=1,IMBMX
      MB=IMB-LB-1
      IABSMB=IABS(MB)
      IF (MB.GE.0) MBSIGN=1
      IJB=JB+IMB
      IIB=INDEX(IJB)
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER MD (IMD).
C
C-----------------------------------------------------------------------
      IF ((IJA.EQ.IJB).AND.(ID.GT.IC)) GO TO 60
      IMDMX=LLDP1
      IF (ID.EQ.IB) IMDMX=IMB
      IF ((IJA.EQ.IJB).AND.(IC.EQ.ID)) IMDMX=MIN0(IMDMX,IMC)
      MDSIGN=-1
      DO 50 IMD=1,IMDMX
      MD=IMD-LD-1
      IABSMD=IABS(MD)
      IF (MD.GE.0) MDSIGN=1
      IJD=JD+IMD
      IID=INDEX(IJD)
C-----------------------------------------------------------------------
C
C     DETERMINE THE LIMITS ON THE SIGMA-LOOP.  IF THE LOWER LIMIT
C     IS HIGHER THAN THE UPPER LIMIT OR IF THE SIGNS DON'T MATCH
C     SKIP THE REMAINING LOOPS.
C
C-----------------------------------------------------------------------
      MSUMMN=MIN0(IABSMA+IABSMC,IABSMB+IABSMD)
      MDIFMX=MAX0(IABS(IABSMA-IABSMC),IABS(IABSMB-IABSMD))
      MSKIP=MSUMMN-MDIFMX
C     REPLACE ARITHMETIC IF WITH MODERN IF-ELSE IF STRUCTURE
      IF (MSKIP .LT. 0) THEN
         GO TO 50
      ELSE IF (MSKIP .EQ. 0) THEN
         GO TO 10
      ELSE
         GO TO 20
      END IF
   10 MSKIP=1
   20 ISIGMN=MDIFMX+1
      IF ((MSIGN.NE.MBSIGN*MDSIGN).AND.(IABSMB.NE.IABSMD)) GO TO 50
C-----------------------------------------------------------------------
C
C     SET THE POINTERS.
C
C-----------------------------------------------------------------------
      IAC=IIA+IJC
      IBD=IIB+IJD
      IAD=IIA+IJD
      IBC=IIB+IJC
      IF (IJC.GT.IJB) IBC=IIC+IJB
      IAB=IIA+IJB
      ICD=IIC+IJD
      IF (IJD.GT.IJC) ICD=IID+IJC
C-----------------------------------------------------------------------
C
C     EVALUATE THE DELTA FUNCTIONS.
C
C-----------------------------------------------------------------------
      IDAC=0
      IF (IJA.EQ.IJC) IDAC=1
      IDBD=0
      IF (IJB.EQ.IJD) IDBD=1
      IDAD=0
      IF (IJA.EQ.IJD) IDAD=1
      IDBC=0
      IF (IJB.EQ.IJC) IDBC=1
      IDAB=0
      IF (IJA.EQ.IJB) IDAB=1
      IDCD=0
      IF (IJC.EQ.IJD) IDCD=1
C-----------------------------------------------------------------------
C
C     DETERMINE THE WEIGHT FACTORS.
C
C-----------------------------------------------------------------------
      KAC=2-IDBD
      KBD=2-IDAC
      IF (IAC.EQ.IBD) KBD=0
      KAD=1+IDAD-IDAD*IDBC
      KBC=1+IDBC-IDBC*IDAD
      IF (IAD.EQ.IBC) KBC=0
      KAB=1+IDAB-IDAB*IDCD-IDAC-IDBD+IDAC*IDBD-IDAB*IDAC-IDAB*IDBD+2*
     X     IDAB*IDBC*IDCD
      KCD=1+IDCD-IDCD*IDAB-IDAC-IDBD+IDAC*IDBD-IDCD*IDAC-IDCD*IDBD+2*
     X     IDCD*IDBC*IDAB
      IF (IAB.EQ.ICD) KCD=0
C-----------------------------------------------------------------------
C
C     MU-LOOP:  LOOP OVER THE CLEBSCH-GORDON COEFFICIENTS AND RADIAL
C     FUNCTIONS TO COMPUTE INTEGRALS.
C
C-----------------------------------------------------------------------
      SUMMU=ZERO
      DO 40 IMU=IMUMN,IMUMX,2
      MU=IMU-1
      TUMUP1=2*MU+1
      ISIGMX=MSUMMN+1
      IF (MU.LT.MSUMMN) ISIGMX=ISIGMN
      DO 30 ISIG=ISIGMN,ISIGMX,MSKIP
      ISIGMA=MSIGN*(ISIG-1)
      SUMMU=SUMMU+CGCOEF(CGC,ISIGMA,MA,MC,MU,LA,LC,QREALY)
     X           *CGCOEF(CGC,ISIGMA,MB,MD,MU,LB,LD,QREALY)
     X           *W(IMU)/TUMUP1
   30 CONTINUE
   40 CONTINUE
C-----------------------------------------------------------------------
C
C     PASS THE NORMALIZED TWO-ELECTRON INTEGRAL ALONG WITH ASSOCIATED
C     POINTERS AND THEIR WEIGHT FACTORS TO THE OUTPUT ROUTINE.
C
C-----------------------------------------------------------------------
      IF (SUMMU.EQ.ZERO) GO TO 50
      NTEI=NTEI+1
      TEI=SUMMU*STNORM
      CALL PUTTEI(TEI,TEIBUF)
C-----------------------------------------------------------------------
C
C     END LOOPS OVER  MA, MC, MB, AND MD (IMA, IMC, IMB, IMD).
C
C-----------------------------------------------------------------------
   50 CONTINUE
   60 CONTINUE
   70 CONTINUE
   80 CONTINUE
C-----------------------------------------------------------------------
C
C     END LOOPS OVER NA,LA, NC,LC, NB,LB, AND ND,LD (IA,IC,IB, AND ID).
C
C-----------------------------------------------------------------------
   90 JD=JD+LLDP1
  100 CONTINUE
      JB=JB+LLBP1
  110 CONTINUE
      JC=JC+LLCP1
  120 CONTINUE
      JA=JA+LLAP1
  130 CONTINUE
C-----------------------------------------------------------------------
C
C     FLAG THE END OF THE INTEGRAL LIST WITH A ZERO POINTER.
C
C-----------------------------------------------------------------------
      IAC=0
      CALL PUTTEI(TEI,TEIBUF)
      RETURN
      END
