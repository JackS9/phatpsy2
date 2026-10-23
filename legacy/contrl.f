      SUBROUTINE CONTRL(INDEX,KREC,CGC,YLMNRM,S,W,OMEGAI,OMEGAJ,A,B,D,
     X                  BUFFER,OVLP,EWMAT,POPUL,COORD,LABEL,CHARGE,
     X                  SCRAT,IATOMX,NX,LX,MLX,ETAX,ANORMX,EVALX,OCCX,
     X                  EVECX,CORE)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      CHARACTER*2 LABEL
C-----------------------------------------------------------------------
C
C     CONTRL...
C
C        THIS IS THE MAIN DRIVING ROUTINE OF THE PROGRAM.   CONSTANTS
C     ARE SET, BASIS DIMENSION INFORMATION IS READ IN, THE MAIN INPUT
C     ROUTINE (ATOMIC) IS CALLED, THE INTEGRAL LOOP (ALOOP) IS CALLED,
C     THE EWMO ROUTINE IS CALLED, THE ATOMIC SCF ROUTINE IS CALLED, AND
C     CONVERGENCE CHECKS ON THE SELF-CONSISTENT CYCLE ARE MADE.
C     .
C     .
C     .
C
C  ** MXNORB, MXMSTO, MXNSTO CAN BE ELIMINATED IF INDEX(*) IS
C     REPLACED BY AN ARITHMETIC STATEMENT FUNCTION.
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(1),QRSTRT),
     X            (QPARM(3),QNWBAS),  (QPARM(4),QOPEN),
     X            (QPARM(5),QFIRST),  (QPARM(6),QLAST),
     X            (QPARM(7),QDEBUG),  (QPARM(8),QSEPAT),
     X            (QPARM(9),QCMPTD),  (QPARM(10),QEWMO),
     X            (QPARM(22),QPRINT), (QPARM(23),QAUDMP),
     X            (QPARM(24),QATOM),  (QPARM(28),QMNPRT),
     X            (QPARM(29),QCHECK), (QPARM(31),QGEOM),
     X            (QPARM(35),QTRACE)
      EQUIVALENCE (IPARM(1),NATOM),   (IPARM(2),NELEM),
     X            (IPARM(5),MSTO),    (IPARM(6),NORB),
     X            (IPARM(7),IORBT),   (IPARM(8),NORBT),
     X            (IPARM(9),N2ORBT),  (IPARM(10),IATOM),
     X            (IPARM(12),MXMSTO),
     X            (IPARM(13),N2ATOM), (IPARM(14),MXNORB),
     X            (IPARM(15),MXNSTO), (IPARM(16),MDIFAT),
     X            (IPARM(18),NNMX),   (IPARM(19),NNMXM1),
     X            (IPARM(20),LMAX),   (IPARM(23),LLMXP2),
     X            (IPARM(24),NCGC),   (IPARM(26),IELEM),
     X            (IPARM(27),M2STO),  (IPARM(28),L3MX),
     X            (IPARM(29),NFACT),  (IPARM(30),NABDIM),
     X            (IPARM(31),NVTERM), (IPARM(32),NSTO),
     X            (IPARM(33),ISPIN),  (IPARM(34),NYLM),
     X            (IPARM(35),N2ORB),
     X            (IPARM(37),MXCYCL), (IPARM(38),NCYCL),
     X            (IPARM(39),IUNIT0), (IPARM(40),M4STO),
     X            (IPARM(41),JSTOT),  (IPARM(42),MSTOT)
      EQUIVALENCE (APARM(2),CONVRG),  (APARM(4),CDAMP),
     X            (APARM(5),CACCEL),  (APARM(6),RANGE)
      DIMENSION OVLP(N2ORBT,ISPIN),    KREC(N2ATOM),
     X          CGC(NCGC),             YLMNRM(NYLM),
     X          S(NNMX),               W(NNMXM1),
     X          OMEGAI(NNMXM1,NNMXM1), OMEGAJ(NNMXM1,NNMXM1),
     X          A(NABDIM),             B(NABDIM),
     X          D(L3MX,2),             INDEX(1),
     X          EWMAT(N2ORBT,ISPIN),   POPUL(NORBT,ISPIN),
     X          BUFFER(LENBUF),        SCRAT(LLMXP2,2),
     X          COORD(3,NATOM),        LABEL(NATOM),
     X          CHARGE(3,NATOM),       IATOMX(MSTOT),
     X          NX(MSTOT),             LX(MSTOT),
     X          MLX(MSTOT),            ETAX(MSTOT),
     X          ANORMX(MSTOT),         EVALX(NORBT,ISPIN),
     X          OCCX(NORBT,ISPIN),     EVECX(MSTOT,NORBT,ISPIN)
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(1),IUNIT1),(IUNIT(2),IUNIT2),
     X            (IUNIT(3),IUNIT3),(IUNIT(4),IUNIT4),
     X            (IUNIT(5),IR),    (IUNIT(6),IW),
     X            (IUNIT(8),IUDUMP),(IUNIT(9),IUTEMP),
     X            (IUNIT(10),IUCSF)
      REAL*4 BEGIN,START,FINISH,SECS,TIMEND,TIMREM,OLDTIM,CYCTIM
      COMMON /TIMES/ BEGIN,START,FINISH,SECS,TIMEND
      COMMON /ENERGY/ TOTPE,TOTKE,TMOLE,ZZPOT,ETOT,VIRTHM,DELTAE
      CHARACTER*8 SPIN
      COMMON /CHARGS/ ZNSUM,ELSUM,ELDIF,ELECS(2),SPIN(2),ORBOCC,CHTRAN
      COMMON /MODPOT/ WGT0,WGT1,WGT2,WGTC,WGTV,VDAMP,VACCEL,
     X                TFEXP(4),TFCOEF(4),LTF(4),MXTF
      COMMON /DAUNIT/ NRECS,LENREC,NEXREC,NLRECS,NBYTES,QMXREC
      COMMON /TABLES/ REALS(10),FACT(22),FFAC(19),BINOM(91),HALFPI(8),
     X                ZERO,HALF,ROOT(10),ROOTPI,CONST(10),CONVRT(10)
      EQUIVALENCE (REALS(1),ONE),(REALS(2),TWO)
      NAMELIST /BASDIM/ QNWBAS,NSTO,MSTO,NORB,NVTERM
      CHARACTER*8 ROUTIN
      COMMON /DYNAMC/ LISTAR(500),IPTR(10),NNIA(10),NID(10),NXARG(10),
     X                ROUTIN(10),NROUT
C----------------------------------------------------------------------
C  FIXED CORE ALLOCATION:
C
      COMMON /FIXED/ MXCORE,NWCORE,NWORDS
      DIMENSION CORE(1),IC(40)
C----------------------------------------------------------------------
      COMMON /MOLSTR/ MSPTR
      DATA QREALY/.TRUE./
      DATA CDMP90/0.9D0/

      IWSAV=IW
      IF (QMNPRT) IW=IUDUMP
      IF (QDEBUG) CALL ARRMAP(1)
      ORBOCC=3-ISPIN
      IF (QTRACE) WRITE (9,*) 'Initialized Tables, Open Files, etc...'
C-----------------------------------------------------------------------
C
C     GENERATE ROOTS, INDEXING ARRAY, FACTORIALS, BINOMIAL COEFFICIENTS,
C     AND CLEBSCH-GORDON COEFFICIENTS.
C
C-----------------------------------------------------------------------
      ZERO=0.0D0
      HALF=0.5D0

      DO 5 I=1,10
      REALS(I)=DFLOAT(I)
      ROOT(I)=DSQRT(REALS(I))
    5 CONTINUE

      PI=DATAN2(ONE,ZERO)*TWO
      ROOTPI=DSQRT(PI)

      DO 10 I=1,8
      HALFPI(I)=HALF*REALS(I)*PI
   10 CONTINUE
   
      NDXDIM=MAX0(MXMSTO,NATOM,NNMX)

      DO 20 I=1,NDXDIM
      INDEX(I)=(I*(I-1))/2
   20 CONTINUE
   
      MXFACT=MAX0(4*LMAX+1,NNMX)
      NFACT=MXFACT+1
      CALL GENFAC(FACT,NFACT)
      NBINOM=INDEX(NNMX)
      CALL GENBC(BINOM,NBINOM,NNMXM1,INDEX)
      CALL GENCGC(CGC,NCGC,SCRAT,LLMXP2,YLMNRM,NYLM,FACT,NFACT,QREALY)

      IF (.NOT.QDEBUG) GO TO 40
      WRITE (IUDUMP,1000)

      DO 30 I=1,MXFACT
      IM1=I-1
      WRITE (IUDUMP,2000) IM1,FACT(I)
   30 CONTINUE
   
      WRITE (IUDUMP,3000)
      CALL PUTONE(BINOM,NNMXM1,IUDUMP)
      WRITE (IUDUMP,4000)
      CALL PUTCGC(CGC,NCGC,LMAX,QREALY,IUDUMP)

   40 CONTINUE
C-----------------------------------------------------------------------
C
C     INITIALIZE COUNTERS.
C
C-----------------------------------------------------------------------
	CALL DERASE(EVALX,NORBT*ISPIN)
	CALL DERASE(OCCX,NORBT*ISPIN)
	CALL DERASE(EVECX,MSTOT*NORBT*ISPIN)
      IELEM=0
      IATOM=0
      IORBT=0
	JSTOT=0
      NRATOM=0
      NCYCL=0
      TMOLE=ZERO
      QFIRST=.TRUE.
      QLAST=.FALSE.
      IF (MXCYCL.EQ.0) QFIRST=.FALSE.
      IF (MXCYCL.LE.1) QLAST=.TRUE.
      OPEN (UNIT=IUNIT2,FILE='Phatpsy.Unit2',FORM='UNFORMATTED')
      OPEN (UNIT=IUNIT3,FILE='Phatpsy.Unit3',FORM='UNFORMATTED')
      OPEN (UNIT=IUNIT4,FILE='Phatpsy.Unit4',FORM='UNFORMATTED')
      OPEN (UNIT=IUTEMP,FILE='Phatpsy.Temp',FORM='UNFORMATTED')
      OPEN (UNIT=IUCSF,FILE='Phatpsy.CSF',FORM='UNFORMATTED')
   50 CONTINUE
      IF (QTRACE) WRITE (9,*) 'Reading Basis Dimensions (&BASDIM)...'
C-----------------------------------------------------------------------
C
C     SET DEFAULT BASIS DIMENSIONS.
C
C-----------------------------------------------------------------------
      QNWBAS=(.NOT.QRSTRT)
      NSTO=MXNSTO
      MSTO=MXMSTO
      NORB=MXNORB
      NVTERM=MXTF
      IF (QATOM) NVTERM=1
C-----------------------------------------------------------------------
C
C     READ BASIS DIMENSION INPUT.  IGNORED IF QATOM=T.
C
C     &BASDIM... (NAMELIST INPUT, DEFAULTS ARE IN /'S)
C
C        QNWBAS... =T --> THE BASIS FOR THIS ELEMENT IS NEW.  WRITE
C                  IT AND THE ONE-CENTER INTEGRALS ON IUNIT(10+IELEM).
C                  /DEFAULT IF QRSTRT=F/
C                  =F --> BASIS AND ONE-CENTER INTEGRALS ARE TO BE READ
C                  FROM IUNIT(10+IELEM). /DEFAULT IF QRSTRT=T/
C        NSTO..... NUMBER OF STO'S TO BE READ IN FOR THIS ELEMENT (SEE
C                  ATOMIC), NOT INCLUDING ML-VALUES. /MXNSTO/
C        MSTO..... NUMBER OF STO'S ACTUALLY IN BASIS (INCLUDING THE
C                  ML-VALUES). /MXMSTO/
C        NORB..... NUMBER OF ATOMIC ORBITAL TO BE REPRESENTED BY THIS
C                  BASIS FOR EACH ATOM OF THIS ELEMENT. /MXNORB/
C        NVTERM... NUMBER OF TERMS TO BE USED IN THE MODEL POTENTIALS
C                  FOR THE ATOMS OF THIS ELEMENT. /4/
C                  (DEFAULT IS 1 IF QATOM=T)
C
C     &END
C
C-----------------------------------------------------------------------
      READ (IR,BASDIM,END=240)
C     READ (IR,*,END=240) QNWBAS,NSTO,MSTO,NORB,NVTERM
      IELEM=IELEM+1
      WRITE (IW,5000) IELEM
      IF (QDEBUG) WRITE (IUDUMP,5000) IELEM
      IF (QDEBUG) WRITE (IUDUMP,BASDIM)
      IF (IELEM.GT.NELEM) CALL BOMB(7)
      IUNIT0=IUNIT(IELEM+10)
      OPEN (UNIT=IUNIT0,FILE='Phatpsy.Elem'//CHAR(48+IELEM),
     X                  FORM='UNFORMATTED')
      IF (.NOT.QNWBAS) WRITE (IW,6000) IUNIT0
      IF (QNWBAS) WRITE (IUNIT0) NSTO,MSTO,NORB,NVTERM
      IF (.NOT.QNWBAS) READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      WRITE (IW,7000) NSTO,MSTO
      WRITE (IW,8000) NORB
      IF (MSTO.GT.MXMSTO) CALL BOMB(5)
      M2STO=INDEX(MSTO)+MSTO
      M4STO=(M2STO*(M2STO+1))/2
      N2ORB=INDEX(NORB)+NORB
C      IF (NVTERM.EQ.2) NVTERM=3
      IF (NVTERM.GT.1) WRITE (IW,9000) NVTERM
C-----------------------------------------------------------------------
C
C     READ BASIS, STARTING ATOMIC VECTORS, THEIR ENERGIES, THE MODEL
C     POTENTIAL PARAMETERS, THE COORDINATES, THE RESTART DATASET, AND
C     COMPUTE THE ONE-CENTER INTEGRALS (IF NECESSARY).
C
C-----------------------------------------------------------------------
      IPTR2=IPTR(2)
      NRTNS=2
C----------------------------------------------------------------------
C  FIXED CORE ALLOCATION OFFSETS:
C
      IC(1)=1
      IC(2)=IC(1)+(NSTO+1)/2
      IC(3)=IC(2)+(NSTO+1)/2
      IC(4)=IC(3)+(NSTO+1)/2
      IC(5)=IC(4)+NSTO
      IC(6)=IC(5)+NSTO
      IC(7)=IC(6)+M2STO
      IC(8)=IC(7)+M2STO
      IC(9)=IC(8)+M2STO*NVTERM
      IC(10)=IC(9)+NVTERM
      IC(11)=IC(10)+NVTERM
      IC(12)=IC(11)+NVTERM
      IC(13)=IC(12)+NVTERM
      IC(14)=IC(13)+NVTERM
      IC(15)=IC(14)+M2STO
      IC(16)=IC(15)+MSTO*MSTO
      IC(17)=IC(16)+MSTO*NORB*ISPIN
      IC(18)=IC(17)+NORB*ISPIN
      IC(19)=IC(18)+NORB*ISPIN
      IC(20)=IC(19)+N2ORB*ISPIN
      IC(21)=IC(20)+NORB*ISPIN
      IC(22)=IC(21)+M2STO
      IC(23)=IC(22)+MSTO*MSTO
      IC(24)=IC(23)+MSTO*9
      IC(25)=IC(24)+NVTERM
      IC(26)=IC(25)+M2STO

      NWORDS=IC(26)+M2STO*6*ISPIN
      WRITE (IW,10000) (8*NWORDS)/1024
      NWCORE=NWCORE-NWORDS
      IF (NWCORE.LE.0) CALL BOMB(0)
      IF (QTRACE) WRITE (9,*) 'Calling ATOMIC...'

      CALL ATOMIC(CORE(IC(1)),CORE(IC(2)),CORE(IC(3)),CORE(IC(4)),
     X            CORE(IC(5)),CORE(IC(6)),CORE(IC(7)),CORE(IC(8)),
     X            CORE(IC(9)),CORE(IC(10)),CORE(IC(11)),CORE(IC(12)),
     X            CORE(IC(13)),CORE(IC(14)),CORE(IC(15)),CORE(IC(16)),
     X            CORE(IC(17)),CORE(IC(18)),CORE(IC(19)),CORE(IC(20)),
     X            CORE(IC(21)),CORE(IC(22)),CORE(IC(23)),CORE(IC(24)),
     X            CORE(IC(25)),CORE(IC(26)),INDEX,CGC,S,W,BUFFER,COORD,
     X            LABEL,CHARGE,FACT,D,SCRAT,NRATOM,QREALY,
     X            IATOMX,NX,LX,MLX,ETAX,ANORMX,EVALX,OCCX,EVECX)

      NWCORE=NWCORE+NWORDS
      REWIND IUNIT0

      IF (.NOT.QATOM .AND. IELEM.LT.NELEM) GO TO 50

   60 CONTINUE

	WRITE (IW,10200)
      DO 61 JSTOT=1,MSTOT
      WRITE (IW,10300) IATOMX(JSTOT),NX(JSTOT),LX(JSTOT),MLX(JSTOT),
     X                 ETAX(JSTOT)
   61 CONTINUE
      WRITE (IW,10400)
      DO 62 ISP=1,ISPIN
      IF (QOPEN) WRITE (IW,16000) SPIN(ISP)
      CALL OUTVEC(EVECX(1,1,ISP),EVALX(1,ISP),MSTOT,NORBT,IW)
   62 CONTINUE

      ELSUM=CHARGE(1,1)
      ELDIF=CHARGE(2,1)
      ELECS(ISPIN)=ZERO
      ELECS(1)=HALF*(ELSUM+ELDIF)
      ELECS(ISPIN)=ELECS(ISPIN)+HALF*(ELSUM-ELDIF)

      IF (QATOM) GO TO 65
      WRITE(IW,10500) RANGE,RANGE
      X0=ASUM(COORD(1,1),NATOM,3)/NATOM
      Y0=ASUM(COORD(2,1),NATOM,3)/NATOM
      Z0=ASUM(COORD(3,1),NATOM,3)/NATOM
      CALL XYZMAP(COORD,LABEL,NATOM,IW,RANGE,X0,Y0,Z0)

   65 CONTINUE
      WRITE (IW,11000)
      REWIND IUNIT3
      IF (.NOT.QATOM) REWIND IUNIT4
      WRITE (IW,12000) IUNIT3
      CALL TIMOUT(-2)
      IF (QTRACE) WRITE (9,*) 'Input Complete...'
      NELEM=IELEM
      NATOM=IATOM
      NORBT=IORBT
      IF (QATOM) GO TO 170
C-----------------------------------------------------------------------
C
C     DYNAMIC DEFINE FILE:
C
C     EQUIVALENT TO... DEFINE FILE IUNIT1(NRECS,LENREC,U,IREC)
C
C        WHERE IUNIT1, NRECS AND LENREC ARE INTEGER CONSTANTS.
C
C     THE SPACE PARAMETER FOR IUNIT1 SHOULD BE
C
C        SPACE = (4*LENREC,NRECS)
C-----------------------------------------------------------------------
C   FIXED DA-FILE ALLOCATION
C
C     NRECS=200
C     LENREC=200
C     DEFINE FILE 1 (200,200,U,IREC)
C-----------------------------------------------------------------------
      IF (.NOT.QMXREC) GO TO 70
C   BLOCKING FACTOR SHOULD REFLECT IMBALANCE IN BASIS SET SIZES
      BLKFAC=1
      NLRECS=BLKFAC*(NRATOM*(2*NATOM-NRATOM-1))/2
      LENREC=8*MXMSTO*MXMSTO/BLKFAC
      NPRECS=(N2ATOM+2)/LENREC+1
      NRECS=NPRECS+NLRECS
      NBYTES=NRECS*LENREC

   70 CONTINUE
      IF (QTRACE) 
     X   WRITE (9,*) '   Opening DA Unit 1 (',NRECS,'x',LENREC,')...'
      IF (QCMPTD) THEN
         OPEN (UNIT=IUNIT1,FILE='Phatpsy.DAUnit1',ACCESS='DIRECT',
     X         RECL=LENREC,STATUS='OLD')
      ELSE
         OPEN (UNIT=IUNIT1,FILE='Phatpsy.DAUnit1',ACCESS='DIRECT',
     X         RECL=LENREC,STATUS='UNKNOWN')
      ENDIF
      WRITE (IW,13000) NRECS,LENREC,IUNIT1
C-----------------------------------------------------------------------
C
C     GET READY FOR FIRST CYCLE.
C
C-----------------------------------------------------------------------
      OLDTE=TMOLE
      TOTKE=-TMOLE
      QPRNT=QPRINT
      IF (QSEPAT) QPRINT=.TRUE.

      IF (QAUDMP) GO TO 80
      QSTALL=((CDAMP.GE.CDMP90).AND.(CDAMP.LT.ONE))
      IF (QSTALL) CDAMP=ONE

   80 CONTINUE
      IF (QCMPTD) GO TO 90
      WRITE (IUNIT1,REC=1) KREC
      INQUIRE (IUNIT1,NEXTREC=NEXREC)

   90 CONTINUE
C-----------------------------------------------------------------------
C
C     BEGIN SELF-CONSISTENT CYCLES.
C
C-----------------------------------------------------------------------
      IF (QCHECK) STOP
      IF (QMNPRT) IW=IUDUMP
      IF (QTRACE) WRITE (9,*)
      IF (QTRACE) WRITE (9,*) 'Cycle ',NCYCL,' - (E = ',TMOLE,') ...'
      CALL TIMOUT(0)
C-----------------------------------------------------------------------
C
C     INITIALIZE THE OVERLAP MATRIX (TO A UNIT MATRIX), COUNTERS
C     AND CHARGE SUMS.
C
C-----------------------------------------------------------------------
      CALL DERASE(OVLP,N2ORBT*ISPIN)
      CALL DERASE(EWMAT,N2ORBT*ISPIN)
      CALL DERASE(POPUL,NORBT*ISPIN)
	CALL DERASE(EVALX,NORBT*ISPIN)
	CALL DERASE(OCCX,NORBT*ISPIN)
	CALL DERASE(EVECX,MSTOT*NORBT*ISPIN)
      IATOM=0
      IORBT=0
	JSTOT=0
      NBYTES=8*N2ATOM
      ZNSUM=ZERO
      ELSUM=ZERO
      ELDIF=ZERO
      IF (QCMPTD) READ (IUNIT1,REC=1) KREC
C-----------------------------------------------------------------------
C
C     BEGIN PRIMARY LOOP (A-LOOP) OVER ATOMS TO GENERATE INTERATOMIC
C     INTERACTION INTEGRALS, OVERLAP INTEGRALS, ETC.
C
C-----------------------------------------------------------------------
      DO 110 IELEM=1,NELEM
      IUNIT0=IUNIT(10+IELEM)
      READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      M2STO=INDEX(MSTO)+MSTO
      READ (IUNIT3) MDIFAT
      N2ORB=INDEX(NORB)+NORB
      WRITE (IUNIT4) MDIFAT
      IF (QDEBUG) WRITE (IUDUMP,14000) NCYCL
      IPTR3=IPTR(3)
      NRTNS=3
C----------------------------------------------------------------------
C  FIXED CORE ALLOCATION OFFSETS:
C
      IC(1)=1
      IC(2)=IC(1)+(NSTO+1)/2
      IC(3)=IC(2)+(NSTO+1)/2
      IC(4)=IC(3)+NSTO
      IC(5)=IC(4)+NSTO
      IC(6)=IC(5)+MSTO*NORB*ISPIN
      IC(7)=IC(6)+NORB*ISPIN
      IC(8)=IC(7)+NORB*ISPIN
      IC(9)=IC(8)+N2ORB*ISPIN
      IC(10)=IC(9)+NORB*ISPIN
      IC(11)=IC(10)+M2STO
      IC(12)=IC(11)+M2STO
      IC(13)=IC(12)+M2STO*NVTERM
      IC(14)=IC(13)+NVTERM
      IC(15)=IC(14)+NVTERM
      IC(16)=IC(15)+NVTERM
      IC(17)=IC(16)+M2STO
      IC(18)=IC(17)+M2STO
      IC(19)=IC(18)+M2STO*6*ISPIN
      IC(20)=IC(19)+MSTO*MSTO*ISPIN
      IC(21)=IC(20)+NVTERM
      IC(22)=IC(21)+NVTERM
      IC(23)=IC(22)+MSTO*MSTO*2
      NWORDS=IC(23)
      NWCORE=NWCORE-NWORDS
      IF (NWCORE.LE.0) CALL BOMB(0)
      IF (QTRACE) WRITE (9,*) '   Calling ALOOP/BLOOP...'

      CALL ALOOP(CORE(IC(1)),CORE(IC(2)),CORE(IC(3)),CORE(IC(4)),
     X           CORE(IC(5)),CORE(IC(6)),CORE(IC(7)),CORE(IC(8)),
     X           CORE(IC(9)),CORE(IC(10)),CORE(IC(11)),CORE(IC(12)),
     X           CORE(IC(13)),CORE(IC(14)),CORE(IC(15)),CORE(IC(16)),
     X           CORE(IC(17)),CORE(IC(18)),CORE(IC(19)),CORE(IC(20)),
     X           CORE(IC(21)),CORE(IC(22)),CORE(IC(23)),
     X           COORD,EWMAT,OVLP,CHARGE,POPUL,
     X           YLMNRM,OMEGAI,OMEGAJ,A,B,D,
     X           INDEX,KREC,CGC,QREALY,EVALX,OCCX,EVECX)

      NWCORE=NWCORE+NWORDS
      REWIND IUNIT0
  110 CONTINUE
C-----------------------------------------------------------------------
C
C     END PRIMARY INTEGRAL LOOP.
C
C-----------------------------------------------------------------------
  
      REWIND IUNIT2
      REWIND IUNIT3
      REWIND IUNIT4
      IF (QPRINT) WRITE (IW,12000) IUNIT4
      ELECS(ISPIN)=ZERO
      ELECS(1)=HALF*(ELSUM+ELDIF)
      ELECS(ISPIN)=ELECS(ISPIN)+HALF*(ELSUM-ELDIF)

      IF (QCMPTD) GO TO 120
      WRITE (IUNIT1,REC=1) KREC
      WRITE (IW,13500) NEXREC-1,LENREC,LENREC*(NEXREC-1)/1024,IUNIT1
      WRITE (IW,13600) NLRECS,NBYTES/1024
      WRITE (IW,12000) IUNIT2
      CALL TIMOUT(8)

  120 CONTINUE
  
      IF (.NOT.QDEBUG) GO TO 140
      WRITE (IUDUMP,14000) NCYCL
      WRITE (IUDUMP,15000)

      DO 130 ISP=1,ISPIN
      IF (QOPEN) WRITE (IUDUMP,16000) SPIN(ISP)
      CALL PUTONE(OVLP(1,ISP),NORBT,IUDUMP)
  130 CONTINUE
  
  140 CONTINUE
      IF (QPRINT.AND.QCMPTD) CALL TIMOUT(3)

      IF (QFIRST) GO TO 150
      IPTR4=IPTR(4)
      NRTNS=4
C-----------------------------------------------------------------------
C
C     CONSTRUCT EWMO MATRIX AND SOLVE FOR THE MOLECULAR ORBITALS AND
C     ENERGIES.  COMPUTE TOTAL ENERGY AND PERFORM POPULATION ANALYS.
C
C-----------------------------------------------------------------------
      QEWMOS=QEWMO
      IF (QMNPRT) IW=IWSAV
      IF ((CDAMP.EQ.ONE).AND.(.NOT.QPRINT)) QEWMO=.FALSE.
C----------------------------------------------------------------------
C  FIXED CORE ALLOCATION OFFSETS:
C
      IC(1)=1
      IC(2)=IC(1)+NORBT*ISPIN
      IC(3)=IC(2)+NORBT*ISPIN
      IC(4)=IC(3)+NORBT*NORBT
      IC(5)=IC(4)+NORBT*9
      IC(6)=IC(5)+NORBT*NORBT*ISPIN
      NWORDS=IC(6)+NORBT*NORBT
      NWCORE=NWCORE-NWORDS
      IF (NWCORE.LE.0) CALL BOMB(0)
      IF (QTRACE) WRITE (9,*) '   Calling EWMO...'

      CALL EWMO(CORE(IC(1)),CORE(IC(2)),CORE(IC(3)),CORE(IC(4)),
     X          CORE(IC(5)),CORE(IC(6)),OVLP,EWMAT,POPUL)

      NWCORE=NWCORE+NWORDS
      QEWMO=QEWMOS
      QSEPAT=.FALSE.
C-----------------------------------------------------------------------
C
C     DETERMINE AUTOMATIC DAMPING FROM CHARGE TRANSFER AND EXTRAPOLATED
C     TOTAL ENERGY CHANGE.
C
C-----------------------------------------------------------------------
      CTWOD=CHTRAN
      EXTE1=TMOLE+DELTAE
      IF (QAUDMP) CDAMP=ONE
      IF (QAUDMP.AND.(DELTAE.LT.ZERO)) CDAMP=(CHTRAN/ELSUM)**CACCEL
      CHTRAN=(ONE-CDAMP)*CHTRAN
      EXTE2=EXTE1-CDAMP*DELTAE
      IF (QPRINT.AND..NOT.QLAST) WRITE (IW,17000) CTWOD,EXTE1,CDAMP,
     X     CHTRAN,EXTE2

  150 CONTINUE
C-----------------------------------------------------------------------
C
C     CHECK THE TIME, NUMBER OF ALLOWED CYCLES REMAINING, AND THE
C     CONVERGENCE OF THE TOTAL ENERGY.
C
C-----------------------------------------------------------------------
      NREMCY=MXCYCL-NCYCL
      TIMREM=TIMEND-FINISH
      CYCTIM=FINISH-OLDTIM
      OLDTIM=FINISH
      TEDIFF=DABS(OLDTE-TMOLE)

      IF (QFIRST) GO TO 160
      IF (QPRINT) WRITE (IW,18000) TEDIFF,TIMREM,NREMCY
C     IF (.NOT.QPRINT) WRITE (IW,19000) NCYCL,TMOLE,CYCTIM,CTWOD,EXTE1,
C    X     CDAMP,CHTRAN,EXTE2
      IF (.NOT.QPRINT) WRITE (IW,19000) NCYCL,TMOLE,CYCTIM,CHTRAN,VDAMP
      IF (QPRINT) WRITE (IW,20000) CYCTIM

      IF (QLAST) GO TO 220
      IF (TEDIFF.LT.CONVRG) QLAST=.TRUE.

  160 CONTINUE
      IF (NREMCY.LE.1) QLAST=.TRUE.
      IF (TIMREM.LT.TWO*CYCTIM) QLAST=.TRUE.
      IF (QLAST) WRITE (IW,21000)
      IF (QMNPRT) IW=IUDUMP

  170 CONTINUE
      IF (QLAST) QPRINT=.TRUE.
      IORBT=0
      IATOM=1
      NCYCL=NCYCL+1
      OLDTE=TMOLE
      TMOLE=ZERO
      TOTKE=ZERO
      ZZPOT=ZERO

C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER ATOMIC SCF-ITERATIONS.
C
C-----------------------------------------------------------------------
      DO 200 IELEM=1,NELEM
      IUNIT0=IUNIT(IELEM+10)
      READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      M2STO=INDEX(MSTO)+MSTO

      IF (QATOM) GO TO 180
      READ (IUNIT4) MDIFAT
      WRITE (IUNIT3) MDIFAT
      GO TO 190

  180 CONTINUE
      READ (IUNIT3) MDIFAT

  190 CONTINUE
      IF (QDEBUG) WRITE (IUDUMP,14000) NCYCL
      N2ORB=INDEX(NORB)+NORB
      IPTR5=IPTR(5)
      NRTNS=5
C----------------------------------------------------------------------
C  FIXED CORE ALLOCATION OFFSETS:
C
      IC(1)=1
      IC(2)=IC(1)+(NSTO+1)/2
      IC(3)=IC(2)+(NSTO+1)/2
      IC(4)=IC(3)+NSTO
      IC(5)=IC(4)+NSTO
      IC(6)=IC(5)+M2STO
      IC(7)=IC(6)+M2STO
      IC(8)=IC(7)+M2STO*NVTERM
      IC(9)=IC(8)+(NVTERM+1)/2
      IC(10)=IC(9)+(NVTERM+1)/2
      IC(11)=IC(10)+NVTERM
      IC(12)=IC(11)+M2STO
      IC(13)=IC(12)+MSTO*MSTO
      IC(14)=IC(13)+M2STO
      IC(15)=IC(14)+M2STO*ISPIN
      IC(16)=IC(15)+M2STO
      IC(17)=IC(16)+M2STO*ISPIN     
      IC(18)=IC(17)+M2STO*ISPIN
      IC(19)=IC(18)+M2STO*ISPIN    
      IC(20)=IC(19)+M2STO*ISPIN    
      IC(21)=IC(20)+M2STO     
      IC(22)=IC(21)+MSTO*NORB 
      IC(23)=IC(22)+MSTO*NORB*ISPIN
      IC(24)=IC(23)+MSTO*NORB*ISPIN
      IC(25)=IC(24)+NORB*ISPIN
      IC(26)=IC(25)+NORB*ISPIN 
      IC(27)=IC(26)+N2ORB*ISPIN
      IC(28)=IC(27)+NORB*ISPIN 
      IC(29)=IC(28)+MSTO*MSTO*ISPIN
      IC(30)=IC(29)+(NORB*ISPIN+1)/2
      IC(31)=IC(30)+MSTO*MSTO*2
      IC(32)=IC(31)+MSTO*9    
      IC(33)=IC(32)+MSTO*MSTO*ISPIN
      IC(34)=IC(33)+NVTERM
      IC(35)=IC(34)+NVTERM
      NWORDS=IC(35)+M2STO*6*ISPIN
      NWCORE=NWCORE-NWORDS
      IF (NWCORE.LE.0) CALL BOMB(0)
      IF (QTRACE) WRITE (9,*) '   Calling SCF for element ',IELEM,' ...'

      CALL SCF(CORE(IC(1)),CORE(IC(2)),CORE(IC(3)),CORE(IC(4)),
     X         CORE(IC(5)),CORE(IC(6)),CORE(IC(7)),CORE(IC(8)),
     X         CORE(IC(9)),CORE(IC(10)),CORE(IC(11)),CORE(IC(12)),
     X         CORE(IC(13)),CORE(IC(14)),CORE(IC(15)),CORE(IC(16)),
     X         CORE(IC(17)),CORE(IC(18)),CORE(IC(19)),CORE(IC(20)),
     X         CORE(IC(21)),CORE(IC(22)),CORE(IC(23)),CORE(IC(24)),
     X         CORE(IC(25)),CORE(IC(26)),CORE(IC(27)),CORE(IC(28)),
     X         CORE(IC(29)),CORE(IC(30)),CORE(IC(31)),CORE(IC(32)),
     X         CORE(IC(33)),CORE(IC(34)),CORE(IC(35)),
     X         BUFFER,CHARGE,POPUL,INDEX,D)

      NWCORE=NWCORE+NWORDS
      REWIND IUNIT0
C-----------------------------------------------------------------------
C
C     END LOOP OVER ATOMIC SCF-ITERATIONS.
C
C-----------------------------------------------------------------------
  200 CONTINUE
  
      REWIND IUNIT3
      IF (.NOT.QATOM) REWIND IUNIT4
      IF (QPRINT) WRITE (IW,12000) IUNIT3

      IF (QATOM) GO TO 220
      QCMPTD=.TRUE.
C-----------------------------------------------------------------------
C
C     ACCELERATE MANUAL DAMPING FACTORS AND BEGIN NEW CYCLE.
C
C-----------------------------------------------------------------------
      IF (QAUDMP) GO TO 210
      IF (QRSTRT.OR.(.NOT.QFIRST)) CDAMP=CDAMP**CACCEL
      IF (QSTALL.AND.(NCYCL.EQ.2)) CDAMP=CDMP90

  210 CONTINUE
      VDAMP=VDAMP**VACCEL
      IF (QMNPRT) IW=IWSAV
      IF (QFIRST.AND..NOT.QPRINT) WRITE (IW,22000)
      QFIRST=.FALSE.

      IF (.NOT.QSEPAT) GO TO 90
      QPRINT=QPRNT
      IF (.NOT.QPRINT) WRITE (IW,22000)
      GO TO 90

  220 CONTINUE
      CLOSE(IUNIT1)
      CLOSE(IUNIT2)
      CLOSE(IUNIT3)
      CLOSE(IUNIT4)
      CLOSE(IUTEMP)

      DO 230 IELEM=1,NELEM
      CLOSE(UNIT=IELEM+10)
  230 CONTINUE
  
      IF (QGEOM) GOTO 40
      GOTO 250

  240 CONTINUE
      CLOSE(IUNIT2)
      CLOSE(IUNIT3)
      CLOSE(IUNIT4)
      CLOSE(IUTEMP)

  250 CONTINUE

      WRITE (IW,10400)
      DO 260 ISP=1,ISPIN
      IF (QOPEN) WRITE (IW,16000) SPIN(ISP)
      CALL OUTVEC(EVECX(1,1,ISP),EVALX(1,ISP),MSTOT,NORBT,IW)
  260 CONTINUE

C
C  Map data into MolStruct structures
C
C      CALL MS_map(NELEM,NATOM,NBOND,NORBT,NORBT,MSPTR,IER);
C-----------------------------------------------------------------------
C
C     THE END...
C
C-----------------------------------------------------------------------
      CALL TIMOUT(-7)
C     IF (QDEBUG) CALL DUMP(NRTNS)
  
      RETURN

 1000 FORMAT(/' FACTORIALS...'//
     X       '   N           FACTORIAL'/
     X       '   -           ---------'/)
 2000 FORMAT(' ',I3,F20.0)
 3000 FORMAT(/' BINOMIAL COEFFICIENTS...')
 4000 FORMAT(/' CLEBSCH-GORDON COEFFICIENTS...')
 5000 FORMAT(//' === ELEMENT',I3,' ==='//)
 6000 FORMAT(' THE BASIS FOR THIS ELEMENT AND THE ONE-CENTER INTEGRALS',
     X       ' WILL BE READ FROM UNIT',I3,'.')
 7000 FORMAT(' THE BASIS CONTAINS',I3,' STO''S (',I3,' COUNTING',
     X       ' ML-VALUES).')
 8000 FORMAT(' THIS ELEMENT IS DESCRIBED BY',I3,' ATOMIC ORBITALS.')
 9000 FORMAT(' THE MODEL POTENTIAL CONTAINS',I3,' TERMS.')
10000 FORMAT(' ...',I6,'K BYTES OF CORE TO BE ALLOCATED FOR ATOMIC ...')
10200 FORMAT(//' COMPLETE MOLECULAR BASIS ...'//
     X       ' ATOM  N  L  ML       ETA'/
     X       ' ----  -  -  --       ---')
10300 FORMAT(I4,1X,I3,I3,I4,2X,F10.5)
10400 FORMAT(//'ATOMIC ORBITALS OVER MOLECULAR BASIS...')
10500 FORMAT(//' COORDINATE PROJECTIONS...'/' (EACH FRAME IS ',
     X       F4.1,' BY ',F4.1,')'//)
11000 FORMAT(//' ... ALL INPUT IS COMPLETE ...')
12000 FORMAT(' ... WRITES TO UNIT',I3,' ARE COMPLETE ...')
13000 FORMAT(' ...',I4,' DIRECT-ACCESS RECORDS, 4*',I5,' BYTES EACH',
     X       ' HAVE BEEN ALLOCATED ON UNIT',I3,' ...')
13500 FORMAT(' ...',I4,I5,'-BYTE RECORDS [',I4,'K] WRITTEN TO UNIT',I3)
13600 FORMAT('    ',I4,'   LOGICAL RECORDS [',I4,'K]...')
14000 FORMAT(//' ',T100,'... CYCLE',I3,' ...')
15000 FORMAT(/' OVERLAP MATRIX OVER THE ATOMIC BASIS...')
16000 FORMAT(/' ',A8)
17000 FORMAT(//
     X       ' -----------------------------------------------------'/
     X       ' PREDICTED CHARGE TRANSFER (W/O DAMPING)....',F10.4/
     X       ' EXTRAPOLATED TOTAL ENERGY (W/O DAMPING)....',F10.4/
     X       ' DAMPING FACTOR TO BE USED FOR NEXT CYCLE...',F10.4/
     X       ' ALLOWED CHARGE TRANSFER (WITH DAMPING).....',F10.4/
     X       ' EXTRAPOLATED TOTAL ENERGY (WITH DAMPING)...',F10.4/
     X       ' -----------------------------------------------------')
18000 FORMAT(//
     X       ' -------------------------------------'/
     X       ' TOTAL ENERGY CONVERGENCE...',F10.5/
     X       ' CPU SECONDS REMAINING......',F10.3/
     X       ' ALLOWED CYCLES REMAINING...',I10/
     X       ' -------------------------------------')
C19000 FORMAT(' ',I3,F16.4,F13.4,F19.4,F17.4,F14.4,F17.4,F17.4)
19000 FORMAT(' ',I3,F16.4,F13.4,F17.4,F14.4,T76,'<te>')
20000 FORMAT(' ... IT TOOK',F6.2,' SECONDS TO COMPLETE THIS CYCLE ...')
21000 FORMAT(//' ... LAST CYCLE FOLLOWS ...'/)
C22000 FORMAT(//
C     X       ' CYCLE   TOTAL ENERGY    CYCLE TIME      CHARGE TRANSFER',
C     X       '  EXTRAPOLATED   DAMPING FACTOR  CHARGE TRANSFER',
C     X       '  EXTRAP. T.E.'/
C     X       '   #      THIS CYCLE     (SECONDS)        (W/O DAMPING) ',
C     X       '  TOTAL ENERGY   FOR NEXT CYCLE   (W/ DAMPING)  ',
C     X       '    (DAMPED)   '/
C     X       ' -----   ------------    ----------      ---------------',
C     X       '  ------------   --------------  ---------------',
C     X       '  ------------'/)
22000 FORMAT(//
     X       ' CYCLE   TOTAL ENERGY    CYCLE TIME    CHARGE TRANSFER',
     X       '     DAMPING   ',T76,'<te>'/
     X       '   #      THIS CYCLE     (SECONDS)      FOR NEXT CYCLE',
     X       '     FACTOR    ',T76,'<te>'/
     X       ' -----   ------------    ----------    ---------------',
     X       '     -------   ',T76,'<te>'/)
      END
