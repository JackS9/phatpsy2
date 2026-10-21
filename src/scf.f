      SUBROUTINE SCF(N,L,ETA,ANORM,STOVL,STKIN,STPOT,
     X               LV,MV,VEXP,STHMAT,ONTRAN,
     X               CPOT2,XPOT2,ZPOT2,
     X               STDEN,STFOCK,STCOUL,STEXCH,
     X               ONFOCK,ONEVEC,
     X               STEVEC,OLDVEC,EVAL,OCC,
     X               AOFOCK,GAMMA,PROJEC,
     X               IORDER,SCRAT1,SCRAT2,TRANX,
     X               VFIT2,VCOEF,PKPOT,
     X               BUFFER,CHARGE,POPUL,INDEX,D)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      CHARACTER*2 LATOM
C-----------------------------------------------------------------------
C
C     SCF...
C
C        THIS ROUTINE PERFORMS THE SELF-CONSISTENT-FIELD ITERATIONS FOR
C     THE ATOMIC CALCULATION INCLUDING ANY EXTERNAL POTENTIALS.  NEW
C     VECTORS, ORBITAL ENERGIES, OCCUPATION NUMBERS (DAMPED), TOTAL
C     ENERGY AND COEFFICIENTS FOR THE LOCAL MODEL POTENTIAL ARE
C     GENERATED.  THE SCF IS FOLLOWED BY AN ENERGY, CHARGE, AND SPIN
C     ANALYSIS.
C
C     VARIABLE DEFINITIONS:
C
C        STEVEC(*,*)... EIGENVECTORS OVER THE PRIMITIVE BASIS.
C        OLDVEC(*,*)... TEMPORARY STORAGE FOR STEVEC(*,*).
C        EVAL(*)....... EIGENVALUES.
C        OCC(*)........ OCCUPATION NUMBERS.
C        IORDER(*,*)... A REORDERING ARRAY FOR THE OCCUPATION NUMBERS.
C        STDEN(*)...... ONE-ELECTRON (FOCK-DIRAC) DENSITY MATRIX.
C        STFOCK(*)..... FOCK MATRIX.
C        STCOUL(*,*)... COULOMB PART OF FOCK MATRIX.
C        STEXCH(*,*)... EXCHANGE PART OF FOCK MATRIX.
C        ZPOT2(*)...... 2-CENTER NUCLEAR ATTRACTION POTENTIAL.
C        VFIT2(*)...... MODEL POTENTIAL APPROX. TO ZPOT2(*).
C        CPOT2(*)...... 2-CENTER COULOMB (MODEL) POTENTIAL MATRIX.
C        XPOT2(*,*).... 2-CENTER EXCHANGE POTENTIAL MATRIX.
C        AOFOCK(*,*)... FOCK MATRIX OVER ATOMIC ORBITALS.
C        GAMMA(*,*).... ADJUSTED ORBITAL ENERGIES.                 
C        PROJEC(*,*)... PROJECTION MATRIX (MODIFIED BY METRIC).
C        PKPOT(*,1,*).. 2-CENTER NON-LOCAL PROJECTION MATRIX.
C             (*,2,*).. 2-CENTER NON-LOCAL PK-PSEUDOPOTENTIAL MATRIX.
C        STOVL(*)...... OVERLAP MATRIX.
C        STKIN(*)...... KINETIC ENERGY MATRIX.
C        STPOT(*,*).... POTENTIAL ENERGY MATRIX (BY TERM).
C        LV(*)......... L-VALUES IN THE MODEL POTENTIAL EXPRESSION.
C        MV(*)......... M-VALUES IN THE MODEL POTENTIAL EXPRESSION.
C        VEXP(*)....... POTENTIAL ENERGY EXPRESSION EXPONENTS.
C        VCOEF(*)...... COEFFICIENTS IN THE LOCAL MODEL POTENTIAL.
C        STHMAT(*)..... ONE-ELECTRON INTEGRAL MATRIX.
C        N(*).......... N-QUANTUM NUMBERS IN BASIS.
C        L(*).......... L-QUANTUM NUMBERS IN BASIS.
C        ETA(*)........ SLATER EXPONENTS FOR BASIS.
C        ANORM(*)...... NORMALIZATION CONSTANTS FOR THE STO'S.
C        ONFOCK(*)..... FOCK MATRIX OVER THE ORTHONORMAL BASIS.
C        ONTRAN(*,*)... ORTHONORMAL (LOWDIN) TRANSFORMATION MATRIX.
C        SCRAT1(*,*)... SCRATCH ARRAY.
C        SCRAT2(*,*)... SCRATCH ARRAY.
C        ONEVEC(*,*)... EIGENVECTORS OVER THE ORTHONORMAL BASIS.
C        BUFFER(*)..... A BUFFER FOR HOLDING TWO-ELECTRON INTEGRALS
C        CHARGE(1,*)... ATOMIC NUCLEAR CHARGES.
C              (2,*)... ATOMIC ELECTRONIC CHARGES.
C              (3,*)... ATOMIC SPIN DIFFERENCES (NET ALPHA SPIN).
C        POPUL(*,*).... POPULATION ANALYSIS OF MOLECULAR ORBITALS OVER
C                       THE ATOMIC BASIS.  USED TO DETERMINE THE NEW
C                       OCCUPATION NUMBERS (DAMPED BY CDAMP).
C        INDEX(N)...... =N*(N-1)/2, INDEXING ARRAY FOR SYMMETRY PACKING.
C        D(*).......... ROTATION COEFFICIENTS.
C        NATOM......... TOTAL NUMBER OF ATOMS IN THE MOLECULE.
C        IATOM......... INDEX OF CURRENT ATOM.
C        MDIFAT........ NUMBER OF DIFFERENT ATOMS OF THIS ELEMENT.
C        NSTO.......... NUMBER OF STO'S (NOT INCLUDING ML-VALUES).
C        MSTO.......... NUMBER OF STO'S (INCLUDING ML-VALUES).
C        M2STO......... =INDEX(MSTO+1).
C        LMAX.......... MAXIMUM L-VALUE.
C        LLMXP1........ =2*LMAX+1.
C        L3MX.......... =(LLMXP1)*(LLMXP1+1)*(LLMXP1+2)/6.
C        NORB.......... NUMBER OF ATOMIC ORBITALS ON THIS ATOM.
C        NORBT......... TOTAL NUMBER OF ATOMIC ORBITALS IN THE MOLECULE.
C        IORBT......... INDEX OF CURRENT ATOMIC ORBITAL.
C        NVTERM........ NUMBER OF TERMS IN POTENTIAL ENERGY EXPRESSION.
C        VDAMP......... FRACTION OF PRIOR EXTERNAL POTENTIALS TO BE MIXED.
C        QATOM......... =T --> ATOMIC CALCULATION ONLY.
C        QRSTRT........ =T --> THIS IS A RESTART.
C        QFIRST........ =T --> THIS IS THE FIRST CYCLE.
C        QLAST......... =T --> THIS IS THE LAST CYCLE.
C        QOPEN......... =T --> SPIN-UNRESTRICTED (OPEN SHELL) CASE.
C                       =F --> SPIN-RESTRICTED CASE.
C        QSEPAT........ =T --> THE 1-ST CYCLE IS A NEAR-SEPARATED-ATOM
C                       LIMIT CALCULATION.
C        QPRINT........ =T --> INTERMEDIATE RESULTS FOR EACH CYCLE
C                       WILL BE PRINTED.
C        QDEBUG........ =T --> THIS IS A DEBUG (DIAGNOSTIC) RUN.
C        QTRACE........ =T --> A RUN TRACE WILL BE WRITTEN TO STDERR.
C        QPLOT......... =T --> TOTAL ATOMIC DENSITY WILL BE PLOTTED.
C        QFLIP......... =T --> ALPHA AND BETA SPINS "FLIPPED".
C        QVIRTL........ =T --> VIRTUAL ATOM ("SEEN" BUT NOT INCLUDED).
C        QFIXED........ =T --> NO SCF FOR THAT ATOM (FIXED VECTORS).
C        QPNCHV........ =T --> EIGENVECTORS ARE PUNCHED ON LAST CYCLE.
C        QC2FIT........ =T --> 2-CENTER COULOMB POTENTIAL TO BE FIT. 
C        QX2FIT........ =T --> 2-CENTER EXCHANGE POTENTIAL TO BE FIT. 
C        ISPIN......... =1, IF QOPEN=F; ISP=1 ALWAYS.
C                       =2, IF QOPEN=T; ISP=1 FOR ALPHA, =2 FOR BETA.
C        IUNIT3........ FORTRAN I/O UNIT FOR WRITING THE NEW ENERGIES,
C                       VECTORS, AND OCCUPATIONS (ETC.).
C        IUNIT4........ FORTRAN I/O UNIT CONTAINING THE ENERGIES, VECT-
C                       ORS, AND OCCUPATIONS (ETC.) ON INPUT.
C        IW............ FORTRAN I/O UNIT FOR PRINTING.
C        IP............ FORTRAN I/O UNIT FOR PUNCHING.
C        IUDUMP........ FORTRAN I/O UNIT FOR DEBUG (DIAGNOSTIC) OUTPUT.
C        IUTEMP........ FORTRAN I/O UNIT FOR TEMPORARY STORAGE.
C        IUNIT0........ FORTRAN I/O UNIT CONTAINING ATOMIC BASIS AND
C                       INTEGRALS.
C        LENBUF........ LENGTH OF TWO-ELECTRON INTEGRAL BUFFER,
C        NCYCL......... NUMBER OF COMPLETED CYCLES.
C        MXITER........ MAXIMUM NUMBER OF ITERATIONS ALLOWED.
C        CONVRG........ TOTAL ENERGY DIFFERENCE CONVERGENCE THRESHOLD.
C        CDAMP......... DAMPING FACTOR FOR DETERMING NEW OCCUPATION
C                       NUMBERS FROM THE POPULATION ANALYS.
C        PKSCAL........ SCALE FOR PHILLIPS-KLEINMAN PSEUDOPOTENTIAL
C        PKE........... ENERGY IN PHILLIPS-KLEINMAN PSEUDOPOTENTIAL
C        QPSMET........ =T --> PSEUDOMETRIC USED INSTEAD OF CONSTANT E.
C        TMOLE.......... TOTAL MOLECULAR ENERGY.
C        TOTKE......... TOTAL KINETIC ENERGY.
C        ZZPOT......... TOTAL NUCLEAR REPULSION ENERGY.
C        SPIN(*)....... SPIN LABELS.
C        ORBOCC........ =3-ISPIN, MAXIMUM ORBITAL OCCUPATION.
C
C     ROUTINES CALLED:  DENMAT, FOKMAT, UTHU, DMPAB, DCOPY, CTCS, TRED3,
C                       IMTQLV, TINVIT, TRBAK3, BOMB, LOWDIN, NORMLZ,
C                       AVERAG, DPROD, MAXOVL, NEWORD, SUM, TIMOUT,
C                       ARRMAP, OUTVEC, PUTONE, SYMOP, FLIP, SYMPRO,
C                       PUNCHV, ADDMAT(SUBMAT); DABS
C
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM(1)(=NATOM),   IPARM(5)(=MSTO),
C                        IPARM(6)(=NORB),    IPARM(7)(=IORBT),
C                        IPARM(8)(=NORBT),   IPARM(10)(=IATOM),
C                        IPARM(16)(=MDIFAT), IPARM(20)(=LMAX),
C                        IPARM(22)(=LLMXP1), IPARM(25)(=MXITER),
C                        IPARM(27)(=M2STO),  IPARM(28)(=L3MX),
C                        IPARM(31)(=NVTERM),
C                        IPARM(32)(=NSTO),   IPARM(33)(=ISPIN),
C                        IPARM(38)(=NCYCL),  IPARM(39)(=IUNIT0),
C                        QPARM(1)(=QRSTRT),
C                        QPARM(4)(=QOPEN),   QPARM(5)(=QFIRST),
C                        QPARM(6)(=QLAST),   QPARM(7)(=QDEBUG),
C                        QPARM(8)(=QSEPAT),  QPARM(11)(=QFIXED),
C                        QPARM(12)(=QVIRTL), QPARM(15)(=QFLIP),
C                        QPARM(16)(=QPNCHV), QPARM(17)(=QPLOT),
C                        QPARM(22)(=QPRINT), QPARM(24)(=QATOM),
C                        QPARM(32)(=QC2FIT), QPARM(33)(=QX2FIT),
C                        QPARM(34)(=QPSMET), QPARM(35)(=QTRACE),
C                        APARM(2)(=CONVRG),  APARM(4)(=CDAMP),
C                        APARM(7)(=PKSCAL),  APARM(8)(=PKE)
C
C                 SETS - IPARM(7)(=IORBT),   IPARM(10)(=IATOM),
C
C        /IODATA/ USES - IUNIT(3)(=IUNIT3), IUNIT(4)(=IUNIT4),
C                        IUNIT(6)(=IW),     IUNIT(7)(=IP),
C                        IUNIT(8)(=IUDUMP),
C                        IUNIT(9)(=IUTEMP), LENBUF
C
C        /ENERGY/ SETS - TOTKE, TMOLE, ZZPOT
C
C        /CHARGS/ USES - SPIN(*), ORBOCC
C
C        /MODPOT/ USES - VDAMP
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (APARM(2),CONVRG),  (APARM(4),CDAMP),
     X            (APARM(7),PKSCAL),  (APARM(8),PKE)
      EQUIVALENCE (IPARM(1),NATOM),   (IPARM(5),MSTO),
     X            (IPARM(6),NORB),    (IPARM(7),IORBT),
     X            (IPARM(8),NORBT),   (IPARM(10),IATOM),
     X            (IPARM(16),MDIFAT), (IPARM(20),LMAX),
     X            (IPARM(22),LLMXP1), (IPARM(25),MXITER),
     X            (IPARM(27),M2STO),  (IPARM(28),L3MX),
     X            (IPARM(31),NVTERM),
     X            (IPARM(32),NSTO),   (IPARM(33),ISPIN),
     X            (IPARM(35),N2ORB),
     X            (IPARM(38),NCYCL),  (IPARM(39),IUNIT0)
      DIMENSION QFLAGS(8)
      EQUIVALENCE (QPARM(1),QRSTRT),
     X            (QPARM(4),QOPEN),   (QPARM(5),QFIRST),
     X            (QPARM(6),QLAST),   (QPARM(7),QDEBUG),
     X            (QPARM(8),QSEPAT),  (QPARM(11),QFLAGS(1),QFIXED),
     X            (QPARM(12),QVIRTL), (QPARM(15),QFLIP),
     X            (QPARM(16),QPNCHV), (QPARM(17),QPLOT),
     X            (QPARM(22),QPRINT), (QPARM(24),QATOM),
     X            (QPARM(32),QC2FIT), (QPARM(33),QX2FIT),
     X            (QPARM(34),QPSMET), (QPARM(35),QTRACE)
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(3),IUNIT3), (IUNIT(4),IUNIT4),
     X            (IUNIT(6),IW),     (IUNIT(7),IP),
     X            (IUNIT(8),IUDUMP), (IUNIT(9),IUTEMP)
      COMMON /ENERGY/ TOTPE,TOTKE,TMOLE,ZZPOT,ETOT,VIRTHM,DELTAE
      DIMENSION STEVEC(MSTO,NORB,ISPIN), EVAL(NORB,ISPIN),
     X          OLDVEC(MSTO,NORB,ISPIN), IORDER(NORB,ISPIN),
     X          OCC(NORB,ISPIN),         AOFOCK(N2ORB,ISPIN),
     X          GAMMA(NORB,ISPIN),       STDEN(M2STO,ISPIN),
     X          STCOUL(M2STO,ISPIN),     STEXCH(M2STO,ISPIN),
     X          STOVL(M2STO),            STKIN(M2STO),
     X          LV(NVTERM),              MV(NVTERM),
     X          STPOT(M2STO,NVTERM),     VEXP(NVTERM),
     X          STFOCK(M2STO,ISPIN),     STHMAT(M2STO),
     X          ONTRAN(MSTO,MSTO),       ONFOCK(M2STO),
     X          SCRAT1(MSTO,MSTO,2),     SCRAT2(MSTO,9),
     X          ONEVEC(MSTO,NORB),       INDEX(MSTO),
     X          N(NSTO),                 L(NSTO),
     X          ETA(NSTO),               ANORM(NSTO),
     X          BUFFER(LENBUF),          VCOEF(NVTERM),
     X          CPOT2(M2STO),            XPOT2(M2STO,ISPIN),
     X          PKPOT(M2STO,6,ISPIN),    TRANX(MSTO,MSTO,ISPIN),
     X          ZPOT2(M2STO),            VFIT2(NVTERM),
     X          POPUL(NORBT,ISPIN),      CHARGE(3,NATOM),
     X          D(L3MX,2),               PROJEC(MSTO,MSTO,ISPIN)
      CHARACTER*8 SPIN
      COMMON /CHARGS/ ZNSUM,ELSUM,ELDIF,ELECS(2),SPIN(2),ORBOCC,CHTRAN
      COMMON /MODPOT/ WGT0,WGT1,WGT2,WGTC,WGTV,VDAMP,VACCEL,
     X                TFEXP(4),TFCOEF(4),LTF(4),MXTF
      DATA ZERO/0.0D0/,ONE/1.0D0/,TWO/2.0D0/
      DATA ESCALE/1.55D0/

      QX2INC=.TRUE.
      QNRMLZ=.TRUE.
      QTRUNC=.FALSE.

      REWIND IUNIT0
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER NON-EQUIVALENT ATOMS.
C
C-----------------------------------------------------------------------
      DO 290 JDIFAT=1,MDIFAT
      IF (QTRACE) WRITE (9,*) '      SCF for non-equivalent atom ',JDIFAT,' ...'
      CALL TIMOUT(0)
C-----------------------------------------------------------------------
C
C     READ ORBITAL AND MODEL POTENTIAL BASES AND ONE-CENTER INTEGRALS
C     FROM IUNIT0.
C
C-----------------------------------------------------------------------
      READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      READ (IUNIT0) LATOM,NUCZ,N,L,ETA,ANORM
      READ (IUNIT0) STOVL,STKIN,STPOT,LV,MV,VEXP
      READ (IUNIT0) STHMAT,ONTRAN
      REWIND IUNIT0
      WRITE (IUTEMP) STOVL,ONTRAN
      REWIND IUTEMP
C-----------------------------------------------------------------------
C
C     READ ORBITAL ENERGIES, OCCUPATION NUMBERS, VECTORS, ETC. FROM
C     IUNIT4.
C
C-----------------------------------------------------------------------
      IUNITX=IUNIT4
      IF (QATOM) IUNITX=IUNIT3
      READ (IUNITX) QFLAGS,NEQAT,X,Y,Z,STEVEC,EVAL,OCC,AOFOCK,GAMMA,
     X     CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      IF (QATOM) REWIND IUNIT3
      TKE=-TOTEN
      TE=TOTEN
      IF (QFIRST.AND..NOT.QRSTRT) TE=ZERO
      ENDIFF=-TOTEN
      IF (QATOM) ZZREP=ZERO

      IF (QVIRTL) GO TO 250
C-----------------------------------------------------------------------
C
C     DETERMINE NEW OCCUPATION NUMBERS (WITH DAMPING) FROM THE
C     POPULATION ANALYSIS.
C
C-----------------------------------------------------------------------
 
      IF (QFIRST) GO TO 30

      DO 20 ISP=1,ISPIN
      DO 10 IORB=1,NORB
      OCC(IORB,ISP)=CDAMP*OCC(IORB,ISP)
     X              +(ONE-CDAMP)*POPUL(IORBT+IORB,ISP)
   10 CONTINUE
   20 CONTINUE

   30 CONTINUE
C-----------------------------------------------------------------------
C
C     WRITE PAGE HEADING.
C
C-----------------------------------------------------------------------

      IF (.NOT.QPRINT) GO TO 40
      WRITE (IW,1000) LATOM,JDIFAT
      WRITE (IW,2000) NCYCL
      IF (QSEPAT) WRITE (IW,3000)
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 40
      WRITE (IUDUMP,1000) LATOM,JDIFAT
      WRITE (IUDUMP,2000) NCYCL
      WRITE (IUDUMP,4000)
      CALL PUTONE(STKIN,MSTO,IUDUMP)
      WRITE(IUDUMP,4100)
      DO 35 NV=1,NVTERM
      WRITE(IUDUMP,4105) NV,VCOEF(NV),LV(NV)-1,LV(NV),MV(NV),VEXP(NV)
      CALL PUTONE(STPOT(1,NV),MSTO,IUDUMP)
   35 CONTINUE
      WRITE(IUDUMP,4200)
      CALL PUTONE(STOVL,MSTO,IUDUMP)
      WRITE(IUDUMP,4500)
      CALL PUTMAT(ONTRAN,MSTO,MSTO,IUDUMP)
      WRITE (IUDUMP,5000)
      CALL PUTONE(ZPOT2,MSTO,IUDUMP)
      WRITE(IUDUMP,9000)
      CALL PUTONE(CPOT2,MSTO,IUDUMP)

   40 CONTINUE
   
      DO 55 ISP=1,ISPIN
C-----------------------------------------------------------------------
C
C     CONSTRUCT PSEUDOMETRIC AND SINGLE-CENTER PROJECTION MATRIX.
C
C-----------------------------------------------------------------------
      READ(IUTEMP) STOVL
      REWIND IUTEMP
      IF (.NOT.QPSMET) GO TO 52
      CALL SUBMAT(STOVL,PKPOT(1,1,ISP),STOVL,M2STO)
      CALL TRISQ(MSTO,STOVL,ONTRAN,PROJEC(1,1,ISP))
      IF (.NOT.QDEBUG) GO TO 50
      WRITE(IUDUMP,10000)
      CALL PUTONE(STOVL,MSTO,IUDUMP)
   50 CONTINUE
   52 CONTINUE
C-----------------------------------------------------------------------
C
C     CONSTRUCT THE ORTHONORMAL TRANSFORMATION MATRIX.
C
C-----------------------------------------------------------------------
      CALL LOWDIN(STOVL,TRANX(1,1,ISP),SCRAT1,MSTO,MSTO,1,QINDEF)
C-----------------------------------------------------------------------
C
C     RESTORE ORIGINAL METRIC IF PSEUDOMETRIC INDEFINITE.
C
C-----------------------------------------------------------------------
      IF (.NOT.QINDEF) GO TO 54
      IF (.NOT.QPSMET) CALL BOMB(0)
      READ(IUTEMP) STOVL
      REWIND IUTEMP
      CALL TRISQ(MSTO,STOVL,ONTRAN,PROJEC(1,1,ISP))
      CALL LOWDIN(STOVL,TRANX(1,1,ISP),SCRAT1,MSTO,MSTO,1,QINDEF)
      IF (QINDEF) CALL BOMB(0)

   54 CONTINUE
C-----------------------------------------------------------------------
C     EXTERNAL EXCHANGE POTENTIAL
C-----------------------------------------------------------------------
      CALL DCOPY(PKPOT(1,3,ISP),M2STO,XPOT2(1,ISP))

   55 CONTINUE
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 59

      DO 58 ISP=1,ISPIN
      WRITE(IUDUMP,1000) LATOM,JDIFAT
      WRITE(IUDUMP,2000) NCYCL
      IF (QOPEN) WRITE(IUDUMP,23000) SPIN(ISP)
      WRITE(IUDUMP,6000)
      CALL PUTONE(PKPOT(1,1,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,7000)
      CALL PUTONE(PKPOT(1,5,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,8000)
      CALL PUTONE(XPOT2(1,ISP),MSTO,IUDUMP)
   58 CONTINUE
   
   59 CONTINUE
C-----------------------------------------------------------------------
C
C     BEGIN SCF ITERATION LOOP.
C
C-----------------------------------------------------------------------
      NITER=0
      QDIVRG=.FALSE.
      QDONE=(QFIXED.OR.(MXITER.LE.0))
      QREDO=.FALSE.

   70 CONTINUE
      IF (QTRACE) WRITE (9,*) '         Iteration ',NITER,' ...'
C-----------------------------------------------------------------------
C
C     GENERATE DENSITY (FOCK-DIRAC) MATRIX.
C
C-----------------------------------------------------------------------
      DO 80 ISP=1,ISPIN
      CALL DENMAT(STDEN(1,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,M2STO,
     X     NORB)
   80 CONTINUE
C-----------------------------------------------------------------------
C
C     READ ATOMIC DATA UP TO THE TWO-ELECTRON INTEGRALS ON IUNIT0.
C
C-----------------------------------------------------------------------
      READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      READ (IUNIT0) LATOM,NUCZ,N,L,ETA,ANORM
      READ (IUNIT0) STOVL,STKIN,STPOT,LV,MV,VEXP
      READ (IUNIT0) STHMAT,ONTRAN
C-----------------------------------------------------------------------
C
C     GENERATE FOCK MATRIX.
C
C-----------------------------------------------------------------------
      CALL FOKMAT(STCOUL,STEXCH,STDEN,BUFFER)
      REWIND IUNIT0

      DO 110 ISP=1,ISPIN

      DO 100 IJ=1,M2STO
      STFOCK(IJ,ISP)=STHMAT(IJ)+STCOUL(IJ,ISP)+STEXCH(IJ,ISP)
     X               +ZPOT2(IJ)+CPOT2(IJ)
      IF (QX2INC) STFOCK(IJ,ISP) = STFOCK(IJ,ISP)+XPOT2(IJ,ISP)
  100 CONTINUE
  
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 110
      WRITE (IUDUMP,1000) LATOM,JDIFAT
      WRITE (IUDUMP,2000) NCYCL
      WRITE (IUDUMP,11000) NITER
      IF (QOPEN) WRITE (IUDUMP,23000) SPIN(ISP)
      WRITE (IUDUMP,12000)
      CALL PUTONE(STDEN(1,ISP),MSTO,IUDUMP)
      WRITE (IUDUMP,13000)
      CALL PUTONE(STCOUL(1,ISP),MSTO,IUDUMP)
      WRITE (IUDUMP,14000)
      CALL PUTONE(STEXCH(1,ISP),MSTO,IUDUMP)
      WRITE (IUDUMP,15000)
      CALL PUTONE(STFOCK(1,ISP),MSTO,IUDUMP)

  110 CONTINUE
  
C-----------------------------------------------------------------------
C
C     COMPUTE APPROXIMATE TOTAL ENERGY AND CHECK FOR CONVERGENCE
C     OR DIVERGENCE.
C
C-----------------------------------------------------------------------

      IF (QDONE) GO TO 180
      NITER=NITER+1
      OLDTE=TE
      TE=AVERAG(STFOCK,STDEN,MSTO,M2STO)
      IF (QOPEN) TE=TE+AVERAG(STFOCK(1,2),STDEN(1,2),MSTO,M2STO)
      TE=TE*ESCALE*ORBOCC
      OLDIFF=ENDIFF
      ENDIFF=DABS(TE-OLDTE)

      IF ((NITER.GE.MXITER).OR.(ENDIFF.LT.CONVRG)) QDONE=.TRUE.

      IF (TE.LT.OLDTE) GO TO 125
      IF (ENDIFF.LE.OLDIFF) GO TO 125
      IF (ENDIFF/TE.LE.0.001/NITER) GO TO 125
      IF (NITER.LE.2) GO TO 125
      IF (QDEBUG) GO TO 125

      IF (.NOT.QDIVRG) GO TO 120
      WRITE (IW,16000) LATOM,JDIFAT
      ITERM1=NITER-1
      WRITE(IW,17000) ITERM1,OLDTE,OLDIFF,NITER,TE,ENDIFF
      QDONE=.TRUE.

  120 QDIVRG=.TRUE.
      GO TO 130

  125 QDIVRG=.FALSE.
  130 CONTINUE
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 140
      WRITE (IUDUMP,1000) LATOM,JDIFAT
      WRITE (IUDUMP,2000) NCYCL
      WRITE (IUDUMP,11000) NITER
      WRITE (IUDUMP,19000) TE

  140 CONTINUE
  
      DO 170 ISP=1,ISPIN
C-----------------------------------------------------------------------
C
C     SUBTRACT THE PSEUDOPOTENTIAL MATRIX FROM THE FOCK MATRIX.
C
C-----------------------------------------------------------------------
      CALL SUBMAT(STFOCK(1,ISP),PKPOT(1,2,ISP),STFOCK(1,ISP),M2STO)

      IF (QDEBUG) THEN
      WRITE (IUDUMP,15400)
      CALL PUTONE(STFOCK(1,ISP),MSTO,IUDUMP)
      END IF
C-----------------------------------------------------------------------
C
C     TRANSFORM THE FOCK MATRIX TO AN ORTHONORMAL BASIS.
C
C-----------------------------------------------------------------------
      CALL UTHU(MSTO,STFOCK(1,ISP),TRANX(1,1,ISP),SCRAT1,ONFOCK)

      IF (QDEBUG) THEN
      WRITE (IUDUMP,15600)
      CALL PUTONE(ONFOCK,MSTO,IUDUMP)
      END IF
C-----------------------------------------------------------------------
C
C     FIND THE NORB LOWEST EIGENVALUES AND EIGENVECTORS.
C
C-----------------------------------------------------------------------
      CALL EISPAK(ONFOCK,SCRAT1,ONEVEC,SCRAT2,MSTO,MSTO,M2STO,
     X            NORB,IERR)

C     REPLACE ARITHMETIC IF WITH MODERN IF-ELSE IF STRUCTURE
      IF (IERR .LT. 0) THEN
         GO TO 150
      ELSE IF (IERR .EQ. 0) THEN
         GO TO 152
      ELSE
         GO TO 151
      END IF

  150 CALL BOMB(17)
  
  151 CALL WARN(15,QDUMMY)
      IF (IERR.LE.NORB) CALL BOMB(16)

  152 CONTINUE
  
      CALL DCOPY(SCRAT1,NORB,EVAL(1,ISP))
      IF (QDEBUG) THEN
      WRITE (IUDUMP,15700)
      CALL OUTVEC(ONEVEC,EVAL(1,ISP),MSTO,NORB,IUDUMP)
      END IF
C-----------------------------------------------------------------------
C
C     SAVE OLD VECTORS FOR MAXIMUM OVERLAP ORDERING.
C
C-----------------------------------------------------------------------
      CALL DCOPY(STEVEC(1,1,ISP),MSTO*NORB,OLDVEC(1,1,ISP))
C-----------------------------------------------------------------------
C
C     TRANSFORM THE EIGENVECTORS BACK TO THE NONORTHOGONAL BASIS.
C
C-----------------------------------------------------------------------
      CALL DMPAB(TRANX(1,1,ISP),MSTO,MSTO,MSTO,MSTO,ONEVEC,MSTO,
     X           NORB,MSTO,NORB,STEVEC(1,1,ISP),MSTO,NORB)

      IF (QDEBUG) THEN
      WRITE (IUDUMP,15800)
      CALL OUTVEC(STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,NORB,IUDUMP)
      END IF
C-----------------------------------------------------------------------
C
C     CONSTRUCT NORMALIZED SINGLE-CENTER PROJECTION OF EIGENVECTORS.
C     (USE LOWDIN SYMMETRIC ORTHONORMALIZATION)
C
C-----------------------------------------------------------------------
      IF (.NOT.QPSMET.OR.QTRUNC) GO TO 160
      CALL DMPATB(PROJEC(1,1,ISP),MSTO,MSTO,MSTO,MSTO,STEVEC(1,1,ISP),
     X            MSTO,NORB,MSTO,NORB,ONEVEC,MSTO,NORB)
      CALL CTSC(ONEVEC,STOVL,SCRAT1,STOVL,MSTO,NORB)
      CALL LOWDIN(STOVL,ONTRAN,SCRAT1,NORB,NORB,2,QINDEF)

      IF (QINDEF) GO TO 160
      CALL DMPAB(ONEVEC,MSTO,NORB,MSTO,NORB,ONTRAN,NORB,NORB,NORB,NORB,
     X           STEVEC(1,1,ISP),MSTO,NORB)
      READ(IUTEMP) STOVL,ONTRAN
      REWIND IUTEMP

  160 CONTINUE
      IF (.NOT.QNRMLZ) GO TO 165
      READ(IUTEMP) STOVL,ONTRAN
      REWIND IUTEMP
      CALL NORMLZ(STEVEC(1,1,ISP),NORB,STOVL,MSTO,M2STO)

C-----------------------------------------------------------------------
C     REMOVE PSEUDOPOTENTIAL TERM FROM FOCK MATRIX
C-----------------------------------------------------------------------
  165 CONTINUE
      CALL ADDMAT(STFOCK(1,ISP),PKPOT(1,2,ISP),STFOCK(1,ISP),M2STO)
C-----------------------------------------------------------------------
C
C     DETERMINE THE MAXIMUM OVERLAP ORDERING AND REORDER OCCUPATION
C     NUMBERS IF NECESSARY.
C
C-----------------------------------------------------------------------
      CALL MAXOVL(IORDER(1,ISP),OLDVEC(1,1,ISP),STEVEC(1,1,ISP),
     X     STOVL,MSTO,M2STO,NORB,QNWORD)
      IF (QDEBUG.AND.QNWORD) WRITE (IW,*) 
     X                       'Orbital occupations being reordered...'
      IF (QNWORD) CALL NEWORD(OCC(1,ISP),SCRAT1,IORDER(1,ISP),NORB)
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------
      IF (.NOT.QDEBUG) GO TO 170
      WRITE (IUDUMP,22000)
      IF (QOPEN) WRITE (IUDUMP,23000) SPIN(ISP)
      CALL OUTVEC(STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,NORB,IUDUMP)
      IF (QNWORD) WRITE (IUDUMP,18000) (IORDER(K,ISP),K=1,NORB)
  170 CONTINUE
C-----------------------------------------------------------------------
C
C     END THE SCF ITERATION LOOP.
C
C-----------------------------------------------------------------------
      GO TO 70

  180 CONTINUE
C-----------------------------------------------------------------------
C
C     PERFORM THE ENERGY, CHARGE, SPIN ANALYSIS AND DETERMINE THE NEW
C     LOCAL MODEL POTENTIAL COEFFICIENTS.
C
C-----------------------------------------------------------------------
      IF (QTRACE) WRITE (9,*) '         Performing Analysis...'
      IF ((ENDIFF.GT.CONVRG).AND.(.NOT.QFIXED)) 
     X    WRITE (IW,20010) ENDIFF,NITER
      IF (QPRINT.AND.(ENDIFF.LE.CONVRG)) WRITE (IW,20000) ENDIFF,NITER
      IF (QPRINT.AND.(VDAMP.GT.ZERO)) WRITE (IW,21000) VDAMP
      TCOUL0=ORBOCC*AVERAG(STPOT,STDEN,MSTO,M2STO)
      IF (QOPEN) TCOUL0=TCOUL0+AVERAG(STPOT,STDEN(1,2),
     X                                MSTO,M2STO)
      CALL ANALYS(STKIN,STPOT,STCOUL,STEXCH,CPOT2,ZPOT2,VFIT2,VCOEF,
     X     VEXP,LV,PKPOT,XPOT2,STEVEC,EVAL,OCC,GAMMA,INDEX,TOTEN,TKE,
     X     ZZREP,STOVL,CHARGE,TCOUL0)
      IF (QDEBUG) CALL PLOTV(LV,MV,VEXP,VCOEF,NVTERM,IUDUMP)

  185 CONTINUE
C-----------------------------------------------------------------------
C
C     CONSTRUCT NEW CONTRIBUTIONS TO THE PROJECTION, PSEUDOPOTENTIAL,
C     AND EXCHANGE TO BE USED EXTERNALLY.
C
C-----------------------------------------------------------------------
      DO 200 ISP=1,ISPIN

      IJORB = 0
      DO 189 IORB=1,NORB
      DO 190 JORB=1,IORB
      IJORB = IJORB + 1
      AOFOCK(IJORB,ISP) = ZERO
      IF (JORB.EQ.IORB) AOFOCK(IJORB,ISP) = GAMMA(IORB,ISP)
  190 CONTINUE
  189 CONTINUE
  
      CALL DENMAT(PKPOT(1,4,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,4,ISP),PKSCAL,M2STO)
      CALL DENMAT(PKPOT(1,5,ISP),STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,5,ISP),PKSCAL,M2STO)
      IF (.NOT.QPSMET) PKPOT(IJ,5,ISP) = PKPOT(IJ,5,ISP)
     X                             - PKE*PKPOT(IJ,4,ISP)

      DO 196 IJ=1,M2STO
      ONFOCK(IJ) = ZERO
      IF (QX2FIT) ONFOCK(IJ) = ONFOCK(IJ) + STEXCH(IJ,ISP)
      IF (.NOT.QC2FIT) GOTO 195
      ONFOCK(IJ) = ONFOCK(IJ) + STCOUL(IJ,ISP)
      DO 194 NV=1,NVTERM
      ONFOCK(IJ) = ONFOCK(IJ) - VCOEF(NV)*STPOT(IJ,NV)
  194 CONTINUE
  195 CONTINUE
  196 CONTINUE
      CALL UTHU(MSTO,ONFOCK,ONTRAN,SCRAT1,PKPOT(1,6,ISP))
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 200
      WRITE(IUDUMP,1000) LATOM,JDIFAT
      WRITE(IUDUMP,2000) NCYCL
      IF (QOPEN) WRITE(IUDUMP,23000) SPIN(ISP)
      WRITE(IUDUMP,25000)
      CALL PUTONE(PKPOT(1,4,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,26000)
      CALL PUTONE(PKPOT(1,5,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,27000)
      CALL PUTONE(PKPOT(1,6,ISP),MSTO,IUDUMP)

  200 CONTINUE
  
C-----------------------------------------------------------------------
C
C     PRINT THE FINAL VECTORS, ENERGIES AND ESCA INTENSITIES.
C
C-----------------------------------------------------------------------

      IF (.NOT.QPRINT) GO TO 240
      WRITE (IW,1000) LATOM,JDIFAT
      WRITE (IW,2000) NCYCL
      IF (QSEPAT) WRITE (IW,3000)
      WRITE (IW,22000)

      DO 230 ISP=1,ISPIN
      IF (QOPEN) WRITE (IW,23000) SPIN(ISP)
      CALL OUTVEC(STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,NORB,IW)
      CALL ESCA(STEVEC(1,1,ISP),EVAL(1,ISP),N,L,ETA,
     X          NORB,MSTO,NSTO,IW)
  230 CONTINUE
  
      WRITE (IW,28000) LATOM,JDIFAT
      CALL TIMOUT(5)

  240 CONTINUE
C-----------------------------------------------------------------------
C
C     PUNCH VECTORS, ENERGIES, OCCUPATIONS, ETC.
C
C-----------------------------------------------------------------------
      IF (QLAST.AND.QPNCHV) CALL PUNCHV(STEVEC,EVAL,OCC,L,NSTO,MSTO,
     X     NORB,ISPIN,IP)
      IF (QLAST.AND.QPNCHV) WRITE (IW,24000) LATOM,JDIFAT,IP
C-----------------------------------------------------------------------
C
C     INCREMENT SOME MOLECULAR QUANTITIES.
C
C-----------------------------------------------------------------------
      CALL DCOPY(OCC,NORB,POPUL(IORBT+1,1))
      IF (QOPEN) CALL DCOPY(OCC(1,2),NORB,POPUL(IORBT+1,2))
      IORBT=IORBT+NORB

  250 CONTINUE
      IATOM=IATOM+1
      TMOLE=TMOLE+TOTEN
      TOTKE=TOTKE+TKE
      ZZPOT=ZZPOT+ZZREP/TWO
C-----------------------------------------------------------------------
C
C     WRITE VECTORS, ENERGIES, OCCUPATIONS, ETC. ON IUNIT3.
C
C-----------------------------------------------------------------------
      IF (QATOM) WRITE (IUNIT3) MDIFAT
      WRITE (IUNIT3) QFLAGS,NEQAT,X,Y,Z,STEVEC,EVAL,OCC,AOFOCK,GAMMA,
     X     CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
C-----------------------------------------------------------------------
C
C     LOOP OVER EQUIVALENT ATOMS.
C
C-----------------------------------------------------------------------

      IF (NEQAT.LE.0) GO TO 290
      WRITE(IUTEMP) STOVL,ONTRAN
      WRITE(IUTEMP) STEVEC,EVAL,OCC,PKPOT,AOFOCK,GAMMA,VCOEF,TOTEN
      ZZREP0=ZZREP
      REWIND IUTEMP

      DO 280 IEQAT=1,NEQAT
      IF (QTRACE) WRITE (9,*) '      Equivalent Atom ',IEQAT,' ...'
      READ (IUNIT4) QFLAGS,ISYM,IXYZ,X,Y,Z,STEVEC,EVAL,OCC,AOFOCK,
     X     GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      READ(IUTEMP) STOVL,ONTRAN
      READ(IUTEMP) STEVEC,EVAL,OCC,PKPOT,AOFOCK,GAMMA,VCOEF,TOTEN
      REWIND IUTEMP
      CHARGE(2,IATOM)=CHARGE(2,IATOM-1)
      CHARGE(3,IATOM)=CHARGE(3,IATOM-1)

      IF (ISYM.EQ.0) GO TO 258
      CALL SYMOP(ISYM,IXYZ,STEVEC,VCOEF,L,LV,D,SCRAT1,NSTO,MSTO,NORB,
     X           ISPIN,NVTERM,LMAX,LLMXP1)
C-----------------------------------------------------------------------
C
C     RE-CONSTRUCT COULOMB AND EXCHANGE MATRICES
C
C-----------------------------------------------------------------------
      DO 265 ISP=1,ISPIN
      CALL DENMAT(STDEN(1,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,M2STO,
     X     NORB)
 265  CONTINUE
      READ (IUNIT0) NSTO,MSTO,NORB,NVTERM
      READ (IUNIT0) LATOM,NUCZ,N,L,ETA,ANORM
      READ (IUNIT0) STOVL,STKIN,STPOT,LV,MV,VEXP
      READ (IUNIT0) STHMAT,ONTRAN
      CALL FOKMAT(STCOUL,STEXCH,STDEN,BUFFER)
      REWIND IUNIT0
C-----------------------------------------------------------------------
C
C     RE-CONSTRUCT CONTRIBUTIONS TO THE PROJECTION, PSEUDOPOTENTIAL,
C     AND EXCHANGE MATRICES TO BE USED EXTERNALLY.
C
C-----------------------------------------------------------------------

      DO 256 ISP=1,ISPIN

      IJORB = 0
      DO 251 IORB=1,NORB
      DO 252 JORB=1,IORB
      IJORB = IJORB + 1
      AOFOCK(IJORB,ISP) = ZERO
      IF (JORB.EQ.IORB) AOFOCK(IJORB,ISP) = GAMMA(IORB,ISP)
  252 CONTINUE
  251 CONTINUE
  
      CALL DENMAT(PKPOT(1,4,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,4,ISP),PKSCAL,M2STO)
      CALL DENMAT(PKPOT(1,5,ISP),STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,5,ISP),PKSCAL,M2STO)
      IF (.NOT.QPSMET) PKPOT(IJ,5,ISP) = PKPOT(IJ,5,ISP) 
     X                             - PKE*PKPOT(IJ,4,ISP)

      DO 255 IJ=1,M2STO
      ONFOCK(IJ) = ZERO
      IF (QX2FIT) ONFOCK(IJ) = ONFOCK(IJ) + STEXCH(IJ,ISP)
      IF (.NOT.QC2FIT) GOTO 254
      ONFOCK(IJ) = ONFOCK(IJ) + STCOUL(IJ,ISP)
      DO 253 NV=1,NVTERM
      ONFOCK(IJ) = ONFOCK(IJ) - VCOEF(NV)*STPOT(IJ,NV)
  253 CONTINUE
  254 CONTINUE
  255 CONTINUE
      CALL UTHU(MSTO,ONFOCK,ONTRAN,SCRAT1,PKPOT(1,6,ISP))
C-----------------------------------------------------------------------
C
C     DIAGNOSTIC OUTPUT.
C
C-----------------------------------------------------------------------

      IF (.NOT.QDEBUG) GO TO 256
      WRITE(IUDUMP,1000) LATOM,JDIFAT
      WRITE(IUDUMP,1500) IEQAT
      WRITE(IUDUMP,2000) NCYCL
      IF (QOPEN) WRITE(IUDUMP,23000) SPIN(ISP)
      WRITE(IUDUMP,25000)
      CALL PUTONE(PKPOT(1,4,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,26000)
      CALL PUTONE(PKPOT(1,5,ISP),MSTO,IUDUMP)
      WRITE(IUDUMP,27000)
      CALL PUTONE(PKPOT(1,6,ISP),MSTO,IUDUMP)

  256 CONTINUE
  
  258 CONTINUE
  
      IF (.NOT.QFLIP) GO TO 260
      CHARGE(3,IATOM)=-CHARGE(3,IATOM)
      CALL FLIP(STEVEC,1,ISPIN,STEVEC,ISPIN,ISPIN,MSTO*NORB)
      CALL FLIP(EVAL,1,ISPIN,EVAL,ISPIN,ISPIN,NORB)
      CALL FLIP(OCC,1,ISPIN,OCC,ISPIN,ISPIN,NORB)
      CALL FLIP(AOFOCK,1,ISPIN,AOFOCK,ISPIN,ISPIN,N2ORB)
      CALL FLIP(GAMMA,1,ISPIN,GAMMA,ISPIN,ISPIN,NORB)
      CALL FLIP(PKPOT(1,4,1),1,1,PKPOT(1,4,ISPIN),1,1,3*M2STO)

  260 CONTINUE
      IATOM=IATOM+1
      TMOLE=TMOLE+TOTEN
      TOTKE=TOTKE+TKE
      IF (ISYM.EQ.0) ZZREP=ZZREP0
      ZZPOT=ZZPOT+ZZREP/TWO

      IF (QVIRTL) GO TO 270
      CALL DCOPY(OCC,NORB,POPUL(IORBT+1,1))
      IF (QOPEN) CALL DCOPY(OCC(1,2),NORB,POPUL(IORBT+1,2))
      IORBT=IORBT+NORB

  270 CONTINUE
      IF (QLAST.AND.QPNCHV) CALL PUNCHV(STEVEC,EVAL,OCC,L,NSTO,MSTO,
     X     NORB,ISPIN,IP)
      MATOM=IEQAT*MDIFAT+JDIFAT
      IF (QLAST.AND.QPNCHV) WRITE (IW,24000) LATOM,MATOM,IP
      WRITE (IUNIT3) QFLAGS,ISYM,IXYZ,X,Y,Z,STEVEC,EVAL,OCC,AOFOCK,
     X     GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      IF (IEQAT.GE.NEQAT) GO TO 280
      WRITE(IUTEMP) STOVL,ONTRAN
      WRITE(IUTEMP) STEVEC,EVAL,OCC,PKPOT,AOFOCK,GAMMA,VCOEF,TOTEN
      REWIND IUTEMP
C-----------------------------------------------------------------------
C
C     END LOOPS OVER ATOMS.
C
C-----------------------------------------------------------------------
  280 CONTINUE
  
  290 CONTINUE
  
      RETURN
  
 1000 FORMAT(///' ',A2,I2,/' ____')
 1500 FORMAT('      (EQUIVALENT ATOM ',I2,')')
 2000 FORMAT(' ',T100,'... CYCLE',I3,' ...')
 3000 FORMAT(' ',T45,'+++ NEAR SEPARATED-ATOM LIMIT +++')
 4000 FORMAT(/' KINETIC ENERGY MATRIX...')
 4100 FORMAT(/' NUCLEAR ATTRACTION (MODEL POTENTIAL) MATRIX...')
 4105 FORMAT(/' (',I3,')',F7.3,' R**(',I2,') Y(',I1,',',I2,') EXP(',
     X          F6.3,' R)...'/)
 4200 FORMAT(/' METRIC MATRIX...')
 4500 FORMAT(/' ORTHONORMAL TRANSFORMATION MATRIX...')
 5000 FORMAT(/' EXTERNAL NUCLEAR ATTRACTION MATRIX...')
 6000 FORMAT(/' EXTERNAL PROJECTION MATRIX...')
 7000 FORMAT(/' EXTERNAL PSEUDOPOTENTIAL MATRIX...')
 8000 FORMAT(/' EXTERNAL EXCHANGE MATRIX...')
 9000 FORMAT(/' 2-CENTER COULOMB (LOCAL MODEL) POTENTIAL MATRIX...')
10000 FORMAT(/' MODIFIED METRIC MATRIX...')
11000 FORMAT(/' ... ITERATION',I3,' ...')
12000 FORMAT(/' DENSITY MATRIX...')
13000 FORMAT(/' HF COULOMB MATRIX...')
14000 FORMAT(/' HF EXCHANGE MATRIX...')
15000 FORMAT(/' FOCK MATRIX...')
15400 FORMAT(/' FOCK MATRIX LESS PSEUDOPOTENTIAL...')
15600 FORMAT(/' FOCK MATRIX IN ORTHONORMAL BASIS...')
15700 FORMAT(/' EIGENVECTORS AND EIGENENERGIES IN ORTHONORMAL BASIS...')
15800 FORMAT(/' UNPROJECTED EIGENVECTORS AND EIGENENERGIES...')
16000 FORMAT(/' *** CONVERGENCE DIFFICULTY MET FOR ',A2,I2,' ***')
17000 FORMAT(/' SUMMARY OF TOTAL ENERGY CONVERGENCE...'//
     X       ' OLD TOTAL ENERGY (',I2,')...',1PD15.8/
     X       ' OLD ENERGY DIFFERENCE.......',1PD8.1/
     X       ' NEW TOTAL ENERGY (',I2,')...',1PD15.8/
     X       ' NEW ENERGY DIFFERENCE.......',1PD8.1)
18000 FORMAT(/' OCCUPATION NUMBERS REORDERED BY:',35I3)
19000 FORMAT(/' APPROXIMATE TOTAL ENERGY...',F9.4)
20000 FORMAT(/' THE TOTAL ENERGY CONVERGED TO',1PD8.1,' AFTER',I3,
     X       ' ITERATIONS.')
20010 FORMAT(/' THE TOTAL ENERGY *NOT* CONVERGED (',1PD8.1,') AFTER',I3,
     X       ' ITERATIONS.')
21000 FORMAT(/' THE EXTERNAL POTENTIALS WERE DAMPED THIS CYCLE BY',F7.4)
22000 FORMAT(/' FINAL VECTORS AND ENERGIES...')
23000 FORMAT(/' ',A8)
24000 FORMAT(/' ... ATOMIC ORBITALS FOR ',A2,I2,' HAVE BEEN PUNCHED',
     X       ' ON UNIT',I3,' ...')
25000 FORMAT(/' CONTRIB TO PROJECTION MATRIX...')
26000 FORMAT(/' CONTRIB TO PSEUDOPOTENTIAL MATRIX...')
27000 FORMAT(/' CONTRIB TO EXTERNAL EXCHANGE MATRIX...')
28000 FORMAT(/' ... END OF SCF FOR ',A2,I2,' ...'//)
      END
