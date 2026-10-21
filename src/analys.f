      SUBROUTINE ANALYS(STKIN,STPOT,STCOUL,STEXCH,CPOT2,ZPOT2,VFIT2,
     X                  VCOEF,VEXP,LV,PKPOT,XPOT2,STEVEC,EVAL,OCC,
     X                  GAMMA,INDEX,TOTEN,TKE,ZZREP,STOVL,CHARGE,TCOUL0)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     ANALYS...
C
C        THIS ROUTINE COMPUTES AND PRINTS AN ANALYSIS OF VARIOUS CONTRI-
C     BUTIONS TO THE TOTAL ENERGY, SPIN, AND CHARGE.  NEW COEFFICIENTS
C     FOR THE LOCAL MODEL POTENTIAL ARE ALSO DETERMINED.
C
C     VARIABLE DEFINITIONS:
C
C        STKIN(*)........ KINETIC ENERGY MATRIX.
C        STPOT(*,*)...... POTENTIAL ENERGY (1-CENTER) MATRIX (BY TERM).
C        STCOUL(*,*)..... COULOMB PART OF THE HF POTENTIAL MATRIX.
C        STEXCH(*,*)..... EXCHANGE PART OF HF POTENTIAL MATRIX.
C        ZPOT2(*)........ EXTERNAL CONTRIBUTION TO NUCLEAR ATTRACTION.
C        CPOT2(*)........ EXTERNAL COULOMB (LOCAL MOD) POTENTIAL MATRIX.
C        XPOT2(*,*)...... EXTERNAL EXCHANGE POTENTIAL MATRIX.
C        PKPOT(*,1,*).... PROJECTION MATRIX.
C             (*,2,*).... PSEUDOPOTENTIAL MATRIX.
C             (*,3,*).... EXTERNAL EXCHANGE (=XPOT2)
C        VFIT2(*)........ MODEL POTENTIAL APPROX. TO ZPOT2(*).
C        VCOEF(*)........ COEFFICIENTS IN THE LOCAL MODEL POTENTIAL.
C        VEXP(*)......... EXPONENTS IN THE MODEL POTENTIAL.
C        LV(*)........... L-VALUES IN THE MODEL POTENTIAL EXPRESSION.
C        WGT2............ RELATIVE WEIGHT GIVEN TO THE CONDITION THAT
C                         THE MODEL POTENTIAL REPRODUCES THE EXTERNAL
C                         NUCLEAR ATTRACTION ENERGY. (ALL WEIGHTS ARE
C                         RELATIVE TO THE Z/R BEHAVIOR AT THE NUCLEUS).
C        WGTV............ RELATIVE WEIGHT GIVEN TO THE CONDITION THAT
C                         THE MODEL POTENTIAL REPRODUCES THE VALENCE
C                         HF COULOMB ENERGY.
C        WGTC............ RELATIVE WEIGHT GIVEN TO THE CONDITION THAT
C                         THE MODEL POTENTIAL REPRODUCES THE CORE
C                         HF COULOMB ENERGY.
C        STEVEC(*,*,*)... HF EIGENVECTORS (ORBITALS).
C        EVAL(*,*)....... HF EIGENVALUES (ORBITAL ENERGIES).
C        OCC(*,*)........ ORBITAL OCCUPATION NUMBERS.
C        GAMMA(*,*)...... ADJUSTED ORBITAL ENERGIES.
C        INDEX(I)........ I*(I-1)/2, INDEXING ARRAY.
C        TOTEN........... ATOMIC CONTRIBUTION TO MOLECULAR TOTAL ENERGY.
C        TKE............. TOTAL ATOMIC KINETIC ENERGY.
C        ZZREP........... NUCLEAR REPULSION FOR THIS ATOM.
C        STOVL(*)........ BASIS METRIC MATRIX.
C        CHARGE(1,*)..... ATOMIC NUCLEAR CHARGES.
C              (2,*)..... ATOMIC ELECTRONIC CHARGES.
C              (3,*)..... ATOMIC NET SPINS.
C        TCOUL0.......... TOTAL COULOMB POTENTIAL AT NUCLEUS.
C        ELECS(*)........ TOTAL SPINS IN MOLECULE.
C        ELSUM........... =ELECS(1)+ELECS(2), TOTAL ELECTRONIC CHARGE.
C        SPIN(*)....... ..SPIN LABELS.
C        NATOM........... TOTAL NUMBER OF ATOMS IN THE MOLECULE.
C        IATOM........... INDEX OF THE CURRENT ATOM.
C        MSTO............ NO. OF STO'S IN THE BASIS (INCLUDING ML'S).
C        NORB............ NO. OF ATOMIC ORBITALS.
C        M2STO........... =INDEX(MSTO+1).
C        NVTERM.......... NO. OF TERMS IN LOCAL POTENTIAL.
C        ISPIN........... =1, IF (.NOT.QOPEN).
C                         =2, IF (QOPEN).
C        QOPEN........... =T --> SPIN-UNRESTRICTED CASE (OPEN SHELL).
C                         =F --> SPIN-RESTRICTED CASE.
C        QPRINT.......... =T --> INTERMEDIATE RESULTS FOR EACH CYCLE
C                         WILL BE PRINTED.
C        QATOM........... =T --> ATOMIC CALCULATION ONLY.
C        QSEPAT.......... =T --> THE 1-ST CYCLE IS A SEPARATED-ATOM
C                         CALCULATION SO STARTING VECTORS, ETC. ARE
C                         NOT NECESSARILY RELIABLE.
C        QFIXED.......... =T --> MODEL POTENTIAL IS FIXED.
C        IW.............. FORTRAN I/O UNIT FOR WRITING.
C
C     ENTRY:  CHGAN...
C
C        THIS ENTRY WRITES A CHARGE AND SPIN ANALYSIS ONLY.
C
C     ROUTINES CALLED:  DERASE, DCOPY, ASUM; DMIN1, DSQRT, DABS, DEXP
C
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM(1)(=NATOM),   IPARM(5)(=MSTO),
C                        IPARM(6)(=NORB),    IPARM(10)(=IATOM),
C                        IPARM(27)(=M2STO),
C                        IPARM(31)(=NVTERM), IPARM(33)(=ISPIN),
C                        QPARM(4)(=QOPEN),   QPARM(8)(=QSEPAT),
C                        QPARM(11)(=QFIXED),
C                        QPARM(22)(=QPRINT), QPARM(24)(=QATOM),
C                        QPARM(25)(=QMODVO)
C                        QPARM(32)(=QC2FIT)
C
C        /IODATA/ USES - IUNIT(6)(=IW), IUNIT(8)(=IUDUMP)
C
C        /MODPOT/ USES - WGT2, WGTV, WGTC
C
C        /CHARGS/ USES - ELSUM, ELECS(*), SPIN(*)
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(4),QOPEN),   
     X            (QPARM(7),QDEBUG),  (QPARM(8),QSEPAT),
     X            (QPARM(11),QFIXED),
     X            (QPARM(22),QPRINT), (QPARM(24),QATOM),
     X            (QPARM(25),QMODVO), (QPARM(32),QC2FIT)
      EQUIVALENCE (IPARM(1),NATOM),   (IPARM(5),MSTO),
     X            (IPARM(6),NORB),    (IPARM(10),IATOM),
     X            (IPARM(27),M2STO),  (IPARM(31),NVTERM),
     X            (IPARM(33),ISPIN)
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(6),IW),      (IUNIT(8),IUDUMP)
      DIMENSION STKIN(M2STO),                 STPOT(M2STO,NVTERM),      
     X          STCOUL(M2STO,ISPIN),          STEXCH(M2STO,ISPIN),
     X          VCOEF(NVTERM),                LV(NVTERM),
     X          VEXP(NVTERM),                 XPOT2(M2STO,ISPIN),
     X          CPOT2(M2STO),                 STEVEC(MSTO,NORB,ISPIN),
     X          EVAL(NORB,ISPIN),
     X          OCC(NORB,ISPIN),              INDEX(MSTO),
     X          GAMMA(NORB,ISPIN),            ZPOT2(M2STO),
     X          VFIT2(NVTERM),                STOVL(M2STO),
     X          PKPOT(M2STO,6,ISPIN),         CHARGE(3,NATOM)
      DIMENSION CHARG(6),OE(9),TE(9),
     X          EPOT(20),VSUM(20),VMAT(210),VINV(20,20),SCRAT1(20,20)
      EQUIVALENCE (OE(1),EORB),  (OE(2),EKIN),  (OE(3),ENUC),
     X            (TE(1),TEORB), (TE(2),TEKIN), (TE(3),TENUC),
     X            (OE(4),ECHF),  (OE(5),EXHF),  (OE(6),EVZ2),
     X            (TE(4),TECHF), (TE(5),TEXHF), (TE(6),TEVZ2),
     X            (OE(7),EVC2),  (OE(8),EVX2),  (OE(9),PKV2),
     X            (TE(7),TEVC2), (TE(8),TEVX2), (TE(9),TPKV2)
      EQUIVALENCE (CHARG(1),ASPIN), (CHARG(2),BSPIN), (CHARG(3),SPDIFF),
     X            (CHARG(4),ZNUC),  (CHARG(5),ECHARG),(CHARG(6),TOTCHG)
	CHARACTER*8 SPIN
      COMMON /CHARGS/ ZNSUM,ELSUM,ELDIF,ELECS(2),SPIN(2),ORBOCC,CHTRAN
      COMMON /MODPOT/ WGT0,WGT1,WGT2,WGTC,WGTV,VDAMP,VACCEL,
     X                TFEXP(4),TFCOEF(4),LTF(4),MXTF
      DATA ZERO/0.D0/,HALF/0.5D0/,ONE/1.D0/,TWO/2.D0/

      QPKINC=.TRUE.
C-----------------------------------------------------------------------
C
C     INITIALIZE ENERGIES, CHARGES, POTENTIAL FIT MATRICES, ETC.
C
C-----------------------------------------------------------------------
      ORBOCC=3-ISPIN
      CHARG(1)=ASUM(OCC,NORB,1)
      CHARG(ISPIN)=ASUM(OCC(1,ISPIN),NORB,1)
      SUMOCC=CHARG(1)+CHARG(ISPIN)
      ZNUC=CHARGE(1,IATOM)
      OLDCHG=CHARGE(2,IATOM)
      NELEC=DMIN1(OLDCHG,ZNUC)
      NCORE=NOBLE(NELEC)
      CALL DERASE(TE,9)
      CALL DERASE(VSUM,20)
      CALL DERASE(VMAT,210)
      CALL DERASE(VINV,400)
      VDEVV=ZERO
      VDEVC=ZERO
      VCALCT=ZERO
      TEVAL=ZERO
      TPKERR=ZERO
	  
      WGT2SV=WGT2
      IF (QATOM.OR.QSEPAT) WGT2=ZERO   
      QNOFIT = .FALSE.
      IF (QFIXED.OR..NOT.QC2FIT) QNOFIT=.TRUE.
      IF (WGTC+WGTV+WGT2.LE.0.1) QNOFIT=.TRUE.
      IF (NVTERM.LT.3) QNOFIT=.TRUE.
      IF (QPRINT) WRITE (IW,1000)
	  
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER SPIN.
C
C-----------------------------------------------------------------------
      DO 190 ISP=1,ISPIN
C-----------------------------------------------------------------------
C
C     WRITE HEADINGS.
C
C-----------------------------------------------------------------------
      IF (QPRINT.AND.QOPEN) WRITE (IW,2000) SPIN(ISP)
      IF (QPRINT) WRITE (IW,3000)
	  
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER ORBITALS.
C
C-----------------------------------------------------------------------
      DO 180 IORB=1,NORB
C-----------------------------------------------------------------------
C
C     INTIALIZE THE ORBITAL ENERGY CONTRIBUTIONS WITH THE (1,1) ELEMENT.
C
C-----------------------------------------------------------------------
      C1C1=STEVEC(1,IORB,ISP)**2
      EKIN=STKIN(1)*C1C1
      ECHF=STCOUL(1,ISP)*C1C1
      EXHF=STEXCH(1,ISP)*C1C1
      EVZ2=ZPOT2(1)*C1C1
      EVC2=CPOT2(1)*C1C1
      EVX2=XPOT2(1,ISP)*C1C1
      PROJ=PKPOT(1,1,ISP)*C1C1
      PKV2=PKPOT(1,2,ISP)*C1C1
	  
      DO 60 NV=1,NVTERM
      EPOT(NV)=STPOT(1,NV)*C1C1
   60 CONTINUE
   
      IF (MSTO.LT.2) GO TO 110
	  
C-----------------------------------------------------------------------
C
C     ADD REMAINING DIAGONAL ELEMENTS TO ORBITAL ENERGY CONTRIBUTIONS.
C
C-----------------------------------------------------------------------
      DO 100 K=2,MSTO
      INDXK=INDEX(K)
      KK=INDXK+K
      KM1=K-1
      CKCK=STEVEC(K,IORB,ISP)**2
      EKIN=EKIN+STKIN(KK)*CKCK
      ECHF=ECHF+STCOUL(KK,ISP)*CKCK
      EXHF=EXHF+STEXCH(KK,ISP)*CKCK
      EVZ2=EVZ2+ZPOT2(KK)*CKCK
      EVC2=EVC2+CPOT2(KK)*CKCK
      EVX2=EVX2+XPOT2(KK,ISP)*CKCK
      PROJ=PROJ+PKPOT(KK,1,ISP)*CKCK
      PKV2=PKV2+PKPOT(KK,2,ISP)*CKCK
	  
      DO 70 NV=1,NVTERM
      EPOT(NV)=EPOT(NV)+STPOT(KK,NV)*CKCK
   70 CONTINUE
   
C-----------------------------------------------------------------------
C
C     ADD THE OFF-DIAGONAL ELEMENTS TO THE ORBITAL ENERGY CONTRIBUTIONS.
C
C-----------------------------------------------------------------------
      DO 90 L=1,KM1
      KL=INDXK+L
      CKCL2=TWO*STEVEC(K,IORB,ISP)*STEVEC(L,IORB,ISP)
      EKIN=EKIN+STKIN(KL)*CKCL2
      ECHF=ECHF+STCOUL(KL,ISP)*CKCL2
      EXHF=EXHF+STEXCH(KL,ISP)*CKCL2
      EVZ2=EVZ2+ZPOT2(KL)*CKCL2
      EVC2=EVC2+CPOT2(KL)*CKCL2
      EVX2=EVX2+XPOT2(KL,ISP)*CKCL2
      PROJ=PROJ+PKPOT(KL,1,ISP)*CKCL2
      PKV2=PKV2+PKPOT(KL,2,ISP)*CKCL2
	  
      DO 80 NV=1,NVTERM
      EPOT(NV)=EPOT(NV)+STPOT(KL,NV)*CKCL2
   80 CONTINUE
   
   90 CONTINUE
   
  100 CONTINUE
  
  110 CONTINUE
      ENUC=-ZNUC*EPOT(1)
      OCCUP=OCC(IORB,ISP)*ORBOCC
C----------------------------------------------------------------------
C
C     DETERMINE PARTIAL SUMS FOR THE MODEL POTENTIAL PARMETERS.
C
C----------------------------------------------------------------------
      VDEV=ORBOCC*ECHF
	  
      DO 113 NV=1,NVTERM
         VDEV=VDEV-ORBOCC*VCOEF(NV)*EPOT(NV)
  113 CONTINUE
  
      IF (IORB.LE.NCORE) VDEVC=VDEVC+VDEV*VDEV
      IF (IORB.GT.NCORE) VDEVV=VDEVV+VDEV*VDEV

      WGT=WGTV/(ORBOCC*(NORB-NCORE))
      IF (IORB.LE.NCORE) WGT=WGTC/(ORBOCC*NCORE)
C     WGT=WGT/(ECHF+HALF)
	  
      VCALC = VCOEF(1)*EPOT(1)
      IJ=0
      DO 144 I=2,NVTERM  
      DO 142 J=2,I
      IJ=IJ+1
      VMAT(IJ)=VMAT(IJ)+EPOT(I)*WGT*EPOT(J)
  142 CONTINUE  
      VSUM(I-1)=VSUM(I-1)+EPOT(I)*WGT*(ECHF-SUMOCC*EPOT(1))
      VCALC = VCALC + VCOEF(I)*EPOT(I)
  144 CONTINUE  
      VCALCT = VCALCT+OCCUP*VCALC
  
C-----------------------------------------------------------------------
C
C     MODIFY THE VIRTUAL ORBITAL ENERGIES TO "FEEL" N-1 POTENTIAL.
C
C-----------------------------------------------------------------------
      IF (.NOT.QMODVO) GO TO 160
      SMAX=NATOM-1
      HOLE=ONE-OCC(IORB,ISP)
      SCREEN=ZERO
      IF (NATOM.GT.1) SCREEN=DLOG(ELSUM/SUMOCC)/SMAX
      SCHOLE=HOLE*DEXP(-SCREEN*PROJ)
      ECHF=ECHF*(SUMOCC-SCHOLE)/SUMOCC
      EL2=ELSUM-SUMOCC
      IF (EL2.NE.ZERO) EVC2=EVC2*(EL2-HOLE+SCHOLE)/EL2
      SCREEN=ZERO
      IF ((NATOM.GT.1).AND.(CHARG(ISP).NE.ZERO)) SCREEN=DLOG(ELECS(ISP)/
     X     CHARG(ISP))/SMAX
      SCHOLE=HOLE*DEXP(-SCREEN*PROJ)
      IF (CHARG(ISP).NE.ZERO) EXHF=EXHF*(CHARG(ISP)-SCHOLE)/CHARG(ISP)
      EL2=ELECS(ISP)-CHARG(ISP)
      IF (EL2.NE.ZERO) EVX2=EVX2*(EL2-HOLE+SCHOLE)/EL2
C-----------------------------------------------------------------------
C
C     DETERMINE PSEUDOPOTENTIAL CONTRIBUTIONS/ERRORS. 
C
C-----------------------------------------------------------------------
  160 CONTINUE
      EORB=ASUM(OE(2),7,1)
      PKV2=EVAL(IORB,ISP)*PROJ-PKV2
      IF (QPKINC) EORB = EORB+PKV2
      PKERR = EVAL(IORB,ISP)-EORB
C-----------------------------------------------------------------------
C
C     DETERMINE THE PARTIAL SUMS FOR THE TOTAL ENERGY.
C
C-----------------------------------------------------------------------
      DO 170 I=1,9
      TE(I)=TE(I)+OCCUP*OE(I)
  170 CONTINUE

      TPKERR=TPKERR+PKERR*OCCUP
      TEVAL=TEVAL+EVAL(IORB,ISP)*OCCUP
C-----------------------------------------------------------------------
C
C     WRITE ORBITAL ENERGY CONRIBUTIONS.
C
C-----------------------------------------------------------------------
      IF (QPRINT) WRITE (IW,4000) IORB,OE,OCCUP
      IF (QPRINT) WRITE (IW,4500) EVAL(IORB,ISP),VCALC,PKERR

      GAMMA(IORB,ISP)=(EVAL(IORB,ISP)+OE(1)+OE(2))/TWO
C     GAMMA(IORB,ISP)=EVAL(IORB,ISP)
C     IF (OCC(IORB,ISP).LT.0.001.OR.GAMMA(IORB,ISP).GT.ZERO) 
C    X    GAMMA(IORB,ISP)=ZERO
C-----------------------------------------------------------------------
C
C     END LOOP OVER ORBITALS.
C
C-----------------------------------------------------------------------
  180 CONTINUE
  
      CHARG(ISP)=ASUM(OCC(1,ISP),NORB,1)
C-----------------------------------------------------------------------
C
C     END LOOP OVER SPIN.
C
C-----------------------------------------------------------------------
  190 CONTINUE
  
C-----------------------------------------------------------------------
C
C     DETERMINE NEW COEFFICIENTS FOR THE LOCAL MODEL POTENTIAL.
C
C-----------------------------------------------------------------------
	    
      VCALC0 = VCOEF(1)*ONE
C     VCALC1 = VCOEF(1)*VEXP(1)
      VCALC2 = VCOEF(1)*VFIT2(1)
      IJ=0
      DO 262 I=2,NVTERM  
      DO 261 J=2,I
      IJ=IJ+1
      VMAT(IJ) = VMAT(IJ) + ONE*WGT0*ONE/SUMOCC
C     IF (LV(I).EQ.0.AND.LV(J).EQ.0) VMAT(IJ)=VMAT(IJ)+
C    X                                        ONE*WGT0*ONE/SUMOCC
C     IF (LV(I).EQ.0.AND.LV(J).EQ.0) VMAT(IJ)=VMAT(IJ)+
C    X                                        VEXP(I)*WGT1*VEXP(J)/
C    X                                        SUMOCC
      VMAT(IJ)=VMAT(IJ)+VFIT2(I)*WGT2*VFIT2(J)/(TEVZ2-HALF)
  261 CONTINUE  
      VSUM(I-1) = VSUM(I-1) + ONE*WGT0*(ZERO-SUMOCC*ONE)/SUMOCC
C     IF (LV(I).EQ.0) VSUM(I-1)=VSUM(I-1)+
C    X                          ONE*WGT0*(ZERO-SUMOCC*ONE)/SUMOCC
C     IF (LV(I).EQ.0) VSUM(I-1)=VSUM(I-1)+
C    X                          VEXP(I)*WGT1*((-TCOUL0)-SUMOCC*VEXP(1))/
C    X                          SUMOCC
      VSUM(I-1)=VSUM(I-1)+VFIT2(I)*WGT2*(TEVZ2-SUMOCC*VFIT2(1))/
     X                    (TEVZ2-HALF)
      VCALC0 = VCALC0 + VCOEF(I)*ONE
C     IF (LV(I).EQ.0) VCALC0 = VCALC0 + VCOEF(I)*ONE
C     IF (LV(I).EQ.0) VCALC1 = VCALC1 + VCOEF(I)*VEXP(I)
      VCALC2 = VCALC2 + VCOEF(I)*VFIT2(I)
  262 CONTINUE

      VDEV2 = TEVZ2-VCALC2
      IF (DABS(TEVZ2).LT.0.1) VDEV2 = -ZZREP-VCALC2
      IF (DABS(ZZREP).LT.0.1) VDEV2 = ZERO

      IF (QDEBUG) WRITE(IUDUMP,'(/A/)') 'VMAT...'
      IF (QDEBUG) CALL PUTONE(VMAT,NVTERM-1,IUDUMP)
	  
      IF (QNOFIT) GOTO 270
      CALL LOWDIN(VMAT,VINV,SCRAT1,NVTERM-1,20,0,QINDEF)
      IF (QDEBUG) WRITE(IUDUMP,'(/A/)') 'VMAT (Diagonalized)...'
      IF (QDEBUG) CALL PUTONE(VMAT,NVTERM-1,IUDUMP)
      IF (.NOT.QINDEF) GOTO 265
      CALL WARN(18,QRETRY)
      IF (NVTERM.LT.4) GO TO 270
      VCOEF(1)=SUMOCC
      DO 264 NV=2,NVTERM
         VCOEF(NV)=ZERO
  264 CONTINUE
      VCOEF(3)=-SUMOCC
      GO TO 270
	  
  265 CONTINUE
      IF (QDEBUG) WRITE(IUDUMP,'(/A/)') 'VINV...'  
      VCOEF(1)=SUMOCC 
      DO 268 I=2,NVTERM
      VCOEF(I)=ZERO
      DO 266 J=2,NVTERM
      VCOEF(I)=VCOEF(I)+VINV(I-1,J-1)*VSUM(J-1)
  266 CONTINUE	  
C     IF ((LV(I).LT.0).AND.(DABS(VCOEF(I)).GT.SUMOCC)) VCOEF(I) = ZERO
      IF (QDEBUG) WRITE(IUDUMP,'(10F9.5)') (VINV(I-1,K),K=1,NVTERM-1)
  268 CONTINUE
  
C-----------------------------------------------------------------------
C
C     WRITE ENERGY, SPIN, AND CHARGE ANALYSIS.
C
C-----------------------------------------------------------------------
  270 CONTINUE
      TEORB=ASUM(TE(2),7,1)
      IF (QPKINC) TEORB=TEORB+TPKV2
      IF (QPRINT) WRITE (IW,5000) TE,SUMOCC
      IF (QPRINT) WRITE (IW,5500) TEVAL,VCALCT,VCALC2,TPKERR,
     X                            SUMOCC-VCALC0
      TKE=TEKIN
      TETOT=TEORB-HALF*(TECHF+TEXHF)
      TPE=TETOT-TKE
      VIRIAL=TPE/TKE
      TOTEN=TETOT-HALF*(TEVC2+TEVX2-ZZREP)

      IF (QPKINC) TOTEN=TOTEN-HALF*TPKV2
      IF (QPKINC) TOTEN=TOTEN+HALF*TPKERR

      IF (QPRINT) WRITE (IW,6000) TETOT,TKE,TPE,VIRIAL,ZZREP,TOTEN
      IF (.NOT.QOPEN) BSPIN=ASPIN
      ECHARG=ASPIN+BSPIN
      SPDIFF=ASPIN-BSPIN
      CHARGE(2,IATOM)=ECHARG
      CHARGE(3,IATOM)=SPDIFF
      TOTCHG=ZNUC-ECHARG
      CALL CHGAN(CHARGE)
	  
C-----------------------------------------------------------------------
C
C     WRITE NEW MODEL POTENTIAL EXPRESSION.
C
C-----------------------------------------------------------------------
      IF (.NOT.QPRINT) GOTO 300
      IF (NORB*ISPIN.GT.20) WRITE(IW,8000)
      VDEVC=DSQRT(VDEVC)
      VDEVV=DSQRT(VDEVV)
      WRITE(IW,9000) VDEVC,VDEVV,VDEV2
      IF (QNOFIT) WRITE(IW,9500)
      IF (.NOT.QNOFIT) WRITE (IW,10000) (NV,VCOEF(NV),NV=1,NVTERM)
  
  300 CONTINUE
      WGT2=WGT2SV

      RETURN
	  
 1000 FORMAT(/' ENERGY ANALYSIS...')
 2000 FORMAT(/' ',A8)
 3000 FORMAT(/'  #',4X,'ORBITAL',3X,'KINETIC',4X,'INT',7X,'INT',
     X 7X,'INT',7X,'EXT',7X,'EXT',7X,'EXT',
     X 5X,'PSEUDO-',6X,'ORB'/
     X ' ',6X,'ENERGY',4X,'ENERGY',3X,'NUC ATTR',2X,'COULOMB',
     X 3X,'EXCHANGE',2X,'NUC ATTR',2X,'COULOMB',3X,'EXCHANGE',
     X 2X,'POTENTIAL',3X,'OCCUP'/)
 4000 FORMAT(' ',I2,10F10.3)
 4500 FORMAT(' ',3X,'(',F8.3,')',20X,'(',F8.3,')',42X,'(',F6.3,')') 
 5000 FORMAT(/' =>',10F10.3)
 5500 FORMAT(' ',3X,'(',F8.3,')',20X,'(',F8.3,')',9X,'(',F9.3,')',22X,
     X '(',F6.3,')','(',F8.3,')')
 6000 FORMAT(/'    TOTAL ELECTRONIC ENERGY...',F11.4/
     X        '    TOTAL KINETIC ENERGY......',F11.4/
     X        '    TOTAL POTENTIAL ENERGY....',F11.4/
     X        '    VIRIAL THEOREM (V/T)......',F11.4//
     X        '    NUCLEAR REPULSION.........',F11.4/
     X        '    CONTR. TO MOLEC. ENERGY...',F11.4)
 8000 FORMAT(///' ')
 9000 FORMAT(/' STD DEVS FOR THE LAST MODEL POTENTIAL FIT:'//
     X       '    CORE HF COULOMB POTENTIAL........',F9.4/
     X       '    VALENCE HF COULOMB POTENTIAL.....',F9.4/
     X       '    EXTERNAL HF NUCLEAR ATTRACTION...',F9.4)
 9500 FORMAT(/' MODEL POTENTIAL HELD FIXED...')
10000 FORMAT(/' NEW LOCAL MODEL POTENTIAL COEFFICIENTS ...'//
     X      (' ',I2,2X,F10.4))
      END
C
C...
      SUBROUTINE CHGAN(CHARGE)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(22),QPRINT)
      EQUIVALENCE (IPARM(1),NATOM),   (IPARM(10),IATOM)
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(6),IW)
      DIMENSION CHARGE(3,NATOM)
      DIMENSION CHARG(6)
      EQUIVALENCE (CHARG(1),ASPIN), (CHARG(2),BSPIN), (CHARG(3),SPDIFF),
     X            (CHARG(4),ZNUC),  (CHARG(5),ECHARG),(CHARG(6),TOTCHG)
      DATA HALF/0.5D0/
	  
      ZNUC=CHARGE(1,IATOM)
      ECHARG=CHARGE(2,IATOM)
      SPDIFF=CHARGE(3,IATOM)
      ASPIN=HALF*(ECHARG+SPDIFF)
      BSPIN=HALF*(ECHARG-SPDIFF)
      TOTCHG=ZNUC-ECHARG
      IF (QPRINT) WRITE (IW,1000) CHARG
      RETURN
	  
 1000 FORMAT(/' SPIN ANALYSIS...'//
     X       '    ALPHA SPIN...',F9.4/
     X       '    BETA SPIN....',F9.4/
     X       '    NET SPIN.....',F9.4//
     X       ' CHARGE ANALYSIS...'//
     X       '    NUCLEAR CHARGE......',F9.4/
     X       '    ELECTRONIC CHARGE...',F9.4/
     X       '    NET CHARGE..........',F9.4)
	  END
