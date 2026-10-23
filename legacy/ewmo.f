      SUBROUTINE EWMO(EVALMO,OCCMO,SCRAT1,SCRAT2,EVECMO,ONTRAN,OVLP,
     X                EWMAT,POPUL)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     EWMO...
C
C        THIS ROUTINE PERFORMS A EWMO (ENERGY WEIGHTED MAXIMUM OVERLAP)
C     CALCULATION, A POPULATION ANALYSIS WITH RESPECT TO THE ATOMIC
C     BASIS AND A TOTAL ENERGY ANALYSIS.
C
C     VARIABLE DEFINITIONS:
C
C        OVLP(*,*)....... OVERLAP MATRIX OVER THE ATOMIC BASIS.
C        EWMAT(*,*)...... ATOMIC BLOCK DIAGONAL FOCK MATRIX USED TO
C                         GENERATE THE EWMO MATRIX.
C        EVECMO(*,*,*)... MOLECULAR ORBITALS (EIGENVECTORS) OVER THE
C                         ATOMIC BASIS.
C        ONTRAN(*,*)..... (LOWDIN) ORTHONORMAL TRANSFORMATION MATRIX.
C        EVALMO(*,*)..... MOLECULAR ORBITAL ENERGIES (EIGENVALUES) IN
C                         ATOMIC UNITS.
C        OCCMO(*,*)...... MOLECULAR ORBITAL OCCUPATION NUMBERS.
C        POPUL(*,*)...... POPULATION ANALYSIS OVER THE ATOMIC BASIS.
C        ZNSUM........... SUM OF NUCLEAR CHARGES.
C        ELSUM........... SUM OF ELECTRONIC CHARGES.
C        ELDIF........... SUM OF SPIN DIFFERENCES (NET SPIN).
C        SPIN(*)......... SPIN LABELS.
C        ORBOCC.......... =3-ISPIN, MAXIMUM ORBITAL OCCUPATION.
C        ELECS(1)........ TOTAL ALPHA SPIN.
C             (2)........ TOTAL BETA SPIN.
C        CHTRAN.......... PREDICTED CHARGE TRANSFER IN MOLECULE.
C        SCRAT1(*,*)..... SCRATCH ARRAY.
C        SCRAT2(*,*)..... SCRATCH ARRAY.
C        QOPEN........... =T --> SPIN-UNRESTRICTED CASE.
C        ISPIN........... =1 IF (.NOT.QOPEN), =2 IF (QOPEN).
C        QEWMO........... =F --> NO EWMO CALCULATION, JUST A POPULATION
C                         AND TOTAL ENERGY ANALYSIS.
C        QPRINT.......... =T --> INTERMEDIATE RESULT FOR EACH CYCLE
C                         WILL BE PRINTED.
C        QDEBUG.......... =T --> THIS IS A DEBUG RUN, THE OVERLAP MATRIX
C                         AND THE EWMO MATRIX WILL ALSO BE WRITTEN OUT.
C        QSEPAT.......... =T --> 'NEAR SEPARATED-ATOM-LIMIT' CYCLE.
C        QGSTAT.......... =T --> GROUND STATE.
C        NORBT........... TOTAL NUMBER OF ATOMIC ORBITALS IN MOLECULE.
C        N2ORBT.......... =NORBT*(NORBT+1)/2, DIMENSION OF OVERLAP AND
C                         EWMO MATRICES.
C        NCYCL........... NUMBER OF COMPLETED CYCLES TO THIS POINT.
C        TOTPE........... TOTAL MOLECULAR ELECTRONIC POTENTIAL ENERGY.
C        TOTKE........... TOTAL MOLECULAR KINETIC ENERGY.
C        TMOLE............ TOTAL MOLECULAR ELECTRONIC ENERGY.
C        ZZPOT........... MOLECULAR NUCLEAR REPULSION ENERGY.
C        ETOT........... TOTAL MOLECULAR MOLECULAR ENERGY.
C        VIRTHM.......... VIRIAL THEOREM (V/T).
C        DELTAE.......... EXTRAPOLATED CHANGE IN TOTAL ENERGY.
C        IW.............. FORTRAN I/O UNIT FOR WRITING.
C        IUDUMP.......... FORTRAN I/O UNIT FOR THE DEBUG OUTPUT.
C        IUTEMP.......... FORTRAN I/O UNIT FOR TEMPORARY STORAGE.
C        IUCSF........... FORTRAN I/O UNIT FOR STORING FINAL VECTORS.
C
C     ROUTINES CALLED:  SUM, DERASE, PUTONE, TRED3, IMTQLV, TINVIT,
C                       TRBAK3, TIMOUT, ARRMAP, EXCITE,
C                       DCOPY, LOWDIN, UTHU, DMPAB, DENMAT;
C                       DABS, DMIN1, DFLOAT
C
C     COMMON USAGE:
C
C        /PARMS/  USES - QPARM(4)(=QOPEN),   QPARM(7)(=QDEBUG),
C                        QPARM(8)(=QSEPAT),  QPARM(10)(=QEWMO),
C                        QPARM(7)(=QDEBUG),  QPARM(8)(=QSEPAT),
C                        QPARM(19)(=QGSTAT), QPARM(22)(=QPRINT)
C                        IPARM(8)(=NORBT),
C                        IPARM(9)(=N2ORBT),  IPARM(33)(=ISPIN),
C                        IPARM(38)(=NCYCL)
C
C        /IODATA/ USES - IUNIT(6)(=IW),      IUNIT(8)(=IUDUMP),
C                        IUNIT(9)(=IUTEMP),  IUNIT(10)(=IUCSF)
C
C        /ENERGY/ USES - TOTKE, TMOLE, ZZPOT
C                        TOTPE, ETOT, VIRTHM, DELTAE
C
C        /CHARGS/ USES - ZNSUM, ELSUM, ELDIF, SPIN(*), ORBOCC
C                        ELECS(*), CHTRAN
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(4),QOPEN),   (QPARM(7),QDEBUG),
     X            (QPARM(8),QSEPAT),  (QPARM(10),QEWMO),
     X            (QPARM(19),QGSTAT), (QPARM(22),QPRINT)
      EQUIVALENCE (IPARM(8),NORBT),
     X            (IPARM(9),N2ORBT),  (IPARM(33),ISPIN),
     X            (IPARM(38),NCYCL)
      DIMENSION OVLP(N2ORBT,ISPIN),           EWMAT(N2ORBT,ISPIN),
     X          EVECMO(NORBT,NORBT,ISPIN),    EVALMO(NORBT,ISPIN),
     X          OCCMO(NORBT,ISPIN),           POPUL(NORBT,ISPIN),
     X          SCRAT1(NORBT,NORBT),          SCRAT2(NORBT,9),
     X          ONTRAN(NORBT,NORBT)
      COMMON /IODATA/ IUNIT(20),LENBUF
      COMMON /ENERGY/ TOTPE,TOTKE,TMOLE,ZZPOT,ETOT,VIRTHM,DELTAE
	CHARACTER*8 SPIN
      COMMON /CHARGS/ ZNSUM,ELSUM,ELDIF,ELECS(2),SPIN(2),ORBOCC,CHTRAN
      EQUIVALENCE (IUNIT(6),IW),       (IUNIT(8),IUDUMP),
     X            (IUNIT(9),IUTEMP),   (IUNIT(10),IUCSF)
      DATA ZERO/0.D0/,HALF/0.5D0/,ONE/1.D0/
      DATA SMALL/5.D-5/
      DATA QNOPOP/.TRUE./

      CALL TIMOUT(0)
C      IF (QDEBUG) CALL ARRMAP(4)
C-----------------------------------------------------------------------
C
C     PREPARE FOR OCCUPATION ASSIGNMENTS.
C
C-----------------------------------------------------------------------
      CALL DERASE(OCCMO,NORBT*ISPIN)
C-----------------------------------------------------------------------
C
C     PRINT HEADINGS.
C
C-----------------------------------------------------------------------
      IF (.NOT.QEWMO) GO TO 10
      IF (QDEBUG) WRITE (IUDUMP,1000)
      IF (QPRINT) WRITE (IW,2000)
      IF (QPRINT) WRITE (IW,3000) NCYCL
      WRITE (IUTEMP) OVLP,EWMAT
      REWIND IUTEMP
   10 CONTINUE
      CALL DERASE(EVECMO,NORBT*NORBT*ISPIN)
      CALL DERASE(EVALMO,NORBT*ISPIN)
      DO 15 ISP=1,ISPIN
      DO 14 IORBT=1,NORBT
      IIORBT=(IORBT*(IORBT+1))/2
      EVECMO(IORBT,IORBT,ISP)=ONE
      IF (EWMAT(IIORBT,ISP).GE.ZERO) THEN
         EVALMO(IORBT,ISP) = ZERO
      ELSE
         EVALMO(IORBT,ISP) = EWMAT(IIORBT,ISP)
      END IF
   14 CONTINUE
   15 CONTINUE
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER SPIN.
C
C-----------------------------------------------------------------------
      DO 150 ISP=1,ISPIN
      IF (.NOT.QEWMO) GO TO 100
      IJ = 0
      DO 40 IORBT=1,NORBT
      II = (IORBT*(IORBT-1))/2
      DO 30 JORBT=1,IORBT
      JJ = (JORBT*(JORBT-1))/2
      IJ = IJ + 1
      EWMAT(IJ,ISP) = -DSQRT(EVALMO(IORBT,ISP)*EVALMO(JORBT,ISP))
     X                *OVLP(IJ,ISP)
C     EWMAT(IJ,ISP) = ZERO
C     DO 20 KORBT=1,NORBT
C     KK = (KORBT*(KORBT-1))/2
C     IK = II + KORBT
C     IF (IORBT.LT.KORBT) IK = KK + IORBT
C     KJ = JJ + KORBT
C     IF (JORBT.LT.KORBT) KJ = KK + JORBT
C     EWMAT(IJ,ISP) = EWMAT(IJ,ISP) + 
C    X                EVALMO(KORBT,ISP)*OVLP(IK,ISP)*OVLP(KJ,ISP)
C  20 CONTINUE
   30 CONTINUE
   40 CONTINUE
C-----------------------------------------------------------------------
C
C     GENERATE SYMMETRIC ORTHONORMAL TRANSFORMATION MATRIX.
C
C-----------------------------------------------------------------------
C     CALL LOWDIN(OVLP(1,ISP),ONTRAN,SCRAT1,NORBT,NORBT,4,QINDEF)
C     IF (QINDEF) STOP 91
C-----------------------------------------------------------------------
C
C     GENERATE EWMO MATRIX IN ORTHONORMAL BASIS.
C
C-----------------------------------------------------------------------
      IF (.NOT.QDEBUG) GO TO 50
      IF (QOPEN) WRITE (IUDUMP,4000) SPIN(ISP)
      CALL PUTONE(EWMAT(1,ISP),NORBT,IUDUMP)
   50 CONTINUE
C     CALL CTSC(ONTRAN,EWMAT(1,ISP),SCRAT1,EWMAT(1,ISP),NORBT,NORBT)
C-----------------------------------------------------------------------
C
C     FIND THE EIGENVALUES AND EIGENVECTORS OF THE EWMO MATRIX.
C
C-----------------------------------------------------------------------
      CALL EISPAK(EWMAT(1,ISP),EVALMO(1,ISP),EVECMO(1,1,ISP),SCRAT2,
C                       IMTQLV, TINVIT, TRBAK3, BOMB, LOWDIN, NORMLZ,
     X            NORBT,NORBT,N2ORBT,NORBT,IERR)
      IF (IERR.NE.0) CALL BOMB(17)
C-----------------------------------------------------------------------
C
C     IF QPRINT=T TRANSFORM THE EIGENVECTORS BACK TO THE ORIGINAL NON-
C     ORTHOGONAL ATOMIC ORBITAL BASIS AND PRINT THEM.
C
C-----------------------------------------------------------------------
      READ (IUTEMP) OVLP,EWMAT
      REWIND IUTEMP

      IF (QPRINT.AND.QOPEN) WRITE (IW,4000) SPIN(ISP)

C     CALL LOWDIN(OVLP(1,ISP),ONTRAN,SCRAT1,NORBT,NORBT,2,QINDEF)
C     IF (QINDEF) GOTO 200
C     CALL DMPAB(ONTRAN,NORBT,NORBT,NORBT,NORBT,EVECMO(1,1,ISP),NORBT,
C    X     NORBT,NORBT,NORBT,SCRAT1,NORBT,NORBT)
C     CALL DCOPY(SCRAT1,NORBT*NORBT,EVECMO(1,1,ISP))

      DO 70 IORBT=1,NORBT
      IIORBT=(IORBT*(IORBT+1))/2
      DO 60 JORBT=1,NORBT
      IF (EWMAT(IIORBT,ISP).GE.ZERO) 
     X   THEN
             EVECMO(IORBT,JORBT,ISP) = ZERO
         ELSE
             EVECMO(IORBT,JORBT,ISP) = EVECMO(IORBT,JORBT,ISP)
     X          /DSQRT(DABS(EWMAT(IIORBT,ISP)))
      END IF
   60 CONTINUE
   70 CONTINUE
      CALL NORMLZ(EVECMO(1,1,ISP),NORBT,OVLP(1,ISP),NORBT,N2ORBT)

      IF (.NOT.QPRINT) GO TO 100
      CALL OUTVEC(EVECMO(1,1,ISP),EVALMO(1,ISP),NORBT,NORBT,IW)

      READ (IUTEMP) OVLP,EWMAT
      REWIND IUTEMP
C-----------------------------------------------------------------------
C
C     GENERATE THE GROUND STATE OCCUPATION NUMBERS.
C
C-----------------------------------------------------------------------
  100 CONTINUE
      REMAIN=ELECS(ISP)/ORBOCC
      IORBT=1
  110 IDEGEN=1
  120 SPLIT=DABS(EVALMO(IORBT,ISP)-EVALMO(IORBT+IDEGEN,ISP))
      IF (SPLIT.GT.SMALL) GO TO 130
      IDEGEN=IDEGEN+1
      IF (IORBT+IDEGEN.LE.NORBT) GO TO 120
  130 CONTINUE
      OCCUP=DMIN1(ONE,REMAIN/DFLOAT(IDEGEN))
      DO 140 ID=1,IDEGEN
      OCCMO(IORBT,ISP)=OCCUP
      IORBT=IORBT+1
      REMAIN=REMAIN-OCCUP
  140 CONTINUE
      IF ((IORBT.LE.NORBT).AND.(REMAIN.GT.ZERO)) GO TO 110
C-----------------------------------------------------------------------
C
C     END LOOP OVER SPIN.
C
C-----------------------------------------------------------------------
  150 CONTINUE
C-----------------------------------------------------------------------
C
C     GENERATE EXCITED-STATE OCCUPATIONS (IF .NOT.QGSTAT) AND PERFORM A
C     POPULATION ANALYSIS WITH RESPECT TO THE ORTHONORMAL BASIS.
C
C-----------------------------------------------------------------------
      CHTRAN=ZERO
      DELTAE=ZERO
      IF (QNOPOP) GOTO 185
      DO 180 ISP=1,ISPIN
      IF (.NOT.QGSTAT) CALL EXCITE(OCCMO,NORBT,ISPIN)
      DO 170 IORBT=1,NORBT
      SUM=ZERO
      OLDPOP=POPUL(IORBT,ISP)
      DO 160 JORBT=1,NORBT
      SUM=SUM+OCCMO(JORBT,ISP)*EVECMO(IORBT,JORBT,ISP)**2
  160 CONTINUE
      POPUL(IORBT,ISP)=SUM
      CT=ORBOCC*(SUM-OLDPOP)
      CHTRAN=CHTRAN+DABS(CT)*HALF
      DELTAE=DELTAE+EVALMO(IORBT,ISP)*CT
  170 CONTINUE
  180 CONTINUE
  185 CONTINUE
C-----------------------------------------------------------------------
C
C     WRITE EIGENVECTORS TO UNIT IUCSF AND PRINT THE ORBITAL ENERGIES
C     AND OCCUPATIONS.
C
C-----------------------------------------------------------------------
C      WRITE (IUCSF) EVECMO   !Switch to CSF (MolStruct) Format
C      REWIND CSF
      IF (.NOT.QPRINT) RETURN
      WRITE (IW,5000)
      WRITE (IW,3000) NCYCL
      CALL POPOUT(EVALMO,OCCMO,ZNSUM,ELECS(1),ELECS(2),NORBT)
C-----------------------------------------------------------------------
C
C     PRINT TOTAL ENERGY ANALYSIS.
C
C-----------------------------------------------------------------------
      ETOT=TMOLE-ZZPOT
      TOTPE=TMOLE-TOTKE
      VIRTHM=TOTPE/TOTKE
      WRITE (IW,6000) TOTPE,TOTKE,ETOT,ZZPOT,TMOLE,VIRTHM
C-----------------------------------------------------------------------
C
C     PRINT POPULATION ANALYSIS.
C
C-----------------------------------------------------------------------
      IF (.NOT.QEWMO) RETURN
      WRITE (IW,7000)
      WRITE (IW,3000) NCYCL
      IF (QOPEN) WRITE (IW,8000)
      DO 190 IORBT=1,NORBT
      WRITE (IW,9000) IORBT,(POPUL(IORBT,ISP),ISP=1,ISPIN)
  190 CONTINUE
      ELECS(1)=ELECS(1)*ORBOCC
      WRITE (IW,10000) (ELECS(ISP),ISP=1,ISPIN)
      CALL TIMOUT(6)
  200 RETURN
 1000 FORMAT(/' EWMO MATRIX...')
 2000 FORMAT(///' MOLECULAR ORBITALS AND ENERGIES...')
 3000 FORMAT(' ',T100,'... CYCLE',I3,' ...')
 4000 FORMAT(/' ',A8)
 5000 FORMAT(///' MOLECULAR ORBITAL ENERGIES AND OCCUPATIONS...')
 6000 FORMAT(///' TOTAL ENERGY ANALYSIS...'//
     X       '    TOTAL POTENTIAL ENERGY.....',F12.4/
     X       '    TOTAL KINETIC ENERGY.......',F12.4/
     X       '    TOTAL ELECTRONIC ENERGY....',F12.4/
     X       '    NUCLEAR REPULSION ENERGY...',F12.4//
     X       '    TOTAL MOLECULAR ENERGY.....',F12.4//
     X       '    VIRIAL THEOREM (V/T).......',F12.6) 
 7000 FORMAT(///' GROSS ATOMIC ORBITAL POPULATIONS...'//)
 8000 FORMAT(/'              ALPHA       BETA'/
     X        '              _____       ____'/)
 9000 FORMAT(' ',6X,I3,F9.4,F12.4)
10000 FORMAT(/'     TOTAL',F9.4,F12.4)
      END
