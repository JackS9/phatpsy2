      SUBROUTINE ATOMIC(N,L,M0,ETA,ANORM,STOVL,STKIN,STPOT,STPOTX,
     X                  LV,MV,VEXP,VCOEF,STHMAT,ONTRAN,STEVEC,EVAL,OCC,
     X                  FOCK,GAMMA,ZPOT2,SCRAT1,SCRAT2,
     X                  VFIT2,CPOT2,PKPOT,INDEX,CGC,S,W,BUFFER,COORD,
     X                  LABEL,CHARGE,FACT,D,SCRAT3,NRATOM,QREALY,
     X                  IATOMX,NX,LX,MLX,ETAX,ANORMX,EVALX,OCCX,EVECX)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      CHARACTER*1 LTYPE
      CHARACTER*2 LATOM,LABEL
      CHARACTER*40 DESCR
C-----------------------------------------------------------------------
C
C    ATOMIC...
C
C       THIS ROUTINE PROCESSES MOST OF THE INPUT FOR A GIVEN ELEMENT
C    AND COMPUTES ALL THE 1-CENTER INTEGRALS OVER THE PRIMITIVE BASIS.
C    SEE MAIN AND CONTRL FOR THE REST OF THE CONTROL AND DIMENSION
C    INPUT.  THE INPUT HERE CONSISTS OF SIX TYPES OF CARD IMAGES TO BE
C    READ IN LOGICAL SEQUENCE.  THE SIX TYPES ARE AS FOLLOWS:
C
C       TYPE-1:  (NAMELIST - &ELEM)
C
C                DESCRIPTION OF ELEMENT (IELEM).
C
C                   LATOM....... 2-CHARACTER LABEL FOR THIS ELEMENT.
C                                (IN QUOTES).
C                   DESCR(*).... 40-CHARACTER DESCRIPTION OF THIS
C                                ELEMENT AND ITS BASIS SET (IN
C                                QUOTES).  (DEFAULT IS BLANKS).
C                   NUCZ........ NUCLEAR CHARGE.
C                   MDIFAT...... NUMBER OF DIFFERENT (NON-EQUIVALENT)
C                                ATOMS IN THE MOLECULE REPRESENTED BY
C                                THIS ELEMENT, I.E., NUMBER OF CARDS
C                                OF TYPE-3 TO FOLLOW.  (DEFAULT IS 1).
C                   QMNBAS...... =T --> EACH ATOMIC ORBITAL REPRESENTED
C                                BY A SINGLE STO. THE COEFFICIENT MATRIX
C                                IS DEFAULTED TO A UNIT MATRIX.
C                                (DEFAULT IS .FALSE.)
C                   QTFPOT...... =T --> TF-APPROX. WILL BE USED FOR THE
C                                FIRST 4 TERMS OF THE MODEL POTENTIAL.
C                                ADDITIONAL TERMS WILL BE READ IN ON
C                                TYPE-4 CARDS. (DEFAULT)
C                                =F --> ALL TERMS READ FROM TYPE-3
C                                CARDS.
C                   RSCALE...... SCALING USED FOR THE RADIAL COORDINATE
C                                IN THE SCREENING FUNCTION OF THE MODEL
C                                POTENTIAL. (DEFAULT IS 1.12956*Z**(1/3)
C                                OR 1.0 FOR Z=1).
C
C       TYPE-2:  (2I3,F10.6)
C
C                DESCRIPTION OF STO (ISTO).  IGNORED IF QNWBAS=F.
C
C                   N(ISTO)..... N-QUANTUM NUMBER OF THIS STO.
C                   L(ISTO)..... L-QUANTUM NUMBER OF THIS STO.
C                   ETA(ISTO)... SLATER EXPONENT OF THIS STO.
C
C                   WHERE
C                          STO = R**(N-1)*EXP(-ETA*R)*Y(L,ML)
C                          ML  = -L,...,L
C
C       TYPE-3:  (2I3,3F10.6)
C
C                DESCRIPTION OF MODEL POTENTIAL TERM (NV).  IGNORED IF
C                QNWBAS=F.  (SEE QTFPOT AND RSCALE ABOVE).
C
C                   LV(NV)........ L-TYPE OF TERM.
C                   MV(NV)........ M-TYPE OF TERM.
C                   VEXP(NV)...... EXPONENT OF TERM. SCALED BY RSCALE
C                                  IF .GT.0.  UNSCALED ABSOLUTE VALUE
C                                  USED IF .LT.0.
C                   VCOEF(NV)..... STARTING COEFFICIENT.  SCALED BY
C                                  NUMBER OF ELECTRONS AND RSCALE**LV
C                                  (OR BY RSCALE**(1-LV)).
C                                  IGNORED IF QRSTRT=T.  (SEE QNWPOT
C                                  BELOW IF QNWBAS=F).
C
C                   LV.GE.0
C                           V = R**(LV-1)*EXP(-VEXP*R)
C                   LV.LT.0
C                           V = R**-LV*EXP(-VEXP*R)*Y(-LV,MV)
C
C                NOTE:  THE FIRST TERM SHOULD ALWAYS BE Z/R (0,0,0.;-1)
C                       THE 2ND TERM SHOULD BE OF 0S-TYPE AND THE
C                       3RD OF 0S- OR 1S-TYPE.
C                       SUCCESSIVE TERMS SHOULD BE GIVEN IN DECREASING
C                       ORDER OF IMPORTANCE WITH THE POLARIZATION TYPE
C                       LAST (TO AID IN FIXUP OF INDETERMINANT CASES).
C
C                NOTE:  IF SYMMETRY OPERATIONS ARE TO BE USED FOR
C                       THIS ELEMENT THEN EACH (-L)-VALUE SHOULD BE
C                       ACCOMPANIED BY A FULL SET OF ML-VALUES
C                       (IN INCREASING ORDER).
C
C       TYPE-4:  (NAMELIST - &ATOM)
C
C                DESCRIPTION OF ATOM (IATOM).  IGNORED IF QATOM=T.
C
C                   XYZ(1)...... X-COMPONENT OF THE POSITION VECTOR.
C                      (2)...... Y-CONPONENT.
C                      (3)...... Z-COMPONENT.
C                                (DEFAULTS ARE  0.,0.,0.).
C
C                      OR THE SYMMETRY OPERATIONS REQUIRED TO BRING AN
C                      EQUIVALENT ATOM (IATOM+IEQAT) INTO THE LAST ATOM
C                      (IATOM+IEQAT-1).
C
C                   ISYM... TYPE OF SYMMETRY OPERATION.
C                           = N --> N-FOLD ROTATION (COUNTER-CLOCKWISE).
C                           = 1 --> IDENTITY.
C                           = 0 --> IDENTITY.         (DEFAULT)
C                           =-1 --> PURE REFLECTION.  (SEE NOTE)
C                           =-2 --> INVERSION.
C                           =-N --> N-FOLD ROTO-REFLECTION.
C                   IXYZ... SYMMETRY ELEMENT WITH RESPECT TO WHICH THIS
C                           OPERATION IS TAKEN. (AXIS FOR ROTATION OR
C                           PERPENDICULAR PLANE FOR REFLECTION).
C                           E.G.,
C                           =001, Z-AXIS (OR XY-PLANE) (DEFAULT).
C                           =100, X-AXIS (OR YZ-PLANE).
C                           =210, REPRESENTS LINE PASSING THROUGH
C                                 (0,0,0) AND (-1,1,0) (WHERE 2 MEANS
C                                 -1) OR THE PLANE PERPENDICULAR TO
C                                 IT AND CONTAINING (0,0,0).
C                           ETC.
C
C                   NEQAT....... NUMBER OF ATOMS WHICH ARE EQUIVALENT
C                                TO THIS ONE, THAT IS, DIFFER ONLY IN
C                                A SYMMETRY OPERATION.  THE NUMBER OF
C                                CARDS OF THIS TYPE TO FOLLOW.
C                                (DEFAULT IS 0).
C                   QNWVEC...... =T --> NEW STARTING VECTORS READ FROM
C                                TYPE-5 AND -6 CARDS ON RESTART.  ONLY
C                                CHANGES FROM LAST RUN NEEDED.
C                                =F --> VECTORS ARE READ FROM IUNIT4
C                                (IUNIT3 IF QBAKUP=T) ON RESTART.
C                                (DEFAULT IS .FALSE.).
C                   QDFVEC...... =T --> DIFFERENT VECTORS (ON TYPE-5
C                                AND -6 CARDS) READ FOR THIS ATOM.  ONLY
C                                CHANGES FROM LAST ATOM NEEDED.
C                                =F --> VECTORS FROM LAST ATOM ARE USED.
C                                (DEFAULT IS .FALSE. EXCEPT FOR FIRST
C                                ATOM OF EACH ELEMENT).
C                   QFLIP....... =T --> THE DEFAULT ORBITAL ENERGIES,
C                                OCCUPATIONS, AND COEFFICIENTS WILL BE
C                                THOSE OF THE LAST ATOM BUT WITH THE
C                                ROLES OF ALPHA AND BETA "FLIPPED".
C                                (ONLY HAS MEANING IF QOPEN=T, QRSTRT=F)
C                                (DEFAULT IS .FALSE.).
C                   QMOVE....... =T --> NEW COORDINATES OR SYMMETRY
C                                OPERATIONS ARE READ.
C                                IGNORED IF QBAKUP=T.
C                                =F --> COORDINATES OR SYMMETRY
C                                OPERATIONS READ FROM IUNIT4 (IUNIT3
C                                IF QBAKUP=T).
C                                (DEFAULT = .NOT.QRSTRT).
C                   QFIXED...... =T --> THE ORBITALS AND MODEL POTENTIAL
C                                ON THIS ATOM WILL BE FIXED (NO SCF).
C                                (DEFAULT IS .FALSE.).
C                   QVIRTL...... =T --> THIS ATOM WILL NOT BE INCLUDED
C                                IN THE EWMO CALCULATION.  IT WILL BE
C                                USED AS A VIRTUAL ATOM AND ONLY SEEN
C                                AS AN EXTERNAL POTENTIAL.  IT SHOULD
C                                BE EQUIVALENT TO A 'REAL' ATOM.
C                                (DEFAULT IS .FALSE.).
C                   QPNCHV...... =T --> ATOMIC EIGENVECTORS WILL BE
C                                PUNCHED ON THE LAST CYCLE IN A FORM
C                                READABLE BY THIS ROUTINE.
C                                (DEFAULT IS .FALSE.)
C                   QPLOT....... =T --> TOTAL ATOMIC DENSITY WILL BE
C                                PLOTTED ON THE LAST CYCLE.
C                                (DEFAULT IS .FALSE.)
C                   QNWPOT...... =T --> NEW MODEL POTENTIAL COEFFICIENTS
C                                WILL BE READ.  (DEFAULT IS .FALSE.).
C                                (SEE VCOEF0(*)).
C                   VCOEF0(*)... NEW MODEL POTENTIAL COEFFICIENTS IF
C                                QNWPOT=T AND QNWBAS=F.  ONLY CHANGES
C                                NEED BE GIVEN.  DEFAULTS ARE INPUT OR
C                                RESTART VALUES (SEE QTFPOT, QETPOT AND
C                                TYPE-3 CARDS ABOVE).
C                   RMAX........ MAXIMUM INTERATOMIC DISTANCE FOR
C                                WHICH INTERACTIONS ARE COMPUTED.
C                                (DEFAULT IS 3.0*Z**(1/3), OR THAT OF
C                                LAST RUN FOR THIS ATOM IF QRSTRT=T).
C                   IUATOM...... FORTRAN I/O UNIT FOR ATOMIC ORBITAL
C                                INPUT (NAMELIST AND COEFFICIENTS).
C                                (THE DEFAULT IS 'IR' OR THAT FOR LAST
C                                ATOM OF THIS ELEMENT).
C                   TE.......... STARTING TOTAL ENERGY. USEFUL FOR FIXED
C                                ATOM OPTION. (DEFAULT IS SUM OF ORBITAL
C                                ENERGIES OR RESTART VALUE ON IURST).
C
C                NOTE:  INTERNALLY, ALL REFLECTIONS ARE DECOMPOSED
C                       INTO ROTATIONS AND INVERSIONS.
C
C       TYPE-5:  (NAMELIST - &ATORB)
C
C                INITIAL DESCRIPTION OF ATOMIC ORBITAL (IORB).
C
C                   NCOEF....... NUMBER OF NON-ZERO COEFFICIENTS TO BE
C                                READ, I.E., NUMBER OF CARDS OF TYPE-6
C                                TO FOLLOW.  IF .LE.0, THE LAST ORBITAL
C                                WILL BE ASSUMED WITH ML+1.  IF .LE.0
C                                AND QNWVEC, ONLY CHANGES FROM THE LAST
C                                ATOM ARE NEEDED.  IF .LE.0 AND QRSTRT,
C                                ONLY CHANGES FROM LAST RUN ARE NEEDED.
C                                (DEFAULT IS 0).
C                   EIGVAL(1)... EIGENVALUE (IN HARTREES) OF THE (ALPHA-
C                                SPIN) ORBITAL. (DEFAULT IS THE LAST
C                                VALUE, SEE BELOW *).
C                         (2)... EIGENVALUE OF THE BETA-SPIN ORBITAL.
C                                (DEFAULT IS THE LAST VALUE).
C                   OCCNO(1).... OCCUPATION NUMBER OF THE (ALPHA-
C                                SPIN) ORBITAL. (DEFAULT IS LAST VALUE).
C                        (2).... OCCUPATION NUMBER OF THE BETA-SPIN
C                                ORBITAL. (DEFAULT IS LAST VALUE).
C
C                NOTE:  IF (.NOT.QOPEN) OCCNO(1)=(OCCNO(1)+OCCNO(2))/2
C
C                *   'LAST VALUE' MEANS THE VALUE FROM THE LAST RUN IF
C                    THIS IS RESTART (QRSTRT=T), OTHERWISE IT MEANS THE
C                    VALUE FROM THE PREVIOUS ORBITAL (IORB-1), AND THE
C                    'FIRST VALUE' (IORB=1) IS DEFAULTED TO 0 AND 1
C                    FOR THE EIGENVALUES (EIGVAL) AND OCCUPATION NUMBERS
C                    (OCCNO), RESPECTIVELY.
C
C       TYPE-6:  (2I3,2F10.6)
C
C                STARTING VECTOR (COEFFICIENTS) FOR THIS ATOMIC ORBITAL.
C                (NON-ZERO COEFFICIENTS ONLY).
C
C                   ISTO........ THE STO OF THIS ELEMENT (IELEM) TO
C                                WHICH THIS COEFFICIENT REFERS.
C                   ML.......... THE ML-QUANTUM NUMBER TO BE USED
C                                WITH THIS STO-COEFFICIENT.
C                   STEVEC(M0(ISTO)+ML,IORB,1)... COEFFICIENT FOR THE
C                                (ALPHA-SPIN-) ORBITAL.
C                   STEVEC(M0(ISTO)+ML,IORB,2)... COEFFICIENT FOR THE
C                                BETA-SPIN-ORBITAL IF QOPEN.  IF =0.,
C                                AND EIGVAL(1) .EQ. EIGVAL(2) THEN
C                                THE ALPHA-VALUE WILL BE ASSUMED.
C
C                   NOTE:  M0(*) IS AN INTERNAL POINTER ARRAY.
C
C     OTHER DEFINITIONS:
C
C        IATOM.... THE INDEX FOR THE CURRENT ATOM.
C        NATOM.... TOTAL NUMBER OF ATOMS IN SYSTEM.
C        NRATOM... NUMBER OF REAL (NON-VIRTUAL) ATOMS IN SYSTEM.
C        MSTO..... THE NUMBER OF STO'S (INCLUDING ML-VALUES) IN THE
C                  BASIS FOR THE CURRENT ELEMENT.
C        JSTOT.....INDEX FOR CURRENT STO (INCLUDING ML-VALUES) AT THE
C                  MOLECULAR LEVEL
C        QNWBAS... =T --> BASIS READ FROM TYPE-2 CARDS ABOVE (DEFAULT
C                  IF QRSTRT=F).
C                  =F --> BASIS READ FROM IUNIT0 (DEFAULT IF QRSTRT=T).
C        M2STO.... =MSTO*(MSTO+1)/2, DIMENSION OF STOVL(*), ETC.
C        NSTO..... THE NUMBER OF STO'S (NOT INCLUDING ML-VALUES) IN
C                  THE BASIS FOR THE CURRENT ELEMENT.  THE NUMBER OF
C                  TYPE-2 CARDS.
C        NORB..... THE NUMBER OF ORBITALS ON EACH ATOM OF THIS ELEMENT.
C                  NUMBER OF TYPE-5 CARDS FOR EACH.
C        N2ORB.... =NORB*(NORB+1)/2.
C        NORBT.... TOTAL NUMBER OF ATOMIC ORBITALS IN THIS MOLECULE.
C        IORBT.... INDEX FOR THE CURRENT ATOMIC ORBITAL.
C        NMAX..... MAXIMUM VALUE OF N-QUANTUM NUMBER TO BE ALLOWED.
C        NNMX..... =2*NMAX, DIMENSION OF S(*).
C        NNMXM1... =2*NMAX-1, DIMENSION OF W(*).
C        LMAX..... MAXIMUM VALUE OF L-QUANTUM NUMBER TO BE ALLOWED.
C        LLMXP1... =2*LMAX+1, DIMENSION OF SCRAT3(*).
C        L3MX..... =(LLMXP1)*(LLMXP1+1)*(LLMXP1+2)/6, DIMENSION OF D(*).
C        QOPEN.... =T --> OPEN SHELL CASE (UNRESTRICTED).
C                  =F --> CLOSED SHELL CASE (RESTRICTED).  DEFAULT.
C        ISPIN.... =1 IF (.NOT.QOPEN), =2 IF (QOPEN).  RANGE OF ISP.
C        IUNIT0... =IUNIT(IELEM+10), FORTRAN I/O UNIT FOR HOLDING THE
C                  BASIS AND INTEGRALS FOR THIS ELEMENT (IELEM).
C        IR....... FORTRAN I/O UNIT USED FOR READING INPUT.
C        IW....... FORTRAN I/O UNIT USED FOR WRITING OUTPUT.
C        IP....... FORTRAN I/O UNIT USED FOR PUNCHING OUTPUT.
C        IUNIT3... FORTRAN I/O UNIT FOR STORING INITIAL DATA OR ON
C                  RESTART CONTAINS DATA FROM LAST RUN AFTER SCF.
C        IUNIT4... FORTRAN I/O UNIT REFERENCING DATA EXISTING FROM LAST
C                  RUN AFTER EWMO CALCULATION (USUAL RESTART DATA).
C        IUTEMP... FORTRAN I/O UNIT FOR TEMPORARY STORAGE OF VECTORS,
C                  ENERGIES, OCCUPATIONS, ETC.
C        IUDUMP... FORTRAN I/O UNIT FOR DIAGNOSTIC OUTPUT.
C        QATOM.... =T --> ATOMIC CALCULATION ONLY.
C        QRSTRT... =T --> THIS IS A RESTART AND OLD DATA IN PART
C                  OR IN WHOLE WILL BE USED.
C                  =F --> ALL DATA (EXCEPT POSSIBLY THE BASIS) WILL BE
C                  READ FROM CARDS AS DESCRIBED ABOVE.
C        QBAKUP... =T --> THE RESTART WILL USE DATA ON IUNIT3 RATHER
C                  THAN ON IUNIT4, ANY CHANGES ARE IGNORED.
C                  =F --> THE RESTART WILL BEGIN WITH DATA ON IUNIT4.
C        NVTERM... NUMBER OF TERMS IN THE MODEL POTENTIAL EXPRESSION.
C        NCGC..... DIMENSION OF CGC(*).
C        NFACT.... AT LEAST 2*NMAX, DIMENSION OF FACT(*).
C        LENBUF... LENGTH (DIMENSION) OF BUFFER(*).
C        QEWMO.... =T --> EWMO CALCULATION WILL BE PERFORMED.
C        QDEBUG... =T --> THIS IS A DEBUG (DIAGNOSTIC) RUN.
C        QTRACE... =T --> A RUN TRACE WILL BE WRITTEN TO STDERR.
C        QANGST... =T --> ALL DISTANCES WILL BE GIVEN IN ANGSTROMS AND
C                  CONVERTED TO A.U.'S WITHIN.
C        TMOLE..... TOTAL MOLECULAR ELECTRONIC ENERGY.
C        PKSCAL... SCALE FOR PHILLIPS-KLEINMAN PSEUDOPOTENTIAL
C
C        STOVL(*).......... OVERLAP MATRIX OVER THE STO BASIS.
C        STKIN(*).......... KINETIC ENERGY MATRIX.
C        STPOT(*,*)........ POTENTIAL ENERGY MATRIX (BY TERM).
C        STPOTX(*)......... TEMPORARY SCRATCH ARRAY USED IN ONEINT.
C        VEXP(*)........... EXPONENTS IN THE LOCAL POTENTIAL.
C        TFEXP(*).......... TF APPROX. TO VEXP(*) * Z**(-1/3).
C        VCOEF(*).......... COEFFICIENTS IN THE LOCAL POTENTIAL.
C        TFCOEF(*)......... TF APPROX. TO VCOEF(*) / Z.
C        LTF(*)............ L-TYPE OF TERM IN TF MODEL POTENTIAL.
C        STHMAT(*)......... ONE-ELECTRON HAMILTONIAN MATRIX.
C        ONTRAN(*,*)....... CANONICAL ORTHONORMAL TRANSFORMATION MATRIX.
C        BUFFER(*)......... TWO-ELECTRON INTEGRAL BUFFER.
C        CGC(*)............ CLEBSCH-GORDON COEFFICIENTS.
C        S(*).............. S-FUNCTIONS (SEE TWOINT).
C        W(*).............. W-FUNCTIONS (SEE TWOINT).
C        INDEX(N).......... =N*(N-1)/2, INDEXING ARRAY.
C        FACT(N)........... =(N-1)-FACTORIAL.
C        COORD(*,IATOM).... =XYZ(*), CARTESIAN COORDINATES OF THIS ATOM.
C        LABEL(IATOM)...... LABEL (ATOMIC SYMBOL) FOR THIS ATOM.
C        CHARGE(1,IATOM)... NUCLEAR CHARGE (NUCZ) OF THIS ATOM.
C              (2,IATOM)... ELECTRONIC CHARGE (SUM OF OCC. NO.'S).
C              (3,IATOM)... NET ALPHA-SPIN CHARGE (SPIN DIFFERENCE).
C        ZPOT2(*).......... NUC. ATTR. CONTRIBUTION TO EXT. POTENTIAL.
C        CPOT2(*).......... EXTERNAL POTENTIAL ENERGY MATRIX.
C        PKPOT(*,*,*)...... PSEUDOPOTENTIAL. 
C        VFIT2(*).......... MODEL POTENTIAL APPROX. TO ZPOT2(*).
C        FOCK(*,*)......... FOCK MATRIX OVER ATOMIC ORBITALS.
C        GAMMA(*,*)........ ADJUSTED ORBITAL ENERGIES.              
C        D(*).............. ROTATION COEFFICIENTS.
C        SCRAT1(*,*)....... SCRATCH ARRAY.
C        SCRAT3(*)..........SCRATCH ARRAY.
C        IATOMX(*)..........ATOM INDEX FOR MOLECULAR BASIS
C        NX(*)..............N QUANTUM NUMBERS FOR THE MOLECULAR BASIS
C        LX(*)..............L QUANTUM NUMBERS FOR THE MOLECULAR BASIS
C        MLX(*).............ML QUANTUM NUMBERS FOR THE MOLECULAR BASIS
C        ETAX(*)............STO EXPONENTS FOR THE MOLECULAR BASIS
C        ANORMX(*)..........STO NORMS FOR THE MOLECULAR BASIS
C        EVALX(*,*).........EIGENVALUES AT THE MOLECULAR LEVEL
C        OCCX(*,*)..........OCCUPATIONS AT THE MOLECULAR LEVEL
C        EVECX(*,*,*).......EIGENVECTORS AT THE MOLECULAR LEVEL
C
C     ROUTINES CALLED:  ONEINT, TWOINT, LOWDIN, NORMLZ, POPOUT, OUTVEC,
C                       DPROD, DERASE, DCOPY, BOMB, SYMOP, ARRMAP, FLIP,
C                       TEINDX(DMPBUF); IABS, DABS, DSQRT, MAX0, MIN0,
C                       DMIN1
C
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM(1)(=NATOM),   IPARM(5)(=MSTO),
C                        IPARM(6)(=NORB),    IPARM(8)(=NORBT),
C                        IPARM(17)(=NMAX),
C                        IPARM(18)(=NNMX),   IPARM(19)(=NNMXM1),
C                        IPARM(20)(=LMAX),   IPARM(22)(=LLMXP1),
C                        IPARM(24)(=NCGC),   IPARM(27)(=M2STO),
C                        IPARM(28)(=L3MX),   IPARM(29)(=NFACT),
C                        IPARM(31)(=NVTERM), IPARM(32)(=NSTO),
C                        IPARM(33)(=ISPIN),  IPARM(39)(=IUNIT0),
C                        IPARM(41)(=JSTOT),  IPARM(42)(=MSTOT),
C                        QPARM(1)(=QRSTRT),  QPARM(2)(=QBAKUP),
C                        QPARM(3)(=QNWBAS),  QPARM(4)(=QOPEN),
C                        QPARM(7)(=QDEBUG),  QPARM(10)(=QEWMO),
C                        QPARM(11)(=QFIXED), QPARM(12)(=QVIRTL),
C                        QPARM(13)(=QNWVEC), QPARM(14)(=QMOVE),
C                        QPARM(15)(=QFLIP),  QPARM(16)(=QPNCHV),
C                        QPARM(17)(=QPLOT),  QPARM(18)(=QNWPOT),
C                        QPARM(24)(=QATOM),  QPARM(30)(=QANGST),
C                        QPARM(35)(=QTRACE),
C                        APARM(7)(=PKSCAL)
C
C                 SETS - IPARM(7)(=IORBT),   IPARM(10)(=IATOM),
C                        IPARM(16)(=MDIFAT), IPARM(41)(=JSTOT),
C                        QPARM(11)(=QFIXED), QPARM(12)(=QVIRTL),
C                        QPARM(13)(=QNWVEC), QPARM(14)(=QMOVE),
C                        QPARM(15)(=QFLIP),  QPARM(16)(=QPNCHV),
C                        QPARM(17)(=QPLOT),  QPARM(18)(=QNWPOT)
C
C        /IODATA/ USES - IUNIT(3)(=IUNIT3),  IUNIT(4)(=IUNIT4),
C                        IUNIT(5)(=IR),      IUNIT(6)(=IW),
C                        IUNIT(8)(=IUDUMP),  IUNIT(9)(=IUTEMP),
C                        LENBUF
C
C        /ENERGY/ SETS - TMOLE
C
C        /MODPOT/ USES - TFEXP(*), TFCOEF(*), LTF(*)
C
C        /PUNCHK/ USES - KATOM
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      DIMENSION QFLAGS(8)
      EQUIVALENCE (QPARM(11),QFLAGS(1))
      EQUIVALENCE (QPARM(1),QRSTRT),  (QPARM(2),QBAKUP),
     X            (QPARM(3),QNWBAS),  (QPARM(4),QOPEN),
     X            (QPARM(7),QDEBUG),  (QPARM(10),QEWMO),
     X            (QPARM(11),QFIXED), (QPARM(12),QVIRTL),
     X            (QPARM(13),QNWVEC), (QPARM(14),QMOVE),
     X            (QPARM(15),QFLIP),  (QPARM(16),QPNCHV),
     X            (QPARM(17),QPLOT),  (QPARM(18),QNWPOT),
     X            (QPARM(24),QATOM),  (QPARM(30),QANGST),
     X            (QPARM(35),QTRACE)
      EQUIVALENCE (IPARM(1),NATOM),   (IPARM(5),MSTO),
     X            (IPARM(6),NORB),    (IPARM(7),IORBT),
     X            (IPARM(8),NORBT),   (IPARM(10),IATOM),
     X            (IPARM(16),MDIFAT), (IPARM(17),NMAX),
     X            (IPARM(18),NNMX),   (IPARM(19),NNMXM1),
     X            (IPARM(20),LMAX),   (IPARM(22),LLMXP1),
     X            (IPARM(24),NCGC),   (IPARM(27),M2STO),
     X            (IPARM(28),L3MX),   (IPARM(29),NFACT),
     X            (IPARM(31),NVTERM), (IPARM(32),NSTO),
     X            (IPARM(33),ISPIN),  (IPARM(35),N2ORB),
     X            (IPARM(39),IUNIT0), (IPARM(41),JSTOT),
     X            (IPARM(42),MSTOT)
      EQUIVALENCE (APARM(7),PKSCAL)
      COMMON /IODATA/ IUNIT(20),LENBUF
      COMMON /ENERGY/ TOTPE,TOTKE,TMOLE,ZZPOT,ETOT,VIRTHM,DELTAE
      COMMON /MODPOT/ WGT0,WGT1,WGT2,WGTC,WGTV,VDAMP,VACCEL,
     X                TFEXP(4),TFCOEF(4),LTF(4),MXTF
      COMMON /PUNCHK/ RENRMA,RENRMB,KATOM,KORB,KSP,QPNCHK
      EQUIVALENCE (IUNIT(3),IUNIT3), (IUNIT(4),IUNIT4),
     X            (IUNIT(5),IR),     (IUNIT(6),IW),
     X            (IUNIT(7),IP),     (IUNIT(8),IUDUMP),
     X            (IUNIT(9),IUTEMP)
      COMMON /MOLSTR/ MSPTR
      CHARACTER*8 SPIN
      COMMON /CHARGS/ ZNSUM,ELSUM,ELDIF,ELECS(2),SPIN(2),ORBOCC,CHTRAN
      DIMENSION N(NSTO),                  L(NSTO),
     X          M0(NSTO),                 ETA(NSTO),
     X          ANORM(NSTO),              STOVL(M2STO),
     X          STKIN(M2STO),             STPOT(M2STO,NVTERM),
     X          STPOTX(NVTERM),           VEXP(NVTERM),
     X          VCOEF(NVTERM),            LV(NVTERM),
     X          MV(NVTERM),               FOCK(N2ORB,ISPIN),
     X          STHMAT(M2STO),            ONTRAN(MSTO,MSTO),
     X          BUFFER(LENBUF),           CGC(NCGC),
     X          S(NNMX),                  W(NNMXM1),
     X          INDEX(MSTO),              STEVEC(MSTO,NORB,ISPIN),
     X          EVAL(NORB,ISPIN),         OCC(NORB,ISPIN),
     X          CPOT2(M2STO),             COORD(3,NATOM),
     X          CHARGE(3,NATOM),          ZPOT2(M2STO),
     X          VFIT2(NVTERM),            PKPOT(M2STO,6,ISPIN),
     X          FACT(NFACT),              GAMMA(NORB,ISPIN),
     X          D(L3MX,2),                LABEL(NATOM), 
     X          SCRAT1(MSTO,MSTO),        SCRAT2(MSTO,9),
     X          SCRAT3(LLMXP1),           IATOMX(MSTOT),
     X          NX(MSTOT),                LX(MSTOT),
     X          MLX(MSTOT),               ETAX(MSTOT),
     X          ANORMX(MSTOT),            EVALX(NORBT,ISPIN),
     X          OCCX(NORBT,ISPIN),        EVECX(MSTOT,NORBT,ISPIN)
      DIMENSION LTYPE(6),XYZ(3),VCOEF0(10)
      DIMENSION EIGVAL(2),OCCNO(2)
      NAMELIST /ELEM/ DESCR,LATOM,NUCZ,MDIFAT,QTFPOT,RSCALE,QMNBAS
      NAMELIST /ATOM/ QFIXED,QVIRTL,QNWVEC,QDFVEC,QFLIP,QMOVE,
     X                QPNCHV,QPLOT,QNWPOT,
     X                VCOEF0,RMAX,IUATOM,TE,
     X                NEQAT,ISYM,IXYZ,XYZ
      NAMELIST /ATORB/ NCOEF,EIGVAL,OCCNO
      DATA ZERO/0.D0/,HALF/0.5D0/,ONE/1.D0/,TWO/2.D0/,THREE/3.D0/
      DATA LTYPE/'S','P','D','F','G','H'/
      DATA ESCALE/1.55D0/
      DATA ZSCALE/1.12956D0/
C     DATA RMAX0/10.0D0/
      DATA RMAX0/3.0D0/,ZPOWER/0.33333D0/
      DATA BPERA/1.889727D0/,APERB/0.529177D0/
      STNORM(NXX,ETAXX)=(TWO*ETAXX)**NXX*DSQRT(TWO*ETAXX/FACT(2*NXX+1))
C      IF (QDEBUG) CALL ARRMAP(2)
C-----------------------------------------------------------------------
C
C     DEFAULT AND READ ELEMENT DESCRIPTION (&ELEM).
C
C-----------------------------------------------------------------------
      QMNBAS=.FALSE.
      MDIFAT=1
      QTFPOT=.TRUE.
      RSCALE=ZERO
      DESCR='STO BASIS...'
      IURST=IUNIT4
      IF (QBAKUP) IURST=IUNIT3
      IF (QRSTRT) READ (IURST) MDIFAT
      CALL DERASE(STEVEC,MSTO*NORB*ISPIN)
      CALL DERASE(CPOT2,M2STO)      
      CALL DERASE(FOCK,N2ORB*ISPIN)
      CALL DERASE(GAMMA,NORB*ISPIN)
      CALL DERASE(PKPOT,M2STO*6*ISPIN)
      CALL DERASE(VFIT2,NVTERM)
      CALL DERASE(ZPOT2,M2STO)
      ZZREP=ZERO
      IF (QNWBAS) GO TO 10
      READ (IUNIT0) LATOM,NUCZ,N,L,ETA,ANORM
      READ (IUNIT0) STOVL,STKIN,STPOT,LV,MV,VEXP
      READ (IUNIT0) STHMAT,ONTRAN
   10 CONTINUE
      IF (QTRACE) WRITE (9,*) 'Reading Element Description (&ELEM)...'
      READ (IR,ELEM,END=320)
C      READ(IR,*,END=320) LATOM,NUCZ,DESCR,MDIFAT,QMNBAS,QTFPOT,RSCALE
      WRITE (IW,1000) LATOM,NUCZ
      IF (.NOT.QNWBAS) WRITE (IW,2000) LATOM,IUNIT0
      ZNUC=NUCZ
      IF (RSCALE.LE.ZERO) RSCALE=ZSCALE*ZNUC**(ONE/THREE)
      IF ((RSCALE.LE.ZERO).AND.(NUCZ.EQ.1)) RSCALE=ONE
      IF (QDEBUG) WRITE (IUDUMP,ELEM)
      IF (.NOT.QBAKUP) WRITE (IUNIT3) MDIFAT
C-----------------------------------------------------------------------
C
C     READ AND WRITE BASIS.
C
C-----------------------------------------------------------------------
      WRITE (IW,3000) DESCR
      JSTO=0
      IF (QTRACE) WRITE (9,*) 'Reading Basis...'
      DO 40 ISTO=1,NSTO
      IF (.NOT.QNWBAS) GO TO 20
      READ (IR,4000) NI,LI,ETAI
      ANORMI=STNORM(NI,ETAI)
      N(ISTO)=NI
      L(ISTO)=LI
      ETA(ISTO)=ETAI
      ANORM(ISTO)=ANORMI
      GO TO 30
   20 CONTINUE
      NI=N(ISTO)
      LI=L(ISTO)
      ETAI=ETA(ISTO)
      ANORMI=ANORM(ISTO)
   30 CONTINUE
      ETAI=DABS(ETAI)
      JSTO=JSTO+1
      LLI=2*LI
      MLI=-LI
	JSTOT=JSTOT+1
	IATOMX(JSTOT)=IATOM+1
	NX(JSTOT)=NI
	LX(JSTOT)=LI
	MLX(JSTOT)=MLI
	ETAX(JSTOT)=ETAI
	ANORMX(JSTOT)=ANORMI
      WRITE (IW,5000) JSTO,NI,LTYPE(LI+1),MLI,ETAI,ANORMI
      IF (LI.EQ.0) GO TO 37
      DO 35 IML=1,LLI
      MLI=IML-LI
      JSTO=JSTO+1
	JSTOT=JSTOT+1
	IATOMX(JSTOT)=IATOM+1
	NX(JSTOT)=NI
	LX(JSTOT)=LI
	MLX(JSTOT)=MLI
	ETAX(JSTOT)=ETAI
	ANORMX(JSTOT)=ANORMI
      WRITE(IW,5500) JSTO,MLI
   35 CONTINUE
   37 CONTINUE
      IF (NI.GT.NMAX) CALL BOMB(1)
      IF (LI.GT.LMAX) CALL BOMB(2)
      IF (JSTO.GT.MSTO) CALL BOMB(3)
	IF (JSTOT.GT.MSTOT) CALL BOMB(0)
      M0(ISTO)=JSTO-LI
   40 CONTINUE
C-----------------------------------------------------------------------
C
C     READ AND WRITE MODEL POTENTIAL.
C
C-----------------------------------------------------------------------
      LV(1)=0
      MV(1)=0
      VEXP(1)=ZERO
      VCOEF(1)=ZNUC
      IF (NVTERM.LT.2.OR..NOT.QNWBAS) GO TO 70
      WRITE(IW,8000)
      IF (QTRACE) WRITE (9,*) 'Reading Model Potential...'
      DO 60 NV=1,NVTERM
      COEF=0.0
      EXPN=1.0
      LVN=0
      MVN=0
      IF (QTFPOT.AND.NV.LE.MXTF) COEF=TFCOEF(NV)
      IF (QTFPOT.AND.NV.LE.MXTF) EXPN=TFEXP(NV)
      IF (QTFPOT.AND.NV.LE.MXTF) LVN=LTF(NV)
      IF ((.NOT.QTFPOT).OR.(NV.GT.MXTF)) 
     X   READ(IR,14000) LVN,MVN,EXPN,COEF
      LV(NV)=LVN
      MV(NV)=MVN
      VEXP(NV)=EXPN*RSCALE
      IF ((ZNUC.EQ.ZERO).OR.(EXPN.LT.ZERO)) VEXP(NV)=DABS(EXPN)
   50 CONTINUE
      IF (LV(NV).LT.0) GO TO 55
      NTYP=LV(NV)
      LTYP=0
      MTYP=0
      GO TO 57
   55 NTYP=-LV(NV)+1
      LTYP=-LV(NV)
      MTYP=MV(NV)
   57 CONTINUE
      VCOEF(NV)=ZNUC*COEF
      IF (ZNUC.EQ.ZERO) VCOEF(NV)=COEF
      WRITE(IW,10000) NV,NTYP,LTYPE(LTYP+1),MTYP,
     X                VEXP(NV),EXPN,VCOEF(NV),COEF
   60 CONTINUE
   70 CONTINUE
C-----------------------------------------------------------------------
C
C     COMPUTE THE 1-CENTER INTEGRALS AND WRITE ON IUNIT0.
C
C-----------------------------------------------------------------------
      IF (.NOT.QNWBAS) GO TO 90
      IF (QTRACE) WRITE (9,*) 'Computing 1-Center Integrals...'
      CALL TIMOUT(0)
      CALL ONEINT(STOVL,STKIN,STPOT,STPOTX,LV,MV,VEXP,N,L,ETA,ANORM,
     X     INDEX,CGC,QREALY)
      WRITE (IUNIT0) LATOM,NUCZ,N,L,ETA,ANORM
      WRITE (IUNIT0) STOVL,STKIN,STPOT,LV,MV,VEXP
      WRITE (IUTEMP) STOVL
      REWIND IUTEMP
      CALL LOWDIN(STOVL,ONTRAN,SCRAT1,MSTO,MSTO,0,QINDEF)
      IF (QINDEF) CALL BOMB(0)
      READ (IUTEMP) STOVL
      REWIND IUTEMP
      DO 80 IJ=1,M2STO
      STHMAT(IJ)=STKIN(IJ)-ZNUC*STPOT(IJ,1)
   80 CONTINUE
      WRITE (IUNIT0) STHMAT,ONTRAN

      CALL TEINDX(-1,BUFFER)
      CALL TWOINT(NTEI,N,L,ETA,S,W,INDEX,ANORM,CGC,QREALY,BUFFER)
      CALL DMPBUF(BUFFER)

      WRITE (IW,11000) LATOM,IUNIT0
      CALL TIMOUT(4)
   90 CONTINUE
C-----------------------------------------------------------------------
C
C     SET DEFAULT VALUES FOR THE FIRST ORBITAL.
C
C-----------------------------------------------------------------------
      QFIRST=.TRUE.
      OCCNO(2)=ONE
      ORBOCC=3-ISPIN
      DO 100 ISP=1,ISPIN
      OCC(1,ISP)=ONE
      EVAL(1,ISP)=ZERO
      IF (QMNBAS) STEVEC(1,1,ISP)=ONE
  100 CONTINUE
      TE=ZERO
      IUATOM=IR
      JATOM=IATOM
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER DIFFERENT ATOMS (JDIFAT).
C
C-----------------------------------------------------------------------
      WRITE (IW,12000) MDIFAT,LATOM
      DO 310 JDIFAT=1,MDIFAT
      IATOM=IATOM+1
      IF (IATOM.GT.NATOM) CALL BOMB(8)
C-----------------------------------------------------------------------
C
C     DEFAULT AND READ ATOM DESCRIPTION (&ATOM).
C
C-----------------------------------------------------------------------
      IF (QRSTRT) READ (IURST) QFLAGS,NEQAT,XYZ,STEVEC,EVAL,OCC,FOCK,
     X     GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      IF (QRSTRT.AND.QANGST) CALL SCMULT(XYZ,APERB,3)
      IF (QRSTRT.AND.QANGST) RMAX=APERB*RMAX
      ISYM=0
      IXYZ=0
      IF (.NOT.QRSTRT) CALL DERASE(XYZ,3)
      IF (QRSTRT) CALL DCOPY(XYZ,3,COORD(1,IATOM))
C     IF (.NOT.QRSTRT) RMAX=RMAX0
      IF (.NOT.QRSTRT) RMAX=RMAX0*ZNUC**(ZPOWER)
      CALL DCOPY(VCOEF,NVTERM,VCOEF0)
      QNWVEC=.FALSE.
      QDFVEC=QRSTRT
      QMOVE=(.NOT.QRSTRT)
      QFLIP=.FALSE.
      QFIXED=.FALSE.
      QVIRTL=.FALSE.
      QPNCHV=.FALSE.
      QPLOT=.FALSE.
      QNWPOT=.FALSE.
      NEQAT=0
      IF (QTRACE) WRITE (9,*) 'Reading Atom Description (&ATOM)...'
      IF (.NOT.QATOM) READ (IR,ATOM)
C      IF (.NOT.QATOM) READ (IR,*) XYZ,NEQAT,QFLIP,QFIXED,QVIRTL
      IF (QRSTRT) QDFVEC=.TRUE.
      IF (.NOT.QOPEN) QFLIP=.FALSE.
      IF (QFLIP.AND..NOT.QRSTRT) QDFVEC=.TRUE.
      IF (QBAKUP.AND.(QNWVEC.OR.QMOVE)) CALL WARN(12,QNWVEC)
      IF (QVIRTL) WRITE (IW,13000)
      IF (QVIRTL) RMAX=ZERO
      IF (QANGST) CALL SCMULT(XYZ,BPERA,3)
      IF (QANGST) RMAX=BPERA*RMAX
      IF (QDEBUG) WRITE (IUDUMP,ATOM)
      IF (.NOT.(QFIRST.OR.QDFVEC)) GO TO 200
      IF (QRSTRT.AND..NOT.QNWVEC) GO TO 180
      IF (QFLIP.AND..NOT.QRSTRT) GO TO 160
      REMAIN=ZNUC
C-----------------------------------------------------------------------
C
C     BEGIN LOOP OVER ORBITALS (IORB).
C
C-----------------------------------------------------------------------
      IF (QTRACE) WRITE (9,*) 'Reading Orbitals...'
      DO 150 IORB=1,NORB
      IIORB=(IORB*(IORB+1))/2
C-----------------------------------------------------------------------
C
C     DEFAULT AND READ ORBITAL DESCRIPTION (&ATORB).
C
C-----------------------------------------------------------------------
      LAST=MAX0(IORB-1,1)
      IF (QDFVEC) LAST=IORB
      DO 110 ISP=1,ISPIN
      OCCNO(ISP)=DMIN1(REMAIN/ORBOCC,OCC(LAST,ISP))
      EIGVAL(ISP)=EVAL(LAST,ISP)
  110 CONTINUE
      NCOEF=0
      IF (.NOT.QOPEN) OCCNO(1)=TWO*OCCNO(1)-OCCNO(2)
      READ (IUATOM,ATORB)
C      READ (IUATOM,*) NCOEF,EIGVAL,OCCNO
      IF (QDEBUG) WRITE (IUDUMP,ATORB)
      IF (.NOT.QOPEN) OCCNO(1)=HALF*(OCCNO(1)+OCCNO(2))
      DO 120 ISP=1,ISPIN
      OCC(IORB,ISP)=OCCNO(ISP)
      REMAIN=REMAIN-ORBOCC*OCCNO(ISP)
      EVAL(IORB,ISP)=EIGVAL(ISP)
      IF (QRSTRT) GO TO 120
      FOCK(IIORB,ISP)=EIGVAL(ISP)
      GAMMA(IORB,ISP)=EIGVAL(ISP)
      IF (EIGVAL(ISP).GT.ZERO) GAMMA(IORB,ISP)=ZERO
  120 CONTINUE
C-----------------------------------------------------------------------
C
C     DEFAULT AND READ STARTING VECTORS.
C
C-----------------------------------------------------------------------
      IF ((IORB.EQ.LAST).OR.QDFVEC.OR.(NCOEF.GT.0)) GO TO 130
      CALL DCOPY(STEVEC(1,LAST,1),MSTO-1,STEVEC(2,IORB,1))
      IF (QOPEN) CALL DCOPY(STEVEC(1,LAST,2),MSTO-1,STEVEC(2,IORB,2))
      GO TO 150
  130 CONTINUE
      IF (NCOEF.EQ.0) GO TO 150
      NCOEF=IABS(NCOEF)
      DO 140 ICOEF=1,NCOEF
      READ (IUATOM,14000) ISTO,ML,COEF1,COEF2
      IF ((ISTO.GT.NSTO).OR.(ISTO.LE.0)) CALL BOMB(11)
      IF (IABS(ML).GT.L(ISTO)) CALL BOMB(10)
      MI=M0(ISTO)+ML
      STEVEC(MI,IORB,1)=COEF1
      IF (.NOT.QOPEN) GO TO 140
      IF ((COEF2.EQ.ZERO).AND.(EIGVAL(1).EQ.EIGVAL(2))) COEF2=COEF1
      STEVEC(MI,IORB,2)=COEF2
  140 CONTINUE
C-----------------------------------------------------------------------
C
C     END LOOP OVER ORBITALS.
C
C-----------------------------------------------------------------------
  150 CONTINUE
      IF (IUATOM.NE.IR) WRITE (IW,15000) LATOM,JDIFAT,IUATOM
      GO TO 170
  160 CONTINUE
C-----------------------------------------------------------------------
C
C     IF QFLIP=T COPY THE LAST ATOM'S VECTORS, ENERGIES, AND
C     OCCUPATIONS WITH ALPHA AND BETA ROLES "FLIPPED".
C
C-----------------------------------------------------------------------
      CALL FLIP(STEVEC,1,ISPIN,STEVEC,ISPIN,ISPIN,MSTO*NORB)
      CALL FLIP(EVAL,1,ISPIN,EVAL,ISPIN,ISPIN,NORB)
      CALL FLIP(FOCK,1,ISPIN,FOCK,ISPIN,ISPIN,N2ORB)
      CALL FLIP(GAMMA,1,ISPIN,GAMMA,ISPIN,ISPIN,NORB)
      CALL FLIP(OCC,1,ISPIN,OCC,ISPIN,ISPIN,NORB)
      CALL FLIP(PKPOT,1,ISPIN,PKPOT,ISPIN,ISPIN,6*M2STO)
      QDFVEC=.FALSE.
      GO TO 185
  170 CONTINUE
C-----------------------------------------------------------------------
C
C     NORMALIZE AND WRITE STARTING VECTORS.
C
C-----------------------------------------------------------------------
      CALL NORMLZ(STEVEC,NORB,STOVL,MSTO,M2STO)
      IF (QOPEN) CALL NORMLZ(STEVEC(1,1,2),NORB,STOVL,MSTO,M2STO)
  180 CONTINUE
      WRITE (IW,16000) LATOM
      IF (QDFVEC) WRITE (IW,17000) JDIFAT
      WRITE (IW,18000)
  185 CONTINUE
      DO 190 ISP=1,ISPIN
      IF (QFLIP.AND..NOT.QRSTRT) GO TO 187
      IF (QOPEN) WRITE (IW,19000) SPIN(ISP)
      CALL OUTVEC(STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,NORB,IW)
  187 CONTINUE
      IF (QRSTRT) GO TO 190
      CALL DENMAT(PKPOT(1,4,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,4,ISP),PKSCAL,M2STO)
      CALL DENMAT(PKPOT(1,5,ISP),STEVEC(1,1,ISP),EVAL(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,5,ISP),PKSCAL,M2STO)
      CALL DENMAT(PKPOT(1,6,ISP),STEVEC(1,1,ISP),OCC(1,ISP),MSTO,
     X            M2STO,NORB)
      CALL SCMULT(PKPOT(1,6,ISP),ZERO,M2STO)
  190 CONTINUE
C-----------------------------------------------------------------------
C
C     WRITE ORBITAL ENERGIES AND OCCUPATIONS.
C
C-----------------------------------------------------------------------
      IF (QFLIP.AND..NOT.QRSTRT) GO TO 200
      WRITE (IW,16000) LATOM
      IF (QDFVEC) WRITE (IW,17000) JDIFAT
      WRITE (IW,20000)
      CALL POPOUT(EVAL,OCC,ZNUC,ALPHA,BETA,NORB)
      IF (.NOT.QRSTRT) 
     X   TOTEN=DOTPRD(EVAL,OCC,NORB*ISPIN)*ESCALE*(3-ISPIN)
      IF (TE.NE.ZERO) TOTEN=TE
      IF (QRSTRT.AND.QNWVEC) WRITE (IW,21000)
C-----------------------------------------------------------------------
C
C     WRITE STARTING COEFFICIENTS FOR THE MODEL POTENTIAL (SCALED IF
C     THE ATOM IS CHARGED AND QRSTRT=F).
C
C-----------------------------------------------------------------------
  200 IF (.NOT.(QFIRST.OR.QDFVEC.OR.QNWPOT)) GO TO 230
      IF (NVTERM.LT.2) GO TO 220
      IF (QNWPOT.AND..NOT.QFIRST) WRITE (IW,16000) LATOM
      IF (QNWPOT.AND..NOT.QFIRST) WRITE (IW,17000) JDIFAT
      SCALE=ONE
      IF ((.NOT.QRSTRT).AND.(NUCZ.NE.0)) SCALE=(ALPHA+BETA)/ZNUC
      WRITE (IW,22000)

      DO 210 NV=1,NVTERM
      IF (QFIRST.AND..NOT.QNWPOT) VCOEF(NV)=SCALE*VCOEF(NV)
      IF (QNWPOT) VCOEF(NV)=VCOEF0(NV)
      WRITE (IW,23000) NV,VCOEF(NV)
  210 CONTINUE

      IF (QDEBUG) CALL PLOTV(LV,MV,VEXP,VCOEF,NVTERM,IUDUMP)

  220 CONTINUE
C-----------------------------------------------------------------------
C
C     WRITE NUCLEAR COORDINATES.
C
C-----------------------------------------------------------------------
      IF (QATOM) GO TO 260
      WRITE (IW,24000)
  230 CONTINUE
      IF (QMOVE) GO TO 250
      DO 240 I=1,3
      IF (XYZ(I).NE.COORD(I,IATOM)) QMOVE=.TRUE.
  240 CONTINUE
  250 CONTINUE
      CALL DCOPY(XYZ,3,COORD(1,IATOM))
      LABEL(IATOM)=LATOM
      WRITE (IW,25000) LATOM,JDIFAT,XYZ
      IF (QFLIP) WRITE (IW,32000)
      IF (QVIRTL) WRITE (IW,27000)
      IF (QMOVE.AND.QRSTRT) WRITE (IW,28000)
      IF (QFIXED) WRITE (IW,29000)
C     IF (RMAX.LT.RMAX0) WRITE (IW,30000) RMAX
      WRITE (IW,30000) RMAX
  260 CONTINUE
      MATOM=JDIFAT
C     IF (KATOM.NE.0) WRITE (IP,40000) LATOM,MATOM,XYZ
C-----------------------------------------------------------------------
C
C     ASSIGN NUCLEAR AND ELECTRONIC CHARGES.
C
C-----------------------------------------------------------------------
      CHARGE(1,IATOM)=ZNUC
      CHARGE(2,IATOM)=ALPHA+BETA
      CHARGE(3,IATOM)=ALPHA-BETA
C-----------------------------------------------------------------------
C
C     WRITE STARTING VECTORS, ENERGIES, OCCUPATIONS, ETC. ON IUNIT3.
C
C-----------------------------------------------------------------------
      IF (.NOT.QBAKUP) WRITE (IUNIT3) QFLAGS,NEQAT,XYZ,STEVEC,EVAL,OCC,
     X     FOCK,GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      TMOLE=TMOLE+TOTEN
      IF (QVIRTL) GO TO 270
      NRATOM=NRATOM+1
      DO 263 ISP=1,ISPIN
         KORBT=IORBT
	   DO 262 IORB=1,NORB
            KORBT=KORBT+1
            EVALX(KORBT,ISP)=EVAL(IORB,ISP)
            OCCX(KORBT,ISP)=OCC(IORB,ISP)
            KSTOT=JSTOT
            IF (JDIFAT.EQ.1) KSTOT=KSTOT-MSTO
	      DO 261 JSTO=1,MSTO
               KSTOT=KSTOT+1
               IF (ISP.EQ.1.AND.IORB.EQ.1.AND.JDIFAT.GT.1) THEN
                  IATOMX(KSTOT)=IATOM
	            NX(KSTOT)=NX(KSTOT-MSTO)
	            LX(KSTOT)=LX(KSTOT-MSTO)
	            MLX(KSTOT)=MLX(KSTOT-MSTO)
	            ETAX(KSTOT)=ETAX(KSTOT-MSTO)
	            ANORMX(KSTOT)=ANORMX(KSTOT-MSTO)
               ENDIF
               EVECX(KSTOT,KORBT,ISP)=STEVEC(JSTO,IORB,ISP)
  261       CONTINUE
  262    CONTINUE
  263 CONTINUE
      IF (JDIFAT.GT.1) JSTOT=JSTOT+MSTO
	IORBT=IORBT+NORB
      IF (IORBT.GT.NORBT) CALL BOMB(9)
  270 CONTINUE
      QFIRST=.FALSE.
C-----------------------------------------------------------------------
C
C     LOOP OVER EQUIVALENT ATOMS (IEQAT).
C
C-----------------------------------------------------------------------
      IF (NEQAT.LE.0) GO TO 310
      IF (QRSTRT.AND.QNWVEC) WRITE (IUTEMP) STEVEC,EVAL,OCC
      IF (QRSTRT.AND.QNWVEC) REWIND IUTEMP
      IF (QTRACE) WRITE (9,*) 'Reading Equivalent Atoms (&ATOM)...'
      DO 300 IEQAT=1,NEQAT
      IATOM=IATOM+1
      IF (IATOM.GT.NATOM) CALL BOMB(8)
      CHARGE(1,IATOM)=ZNUC
      CHARGE(2,IATOM)=CHARGE(2,IATOM-1)
      CHARGE(3,IATOM)=CHARGE(3,IATOM-1)
      IF (QRSTRT) READ (IURST) FLAGS,ISYM,IXYZ,XYZ,STEVEC,EVAL,OCC,
     X     FOCK,GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,RMAX
      IF (QMOVE) CALL DERASE(XYZ,3)
      QFLIP=.FALSE.
      QPNCHV=.FALSE.
      QPLOT=.FALSE.
      READ (IR,ATOM)
C      READ (IR,*) XYZ,ISYM,IXYZ,QFLIP,QFIXED,QVIRTL
      IF (.NOT.QOPEN) QFLIP=.FALSE.
      IF (QVIRTL) RMAX=ZERO
      IF (QDEBUG) WRITE (IUDUMP,ATOM)
      IF (QRSTRT.AND..NOT.QNWVEC) GO TO 280
      IF (QRSTRT) READ (IUTEMP) STEVEC,EVAL,OCC
      IF (QRSTRT) REWIND IUTEMP
      IF (ISYM.EQ.0) GO TO 275
      CALL SYMOP(ISYM,IXYZ,STEVEC,VCOEF,L,LV,D,SCRAT3,
     X           NSTO,MSTO,NORB,ISPIN,NVTERM,LMAX,LLMXP1)
      IF (QDEBUG) WRITE(IUDUMP,36000) IABS(ISYM),LATOM,JDIFAT,IEQAT
      IF (QDEBUG) CALL PUTDC(D,L3MX,LMAX,IUDUMP)
  275 CONTINUE
      IF (.NOT.QFLIP) GO TO 280
      CALL FLIP(STEVEC,1,2,STEVEC,2,2,MSTO*NORB)
      CALL FLIP(EVAL,1,2,EVAL,2,2,NORB)
      CALL FLIP(OCC,1,2,OCC,2,2,NORB)
      CALL FLIP(FOCK,1,2,FOCK,2,2,N2ORB)
      CALL FLIP(GAMMA,1,2,GAMMA,2,2,NORB)
      CALL FLIP(PKPOT,1,ISPIN,PKPOT,ISPIN,ISPIN,6*M2STO)
  280 CONTINUE
      IF (QFLIP) CHARGE(3,IATOM)=-CHARGE(3,IATOM)
      R2=XYZ(1)**2+XYZ(2)**2+XYZ(3)**2
      IF (QMOVE.AND.R2.LT.1D-4) 
     X   CALL SYMXYZ(ISYM,IXYZ,XYZ,COORD,NATOM,IATOM)
      WRITE (IW,26000) XYZ
      MATOM=IEQAT*MDIFAT+JDIFAT
C     IF (KATOM.NE.0) WRITE (IP,40000) LATOM,MATOM,XYZ
      IF (QFLIP) WRITE (IW,32000)
      IF (QVIRTL) WRITE (IW,27000)
      IF (QMOVE.AND.QRSTRT) WRITE (IW,28000)
      IF (QFIXED) WRITE (IW,29000)
      JSYM=ISYM
      IF (ISYM.LT.0) JSYM=-ISYM
      IF (JSYM.GT.1) WRITE (IW,33000) JSYM
      IF (ISYM.LT.0) WRITE (IW,34000)
      CALL DCOPY(XYZ,3,COORD(1,IATOM))
      LABEL(IATOM)=LATOM
      IF (.NOT.QBAKUP) WRITE (IUNIT3) QFLAGS,ISYM,IXYZ,XYZ,STEVEC,EVAL,
     X     OCC,FOCK,GAMMA,CPOT2,VFIT2,VCOEF,PKPOT,ZPOT2,TOTEN,ZZREP,
     X     RMAX
      TMOLE=TMOLE+TOTEN
      IF (QVIRTL) GO TO 290
      NRATOM=NRATOM+1
      DO 283 ISP=1,ISPIN
         KORBT=IORBT
	   DO 282 IORB=1,NORB
            KORBT=KORBT+1
            EVALX(KORBT,ISP)=EVAL(IORB,ISP)
            OCCX(KORBT,ISP)=OCC(IORB,ISP)
            KSTOT=JSTOT
	      DO 281 JSTO=1,MSTO
               KSTOT=KSTOT+1
               IF (ISP.EQ.1.AND.IORB.EQ.1) THEN
                  IATOMX(KSTOT)=IATOM
	            NX(KSTOT)=NX(KSTOT-MSTO)
	            LX(KSTOT)=LX(KSTOT-MSTO)
	            MLX(KSTOT)=MLX(KSTOT-MSTO)
	            ETAX(KSTOT)=ETAX(KSTOT-MSTO)
	            ANORMX(KSTOT)=ANORMX(KSTOT-MSTO)
               ENDIF
               EVECX(KSTOT,KORBT,ISP)=STEVEC(JSTO,IORB,ISP)
  281       CONTINUE
  282    CONTINUE
  283 CONTINUE
      JSTOT=JSTOT+MSTO
      IORBT=IORBT+NORB
      IF (IORBT.GT.NORBT) CALL BOMB(9)
  290 CONTINUE
      IF (.NOT.(QRSTRT.AND.QNWVEC).OR.(IEQAT.GE.NEQAT)) GO TO 300
      WRITE (IUTEMP) STEVEC,EVAL,OCC
      REWIND IUTEMP
  300 CONTINUE
C-----------------------------------------------------------------------
C
C     END LOOP OVER DIFFERENT ATOMS.
C
C-----------------------------------------------------------------------
  310 CONTINUE
      MATOM=IATOM-JATOM
      WRITE (IW,35000) MATOM,LATOM
  320 RETURN
  
 1000 FORMAT(/' -------------------'/
     X       ' ATOMIC SYMBOL...',1X,A2/
     X       ' ATOMIC NUMBER...',I3/
     X       ' -------------------')
 2000 FORMAT(/' ... BASIS AND ONE-CENTER INTEGRALS FOR ',A2,' READ',
     X       ' FROM UNIT',I3,' ...')
 3000 FORMAT(/' ',A40/
     X       /'  #',5X,'TYPE',5X,'EXPONENT',5X,'NORMALIZATION'
     X       /'  _',5X,'____',5X,'________',5X,'_____________'/)
 4000 FORMAT(2I3,F10.6)
 5000 FORMAT(' ',I2,I6,A1,I2,F13.6,1PD18.6)
 5500 FORMAT(' ',I2,7X,I2)
 8000 FORMAT(/' MODEL POTENTIAL ...'//
     X       '  #',5X,'TYPE',5X,'EXPONENT',2X,'          ',5X,'COEF'/
     X       '  _',5X,'____',5X,'________',2X,'          ',5X,'____'/)
10000 FORMAT(' ',I2,I6,A1,I2,F13.6,2X,'(',F6.3,') ',      
     X                       F10.4,2X,'(',F6.3,')')
11000 FORMAT(/' ... THE BASIS AND ONE-CENTER INTEGRALS FOR ',A2,
     X       ' WERE WRITTEN ON UNIT',I3,' ...')
12000 FORMAT(/' ',I2,' NON-EQUIVALENT ',A2,' ATOMS WILL BE INPUT.')
13000 FORMAT(/' *** WARNING - THE TOTAL ENERGY WILL BE IN ERROR UNLESS',
     X       ' THIS ATOM IS MADE ''REAL'' ***')
14000 FORMAT(2I3,2F10.6)
15000 FORMAT(/' ... ATOMIC ORBITALS FOR ',A2,I2,' READ FROM UNIT',
     X       I3,' ...')
16000 FORMAT(///' ',A2/' __')
17000 FORMAT(' ',I4/'   __')
18000 FORMAT(/' STARTING VECTORS AND ENERGIES...')
19000 FORMAT(/' ',A8)
20000 FORMAT(/' STARTING ORBITAL ENERGIES AND OCCUPATIONS...')
21000 FORMAT(/' ... RESTART DATA HAS BEEN PARTIALLY OVERRIDDEN ...')
22000 FORMAT(/' STARTING COEFFICIENTS FOR THE MODEL POTENTIAL...'/)
23000 FORMAT(' ',I2,2X,F10.4)
24000 FORMAT(/' NUCLEAR COORDINATES...'//' ',10X,'X',14X,'Y',14X,'Z'/
     X                                 ' ',10X,'_',14X,'_',14X,'_')
25000 FORMAT(/' ',A2,I2,3F15.8/' ____')
26000 FORMAT(' ',4X,3F15.8)
27000 FORMAT(' ',T55,'* VIRTUAL        *')
28000 FORMAT(' ',T55,'* MOVED          *')
29000 FORMAT(' ',T55,'* FIXED          *')
30000 FORMAT(' ',T55,'* RMAX =',F8.3,' *')
32000 FORMAT(' ',T55,'* SPIN FLIPPED   *')
33000 FORMAT(' ',T55,'* ROTATED (C',I1,')   *')
34000 FORMAT(' ',T55,'* REFLECTED      *')
35000 FORMAT(/' THE INPUT FOR THE',I3,1X,A2,' ATOMS IS COMPLETE.')
36000 FORMAT(/' D-COEFFICIENTS FOR THE ',I3,'-FOLD ROTATION OF ',
     X       A2,'-',I2,' (',I2,')...')
40000 FORMAT(A2,I2,6X,3F10.5)
      END
