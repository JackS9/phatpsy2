      SUBROUTINE POPOUT(EVAL,OCC,ZCHARG,ALPHA,BETA,NORB)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
      CHARACTER*1 QCC,QSKIP,QNOSKP
C-----------------------------------------------------------------------
C
C     POPOUT...
C
C        THIS ROUTINE WRITES OUT THE ORBITAL ENERGIES AND OCCUPATION
C     NUMBERS IN COLUMN FORM, FOLLOWED BY A CHARGE AND SPIN ANALYSIS.
C
C     DEFINITIONS:
C
C        EVAL(*)... EIGENVALUE ARRAY.
C        OCC(*).... OCCUPATION NUMBERS.
C        ZCHARG.... NUCLEAR CHARGE.
C        ALPHA..... TOTAL ALPHA SPIN (SUM OF OCC(*,1)).
C        BETA...... TOTAL BETA SPIN (SUM OF OCC(*,ISPIN)).
C        NORB...... NUMBER OF ORBITALS.
C        QOPEN..... =T --> OPEN SHELL CASE.
C                   =F --> SPIN-RESTRICTED CASE.
C        ISPIN..... =1 IF QOPEN.
C                   =2 IF .NOT.QOPEN.
C        IW........ FORTRAN I/O UNIT FOR WRITING.
C
C     ROUTINES CALLED:  SUM; DABS
C
C     COMMON USAGE:
C
C        /PARMS/  USES - IPARM(33)(=ISPIN)
C                        QPARM(4)(=QOPEN)
C
C        /IODATA/ USES - IUNIT(6)(=IW)
C
C-----------------------------------------------------------------------
      COMMON /PARMS/ APARM(20),IPARM(50),QPARM(50)
      EQUIVALENCE (QPARM(4),QOPEN)
      EQUIVALENCE (IPARM(33),ISPIN)
      COMMON /IODATA/ IUNIT(20),LENBUF
      EQUIVALENCE (IUNIT(6),IW)
      DIMENSION EVAL(NORB,ISPIN),OCC(NORB,ISPIN)
	CHARACTER*8 SPIN(2)
      DATA ZERO/0.D0/,TWO/2.D0/
      DATA SPIN/'ALPHA...','BETA... '/,QSKIP/' '/,QNOSKP/' '/
      DATA EVSHAR/27.21070D0/,CALSEV/23.060D0/,DEGEN/5.D-5/
C-----------------------------------------------------------------------
C             ANGSTROM    BOHR
C
C ANGSTROM    1.000000  0.529177
C BOHR        1.889727  1.000000
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
C             HARTREES     EV     KCAL/MOL    KJ/MOL    CM-1
C
C HARTREES    1.000000  0.036749  0.001594  0.000381  0.000005
C EV          27.21165  1.000000  0.043367  0.010365  0.000124
C KCAL/MOL    627.4724  23.05896  1.000000  0.239006  0.002859
C KJ/MOL      2625.344  96.47868  4.184000  1.000000  0.011962
C CM-1        219474.6  8065.465  349.7758  83.59842  1.000000
C-----------------------------------------------------------------------
      HART=ZERO
      DO 20 ISP=1,ISPIN
      IF (QOPEN) WRITE (IW,1000) SPIN(ISP)
      WRITE (IW,2000)
      ORBOCC=3-ISPIN
      DO 10 IORB=1,NORB
      OCCUP=OCC(IORB,ISP)*ORBOCC
      OLDEN=HART
      HART=EVAL(IORB,ISP)
      QCC=QSKIP
      IF (DABS(HART-OLDEN).LT.DEGEN) QCC=QNOSKP
      RYDB=HART*TWO
      EVS=HART*EVSHAR
      CALS=EVS*CALSEV
      WRITE (IW,3000) QCC,IORB,HART,CALS,EVS,OCCUP
   10 CONTINUE
   20 CONTINUE
      ALPHA=ASUM(OCC(1,1),NORB,1)
      BETA=ASUM(OCC(1,ISPIN),NORB,1)
      ECHARG=-(ALPHA+BETA)
      TCHARG=ZCHARG+ECHARG
      WRITE (IW,4000) ZCHARG,ECHARG,TCHARG
      IF (.NOT.QOPEN) RETURN
      SPDIF=ALPHA-BETA
      WRITE (IW,5000) ALPHA,BETA,SPDIF
      RETURN
 1000 FORMAT(//' ',A8)
 2000 FORMAT(/'   #',T12,'HARTREES',T27,'KCAL/MOLE',T45,'EVS',T60,'OCC'/
     X       '   _',T12,'________',T27,'_________',T45,'___',T60,'___'/)
 3000 FORMAT(A1,I3,F15.4,F15.2,F15.4,F14.3)
 4000 FORMAT(/' CHARGE ANALYSIS...'//
     X       '    NUCLEAR CHARGE......',F9.4/
     X       '    ELECTRONIC CHARGE...',F9.4/
     X       '    NET CHARGE..........',F9.4)
 5000 FORMAT(/' SPIN ANALYSIS...'//
     X       '    ALPHA SPIN...',F9.4/
     X       '    BETA SPIN....',F9.4/
     X       '    NET SPIN.....',F9.4)
      END
