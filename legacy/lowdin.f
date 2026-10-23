      SUBROUTINE LOWDIN(OVRLAP,ONTRAN,SCRAT1,NBAS,NDIM,ITYPE,QINDEF)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     LOWDIN...
C
C        THIS ROUTINE RETURNS ONE OF THE FOLLOWING ORTHONORMAL TRANS-
C     FORMATION MATRICES
C
C        ITYPE             TYPE
C
C                           -1   T    -1
C          0 ) . . . . . U D    U  = S    (INVERSE)
C
C                           -1/2      -1/2
C          1 ) . . . . . U D       = S    (CANONICAL)
C
C                           -1/2 T    -1/2
C          2 ) . . . . . U D    U  = S    (SYMMETRIC)
C
C                           +1/2 T    +1/2
C          3 ) . . . . .   D    U  = S    (CANONICAL)
C
C                           +1/2 T    +1/2
C          4 ) . . . . . U D    U  = S    (SYMMETRIC)
C
C     WHERE U IS THE UNITARY MATRIX WHICH DIAGONALIZES THE OVERLAP
C     MATRIX S
C
C             T
C            U S U = D
C
C     ROUTINES CALLED:  DSPEV, JACOBI; DSQRT, DABS, MOD
C
C-----------------------------------------------------------------------
      DIMENSION OVRLAP(1),ONTRAN(NDIM,NBAS),SCRAT1(NBAS,NBAS)
      PARAMETER (MXBAS=1000)
      DOUBLE PRECISION W(MXBAS),WORK(3*MXBAS)
      DATA ZERO/0.0D0/,ONE/1.0D0/,TOLER/1.0D-10/
      QINDEF=.FALSE.
C
C     ZERO OUT TRANSFORMATION MATRIX
C
      DO 5 J=1,NBAS
      DO 5 I=1,NDIM
    5 ONTRAN(I,J)=ZERO
C
C     FOR SYMMETRIC POWERS (ITYPE = 0, 2, 4), USE LAPACK DSPEV
C
      IF (MOD(ITYPE,2).NE.0) GO TO 50
      IF (NBAS.GT.MXBAS) THEN
         WRITE (6,*) '*** ERROR: NBAS EXCEEDS MXBAS IN LOWDIN ***'
         QINDEF=.TRUE.
         RETURN
      END IF
C
      CALL DSPEV('V','U',NBAS,OVRLAP,W,SCRAT1,NBAS,WORK,INFO)
      IF (INFO.NE.0) THEN
         WRITE (6,*) '*** ERROR: DSPEV FAILED IN LOWDIN, INFO=',INFO
         QINDEF=.TRUE.
         RETURN
      END IF
C
      DO 20 I=1,NBAS
      DO 20 J=1,NBAS
      SUM=ZERO
      DO 10 K=1,NBAS
      DIAG=W(K)
      IF (DABS(DIAG).GT.TOLER) THEN
         IF (ITYPE.EQ.0) DIAG=ONE/DIAG
         IF (ITYPE.EQ.2) THEN
            IF (DIAG.LT.ZERO) THEN
               QINDEF=.TRUE.
               DIAG=ZERO
            ELSE
               DIAG=ONE/DSQRT(DIAG)
            END IF
         END IF
         IF (ITYPE.EQ.4) THEN
            IF (DIAG.LT.ZERO) THEN
               QINDEF=.TRUE.
               DIAG=ZERO
            ELSE
               DIAG=DSQRT(DIAG)
            END IF
         END IF
      ELSE
         IF (ITYPE.GT.0 .AND. DIAG.LT.ZERO) QINDEF=.TRUE.
         DIAG=ZERO
      END IF
      SUM=SUM+SCRAT1(I,K)*DIAG*SCRAT1(J,K)
   10 CONTINUE
      ONTRAN(I,J)=SUM
   20 CONTINUE
C
C     SET DIAGONAL MATRIX D IN OVRLAP
C
      NN=(NBAS*(NBAS+1))/2
      DO 30 I=1,NN
   30 OVRLAP(I)=ZERO
      DO 40 K=1,NBAS
      KK=(K*(K+1))/2
   40 OVRLAP(KK)=W(K)
      RETURN
C
C     FOR CANONICAL TRANSFORMS (ITYPE = 1, 3), USE JACOBI TO PRESERVE
C     SPATIAL/ATOMIC BASIS ORIENTATION
C
   50 CONTINUE
      CALL JACOBI(NBAS,OVRLAP,SCRAT1,NBAS)
      DO 70 I=1,NBAS
      DO 60 J=1,NBAS
      JJ=(J*(J+1))/2
      DIAG=OVRLAP(JJ)
      IF (DIAG.LT.TOLER) THEN
         IF (DIAG.LT.-1.0D-8) THEN
            WRITE (6,2000) DIAG
            QINDEF=.TRUE.
         END IF
         ONTRAN(I,J)=ZERO
      ELSE
         DIAG=DSQRT(DIAG)
         IF (ITYPE.EQ.1) ONTRAN(I,J)=SCRAT1(I,J)/DIAG
         IF (ITYPE.EQ.3) ONTRAN(J,I)=DIAG*SCRAT1(I,J)
      END IF
   60 CONTINUE
   70 CONTINUE
      RETURN
C
 1000 FORMAT('-*** WARNING - ORTHONORMAL TRANSFORMATION UNSTABLE ***')
 2000 FORMAT('-*** LINEAR DEPENDENCY ENCOUNTERED, METRIC EIGENVALUE =',
     X       1PD10.1,' ***')
      END
