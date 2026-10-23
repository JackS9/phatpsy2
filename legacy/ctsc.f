      SUBROUTINE CTSC(C,S,SCRAT,SS,NBAS,NVEC)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     CTSC...
C
C        THIS ROUTINE IS A MODIFIED FORTRAN VERSION OF THE ROUTINE
C     UTHU.  IT PERFORMS A SIMILARITY TRANSFORMATION OF A SYMMETRIC
C     MATRIX.  THE TRANSFORMATION NEED NOT BE SQUARE.
C
C              T
C        SS = C S C
C
C     VARIABLE DEFINITIONS:
C
C        C(*,*)....... TRANSFORMATION MATRIX.
C        S(*)......... SYMMETRIC (PACKED) MATRIX TO BE TRANSFORMED.
C        SCRAT(*,*)... A SCRATCH MATRIX AT LEAST NBAS BY NVEC.
C        SS(*)........ SYMMETRIC (PACKED) TRANSFORMED MATRIX.
C        NBAS......... NUMBER OF COLUMNS IN C(*,*) AND ORDER OF SS(*).
C        NVEC......... NUMBER OF ROWS IN C(*,*) AND ORDER OF S(*).
C
C     NOTE:  S(*) AND SS(*) MAY BE THE SAME MATRIX.
C
C-----------------------------------------------------------------------
      DIMENSION C(NBAS,NVEC),SCRAT(NBAS,NVEC),S(1),SS(1)
      DATA ZERO/0.0D0/
	  
      DO 30 I=1,NBAS
      II=(I*(I-1))/2
      DO 20 J=1,NVEC
      SUM=ZERO	  
      DO 10 K=1,NBAS
      IK=II+K
      IF (K.GT.I) IK=(K*(K-1))/2+I
      SUM=SUM+S(IK)*C(K,J)
   10 CONTINUE   
      SCRAT(I,J)=SUM
   20 CONTINUE
   30 CONTINUE
   
      DO 60 I=1,NVEC
      II=(I*(I-1))/2
      DO 50 J=1,I
      IJ=II+J
      SUM=ZERO
      DO 40 K=1,NBAS
      SUM=SUM+C(K,I)*SCRAT(K,J)
   40 CONTINUE
      SS(IJ)=SUM
   50 CONTINUE
   60 CONTINUE
   
      RETURN
      END
C
C...
      SUBROUTINE CSCT(C,S,SCRAT,SS,NBAS,NVEC)
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C
C     CSCT...
C
C        THIS ROUTINE IS THE TRANSPOSE OPERATION OF CTSC.
C
C                  T
C        SS = C S C
C
C-----------------------------------------------------------------------
      DIMENSION C(NVEC,NBAS),SCRAT(NBAS,NVEC),S(1),SS(1)
      DATA ZERO/0.0D0/
	  
      DO 30 I=1,NBAS
      II=(I*(I-1))/2
      DO 20 J=1,NVEC
      SUM=ZERO
      DO 10 K=1,NBAS
      IK=II+K
      IF (K.GT.I) IK=(K*(K-1))/2+I
      SUM=SUM+S(IK)*C(J,K)
   10 CONTINUE
      SCRAT(I,J)=SUM
   20 CONTINUE
   30 CONTINUE
   
      DO 60 I=1,NVEC
      II=(I*(I-1))/2
      DO 50 J=1,I
      IJ=II+J
      SUM=ZERO
      DO 40 K=1,NBAS
      SUM=SUM+C(I,K)*SCRAT(K,J)
   40 CONTINUE
      SS(IJ)=SUM
   50 CONTINUE
   60 CONTINUE
   
      RETURN
      END
