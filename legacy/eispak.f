      SUBROUTINE EISPAK(SYMMAT,EVAL,EVEC,SCRAT,NDIM,NBAS,N2BAS,
     X                  NVEC,IERR)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     EISPAK...
C
C        THIS ROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS OF A
C     SYMMETRIC (PACKED) MATRIX.  THIS ROUTINE MAKES USE OF THE SUB-
C     ROUTINES IN THE EIGENSYSTEM PACKAGE (EISPAK) OF ARGONNE NATIONAL
C     LABS.  ALL OR SOME REASONABLE PORTION OF THE EIGENSOLUTIONS ARE
C     ASSUMED TO BE DESIRED.
C
C     VARIABLE DEFINITIONS:
C
C        SYMMAT(*)..... THE SYMMETRIC (PACKED) MATRIX WHOSE EIGEN-
C                       SOLUTIONS ARE SOUGHT.
C        EVAL(*)....... THE EIGENVALUES IN NON-DECREASING ORDER.
C        EVEC(*,*)..... THE CORRESPONDING EIGENVECTORS.
C        SCRAT(*,*).... A SCRATCH ARRAY (AT LEAST 9*NBAS).
C        NDIM.......... ROW DIMENSION OF EVEC(*,*) IN THE CALLING
C                       ROUTINE.
C        NBAS.......... SIZE OF BASIS, ORDER OF SYMMAT.
C        N2BAS......... =(NBAS*(NBAS+1))/2, PACKED DIMENSION.
C        NVEC.......... NUMBER OF EIGENSOLUTIONS SOUGHT (.LE.NBAS).
C        IERR.......... =0 --> NO ERROR CONDITION.
C                       >0 --> ERROR DURING EIGENVALUE DETERMINATION.
C                       <0 --> ERROR DURING EIGENVECTOR DETERMINATION.
C
C     ROUTINES CALLED:  TRED3, IMTQLV, TINVIT, TRBAK3
C
C     NOTE:  IF ONLY A FEW SOLUTIONS (.LT. 7%) ARE SOUGHT AN ALTERNATIVE
C            ALGORITHM MIGHT BE MORE SUITABLE.
C
C     REFERENCE:  EISPAK DOCUMENTATION.
C
C-----------------------------------------------------------------------
      DIMENSION SYMMAT(N2BAS),EVAL(NVEC),EVEC(NDIM,NVEC),SCRAT(NBAS,9)
C-----------------------------------------------------------------------
C
C     REDUCE THE FULL SYMMETRIC MATRIX TO SYMMETRIC TRIDIAGONAL FORM.
C
C-----------------------------------------------------------------------
      CALL TRED3(NBAS,N2BAS,SYMMAT,SCRAT(1,1),SCRAT(1,2),SCRAT(1,3))
C-----------------------------------------------------------------------
C
C     FIND ALL THE EIGENVALUES OF THE TRIDIAGONAL MATRIX USING THE
C     IMPLICIT QL ALGORITHM.
C
C-----------------------------------------------------------------------
      CALL IMTQLV(NBAS,SCRAT(1,1),SCRAT(1,2),SCRAT(1,3),EVAL,SCRAT(1,4),
     X           IERROR,SCRAT(1,5))
      IERR=IERROR
      IF (IERR.NE.0) RETURN
C-----------------------------------------------------------------------
C
C     FIND THE EIGENVECTORS ASSOCIATED WITH NVEC LOWEST EIGENVALUES
C     USING INVERSE ITERATION AND STURM SEQUENCING.
C
C-----------------------------------------------------------------------
      CALL TINVIT(NDIM,NBAS,SCRAT(1,1),SCRAT(1,2),SCRAT(1,3),NVEC,EVAL,
     X           SCRAT(1,4),EVEC,IERROR,SCRAT(1,5),SCRAT(1,6),
     X           SCRAT(1,7),SCRAT(1,8),SCRAT(1,9))
      IERR=-IERROR
      IF (IERR.NE.0) RETURN
C-----------------------------------------------------------------------
C
C     TRANSFORM THE EIGENVECTORS BACK TO THOSE CORRESPONDING TO THE
C     FULL SYMMETRIC MATRIX.
C
C-----------------------------------------------------------------------
      CALL TRBAK3(NDIM,NBAS,N2BAS,SYMMAT,NVEC,EVEC)
      RETURN
      END
