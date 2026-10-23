      SUBROUTINE INSERT(BLOCK,SUPMAT,I,J,NROW,NCOL)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     INSERT...
C
C        THIS ROUTINE INSERTS A RECTANGULAR BLOCK (MATRIX) INTO A LARGER
C     LOWER TRIANGULAR PACKED MATRIX.
C
C     DEFINITIONS:
C
C        BLOCK(*,*)... RECTANGULAR MATRIX TO BE INSERTED.
C        SUPMAT(*).... 'SUPER' MATRIX INTO WHICH THE BLOCK IS TO BE
C                      INSERTED.
C        I............ ROW INDEX IN SUPMAT(*) AT WHICH THE INSERTION
C                      IS TO BEGIN.
C        J............ COLUMN AT WHICH THE INSERTION BEGINS.
C        NROW......... NUMBER OF ROWS IN THE BLOCK.
C        NCOL......... NUMBER OF COLUMNS IN THE BLOCK.
C
C-----------------------------------------------------------------------
      DIMENSION BLOCK(NROW,NCOL),SUPMAT(1)
      IJ=(I*(I-1))/2+J
      IJSKIP=I-NCOL-1
      DO 20 K=1,NROW
      DO 10 L=1,NCOL
      SUPMAT(IJ)=BLOCK(K,L)
      IJ=IJ+1
   10 CONTINUE
      IJ=IJ+IJSKIP+K
   20 CONTINUE
      RETURN
      END
