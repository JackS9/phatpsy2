      SUBROUTINE ERASE(R4,N)
C-----------------------------------------------------------------------
C
C     ERASE...
C
C        THIS ROUTINE ERASES AN ARRAY OR MATRIX WHICH IS STORED 
C     CONTIGUOUSLY.  THE ELEMENTS ARE ALL SET TO ZERO (OR FALSE).
C
C        R4 = 0
C
C     VARIABLE DEFINITIONS:
C
C        R4(*)... ARRAY/MATRIX TO BE ERASED (ZEROED).
C        N....... NUMBER OF CONTIGUOUS STORAGE ELEMENTS TO BE ERASED.
C
C     ENTRY POINTS:
C
C        ERASE....   (REAL*4)
C        DERASE...   (REAL*8)
C        IERASE...   (INTEGER*4)
C        QERASE...   (LOGICAL*1)
C
C-----------------------------------------------------------------------
      REAL*4    R4(N)	  
      DO 10 I=1,N
      R4(I)=0.0
   10 CONTINUE
      RETURN
      END
	  
	  SUBROUTINE DERASE(R8,N)
      REAL*8    R8(N)
      DO 20 I=1,N
      R8(I)=0.0D0
   20 CONTINUE
      RETURN
      END
	  
	  SUBROUTINE IERASE(I4,N)
      INTEGER*4 I4(N)
      DO 30 I=1,N
      I4(I)=0
   30 CONTINUE
      RETURN
      END
	  
	  SUBROUTINE QERASE(Q1,N)
      LOGICAL*1 Q1(N)
      DO 40 I=1,N
      Q1(I)=.FALSE.
   40 CONTINUE
      RETURN
      END
