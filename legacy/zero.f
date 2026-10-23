      FUNCTION QZERO(ARRAY,N)                                           
      IMPLICIT REAL*8(A-H,O,P,R-Z),LOGICAL*1(Q)
C-----------------------------------------------------------------------
C                                                                       
C     QZERO...                                                          
C                                                                       
C        THIS LOGICAL FUNCTION CHECKS FOR NON-ZERO ELEMENTS IN AN       
C     ARRAY OF ELEMENTS.                                                
C                                                                       
C     VARIABLE DEFINITIONS:                                             
C                                                                       
C        ARRAY(*)... ARRAY TO BE CHECKED.                               
C        N.......... NUMBER OF ELEMENTS IN ARRAY(*).                    
C        QZERO...... =T IF ALL ELEMENTS ARE ZERO.                       
C                    =F IF AT LEAST ONE ELEMENT IS NOT ZERO.            
C                                                                       
C-----------------------------------------------------------------------
      DIMENSION ARRAY(N)                                                
      DATA ZERO/0.0D0/                                                  
      QZERO=.TRUE.                                                      
      DO 10 I=1,N                                                       
      IF (ARRAY(I).NE.ZERO) QZERO=.FALSE.                               
   10 CONTINUE                                                          
      RETURN                                                            
      END                                                               
