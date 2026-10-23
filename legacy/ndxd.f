      FUNCTION NDXD(L,M,ISIGMA)
C-----------------------------------------------------------------------
C
C     NDXD...
C
C        THIS FUNCTION RETURNS THE VALUE OF THE COMPACT INDEX USED FOR
C     STORING AND RETRIEVING THE D-COEFICIENTS OF 'GENDC'.  THE FOLLOW-
C     ING RELATION IS ASSUMED TO BE TRUE:
C
C             L GE M GE ISIGMA GE 0
C
C-----------------------------------------------------------------------
      NDX1(I)=(I*(I+1))/2
      NDX2(I)=(I*(I+1)*(I+2))/6
      NDXD=NDX2(L)+NDX1(M)+ISIGMA+1
      RETURN
      END
