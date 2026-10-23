      SUBROUTINE SYMXYZ(ISYM,IXYZ,XYZ,COORD,NATOM,IATOM)
      IMPLICIT REAL*8(A-H,O-Z)
C-----------------------------------------------------------------------
C
C     SYMXYZ...
C
C        THIS ROUTINE DETERMINES THE COORDINATES OF THE EQUIVALENT ATOM.
C     IF THE COORDINATES ARE NON-ZERO ON INPUT THEN NO OPERATION IS
C     PERFORMED.
C
C     VARIABLE DEFINITIONS:
C
C        ISYM........ = N --> N-FOLD ROTATION.
C                     = 1 --> IDENTITY.
C                     = 0 --> IDENTITY.
C                     =-1 --> PURE REFLECTION.
C                     =-2 --> INVERSION.
C                     =-N --> N-FOLD ROTO-REFLECTION.
C        IXYZ........ A PNEUMONIC VECTOR REPRESENTING THE AXIS OF
C                     ROTATION OR THE PERPENDICULAR PLANE OF REFLECTION.
C        XYZ(*)...... CARTESIAN COORDINATES OF THIS ATOM.
C        COORD(*).... CARTESIAN COORDINATES OF ALL THE OTHER ATOMS.
C        NATOM....... NUMBER OF ATOMS.
C        IATOM....... INDEX OF THIS ATOM.
C
C     ROUTINES CALLED:  BOMB; IABS, DFLOAT, DATAN2
C
C-----------------------------------------------------------------------
      DIMENSION XYZ(3),COORD(3,NATOM),ROTATE(3,3)
      DATA ZERO/0.0D0/,ONE/1.0D0/,TWO/2.0D0/
      IF ((ISYM.EQ.0).OR.(ISYM.EQ.1)) RETURN
      PI=DATAN2(ONE,ZERO)*TWO
      NFOLD=IABS(ISYM)
      ANGLE=TWO*PI/DFLOAT(NFOLD)
      LAST=IATOM-1
      PARITY=ONE
C     REPLACE ARITHMETIC IF WITH MODERN IF-ELSE IF STRUCTURE
      IF (ISYM .LT. 0) THEN
         GO TO 60
      ELSE IF (ISYM .EQ. 0) THEN
         GO TO 80
      ELSE
         GO TO 70
      END IF
C-----------------------------------------------------------------------
C
C     REFLECTION.
C
C-----------------------------------------------------------------------
   60 PARITY=-ONE
      ANGLE=ANGLE+PI
C-----------------------------------------------------------------------
C
C     ROTATION.
C
C-----------------------------------------------------------------------
   70 XLAST=COORD(1,LAST)
      YLAST=COORD(2,LAST)
      ZLAST=COORD(3,LAST)
      IF (IXYZ.EQ.1) GO TO 73
      IF (IXYZ.EQ.10) GO TO 74
      IF (IXYZ.EQ.100) GO TO 75
      IF (IXYZ.NE.0) CALL BOMB(4)
   73 ALPHA=ANGLE
      BETA=ZERO
      GAMMA=ZERO
      GO TO 76
   74 ALPHA=PI/TWO
      BETA=ANGLE
      GAMMA=-PI/TWO
      GO TO 76
   75 ALPHA=ZERO
      BETA=ANGLE
      GAMMA=ZERO
   76 CALL GENROT(ALPHA,BETA,GAMMA,ROTATE)
   
C     WRITE(6,*) 'Rotation matrix for Euler angles: ',ALPHA,BETA,GAMMA
C	  CALL PUTMAT(ROTATE,3,3,6)
	  
      XYZ(1)=PARITY*(ROTATE(1,1)*XLAST+ROTATE(2,1)*YLAST
     X                                +ROTATE(3,1)*ZLAST)
      XYZ(2)=PARITY*(ROTATE(1,2)*XLAST+ROTATE(2,2)*YLAST
     X                                +ROTATE(3,2)*ZLAST)
      XYZ(3)=PARITY*(ROTATE(1,3)*XLAST+ROTATE(2,3)*YLAST
     X                                +ROTATE(3,3)*ZLAST)
   80 RETURN
      END
