import numpy as np
from math import pi
class FK():
    """l1 = 0.192 + 0.141
    l4 = 0.121 + 0.195
    a4 = 0.0825
    a5 = 0.0825
    l6 = 0.259 + 0.125
    a7 = 0.088 """

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout"""
        pass 
    def matrixA(ai, alphai, di, thetai):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        mA = np.array([[np.cos(thetai),-np.sin(thetai)*np.cos(alphai),np.sin(thetai)*np.sin(alphai),ai*np.cos(thetai)],
                        [np.sin(thetai),np.cos(thetai)*np.cos(alphai),-np.cos(thetai)*np.sin(alphai),ai*np.sin(thetai)],
                        [0,np.sin(alphai),np.cos(alphai),di],
                        [0,0,0,1]])
        return mA 
                       
    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """
        # Your Lab 1 code starts here
        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)
        
        A1 = FK.matrixA(0,-pi/2,0.333,q[0])
        A2 = FK.matrixA(0,pi/2,0,q[1])
        A3 = FK.matrixA(0.0825,pi/2,0.316,q[2])
        A4 = FK.matrixA(0.0825,pi/2,0,pi+q[3])
        A5 = FK.matrixA(0,pi/2,0.384,pi+q[4])
        A6 = FK.matrixA(0.088,pi/2,0,q[5])
        A7 = FK.matrixA(0,0,0.210,-pi/4+q[6])
        
        T01 = A1
        T02 = T01 @ A2
        T03 = T02 @ A3
        T04 = T03 @ A4
        T05 = T04 @ A5
        T06 = T05 @ A6
        T0e = T06 @ A7

        p22 = [[0],[0],[0.195],[1]]
        p02 = T02 @ p22
        p44 = [[0],[0],[0.125],[1]]
        p04 = T04 @ p44
        p55 = [[0],[0],[-0.015],[1]]
        p05 = T05 @ p55
        p66 = [[0],[0],[0.051],[1]]
        p06 = T06 @ p66

        jointPositions[0,:] = 0, 0, 0.141
        jointPositions[1,:] = T01[0][3], T01[1][3], T01[2][3]
        jointPositions[2,:] = p02[0][0], p02[1][0], p02[2][0]
        jointPositions[3,:] = T03[0][3], T03[1][3], T03[2][3]
        jointPositions[4,:] = p04[0][0], p04[1][0], p04[2][0]
        jointPositions[5,:] = p05[0][0], p05[1][0], p05[2][0]
        jointPositions[6,:] = p06[0][0], p06[1][0], p06[2][0]
        jointPositions[7,:] = T0e[0][3], T0e[1][3], T0e[2][3]
        # Your code ends here
        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1


    # This code is for Lab 2, you can ignore it ofr Lab 1
    """
    def get_axis_of_rotation(self, q):

        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

    def compute_Ai(self, q):
        
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations

        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
        """
        

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
