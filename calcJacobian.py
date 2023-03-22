import numpy as np
from lib.calculateFK import FK
from math import pi

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    J = np.zeros((6, 7))
    ## STUDENT CODE GOES HERE
    fk = FK()
    joint_positions, T0e = fk.forward(q_in)

    #calculate transformation matrices
    q1,q2,q3,q4,q5,q6,q7 = q_in[0:]
    A1 = FK.matrixA(0,-pi/2,0.333,q1)
    A2 = FK.matrixA(0,pi/2,0,q2)
    A3 = FK.matrixA(0.0825,pi/2,0.316,q3)
    A4 = FK.matrixA(0.0825,pi/2,0,pi+q4)
    A5 = FK.matrixA(0,pi/2,0.384,pi+q5)
    A6 = FK.matrixA(0.088,pi/2,0,q6)
    A7 = FK.matrixA(0,0,0.210,-pi/4+q7)

    T01 = A1
    T02 = T01 @ A2
    T03 = T02 @ A3
    T04 = T03 @ A4
    T05 = T04 @ A5
    T06 = T05 @ A6
    T0e = T06 @ A7

    ##jacobian calculation
    z0 = np.array([0,0,1]).reshape(1,3)
    z1 = T01[0:3,2].reshape(1,3)
    z2 = T02[0:3,2].reshape(1,3)
    z3 = T03[0:3,2].reshape(1,3)
    z4 = T04[0:3,2].reshape(1,3)
    z5 = T05[0:3,2].reshape(1,3)
    z6 = T06[0:3,2].reshape(1,3)
    z7 = T0e[0:3,2].reshape(1,3)
    o0 = np.array([0,0,0])
    o1 = joint_positions[1,:].reshape(1,3)
    o2 = joint_positions[2,:].reshape(1,3)
    o3 = joint_positions[3,:].reshape(1,3)
    o4 = joint_positions[4,:].reshape(1,3)
    o5 = joint_positions[5,:].reshape(1,3)
    o6 = joint_positions[6,:].reshape(1,3)
    o7 = joint_positions[7,:].reshape(1,3)
    #print(o7 - o0)
    #print(z1)
    j_v1 = np.cross(z0,(o7 - o0))
    j_v2 = np.cross(z1,(o7 - o1))
    j_v3 = np.cross(z2,(o7 - o2))
    j_v4 = np.cross(z3,(o7 - o3))
    j_v5 = np.cross(z4,(o7 - o4))
    j_v6 = np.cross(z5,(o7 - o5))
    j_v7 = np.cross(z6,(o7 - o6))
    Jv = np.array([j_v1,j_v2,j_v3,j_v4,j_v5,j_v6,j_v7])
    #print(Jv)
    Jv = Jv.transpose()
    #print(Jv)
    Jw = np.array([z0,z1,z2,z3,z4,z5,z6])
    Jw = Jw.transpose()
    #print(Jw)
    J = np.vstack((Jv,Jw))
    J = J.reshape(-1)
    J = J.reshape(6,7)
    return J

#if __name__ == '__main__':
    #q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    #print(np.round(calcJacobian(q),3))
