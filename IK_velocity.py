
import numpy as np
from  calcJacobian import calcJacobian


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q: 0 x 7 vector corresponding to the robot's current configuration.
    :param v: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 0 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
    Jcob = calcJacobian(q_in)
    ein = np.hstack((v_in,omega_in))
    J = []
    e = []

    #J_inv = (J.T)@np.linalg.inv(J@J.T)
    #e  = np.hstack((v_in,omega_in))
    for i in range(0,6):
        if np.isnan(ein[i]):
            pass
        else:
            e.append(ein[i])
            J.append(Jcob[i])

    dq = np.zeros(7)

    if len(e) == 0:
        return dq
    else:
        e = np.array(e)
        J = np.array(J)
        J_aug = np.hstack((J,e.reshape(-1,1)))
        if np.linalg.matrix_rank(J_aug) != np.linalg.matrix_rank(J):
            if len(J.shape) == 2:
              J = J.reshape(len(J),-1)
            else:
              J = J.reshape(-1,len(J))

            dq = np.linalg.lstsq(J,e,rcond=None)[0]
            return dq
        else:
            J_inv = (J.T)@np.linalg.inv(J@J.T)
            dq = J_inv@e
            # print(dq)
            return dq