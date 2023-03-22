import numpy as np
from math import pi, acos
from numpy import sin , cos
from scipy.linalg import null_space

from calcJacobian import calcJacobian
from calculateFK import FK
from IK_velocity import IK_velocity

class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        d_current = current[0:3,3]
        d_target = target[0:3,3]

        R_current = current[0:3,0:3]
        R_target = target[0:3,0:3]

        displacement = d_target-d_current

        R_current_target = R_current.T@R_target
        S = 0.5*(R_current_target-R_current_target.T)
        axis_current = np.array([S[2,1],-S[2,0],S[1,0]])
        axis = R_current@axis_current

        ## END STUDENT CODE

        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H


        """

        ## STUDENT CODE STARTS HERE
        d_G = G[0:3,3]
        d_H = H[0:3,3]

        R_G = G[0:3,0:3]
        R_H = H[0:3,0:3]

        distance = np.linalg.norm(d_G-d_H,2)
        R_G_H = R_G.T@R_H
        arg = (np.trace(R_G_H)-1)*0.5
        arg = max(min(1, arg), -1)
        angle = np.arccos(arg)

        ## END STUDENT CODE

        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        jointPositions, current = self.fk.forward(q)
        distance, angle = self.distance_and_angle(current,target)

        ## Condition 1 : The joint angles are within the joint limits
        if (self.lower<=q).all() and (q<=self.upper).all():
            cond1 = True
        else:
            cond1 = False

        ## Condition 2: The distance between the achieved and target end effector positions is less than linear_tol
        if distance < self.linear_tol:
            cond2 = True
        else:
            cond2 = False

        ## Condition 3: The magnitude of the angle between the achieved and target end effector orientations is less than angular_tol
        if angle < self.angular_tol:
            cond3 = True
        else:
            cond3 = False

        ## Checking conditions 1,2,3
        if cond1 and cond2 and cond3:
            success = True
        else:
            success = False

        ## END STUDENT CODE

        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE
        jointPositions, current = IK.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(target, current)
        dq = IK_velocity(q, displacement, axis)  # IK_velocity(q_in, v_in, omega_in)

        ## END STUDENT CODE

        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []
        iter = 0
        while True:
            iter = iter + 1
            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q,target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            ## STUDENT CODE STARTS HERE

            # Task Prioritization
            J = calcJacobian(q)
            dq_null = null_space(J)[:,0]
            dq = dq_ik + (np.dot(dq_center,dq_null)/np.linalg.norm(dq_null)**2)*dq_null

            # Termination Conditions
            if iter == self.max_steps or np.linalg.norm(dq,2) <= self.min_step_size :
                break



            ## END STUDENT CODE

            q = q + dq

        success = self.is_valid_solution(q,target)
        return q, success, rollout

################################
## Simple Testing Environment ##
################################

# if __name__ == "__main__":

#     np.set_printoptions(suppress=True,precision=5)

#     ik = IK()

#     # matches figure in the handout
#     seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

#     target = np.array([
#         [0,-1,0,0.3],
#         [-1,0,0,0],
#         [0,0,-1,.5],
#         [0,0,0, 1],
#     ])

#     q, success, rollout = ik.inverse(target, seed)

#     for i, q in enumerate(rollout):
#         joints, pose = ik.fk.forward(q)
#         d, ang = IK.distance_and_angle(target,pose)
#         print('iteration:',i,' q =',q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

#     print("Success: ",success)
#     print("Solution: ",q)
#     print("Iterations:", len(rollout))

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotations in the order of x-axis, y-axis and z-axis with respect to intermediate frame.

    INPUT:
    d: translation vector
    rpy: rotation angles where 1st component is x-axis angle, 2nd component is y-asix angle and 3rd component is z-axis angle

    OUTPUT:
    Homogenous transform
    """
    trans = np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

    roll = np.array([
        [ 1,      0     ,       0     , 0 ],
        [ 0, cos(rpy[0]), -sin(rpy[0]), 0 ],
        [ 0, sin(rpy[0]),  cos(rpy[0]), 0 ],
        [ 0,      0     ,       0     , 1 ],
    ])

    pitch = np.array([
        [ cos(rpy[1]), 0, sin(rpy[1]) , 0 ],
        [      0     , 1,     0       , 0 ],
        [-sin(rpy[1]), 0,  cos(rpy[1]), 0 ],
        [      0     , 0,     0       , 1 ],
    ])

    yaw = np.array([
        [ cos(rpy[2]), -sin(rpy[2]), 0, 0 ],
        [ sin(rpy[2]),  cos(rpy[2]), 0, 0 ],
        [      0,            0     , 1, 0 ],
        [      0,            0     , 0, 1 ],
    ])
    return trans @ roll @ pitch @ yaw



if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)
    ik = IK()

    print("** BLUE TEAM  **")
    T_robot_goal_top = transform(np.array([0.562,-0.169,0]),np.array([pi,0,0]))
    T_robot_goal_front = transform(np.array([0.562,-0.169,0]),np.array([-pi/2,0,-pi/2]))  # sideway approach
    T_robot_goal_back = transform(np.array([0.562,-0.169,0]),np.array([pi/2,0,-pi/2]))    # sideway approach
    T_robot_goal_dynamic = transform(np.array([0.562,-0.169,0]),np.array([-pi/2,0,-pi/2])) @ transform(np.array([0,0,0]),np.array([0,-pi/4,0]))

    T_robot_static_setpoint = transform(np.array([0.562,0.169,0.2+0.05*2]),np.array([pi,0,0]))

    T_robot_dynamic = transform(np.array([0,-0.698-0.005,0.2+0.05/2]),np.array([pi,pi/4,0]))
    T_robot_ready = transform(np.array([0,-0.553,0.2+0.05/2-0.03]),np.array([pi,pi/4,0]))

    seed_static_front = np.array([-0.88827 , 0.48444 , 0.31647 ,-1.22539,  1.34645 , 0.9864,  -0.92593])  # sideway approach
    seed_static_back = np.array([ 0.05551 , 0.01429 , 0.03045 ,-1.82928 ,-1.548  ,  1.48791, -0.51166])   # sideway approach
    seed_static_top =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_dynamic =  np.array([0.25,-1.38,-1.82,-1.77,0.25,1.96,-1.09])
    seed =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    q_static_setpoint = np.array([ 0.2158,   0.14197,  0.0795,  -1.96261, -0.01305,  2.10411,  1.08654])
    q_dynamic_ready = np.array([ 0.15706, -0.93944, -1.83014, -2.1291,  -0.16416,  2.27171, -0.84507])
    q_dynamic_dynamic = np.array([ 0.5687,  -1.21852, -1.85342, -1.44783, -0.37109,  1.97274, -0.69434])
    
    print("q_static_top2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_top
        q, success, rollout = ik.inverse(target, seed_static_top)
        print(block_num,success,repr(q))

    print("q_static_front2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_front
        q, success, rollout = ik.inverse(target, seed_static_front)
        print(block_num,success,repr(q))

    print("q_static_back2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_back
        q, success, rollout = ik.inverse(target, seed_static_back)
        print(block_num,success,repr(q))
        
    print("*****************************  RED TEAM  *********************************")
    T_robot_goal_top = transform(np.array([0.562,0.169,0]),np.array([pi,0,0]))
    T_robot_goal_front = transform(np.array([0.562,0.169,0]),np.array([-pi/2,0,-pi/2]))  # sideway approach
    T_robot_goal_back = transform(np.array([0.562,0.169,0]),np.array([pi/2,0,-pi/2]))    # sideway approach
    T_robot_goal_dynamic = transform(np.array([0.562,0.169,0]),np.array([-pi/2,0,-pi/2])) @ transform(np.array([0,0,0]),np.array([0,-pi/4,0]))

    T_robot_static_setpoint = transform(np.array([0.562,-0.169,0.2+0.05*2]),np.array([pi,0,0]))
    T_robot_dynamic = transform(np.array([0,0.698+0.02,0.2+0.05/2]),np.array([0,-pi/2-pi/4,0]))
    T_robot_ready = transform(np.array([0,0.553,0.2+0.05/2-0.03]),np.array([0,-pi/2-pi/4,0]))

    seed =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_static_top =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_static_front = np.array([-0.31383 , 0.47696 , 0.25822 ,-2.03823 , 1.43368  ,1.61479, -1.70826])  # sideway approach
    seed_static_back = np.array([ 0.05551 , 0.01429 , 0.03045 ,-1.82928 ,-1.548  ,  1.48791, -0.51166])   # sideway approach
    seed_dynamic = np.array([ 0.7546  , 1.16678 , 0.80799, -1.28532,  0.7526  , 1.66755 ,-1.31044])

    print("q_static_top2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_top
        q, success, rollout = ik.inverse(target, seed_static_top)
        print(block_num,success,repr(q))

    print("q_static_front2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_front
        q, success, rollout = ik.inverse(target, seed_static_front)
        print(block_num,success,repr(q))

    print("q_static_back2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_back
        q, success, rollout = ik.inverse(target, seed_static_back)
        print(block_num,success,repr(q))

    # np.savetxt('data.txt',q,fmt='%10.5f',delimiter=',')
    # f.write("\n\n")

    # for i, q in enumerate(rollout):
    #     joints, pose = ik.fk.forward(q)
    #     d, ang = IK.distance_and_angle(target,pose)
    #     print('iteration:',i,' q =',q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    print("Success: ",success)
    print("Solution: ",q)
    print("Iterations:", len(rollout))