import numpy as np
from numpy import sin , cos
from math import pi, acos
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
        displacement = np.zeros(3)
        axis = np.zeros(3)
        displacement = target - current
        displacement = displacement[0:3,3]
        #displacement[0] = current[3,0]-target[3,0]
        #displacement[1] = current[3,1]-target[3,1]
        #displacement[2] = current[3,2]-target[3,2]
        R = np.linalg.inv(current[0:3,0:3]) @ target[0:3,0:3]
        S = 1/2 * (R - R.T)
        axis[0] = S[2,1]
        axis[1] = S[0,2]
        axis[2] = S[1,0]
        axis = current[0:3,0:3] @ axis
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

        distance = 0
        angle = 0
        
        R = np.linalg.inv(G) @ H
        distance = np.sqrt((G[3,0]-H[3,0])**2+(G[3,1]-H[3,1])**2+(G[3,2]-H[3,2])**2)
        temp = (np.trace(R)-1)/2
        if temp > 1:
            temp = 1
        if temp < -1:
            temp = -1
        angle = np.arccos(temp)
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

        success = True
        for i in range(0,7):
            if (q[i] > self.upper[i] or q[i] < self.lower[i]):
                success = False 
                break   
        joints, pose = self.fk.forward(q)
        distance, angle = self.distance_and_angle(pose, target)
        if distance > self.linear_tol or angle > self.angular_tol:
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

        dq = np.zeros(7)
        joints, current = FK().forward(q)
        displacement, axis = IK.displacement_and_axis(target, current)
        dq = IK_velocity(q, displacement, axis)
        #print('q,displacement, axis', q, displacement, axis)
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
        steps = 0
        while True:
            steps = steps + 1
            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q,target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            ## STUDENT CODE STARTS HERE
            J = calcJacobian(q)
            dq_null = null_space(J)
            #print('dq_center, dq_null',dq_center,dq_null)

            # Task Prioritization
            # dq_null np.array([[]])
            dq = np.zeros(7) # TODO: implement me!
            #print('update', (np.dot(dq_null.flatten(),dq_center))/(np.linalg.norm(dq_null)**2)*dq_null)
            #print('dq before update:', dq)
            update = (np.dot(dq_null.flatten(),dq_center))/(np.linalg.norm(dq_null)**2)*dq_null
            dq = dq_ik + update.flatten()
            
            # Termination Conditions
            if steps > self.max_steps or np.linalg.norm(dq) < self.min_step_size: # TODO: check termination conditions
                # print('steps:',steps, np.linalg.norm(dq), self.min_step_size)
                break # exit the while loop if conditions are met!

            ## END STUDENT CODE
            # print('q,dq',q,dq)
            q = q + dq

        success = self.is_valid_solution(q,target)
        return q, success, rollout

################################
## Simple Testing Environment ##
################################
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

    T_robot_static_setpoint = transform(np.array([0.562,0.169,0.2+0.05*3]),np.array([pi,0,0]))  # TESTING

    T_robot_dynamic = transform(np.array([0,-0.698-0.005,0.2+0.05/2]),np.array([pi,pi/4,0]))
    T_robot_ready = transform(np.array([0,-0.553,0.2+0.05/2-0.03]),np.array([pi,pi/4,0]))


    seed_static_front = np.array([-0.88827 , 0.48444 , 0.31647 ,-1.22539,  1.34645 , 0.9864,  -0.92593])  # sideway approach
    seed_static_back = np.array([ 0.05551 , 0.01429 , 0.03045 ,-1.82928 ,-1.548  ,  1.48791, -0.51166])   # sideway approach
    seed_static_top =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_dynamic =  np.array([0.25,-1.38,-1.82,-1.77,0.25,1.96,-1.09])
    seed =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    

    print("q_stack_top1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_top    # TESTING
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_top2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_top
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_top3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_top   # TESTING !!!
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_front   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_front
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+4)]),np.array([0,0,0])) @ T_robot_goal_front # TESTING
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_back   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_back
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+4)]),np.array([0,0,0])) @ T_robot_goal_back   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))
    
    print("q_stack_else1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_dynamic # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_else2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_dynamic
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_else3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_dynamic   # TESTING  NEW!
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_dynamic_ready:")
    q, success, rollout = ik.inverse(T_robot_ready, seed_dynamic)
    print(repr(q))
    T_robot_ready = transform(np.array([0,0,0]),np.array([0,0,-pi/9])) @ T_robot_dynamic   # scooping approach  NEW!
    print("NEW  q_dynamic_ready:")
    q, success, rollout = ik.inverse(T_robot_ready, seed_dynamic)
    print(repr(q))
    print("q_dynamic_dynamic:")
    q, success, rollout = ik.inverse(T_robot_dynamic, seed_dynamic)
    print(repr(q))
        
    print("*****************************  RED TEAM  *********************************")
    T_robot_goal_top = transform(np.array([0.562,0.169,0]),np.array([pi,0,0]))
    T_robot_goal_top = transform(np.array([0.562,0.169,0]),np.array([pi,0,0]))
    T_robot_goal_front = transform(np.array([0.562,0.169,0]),np.array([-pi/2,0,-pi/2]))  # sideway approach
    T_robot_goal_back = transform(np.array([0.562,0.169,0]),np.array([pi/2,0,-pi/2]))    # sideway approach
    T_robot_goal_dynamic = transform(np.array([0.562,0.169,0]),np.array([-pi/2,0,-pi/2])) @ transform(np.array([0,0,0]),np.array([0,-pi/4,0]))

    T_robot_static_setpoint = transform(np.array([0.562,-0.169,0.2+0.05*3]),np.array([pi,0,0]))   # TESTING

    T_robot_dynamic = transform(np.array([0,0.698+0.02,0.2+0.05/2]),np.array([0,-pi/2-pi/4,0]))
    T_robot_ready = transform(np.array([0,0.553,0.2+0.05/2-0.03]),np.array([0,-pi/2-pi/4,0]))

    seed =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_static_top =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    seed_static_front = np.array([-0.31383 , 0.47696 , 0.25822 ,-2.03823 , 1.43368  ,1.61479, -1.70826])  # sideway approach
    seed_static_back = np.array([ 0.05551 , 0.01429 , 0.03045 ,-1.82928 ,-1.548  ,  1.48791, -0.51166])   # sideway approach
    seed_dynamic = np.array([ 0.7546  , 1.16678 , 0.80799, -1.28532,  0.7526  , 1.66755 ,-1.31044])

    print("q_stack_top1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_top    # TESTING
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_top2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_top
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_top3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_top   # TESTING !!!
        q, success, rollout = ik.inverse(target, seed_static_top)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_front   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_front
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_front3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+4)]),np.array([0,0,0])) @ T_robot_goal_front # TESTING
        q, success, rollout = ik.inverse(target, seed_static_front)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_back   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_back
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_back3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+4)]),np.array([0,0,0])) @ T_robot_goal_back   # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))
    
    print("q_stack_else1:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_dynamic # TESTING
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_else2:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+1)-0.015]),np.array([0,0,0])) @ T_robot_goal_dynamic
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_stack_else3:")
    for block_num in range(0,6):
        target = transform(np.array([0,0,0.2+0.05*(block_num+3)]),np.array([0,0,0])) @ T_robot_goal_dynamic   # TESTING  NEW!
        q, success, rollout = ik.inverse(target, seed_static_back)
        if success == False:
            print("FFFFFFFFFFFFFFFFFFFFFFFUCK UUUUPPPP!!!!!!!!!!!!!!!!!!")
        print(block_num,repr(q))

    print("q_dynamic_ready:")
    q, success, rollout = ik.inverse(T_robot_ready, seed_dynamic)
    print(repr(q))
    T_robot_ready = transform(np.array([0,0,0]),np.array([0,0,-pi/9])) @ T_robot_dynamic   # scooping approach  NEW!
    print("NEW  q_dynamic_ready:")
    q, success, rollout = ik.inverse(T_robot_ready, seed_dynamic)
    print(repr(q))
    print("q_dynamic_dynamic:")
    q, success, rollout = ik.inverse(T_robot_dynamic, seed_dynamic)
    print(repr(q))
    # np.savetxt('data.txt',q,fmt='%10.5f',delimiter=',')
    # f.write("\n\n")

    # for i, q in enumerate(rollout):
    #     joints, pose = ik.fk.forward(q)
    #     d, ang = IK.distance_and_angle(target,pose)
    #     print('iteration:',i,' q =',q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # print("Success: ",success)
    # print("Solution: ",q)
    # print("Iterations:", len(rollout))

     
