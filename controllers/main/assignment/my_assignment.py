import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


# ---- CLASSES ----

class GateTracker:
    def __init__(self):
        self.state = 0                          # 0: store 1st frame, 1: 2nd frame + triangulation
        self.detected_gates = []                # estimated poses of each gate [x, y, z]
        self.gate_index = 0                     # current gate index. index = 0,...,4
        
        self.possible_gate_corners = []         # detected gate corners list of np.array
        self.prev_sensor = None                 # sensor data [x, y, z, yaw, qx, qy, qz, qw]
        self.avg_distance = 0                   # average distance to the gate
        self.cntr = 0                           # general counter
        self.estimated_gate_pose = None         # estimated pose of targeted gate [x, y, z]
        self.nb_triang = 0                      # nb of triangulations


    ## PUBLIC METHODS    
    def get_command_1st_lap(self, sensor_data, camera_data, get_command_ref, Verbose=False):
        """
        @ pars
        - sensor_data : Data from sensors.
        - camera : np.array of size (300, 300, 4). Current camera image in BGRA format.
        - Verbose : bool. Allows to display debug info.
        @ return
        - control_command : list [x,y,z,yaw]. Setpoint command.
        ---
        @ brief
        - Compute and return the control command to perform the 1st lap.
        """

        ## gate tracking FSM
        # constant pars
        NB_MAX_TRIANG   = 4
        YAW_VAR         = np.deg2rad(30)
        YAW_VAR_THRESH  = np.deg2rad(2)
        FORWARD_CNTR    = 60
        FORWARD_COEFF   = 0.05
        ALTITUDE_CNTR   = 200
        ALTITUDE_CNTR2  = 100
        ALTITUDE_THRESH = 0.7
        DISTANCE_THRESH = 1e-1
        BASE_POSE       = [1.0, 4.0, 0.1]

        # STATE 0 : 1st frame
        if self.state == 0:         
            if Verbose: print("phase 0")

            self.possible_gate_corners = self.__find_gate_corners(camera_data)
            if self.possible_gate_corners is None:
                self.state = 2
            else:
                self.state = 3
            
            self.prev_sensor = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'],
                                sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]

            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
        

        # STATE 1 : 2nd frame + triangulate
        elif self.state == 1:       
            if Verbose: print("phase 1")

            self.nb_triang += 1

            curr_sensor = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'],
                           sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]
            curr_gate_corners = self.__find_gate_corners(camera_data)

            if curr_gate_corners is None or len(curr_gate_corners) != len(self.possible_gate_corners):
                self.nb_triang -= 1
                self.state = 0
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command

            # triangulate all possible gate corners sets
            for i,corners in enumerate(self.possible_gate_corners):

                # triangulate corners set and compute center
                if corners.shape != curr_gate_corners[i].shape:
                    continue

                estimated_corners = []
                for j, coords in enumerate(corners):
                    estimated_corners.append(self.__triangulate_gate(self.prev_sensor, coords, curr_sensor, curr_gate_corners[i][j]))
                
                estimated_center = np.mean(estimated_corners, axis=0)

                # identify correct center knowing the targeted gate's region
                if self.__in_correct_sector(estimated_center):
                    self.estimated_gate_pose = estimated_center
                    break
                    
                # no correct center found

            
            if self.estimated_gate_pose is None:
                self.state = 0
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
                return control_command
            
            # store current data for next triangulation
            self.prev_sensor = curr_sensor
            self.possible_gate_corners = curr_gate_corners

            # move a bit before performing next triangulation
            self.state = 3

            control_command = self.estimated_gate_pose.tolist() + [sensor_data['yaw']]
            return control_command
        

        # STATE 2 : turn YAW_VAR [rad] toward next region if no gate in frame
        elif self.state == 2:       
            if Verbose: print("phase 2")

            # avoid reaching ground
            if sensor_data['z_global'] < 0.5:
                z = 0.7
            else:
                z = sensor_data['z_global']
            
            # compute yaw difference
            ref      = self.prev_sensor[3] + YAW_VAR
            curr     = sensor_data['yaw']
            diff     = curr - ref
            diff_mod = (diff + np.pi)%(2*np.pi) - np.pi

            if abs(diff_mod) < YAW_VAR_THRESH:
                self.state = 0
                control_command = [sensor_data['x_global'], sensor_data['y_global'], z, sensor_data['yaw']]
            else:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], z, self.prev_sensor[3] + YAW_VAR]
                # print("prev_data   yaw: %.3f" %self.prev_sensor[3])
                # print("curr_sensor yaw: %.3f" %sensor_data['yaw'])

            return control_command

        
        # SATTE 3 : move toward setpoint
        elif self.state == 3:       
            if Verbose: print("phase 3")

            self.cntr += 1
            if self.cntr >= FORWARD_CNTR:
                if self.nb_triang <= NB_MAX_TRIANG:
                    self.state = 1
                self.cntr = 0

            # move forward
            if self.estimated_gate_pose is None:
                control_command = [sensor_data['x_global'] + FORWARD_COEFF*np.cos(-sensor_data['yaw']),
                                sensor_data['y_global'] - FORWARD_COEFF*np.sin(-sensor_data['yaw']),
                                sensor_data['z_global'], sensor_data['yaw']]

                if self.nb_triang >= NB_MAX_TRIANG:
                    self.nb_triang = 0
                    self.state = 0

            # move toward gate
            else:                                   
                # correct yaw to point toward gate
                corr_yaw = np.arctan2((self.estimated_gate_pose[1] - sensor_data['y_global']),
                                      (self.estimated_gate_pose[0] - sensor_data['x_global']))
                
                # correct only altitude if dz is too important
                if abs(self.estimated_gate_pose[2] - sensor_data['z_global']) > ALTITUDE_THRESH:
                    self.state = 4

                # XY distance
                distance_to_gate = np.linalg.norm(self.estimated_gate_pose[0:2] - (sensor_data['x_global'], sensor_data['y_global']))

                if distance_to_gate < DISTANCE_THRESH:  # estimated gate pose reached
                    self.__store_gate_and_reset()
                    control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

                else:
                    control_command = self.estimated_gate_pose.tolist() + [corr_yaw]

            return control_command


        # STATE 4 : correct altitude
        elif self.state == 4:
            if Verbose: print("phase 4")

            self.cntr += 1
            if self.cntr >= ALTITUDE_CNTR:
                if self.nb_triang <= NB_MAX_TRIANG:
                    self.state = 1
                self.cntr = 0

            control_command = [sensor_data['x_global'], sensor_data['y_global'], self.estimated_gate_pose[2], sensor_data['yaw']]
            return control_command


        # STATE 5 : reset altitude after passing a gate
        elif self.state == 5:
            if Verbose: print("phase 5")

            self.cntr += 1
            if self.cntr >= ALTITUDE_CNTR2:
                self.state = 0
                self.cntr = 0

            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.4, sensor_data['yaw']]
            return control_command


        # STATE 6 : all gates detected, go back to base
        elif self.state == 6:
            if Verbose: print("phase 6")


            # XY distance
            distance_to_base = np.linalg.norm(np.array([BASE_POSE[0], BASE_POSE[1]]) - (sensor_data['x_global'], sensor_data['y_global']))
            if distance_to_base < DISTANCE_THRESH:

                # 1st lap complete, switch to race mode
                get_command_ref.state = "RACE"
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

            else:
                corr_yaw = np.arctan2((BASE_POSE[1] - sensor_data['y_global']),
                                      (BASE_POSE[0] - sensor_data['x_global']))
                
                control_command = [BASE_POSE[0], BASE_POSE[1], 1.4, corr_yaw]


            return control_command


        # STATE DEFAULT
        else:                       
            if Verbose: print("phase default")

            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
        

    ## PRIVATE METHODS
    def __find_gate_corners(self, camera):

        ## extract magenta region
        image_BGR = camera[:, :, :3]
        hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

        # magenta mask
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        ## extract center of contour(s)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:    # no gate detected in the frame
            self.state = 2
            return None
        
        all_corners = []
        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # corner points
            corners = approx.reshape(-1, 2)

            # order corners (NO, NE, SE, SO)
            if corners.shape == (4,2):
                corners = sorted(corners, key=lambda x: (x[1], x[0]))

                top_two = corners[:2]
                bottom_two = corners[2:]

                top_left, top_right = sorted(top_two, key=lambda x: x[0])
                bottom_left, bottom_right = sorted(bottom_two, key=lambda x: x[0])

                corners = np.array([top_left, top_right, bottom_right, bottom_left])

            all_corners.append(corners)
        
    
        return all_corners      # list of np.array of int


    def __triangulate_gate(self, prev_sensor, prev_center, curr_sensor, curr_center):

        # intrinsic pars and misc
        FOCAL_LENGTH    = 161.013922282                     # focal lentgh in pixels.
        (WIDTH, HEIGHT) = (300, 300)                        # size of image in pixels.
        CAMERA_OFFSET   = np.array([0.03, 0.0, 0.01])       # camera position offset wrt the drone.
        R_cam_to_drone  = np.array([[0,  0, 1],             # Rotation matrix from camera frame to drone frame.
                                    [-1, 0, 0],
                                    [0, -1, 0]])

        # nested functions
        def pixel_to_camera_vector(center):
            uc     = center[0] - 0.5*WIDTH
            vc     = center[1] - 0.5*HEIGHT

            return np.array([uc, vc, FOCAL_LENGTH])
        
        def get_camera_to_world_vector(center, sensor):
            v_cam   = pixel_to_camera_vector(center)
            v_drone = R_cam_to_drone @ v_cam
            Rbw     = rotation_matrix_from_quaternion(sensor[4], sensor[5], sensor[6], sensor[7])

            return Rbw @ v_drone
        
        def get_camera_position(sensor):
            pos_drone = np.array([sensor[0], sensor[1], sensor[2]])
            Rq        = rotation_matrix_from_quaternion(sensor[4], sensor[5], sensor[6], sensor[7])

            return pos_drone + Rq @ CAMERA_OFFSET
        
        def rotation_matrix_from_quaternion(qx, qy, qz, qw):
            norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz

            R = np.array([[1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy)],
                          [2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx)],
                          [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)]])
            
            return R


        # vectors and camera centers
        r = get_camera_to_world_vector(prev_center, prev_sensor)
        s = get_camera_to_world_vector(curr_center, curr_sensor)
        P = get_camera_position(prev_sensor)
        Q = get_camera_position(curr_sensor)

        # solve (least squares)
        rr = r @ r
        rs = r @ s
        ss = s @ s
        b1 = (Q - P) @ r
        b2 = (Q - P) @ s
        A  = np.array([[ rr, -rs],
                    [ rs, -ss]])
        b  = np.array([b1, b2])

        try:
            lam, mu = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            lam, mu = 1.0, 1.0

        F = P + lam*r
        G = Q + mu*s
        H = 0.5*(F + G)     # position [x,y,z]

        return H


    def __in_correct_sector(self, position):
        (CX, CY) = (4.0, 4.0)       # center of wolrd frame area
        
        x     = position[0] - CX
        y     = position[1] - CY
        theta = np.arctan2(y, x)    # angle from center of world frame

        return (np.rad2deg(theta) > self.gate_index*60 - 140 and np.rad2deg(theta) < (self.gate_index + 1)*60 - 160)
    

    def __store_gate_and_reset(self):
        # store gate
        self.detected_gates.append(self.estimated_gate_pose)

        # reset gate and pars
        self.cntr = 0
        self.estimated_gate_pose = None
        self.possible_gate_corners = []
        self.prev_sensor = None
        self.nb_triang = 0

        self.gate_index += 1

        if self.gate_index <= 4:
            self.state = 5
        else:
            self.state = 6      # all estimated gate poses reached

        return



####################################
# --- MAIN ASSIGNMENT FUNCTION --- #
####################################

def get_command(sensor_data, camera_data, dt):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    
    # if sensor_data['z_global'] < 0.49:
    #     control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, segments_yaws[1]]
    #     return control_command

    # ---- YOUR CODE HERE ----


    # ---- GENERAL INFOS ----

    # control_command : <list> [x, y, z, yaw] in the inertial reference frame (in meters and radians).
    # sensor_data : <dictionnary>. Available ground truth state measurements
    # camera_data : <np.array> of size (300, 300, 4). Image im BGRA format, A=alpha is for opacity of the pixel. Pixel datatype is np.uint8. 
    # dt : elapsed time [s] since last call for the planner.


    # ---- FUNCTION ATTRIBUTES ----
    if not hasattr(get_command, "initialized"):     # done at first call to initialize internal FSM
        get_command.initialized = True
        get_command.state = "TAKE_OFF"
        get_command.tracker = GateTracker()
        get_command.racer = None                    # class to be defined


    # ---- TAKE OFF ----
    if get_command.state == "TAKE_OFF":

        if sensor_data['z_global'] < 1.2:
            # take-off
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.4, sensor_data['yaw']]

            return control_command
        
        else:
            get_command.state = "DETECTION"


    # ---- 1st LAP : GATES DETECTION ----
    if get_command.state == "DETECTION":
        
        control_command = get_command.tracker.get_command_1st_lap(sensor_data, camera_data, get_command, Verbose=True)

        # control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.4, sensor_data['yaw']]
        return control_command
    

    # ---- 2nd and 3rd LAP : RACE ----
    elif get_command.state == "RACE":

        print("racing time")

        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command



    

  