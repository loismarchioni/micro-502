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
        self.phase = 0                      # 0: store 1st frame, 1: 2nd frame + triangulation
        self.detected_gates = []            # estimated poses of each gate (x, y, z, yaw)
        self.gate_index = 0                 # current gate index. index = 0,...,4.
        self.gate_passed = False            # flag for passing gate

        self.prev_gate_center = None        # center coord of first frame (cx, cy)
        self.prev_pose = None               # drone's pose at first frame (x, y, z, yaw)
        self.estimated_gate_pose = None     # estimated pose of targeted gate (x, y, z, yaw)

    

    def current_gate_center(camera):
        """
        @ pars
        - camera : np.array of size (300, 300, 4). Current camera image in BGRA format.

        @ return
        - c : tuple (cx, cy). Center of the gate in the camera frame.

        ---
        @ brief
        - Compute the coordonates of the center of the current targeted gate in the camera frame.
        """

        ## extract magenta region
        image_BGR = camera[:, :, :3]
        hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

        # magenta mask
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


        ## extract largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        largest_cnt = None
        largest_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_cnt = cnt   # largest contour
                largest_area = area


        ## compute centroid of largest contour
        M = cv2.moments(largest_cnt)
        if M["m00"] == 0:
            M["m00"] = 1e-6

        c = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        return c
    

    def get_command_from_triangulation(self, sensor_data, camera_data):
        """
        @ pars
        - camera : np.array of size (300, 300, 4). Current camera image in BGRA format.

        @ return
        - 

        ---
        @ brief
        - 
        """

        # gate tracking FSM
        if self.phase == 0:         # 1st frame
            self.prev_gate_center = self.current_gate_center(camera_data)
            self.prev_pose = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            self.phase = 1

            d = 0.02    # [m]
            control_command = [sensor_data['x_global'] + d*np.sin(-sensor_data['yaw']),
                               sensor_data['y_global'] + d*np.cos(-sensor_data['yaw']),
                               sensor_data['z_global'], sensor_data['yaw']]
            return control_command

        elif self.phase == 2:       # 2nd frame + triangulate

            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
        else:                       # default case

            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            return control_command
        


def get_command(sensor_data, camera_data, dt):

    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example

    segments_yaws = [np.deg2rad(-90),   # toward origin
                        np.deg2rad(-45),   # toward gate 1 region
                        np.deg2rad(15),    # toward gate 2 region
                        np.deg2rad(75),    # toward gate 3 region
                        np.deg2rad(135),   # toward gate 4 region
                        np.deg2rad(195),]  # toward gate 5 region
    
    if sensor_data['z_global'] < 0.49:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, segments_yaws[1]]
        return control_command

    # ---- YOUR CODE HERE ----


    # ---- GENERAL INFOS ----

    # control_command : [x, y, z, yaw] in the inertial reference frame
    # sensor_data : <dictionnary>. Available ground truth state measurements
    # camera_data : <np.array> of size (300, 300, 4). Image im BGRA format, A=alpha is for opacity of the pixel. Pixel datatype is np.uint8. 
    # dt : elapsed time [s] since last call for the planner.


    # ---- FUNCTION ATTRIBUTES ----
    if not hasattr(get_command, "initialized"):     # done at first call to initialize internal FSM
        get_command.initialized = True
        get_command.state = "TRACKING"
        get_command.tracker = GateTracker()
        get_command.racer = None


    # ---- TAKE OFF ----



    # ---- 1st LAP : GATES DETECTION ----
    if get_command.state == "TRACKING":
        
        control_command = get_command.tracker.get_command_from_triangulation(sensor_data, camera_data)

        # control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, segments_yaws[1]]
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians



    

  