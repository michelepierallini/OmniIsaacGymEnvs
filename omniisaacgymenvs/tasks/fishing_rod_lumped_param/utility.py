import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = ''.join([r'\usepackage{siunitx}', r'\usepackage{amsmath}'])
import math

def define_csv_save_location(parentDir, expSet):
    print("This run belongs to experiment set:", expSet, "\n")
    path_to_exp_dir =  os.path.join(parentDir, expSet)
    if os.path.isdir(path_to_exp_dir): 
        print("Directory " + expSet + " already exists. New folder is not created.")
    else: 
        os.mkdir(path_to_exp_dir)
        print("New folder with name: " + expSet + " is created.")
    input("Press enter to continue")
    print("\n\n")
    
    return path_to_exp_dir

def plot_and_save_data(plottingData, xAxisLabel, yAxisLabel, label, savingData, 
                        filename, saveDir, display_plot = True, saveData = True, figsize = (6,8)):
    if display_plot:
        # plot the figures
        f, ax = plt.subplots(len(plottingData) ,1,figsize=figsize)
        
        for i in range(len(plottingData)): 
            
            # plotting
            for j in range(len(plottingData[i]) - 1): 
                ax[i].plot(plottingData[i][0], plottingData[i][j+1], label = label[i][j])

            # x axis label
            if xAxisLabel[i] is not None:
                ax[i].set_xlabel(xAxisLabel[i], fontsize = 15)
        
            # y axis title
            if yAxisLabel[i] is not None:
                ax[i].set_ylabel(yAxisLabel[i], fontsize = 15)

            # legend
            if None not in label[i]:
                ax[i].legend(loc = "best")

        plt.title(filename)
        plt.tight_layout()
        plt.show()

    if saveData:
        save_data = np.transpose(np.vstack(savingData))
        np.savetxt(saveDir + '/' + filename + ".csv", save_data, delimiter = ",")

def y_n_prompt(msg):
    prompt = False
    while 1:
        decision = input(msg)
        if decision == "y" or decision == "Y": 
            prompt = True
            break
        elif decision == "n" or decision == "N":
            break
    return prompt

def obtain_csv_filename(save_dir):
    while 1:
        print("\nEnter csv file name. If you don't care, just press enter")
        filename = input()
        if filename == "": 
            filename += "testing_" + str(int(np.random.random() * 1000000))

        files = os.listdir(save_dir)
        breakLoop = True
        for file in files: 
            if file == filename + ".csv": 
                print("You have a file with the same name. If you continue the prvious file will be overriden.")
                breakLoop = y_n_prompt("Are you SURE you want to continue? (y/n)")
        if breakLoop:
            break
    
    print("This data will correspond with csv with name:", filename + ".csv")

    return filename

def mjRef(init_point, set_point, t_f, max_step):
    mjerk = []
    mjerk_dot = []
    mjerk_ddot = []
    time_vett = np.linspace(0, t_f, max_step)

    for time in range(0, len(time_vett)):
        mjerk.append(
            init_point + (set_point - init_point) *
            (10.0 * (time_vett[time]/t_f)**3
            - 15.0 * (time_vett[time]/t_f)**4
            + 6.0 * (time_vett[time]/t_f)**5))

        mjerk_dot.append(
            (1.0/t_f) * (set_point - init_point) *
            (30.0 * (time_vett[time]/t_f)**2.0
            - 60.0 * (time_vett[time]/t_f)**3.0
            + 30.0 * (time_vett[time]/t_f)**4.0))

        mjerk_ddot.append(
            (1.0/t_f**2) * (set_point - init_point) *
            (60.0 * (time_vett[time]/t_f)
            - 180.0 * (time_vett[time]/t_f)**2.0
            + 120.0 * (time_vett[time]/t_f)**3.0))

    return mjerk, mjerk_dot, mjerk_ddot

def rotYawZ(yaw):
    return np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1] ])

def rotPitchY(pitch):
    return np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

def rotRollX(roll):
    return np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

def convertIMUStep(imu):
    """
    This function convertes the IMU data (degree) doing two operations
        1) wrap 180-180 
        2) to radians
    This function acts only to ONE IMU data, i.e., 3x1 np.array
    Args:
        imu (double array): imu data degrees 0-360

    Returns:
        doubel array:  imu data radiands -pi - pi
    """
    # imu \in \mathbb{R}^3
    imu = np.asanyarray(imu, dtype=np.float64).reshape((-1,))
    imu_x = imu[0]
    imu_y = imu[1]
    imu_z = imu[2]

    # wrap to 2pi handmade
    # if abs(imu_x) > 300 and abs(imu_x) < 380:
    #     imu_x = imu_x - 360
    # if abs(imu_y) > 300 and abs(imu_y) < 380:
    #     imu_y = imu_y - 360
    # if abs(imu_z) > 300 and abs(imu_z) < 380:
    #     imu_z = imu_z - 360
    
    imu_x_rad = math.radians(imu_x)
    imu_y_rad = math.radians(imu_y)
    imu_z_rad = math.radians(imu_z)
    return np.array([imu_x_rad, imu_y_rad, imu_z_rad])

def convertIMUAllVector(imu):
    """ 
        It does the operation of convertIMUStep but for the all vector of IMU data 
        acquired during one execution wrt time 
    """
    n_time, dim = np.shape(imu)
    if n_time < dim:
        imu = np.transpose(imu)
        n_time, dim = np.shape(imu)
    else:
        pass

    imu_x_rad = np.zeros((n_time, 1))
    imu_y_rad = np.zeros((n_time, 1))
    imu_z_rad = np.zeros((n_time, 1))

    imu_x = imu[:,0]
    imu_y = imu[:,1]
    imu_z = imu[:,2]
    for i in range(0,n_time):
        # wrap to 2pi handmade
        if (imu_x[i]) > 300 and (imu_x[i]) < 380:
            imu_x[i] = imu_x[i] - 360
        if (imu_y[i]) > 300 and (imu_y[i]) < 380:
            imu_y[i] = imu_y[i] - 360
        if (imu_z[i]) > 300 and (imu_z[i]) < 380:
            imu_z[i] = imu_z[i] - 360
        imu_x_rad[i] = math.radians(imu_x[i])
        imu_y_rad[i] = math.radians(imu_y[i])
        imu_z_rad[i] = math.radians(imu_z[i])
    return np.column_stack([imu_x_rad, imu_y_rad, imu_y_rad])

def pcc2delta0(q_pcc, d):
    phi, theta, dL = q_pcc
    Delta_x = theta * d * np.sin(phi)
    Delta_y = theta * d * np.cos(phi) 
    dL = q_pcc[2]
    return np.array([Delta_x, Delta_y, dL])

def ypr2pcc(angles, L, R = None, p = None):
    """
    From IMU data [angles, p]\in\mathbbf{R}^6 to PCC 
    model [phi theta dL]\in\mathbb{R}^3.
    
    I could also create a rotation matrix with the parametrization I like the most and 
    then, just implement the following formulas. 

    Args:
        R (double): rotational matrix SO(3)
        p (double): position of the robot tip
        angles (double): yaw pitch roll
        L (double): length of the robot

    Returns:
        double: conversion between the IMU output and the PCC model 
    """
    angles = np.asanyarray(angles, dtype=np.float64).reshape((-1,))
    if R is None: 
        # R = rotYawZ(angles[2]) * rotPitchY(angles[1]) * rotRollX(angles[0])
        ## IMU returns Yaw Pitch Roll in this order 
        R = rotYawZ(angles[0]) * rotPitchY(angles[1]) * rotRollX(angles[2])
    else:
        R = R

    if p is None:    
        # check this
        p = np.zeros(3)
    else:
        p = p

    ## if only IMU Data are aviable use: what follows
    D_coordinates = R2Delta(R)
    q_pcc = Dx2PCC(D_coordinates, L, R) # [phi, theta, dL]
    q_delta = np.array([-D_coordinates[0], -D_coordinates[1], q_pcc[-1]])
    return q_pcc, q_delta

def R2Delta(R):
    """
    Get the Delta-param variables from the rotation matrix gained from IMU data 
    Args:
        R (double): orientation matrix in SO(3)

    Returns:
        double: Delta-param variables
    """
    # R = np.asanyarray(R)
    Dx = float(1/2) * (R[2,0] - R[0,2] ) * np.arccos( R[2,2]) / np.sin(np.arccos(R[2,2]))
    Dy = float(1/2) * (R[2,1] - R[1,2] ) * np.arccos( R[2,2]) / np.sin(np.arccos(R[2,2]))
    return np.array([Dx, Dy])

def Dx2PCC(D_coordinates, L, R=None, d=None):
    """
    Get the PCC variables from the Delta-param variables
    Args:
        D_coordinates (double): Delta-param variables
        L (double): Resting length of the beam  
        d (double, optional): Where the actuator is places on the robot tip. Defaults to None.

    Returns:
        double: PCC variables
    """
    if R is None:
        R = np.eye(3)
    else:
        R = R
        
    if d is None:
        d = 1
    else:
        d = d

    Dx = D_coordinates[0]
    Dy = D_coordinates[1]
    Delta = np.sqrt(Dx**2 + Dy**2)
    theta = Delta/d # curvature
    phi = np.arccos(Dx/Delta) - math.pi/2
    ## absolute compression
    ## this has been used in the exp up to 23 but I am not sure it is correct
    # dL = -np.sin(Delta) * Delta**3 - 0 * L
    
    ## numerical issue but correct
    use = np.asanyarray(np.dot(R, np.array([0, 0, L]))).reshape((3,1))
    # dL = (use[2] * Delta**2 - L)/(math.sin(Delta) * Delta)
    ## more robust 
    dL = use[2,0] * theta/math.sin(theta) - L
    return np.array([phi, theta, dL])

def checkControlUpZero(u):
    """
    Tendons can only pull !
    In the model the force action on the head plane are negative wrt to z-axis on the base frame.
    Therefore, the each pulley sees a negative force in that direction. That's why I add a minus here. 
    BUT check if this makes sense otherwise you will break tendons!!
    Args:
        u (doble): control vector action on the tendonds

    Returns:
        double: pulling control input for the tendons
    """
    for i in range(0, len(u)):
        if u[i] < 0:
            u[i] = 0
            # u[i] = -u[i] # check this 
    return u

def removeOutliersVect(vector, threshold=None):
    """
    Removes outliers from a vector using the z-score method.

    Args:
        vector (double): vector which may contains outliears 
        threshold (double, optional): Threshold to detect outlier. Defaults to None.

    Returns:
        double: output vector with no outliers
    """
    if threshold is None:
        median = np.median(vector)
        mad = np.median(np.abs(vector - median))
        threshold = 3 * mad
    else:
        threshold = threshold

    mean = np.mean(vector)
    std = np.std(vector)
    z_scores = [(x - mean) / std for x in vector]
    filtered = [x for x, z in zip(vector, z_scores) if abs(z) < threshold]
    return filtered

def removeOutliersMatrix(matrix):
    """
    Removes outliers from a vector using the z-score method.

    Args:
        matrix (double): matrix which may contains outliears 

    Returns:
        double: output matrix with no outliers
    """
    matrix = np.asanyarray(matrix)
    n_rows, n_cols = np.shape(matrix)
    filtered_matrix = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        column = matrix[:, i]
        median = np.median(column)
        mad = np.median(np.abs(column - median))
        threshold = 3 * mad
        mean = np.mean(column)
        std = np.std(column)
        z_scores = [(x - mean) / std for x in column]
        filtered_column = [x if abs(z) < threshold else np.nan for x, z in zip(column, z_scores)]
        filtered_matrix[:, i] = filtered_column
    filtered_matrix = filtered_matrix[~np.isnan(filtered_matrix).any(axis=1)]
    return filtered_matrix


def lowPassFilter(matrix, window_size = None):
    """
    Applies a low-pass filter to each column of a matrix using a simple moving average window of size 'window_size'.
    
    Args:
        matrix (double): noisy data n_time x n_variables
        window_size (int, optional): windows to smooth the signal on. Defaults to None.

    Returns:
        double: smoothed matrix n_time x n_variables
    """

    if window_size is None:
        # n_max = max(np.shape(matrix))
        # window_size = np.floor(0.1  * n_max)
        window_size = 10
    else:
        window_size = window_size

    matrix = np.asanyarray(matrix)

    n_time, dim = np.shape(matrix)
    if dim > n_time:
        matrix = matrix.T
    else:
        pass

    n_time = np.max(np.shape(matrix))
    dim = np.min(np.shape(matrix))

    filtered_matrix = np.zeros_like(matrix)
    for i in range(0, dim):
        filtered_matrix[:, i] = np.convolve(matrix[:, i], np.ones(window_size) / window_size, mode='same')
    return filtered_matrix


def resampleData(data, n):
    """
    Resamples a matrix with dimensions n-smething x 3 to n x 3 using linear interpolation.
    
    Parameters:
        matrix (double): The input matrix with dimensions n-10 x 3.
        n (int): The desired number of rows in the output matrix, i.e., max_step
    
    Returns:
        double: The resampled matrix with dimensions n x 3.
    """ 
    n_real_data = np.max(np.shape(data))
    step_exp = np.linspace(0, n - 1, n_real_data)
    # Interpolate each column using linear interpolation
    f = interp1d(step_exp, data, axis=0, kind='linear')
    # Create a new array of indices to resample at
    time_real = np.linspace(0, n-1, n)
    # Resample each column using the interpolated function
    resampled_data = f(time_real)
    
    return resampled_data


def stopIterativeLearning(err, number=5):
    """
    Check if the system is still learning the desired trajectory

    Args:
        err (duoble): The RMS error quantifying the tracking perfeormance 
        number (int, optional): The windows of iterations to check. Defaults to 3.

    Returns:
        bool: return a flag to manage the iterative learning 
    """
    if len(err) > number: 
        conta = 0
        for j in range(2, number):
            if err[-j] > err[-j - 1]:
                conta = conta + 1
            else:
                pass
        if conta >= number - 3: 
            ## You can directly kille th execution of the code
            # sys.exit(colored("The system is NOT improving its tracking performances. Killing the process ... ", "red"))
            ## You can save data data 
            return True
        else:
            # pass
            return False
    else:
        # pass
        return False








    
    