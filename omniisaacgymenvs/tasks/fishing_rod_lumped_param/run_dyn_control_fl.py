import numpy as np
from termcolor import colored
import sys
import os
import csv
import shutil
import time
from Fishing_class import DataRobot
import CONST as c
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = ''.join([r'\usepackage{siunitx}', r'\usepackage{amsmath}'])

def get_folder_plot(directory):
    """
    Create a folder to store data
    Args:
        directory (string): path to the new folder to create

    Returns:
        _: create the folder

    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def plot_data_on(time_task, y_des, y, u_new, err, pos, directory, q=None):

    # time_task.append(time_task[-1])
    y.append(y[-1])
    u_new = np.array(u_new)
    u_new = np.squeeze(u_new)

    if len(q_des) < 10:
        if c.n_actuator > 1:
            NotImplementedError
        else:
            y_des = np.vdot(c.C_y, y_des) * np.ones_like(time_task)

    q = np.array(q)
    
    plt.figure(clear=True)
    plt.ylabel(r'Tip Position X $[m]$')
    plt.xlabel(r'Time $[s]$')
    plt.plot(time_task, -pos[:,0], label=r'X', linewidth=c.line_widt)
    plt.grid()
    plt.legend(loc='best', shadow = True, fontsize=c.font_size)
    plt.savefig(directory + '/pos_X.svg', format= 'svg')
    plt.show()
    
    plt.figure(clear=True)
    plt.ylabel(r'Tip Position Z $[m]$')
    plt.xlabel(r'Time $[s]$')
    plt.plot(time_task, pos[:,1], label=r'Z', linewidth=c.line_widt)
    plt.grid()
    plt.legend(loc='best', shadow = True, fontsize=c.font_size)
    plt.savefig(directory + '/pos_Z.svg', format= 'svg')
    plt.show()
    
    plt.figure(clear=True)
    plt.ylabel(r'Action New $[Nm]$')
    plt.xlabel(r'Time $[s]$')
    if c.n_actuator > 1:
        for i in range(0, c.n_actuator):
            # assuming matrix n_time x n_actuators
            # plt.plot(time_task_lin, u_new[:, i], label= 'u_' + str(i + 1), linewidth=C.line_widt)
            plt.plot(time_task, u_new[:, i], label= 'u_' + str(i + 1), linewidth=c.line_widt)
    else:
        plt.plot(time_task, u_new, label='u', linewidth=c.line_widt)
    plt.grid()
    plt.legend(loc='best', shadow = True, fontsize=c.font_size)
    plt.savefig(directory + '/action.svg', format= 'svg')
    if 0:
        plt.show()
    # if c.WANNA_PLOT:
    #     plt.show(block=False)
    #     plt.pause(c.TIME_2_PAUSE_PLOT)
    # plt.close()

    plt.figure(clear=True)
    plt.ylabel(r'q')
    plt.xlabel(r'Time $[s]$')
    for i in range(0, c.n_state):
        # assuming matrix n_time x n_actuators
        plt.plot(time_task, q[:, i], label= r'q_' + str(i + 1), linewidth=c.line_widt)
    plt.grid()
    plt.legend(loc='best', shadow = True, fontsize=c.font_size)
    plt.savefig(directory + '/q.svg', format= 'svg')
    if 0:
        plt.show()
    # if c.WANNA_PLOT:
    #     plt.show(block=False)
    #     plt.pause(c.TIME_2_PAUSE_PLOT)
    # plt.close()

    plt.figure(clear=True)
    plt.ylabel(r'Error $[Nm]$')
    plt.xlabel(r'Time $[s]$')
    if c.n_actuator > 1:
        for i in range(0, c.n_actuator):
            # assuming matrix n_time x n_actuators
            plt.plot(time_task, err[:, i], label= r'err' + str(i + 1), linewidth=c.line_widt)
    else:
        plt.plot(time_task, err, label='err', linewidth=c.line_widt)
    plt.grid()
    plt.legend(loc='best', shadow = True, fontsize=c.font_size)
    if 0:
        plt.show()
    # if c.WANNA_PLOT:
    #     plt.show(block=False)
    #     plt.pause(c.TIME_2_PAUSE_PLOT)
    # plt.close()

def run_dyn_control_fl(robot,
        q0,
        q0_dot,
        m,
        max_step,
        dt,
        C_y,
        underactuation_matrix,
        q_des,
        q_dot_des,
        q_ddot_des,
        q_dddot_des,
        name_task,
        rid = 9):

    """
    This function implements the running of the algorithm.
    It runs over time domain implmenting the dynamic of a specified robot in discrete time

    Args:
        robot (class): Class of the robot to control
        directory (string): name of the folder to store the data per iter.
        m (int): number of actuators of the robot.
        max_step (int): numbr of time steps for each trajectory.
        dt (double): sampling time for each trajectory.
        y_des (double): desired position trajectory.
        y_dot_des (double): desired velocity trajectory.
        y_ddot_des (double): desired acceleration trajectory.
        toll (double): error threshold to stop the learning algorithm.
        max_actuator (int, optional): _description_. Defaults to None.
        u0 (double, optional): _description_. Defaults to None.

    Raises:
        Store the all the data gained from the simulation.
    """

    u0 = np.zeros(m).reshape(-1,)

    y_des = np.dot(C_y, q_des)
    y_dot_des = np.dot(C_y, q_dot_des)
    y_ddot_des = np.dot(C_y, q_ddot_des)
    y_dddot_des = np.dot(C_y, q_dddot_des)

    print(colored('=============================================================================================================================================','yellow'))
    print(colored('\tImplementing a feedforward Controller  ','yellow'))
    print(colored('=============================================================================================================================================\n','yellow'))

    try:                    
        robot = DataRobot(q0,
                        q0_dot,
                        q0_dot,
                        u0,
                        underactuation_matrix=underactuation_matrix,
                        C_y=C_y)

        q_app, q_dot_app, q_ddot_app, pos_app = [], [], [], []
        err_y_app, control_app, time_app, y_app = [], [], [], []

        directory = os.getcwd() + '/data_folder/' + name_task
        if os.path.isdir(directory):
            directory = directory + '_' + str(int(np.floor(np.random.rand(1) * 10000))) 
            get_folder_plot(directory)
        else:
            get_folder_plot(directory)

        path_data = directory + "/test_" + name_task
        
        if os.path.exists(path_data + ".csv"):
            path_data = path_data + '_' + str(int(np.floor(np.random.rand(1) * 10000))) 
        else:
            pass
        path_data = path_data + ".csv"
        inputs = {
                'K': robot.K,
                'D': robot.D,
                'L': c.L,
                'I_zz': c.I_zz,
                'm' : c.m
            }
        file_path = directory + '/input.py'
        robot.write_inputs_to_python(inputs, file_path)

        print(colored('\n=============================================================================================================================================', 'blue'))
        print(colored('[INFO]: Salving data in\n\t{}'.format(path_data), 'blue'))
        print(colored('=============================================================================================================================================\n', 'blue'))

        fileHeader = ["time", "q", "q_dot", "u_new", "y", "err", "pos_x", "pos_y", "pos_z"]
        csvFile = open(path_data, "w")
        writer = csv.writer(csvFile)
        writer.writerow(fileHeader)

        q_app.append(robot.q0)
        q_dot_app.append(robot.q0_dot)
        pos_appoggio = robot.pos.reshape((-1,))
        pos_app.append(pos_appoggio)
        time_start = time.time()
        control_action_pre = -(np.pi + np.deg2rad(0)) * np.sin(2 * np.pi * c.time_task) # * rid #  2 *
        # control_action = robot.getControl(y_des, y_dot_des, y_ddot_des, r, y_dddot_des)

        control_app.append(control_action_pre)
        err_y_app.append(robot.y - y_des)

        for i in range(0, int(max_step)): # time

            # control_action = np.random.uniform(low=3.0, high=3.0, size=(m,)) # u0
            control_action = control_action_pre[i]

            if c.integratorIs == 'Euler':
                x_new = robot.eulerStep(robot.x, control_action, dt)
                q_new = x_new[:robot.n]
                q_dot_new = x_new[-robot.n:]
            elif c.integratorIs == 'RungeKutta':
                x_new = robot.rk4Step(robot.x, control_action, dt)
                q_new = x_new[:robot.n]
                q_dot_new = x_new[-robot.n:]
                q_dot_new = x_new[-robot.n:]
            elif c.integratorIs == 'VariableStep':
                x_new = robot.rkfStep(robot.x, control_action, dt)
                q_new = x_new[:robot.n]
                q_dot_new = x_new[-robot.n:]
            else:
                sys.exit(colored("No suitable integrator was selected. Killing the process ... ", "red"))

            # robot.spinOnes(q_new, q_dot_new, control_action)
            # control_action = robot.getControl(y_des, y_dot_des, y_ddot_des, r, y_dddot_des)
            # ## Update the robot class
            robot.spinOnes(q_new, q_dot_new, control_action)

            q_app.append(robot.q)
            q_dot_app.append(robot.q_dot)
            q_ddot_app.append(robot.q_ddot)
            pos_app.append(robot.pos.reshape((-1,)))
            control_app.append(control_action)
            time_app.append(time.time() - time_start)
            y_app.append(robot.y)
        
            if robot.m > 1:
                err_now = np.dot(robot.C_y, q_des - robot.q)
                if i%c.TIME2PRINT == 0:
                    print("[INFO]:\t\tt: %0.2f [s]\tq: %s\tpos: %s\terr: %0.3f" %(dt * i,['%.3f' % q_val for q_val in robot.q],\
                                                                                            ['%.4f' % pos_val for pos_val in robot.pos], \
                                                                                            ['%.3f' % err_ for err_ in err_now]))
            else:
                err_now = np.vdot(robot.C_y, q_des - robot.q)
                if i%c.TIME2PRINT == 0:
                    # print("[INFO]:\nt: %0.2f [s]\nq: %s\npos: %s\nerr: %0.3f" %(dt * i,\
                    #                                                            ['%.4f' % q_val for q_val in robot.q], \
                    #                                                            ['%.3f' % pos_val for pos_val in robot.pos],
                    #                                                            err_now))
                    print("[INFO]:\t\tt: %0.2f [s]\tpos: %s" %(dt * i, ['%.3f' % pos_val for pos_val in robot.pos]))
                    print("[INFO]:\t\tt: %0.2f [s]\tq: %s" %(dt * i, ['%.3f' % q_val for q_val in robot.q]))


            err_y_app.append(err_now)
            data = [time.time() - time_start, robot.q, robot.q_dot, control_action, robot.y, err_now, -robot.pos[0], robot.pos[1], robot.pos[2]]
            writer.writerow(data)


        time_finish = time.time()
        pos_app = np.array(pos_app)
        print('\n=============================================================================================================================================')
        print('[INFO]: Time is {} s'.format(np.round(abs(time_finish - time_start), 2)))
        csvFile.close()
        plot_data_on(c.time_task, # time_app
                q_des,
                y_app,
                control_action_pre, # control_app,
                err_y_app,
                pos_app,
                directory,
                q=q_app)

    except KeyboardInterrupt:
        print(colored('=============================================================================================================================================','red'))
        print(colored('                TASK HAS BEEN KILLED BY ME!', 'red'))
        print(colored('=============================================================================================================================================','red'))


if __name__ == '__main__':

    # os.system('cls') ## only windows

    m = c.n_actuator
    n = c.n_state
    dt = c.dt
    max_step = c.max_step
    q = np.zeros(n)
    q[0] = np.pi/2
    q_dot = np.zeros(n)
    u0 = np.zeros(m)
    robot_name = 'fishing_rod'
    q_des = np.zeros(n)
    q_dot_des = np.zeros(n)
    q_ddot_des = np.zeros(n)
    q_dddot_des = np.zeros(n)
    r = 2
    name_task = 'new_fishing_rod_'+ str(c.n_state) + '_basic_u_new'  # remove the motor variables 
    print('=============================================================================================================================================\n')

    run_dyn_control_fl(robot_name,
            q,
            q_dot,
            m,
            max_step,
            dt,
            c.C_y,
            c.A_act,
            q_des,
            q_dot_des,
            q_ddot_des,
            q_ddot_des,
            name_task)

