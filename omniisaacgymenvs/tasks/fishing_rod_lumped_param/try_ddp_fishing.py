
import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to

def ddp_planning(tracking_Z_bool, 
                des_y_coordinate, 
                _pos_des, 
                _max_episode_length_s,
                _vel_lin_des, 
                K_matrix,
                D_matrix, 
                name_ddp_robot = "fishing_rod_four",
                u_max = 10, 
                th_stop=1e-3, 
                nthreads=1, 
                length_on_Y=0.07,
                max_iter=200):

    fishing_rod = example_robot_data.load(name_ddp_robot)
    robot_model = fishing_rod.model
    robot_model.gravity.linear = np.array([0, 0, -9.81]) # w.r.t. global frame
    state = crocoddyl.StateMultibody(robot_model)
    actuation = aslr_to.ASRFishing(state)
    nu = actuation.nu

    ## desired 
    if tracking_Z_bool:
        target_pos_ddp = np.array([des_y_coordinate, length_on_Y, _pos_des])
    else:
        target_pos_ddp = np.array([_pos_des, length_on_Y, des_y_coordinate])
        
    target_vel_ddp = np.array([_vel_lin_des, 0.0, 0.0])

    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("Link_EE"),
                                                            pinocchio.SE3(np.eye(3), target_pos_ddp), nu)
    framePlacementVelocity = crocoddyl.ResidualModelFrameVelocity(state, robot_model.getFrameId("Link_EE"),
                                                            pinocchio.Motion(target_vel_ddp, np.zeros(3)),
                                                            pinocchio.WORLD, nu)

    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e1] * state.nv + [1e0] * state.nv)) 
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
    goalVelCost = crocoddyl.CostModelResidual(state, framePlacementVelocity)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
    runningCostModel.addCost("gripperVel", goalVelCost, 1e1)
    runningCostModel.addCost("xReg", xRegCost, 1e0)
    runningCostModel.addCost("uReg", uRegCost, 1e0) # increase to decrease the cost of the control
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2) 
    terminalCostModel.addCost("gripperVel", goalVelCost, 1e1)
            
    runningModel = crocoddyl.IntegratedActionModelEuler(
                            aslr_to.DAM2(state, actuation, runningCostModel, K_matrix, D_matrix), 
                            0.0001
                        )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
                            aslr_to.DAM2(state, actuation, terminalCostModel, K_matrix, D_matrix), 
                            0
                        )

    runningModel.u_lb, runningModel.u_ub = np.array([-u_max]), np.array([u_max])

    q0 = np.zeros(state.nv)
    x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * int(_max_episode_length_s), terminalModel)
    solver = crocoddyl.SolverBoxFDDP(problem)  
                
    solver.problem.nthreads = nthreads
    solver.th_stop = th_stop        

    xs = [x0] * (solver.problem.T + 1)
    us = [np.zeros(1)] * (solver.problem.T)

    solver.solve(xs, us, max_iter, False)

    # pos_final = solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId("Link_EE")].translation.T
    # vel_final = pinocchio.getFrameVelocity(solver.problem.terminalModel.differential.state.pinocchio, 
    #                                         solver.problem.terminalData.differential.multibody.pinocchio, 
    #                                         robot_model.getFrameId("Link_EE")).linear

    # print('[INFO]: Reached Pos: {}\tReached Vel: {}'.format(np.round(pos_final, 3), np.round(vel_final, 3)))
    # print('[INFO]: Desired Pos: {}\tDesired Vel: {}'.format(np.round(target_pos_ddp, 3), np.round(target_vel_ddp, 3)))

    u = solver.us.tolist()
    return u