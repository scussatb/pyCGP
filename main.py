import os
# os.environ['JSBSIM_ROOT_DIR'] = '/home/cussat/Documents/Airbus_CGP/JSBSim/jsbsim-JSBSim-trusty-v2018a'

import sys
from cgpes import CGPES
from cgp import CGP
from cgpfunctions import *
import numpy as np
from evaluator import Evaluator
import gym
import gym_jsbsim
from gym_jsbsim.catalogs.catalog import Catalog as prp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

class SinEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, cgp, it):
        fit = 0.0
        x = 0.0
        while x < 2 * np.pi:
            fit += abs(np.sin(x) - cgp.run([x])[0])
            x += 0.1
        return -fit

    def clone(self):
        return SinEvaluator()


class GymEvaluator(Evaluator):
    def __init__(self, env_name, it_max, ep_max):
        super().__init__()
        self.env_name = env_name
        self.it_max = it_max
        self.ep_max = ep_max
        self.env = gym.make(self.env_name)

    def evaluate(self, cgp, it, with_render=False, display_in_out=False, plot=False):
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(221, projection='3d')
            ay = fig.add_subplot(222)
            az = fig.add_subplot(224)
            traceTraj = np.zeros((100000, 3))
            traceState = np.zeros((100000, len(self.env.observation_space.sample())))
            traceAction = np.zeros((100000, len(self.env.action_space.sample())))
        if with_render:
            traj_string = []
            attitude_string = [] 
        fitnesses = np.zeros(self.ep_max)
        # print(cgp.genome)
        for e in range(self.ep_max):
            end = False
            fit = 0
            states = self.env.reset()
            norm_states = []
            step = 0
            while not end:
                #start_time = time.time()
                #print(self.env.sim.get_property_value(prp.simulation_sim_time_sec), states)
                for s in range(len(states)):
                    st = min(self.env.observation_space[s].high, max(self.env.observation_space[s].low, states[s]))
                    norm_states += [2.0 * (st - self.env.observation_space[s].low) / (self.env.observation_space[s].high - self.env.observation_space[s].low) - 1.0]
                # states = 2.0 * (states - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1.0
                #print(self.env.sim.get_property_value(prp.simulation_sim_time_sec), 'normalized: ', states)
                if plot:
                    traceState[step] = norm_states
                actions = cgp.run(norm_states)
                for a in range(len(actions)):
                    # outputs = outputs * 0.25
                    actions[a] = np.minimum(np.maximum(actions[a], self.env.action_space[a].low),
                                            self.env.action_space[a].high)
                if plot:
                    traceAction[step] = actions
                if display_in_out:
                    print(' ', states, ' => ', actions)

                
                states, reward, end, _ = self.env.step(actions)
                #print("--- main.py::self.env.step(actions) ---", (time.time() - start_time))

                
                

                lon = self.env.sim.get_property_value(prp.position_long_gc_deg)
                lat = self.env.sim.get_property_value(prp.position_lat_geod_deg)
                alt = self.env.sim.get_property_value(prp.position_h_sl_ft)
                psi = self.env.sim.get_property_value(prp.attitude_psi_rad)
                theta = self.env.sim.get_property_value(prp.attitude_theta_rad)
                phi = self.env.sim.get_property_value(prp.attitude_phi_rad)

                sim_time = self.env.sim.get_property_value(prp.simulation_sim_time_sec)

                #print(sim_time, self.env.sim.get_property_value(prp.target_altitude_ft))

                if with_render:
                    # self.env.render()
                    traj_string += [sim_time, lon, lat, alt]
                    attitude_string += [sim_time, psi, theta, phi]
                    #print(sim_time, ',', lon, ',', lat, ',', alt, ',')
                if plot:
                    traceTraj[step] = [lon, lat, alt]
                    if step % 1000 == 0:
                        ax.scatter3D(lon, lat, alt)
                        ay.scatter(step, 0)
                        az.scatter(step, 0)
                # fit += reward * reward
                #if abs(states[0]) < 0.05:
                #    fit += 0.1 * (1.0 - abs(states[0]))
                #if abs(states[1]) < 0.2:
                #    fit += 1.0 - abs(states[1])
                fit += reward
                # fit += 1
                step += 1
            fitnesses[e] = fit
        np.sort(fitnesses)
        fit = 0
        sum_e = 0
        for e in range(self.ep_max):
            fit += fitnesses[e] * (e + 1)
            sum_e += e + 1
        # print(fitnesses)
        # print(fit)
        # print('-----------')

        if plot:
            ay.plot(range(0, step), traceState[0:step, 0], label='delta_alt')
            ay.plot(range(0, step), traceState[0:step, 1], label='delat_head')
            ay.legend()
            az.plot(range(0, step), traceAction[0:step, 0], label='aileron')
            az.plot(range(0, step), traceAction[0:step, 1], label='elevator')
            az.plot(range(0, step), traceAction[0:step, 2], label='rudder')
            az.plot(range(0, step), traceAction[0:step, 3], label='throttle')
            az.legend()

            ax.plot3D(traceTraj[0:step, 0], traceTraj[0:step, 1], traceTraj[0:step, 2], 'black')
            plt.show()
            plt.savefig('outputCGP.png')
        
        if with_render:
            print("\n==========\nTRAJECTOIRE\n==========\n")
            for t in range(0, len(traj_string)-5, +4):
                print(str(traj_string[t]) + "," + str(traj_string[t+1]) + "," + str(traj_string[t+2]) + "," + str(traj_string[t+3]) + ",")
            print("\n==========\nATTITUDE\n==========\n")
            for t in range(0, len(attitude_string)-5, +4):
                print(str(attitude_string[t]) + "," + str(attitude_string[t+1]) + "," + str(attitude_string[t+2]) + "," + str(attitude_string[t+3]) + ",")
            print("target heading", self.env.sim.get_property_value(prp.target_heading_deg)
)
        return fit / sum_e

    def clone(self):
        return GymEvaluator(self.env_name, self.it_max, self.ep_max)


def build_funcLib():
    return [CGP.CGPFunc(f_sum, 'sum', 2),
            CGP.CGPFunc(f_aminus, 'aminus', 2),
            CGP.CGPFunc(f_mult, 'mult', 2),
            CGP.CGPFunc(f_exp, 'exp', 2),
            CGP.CGPFunc(f_abs, 'abs', 1),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1),
            CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2),
            CGP.CGPFunc(f_squared, 'squared', 1),
            CGP.CGPFunc(f_pow, 'pow', 2),
            CGP.CGPFunc(f_one, 'one', 0),
            CGP.CGPFunc(f_zero, 'zero', 0),
            CGP.CGPFunc(f_inv, 'inv', 1),
            CGP.CGPFunc(f_gt, 'gt', 2),
            CGP.CGPFunc(f_asin, 'asin', 1),
            CGP.CGPFunc(f_acos, 'acos', 1),
            CGP.CGPFunc(f_atan, 'atan', 1),
            CGP.CGPFunc(f_min, 'min', 2),
            CGP.CGPFunc(f_max, 'max', 2),
            CGP.CGPFunc(f_round, 'round', 1),
            CGP.CGPFunc(f_floor, 'floor', 1),
            CGP.CGPFunc(f_ceil, 'ceil', 1)
            ]


def evolveGym(env, library, folder_name, col=100, row=1, nb_ind=4, mutation_rate_nodes=0.1, mutation_rate_outputs=0.3,
              n_cpus=1, n_it=1000000, genome=None):
    e = GymEvaluator(env, 10000, 1)
    # cgpFather = CGP.random(len(e.env.observation_space.sample()), len(e.env.action_space.sample()), col, row, library)
    #print(e.env.observation_space.sample())
    #print(e.env.action_space.sample())
    if genome is None:
        cgpFather = CGP.random(len(e.env.observation_space.sample()), len(e.env.action_space.sample()), col, row, library,
                               1.0)
    else:
        cgpFather = CGP.load_from_file(genome, library)
    print(cgpFather.genome)
    es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, folder_name, n_cpus)
    es.run(n_it)


def evo(gym_env, folder_name, genome=None):
    library = build_funcLib()
    # evolveGym('JSBSim-ChangeHeadingControlTask-A320-NoFG-v0', library)
    evolveGym(gym_env, library, folder_name, genome=genome)

def load(gym_env, file_name):
    print('loading ' + file_name)
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    e = GymEvaluator(gym_env, 10000, 1)
    # e = GymEvaluator('JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0', 10000, 5)
    print(e.evaluate(c, 0, True, False, False))
   
def toDot(file_name, out_name):
    print('Exporting ' + file_name + ' in dot ' + out_name + '.dot')
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    #c.to_dot(out_name+'.dot',
    #             ['delta_altitude', 'delta_heading', 'vDown', 'vc', 'p', 'q', 'r'],
    #             ['aileron', 'elevator', 'rudder', 'throttle'])
    c.to_dot(out_name+'.dot', ['velocities_vc_fps', 'shortest_dist', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], ['fcs_steer_cmd_norm', 'fcs_left_brake_cmd_norm', 'fcs_right_brake_cmd_norm', 'fcs_throttle_cmd_norm'])
    print('Converting dot file into pdf in ' + out_name + '.pdf')
    os.system('dot -Tpdf ' + out_name + '.dot' + ' -o ' + out_name + '.pdf')

def displayFunctions(file_name):
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    #c.to_function_string(['delta_altitude', 'delta_heading', 'vDown', 'vc', 'p', 'q', 'r'],
    #             ['aileron', 'elevator', 'rudder', 'throttle'])
    c.to_function_string(['velocities_vc_fps', 'shortest_dist', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], ['fcs_steer_cmd_norm', 'fcs_left_brake_cmd_norm', 'fcs_right_brake_cmd_norm', 'fcs_throttle_cmd_norm'])
    


if __name__ == '__main__':
    if len(sys.argv)==3:
        print('Starting evolution from random genome')
        evo(sys.argv[1], sys.argv[2])
    elif len(sys.argv)==4:
        print('Starting evolution from genome saved in ', sys.argv[3])
        evo(sys.argv[1], sys.argv[2], genome=sys.argv[3])