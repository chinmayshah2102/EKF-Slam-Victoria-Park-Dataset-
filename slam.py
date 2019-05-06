from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
import pdb


def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''
    motion = np.zeros([3,])
    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']
    phi = ekf_state['x'][2]

    vc = u[0]/(1-(np.tan(u[1])*H/L))
    motion[0] = dt*vc*(np.cos(phi) - (1/L)*np.tan(u[1])*(a*np.sin(phi)+b*np.cos(phi)))
    motion[1] = dt*vc*(np.sin(phi) + (1/L)*np.tan(u[1])*(a*np.cos(phi)-b*np.sin(phi)))
    motion[2] = dt*((vc/L)*np.tan(u[1]))

    G = np.zeros([3,3])
    G[0,0] = 1
    G[1,0] = 0
    G[2,0] = 0

    G[0,1] = 0
    G[1,1] = 1
    G[2,1] = 0

    G[0,2] = -dt*vc*(np.sin(phi) + (1/L)*np.tan(u[1])*(a*np.cos(phi) - b*np.sin(phi)))
    G[1,2] =  dt*vc*(np.cos(phi) - (1/L)*np.tan(u[1])*(a*np.sin(phi) + b*np.cos(phi)))
    G[2,2] = 1

    return motion, G


def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    xnoise = pow(sigmas['xy'],2)
    ynoise = pow(sigmas['xy'],2)
    phinoise = pow(sigmas['phi'],2)
    Pprev = ekf_state['P'][0:3,0:3]
    Rt = np.diag([xnoise, ynoise, phinoise])

    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    ekf_state['x'][0:3] = ekf_state['x'][0:3] + motion
    ekf_state['P'][0:3,0:3] = np.matmul(np.matmul(G,Pprev),G.T) + Rt
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''

    xnoise = pow(sigmas['gps'],2)
    ynoise = pow(sigmas['gps'],2)
    Qt = np.diag([xnoise,ynoise])
    H = np.zeros([2,3])
    H[0:2,0:2] = np.eye(2)
    Pprev = ekf_state['P'][0:3,0:3]
    innov = gps - ekf_state['x'][0:2]
    r = np.reshape(innov,[2,1])
    S = np.matmul(H,np.matmul(Pprev,H.T)) + Qt
    Mdist = np.matmul(np.matmul(r.T, np.linalg.inv(S)), r)

    if Mdist < 13.8:
        Kt = np.matmul(np.matmul(Pprev,H.T),np.linalg.inv(S))
        ekf_state['x'][0:3] = ekf_state['x'][0:3] + np.matmul(Kt,r)[:,0]
        ekf_state['P'][0:3,0:3] = np.matmul(np.eye(3)-np.matmul(Kt,H),Pprev)
        ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    else:
        ekf_state = ekf_state

    return ekf_state


def laser_measurement_model(ekf_state, landmark_id):
    '''
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''

    xL = ekf_state['x'][3+2*landmark_id]
    yL = ekf_state['x'][4+2*landmark_id]
    xV = ekf_state['x'][0]
    yV = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    m = ekf_state['num_landmarks']

    zhat = np.zeros([2,1])
    norm = np.sqrt(pow(xL-xV, 2) + pow(yL-yV, 2))
    zhat[0,0] = norm
    zhat[1,0] = slam_utils.clamp_angle(np.arctan2((yL-yV),(xL-xV)) - phi)
    H = np.zeros([2,3+2*m])

    if norm != 0:
        H[0,0] = -(xL-xV)/norm
        H[0,1] = -(yL-yV)/norm
        H[0,2] =       0
        H[0,3+2*landmark_id] = (xL-xV)/norm
        H[0,4+2*landmark_id] = (yL-yV)/norm

        H[1,0] = (yL-yV)/pow(norm,2)
        H[1,1] = -(xL-xV)/pow(norm,2)
        H[1,2] =        -1
        H[1,3+2*landmark_id] = -(yL-yV)/pow(norm,2)
        H[1,4+2*landmark_id] = (xL-xV)/pow(norm,2)

    return zhat, H


def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    r = tree[0]
    if r>75:
        print(r)
    theta = tree[1]
    xV = ekf_state['x'][0]
    yV = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    xL = r*np.cos(slam_utils.clamp_angle(theta+phi)) + xV
    yL = r*np.sin(slam_utils.clamp_angle(theta+phi)) + yV
    ekf_state['x'] = np.append(ekf_state['x'],[xL,yL])
    n = ekf_state['P'].shape[0]
    Ptemp = np.zeros([n+2,n+2])
    Ptemp[0:n,0:n] = ekf_state['P']

    Ptemp[n,n] = 0.5
    Ptemp[n+1,n+1] = 0.5

    ekf_state['P'] = Ptemp
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])

    ekf_state['num_landmarks'] += 1
 
    return ekf_state


def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    num_landmarks = ekf_state['num_landmarks']
    n = len(measurements)

    if n > num_landmarks:
        M = np.full([n,n+num_landmarks],6.01)
    else:
        M = np.zeros([n,num_landmarks])

    rnoise = pow(sigmas['range'],2)
    bnoise = pow(sigmas['bearing'],2)
    Qt = np.diag([rnoise, bnoise])

    assoc = [0]*len(measurements)

    for j in range(num_landmarks):
        for i in range(n):
            r, b, d = measurements[i]
            z = np.array([[r],[b]])
            fi, H = laser_measurement_model(ekf_state, j)
            r = z - fi
            S = np.matmul(np.matmul(H, ekf_state['P']),H.T) + Qt
            M[i, j] = np.matmul(np.matmul(r.T, np.linalg.inv(S)), r)

    Mcopy = np.copy(M)
    pairs = slam_utils.solve_cost_matrix_heuristic(Mcopy)

    for i, j in pairs:
        if j >= num_landmarks:
            mind = np.amin(M[i,0:(num_landmarks-1)])
            if mind > 13.8:
                assoc[i] = -1
            else:
                assoc[i] = -2
        else:
            if M[i, j] > 6:
                if M[i, j] > 13.8:
                    assoc[i] = -1
                else:
                    assoc[i] = -2
            else:
                assoc[i] = j

    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    for i,j in enumerate(assoc):
        if j == -2:
            continue
        elif j == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
        else:
            zhat, H = laser_measurement_model(ekf_state, j)
            Pprev = ekf_state['P']
            rnoise = pow(sigmas['range'],2)
            bnoise = pow(sigmas['bearing'],2)
            Qt = np.diag([rnoise,bnoise])
            Kt = np.matmul(np.matmul(Pprev, H.T), np.linalg.inv(np.matmul(H, np.matmul(Pprev, H.T)) + Qt))
            r, b, d = trees[i]
            z = np.array([[r],[b]])
            n = len(ekf_state['x'])
            ekf_state['x'] = np.reshape((np.reshape(ekf_state['x'],(n,1)) + np.matmul(Kt,(z-zhat))),(n,))
            ekf_state['P'] = np.matmul((np.eye(n)-np.matmul(Kt,H)),Pprev)
            ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))
            print(ekf_state['x'].shape)

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": False,
        "plot_map_covariances": False

    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array([gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)


if __name__ == '__main__':
    main()
