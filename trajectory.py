import constants as c
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rocket_2dof(rocket_params, motor_params, prop_params, launch_params):            
    d_rocket, m_inert = rocket_params
    id_case, L_case, d_throat, d_exit = motor_params
    rho_p, c_star, a, n = prop_params
    launch_angle, rail_length, wind_speed = launch_params

    Cd = 0.3            
    air_density = 1.225  
    s = 0.25 * np.pi * d_rocket**2         

    theta_rail = np.radians(launch_angle)
    e_rail = np.array([np.cos(theta_rail), np.sin(theta_rail)])  # rail direction

    # State: [x (downrange), z (altitude, up), vx, vz, m]
    y0 = [0.0, 0.0, 0.0, 0.0, 1.0]

    def mass(t):
        return m

    def thrust(t):
        return T

    def dynamics(t, y, T, mdot, on_rail):
        x, z, vx, vz = y
        m = mass(t)
        T = thrust(t)

        if on_rail:
            v_rail = vx * e_rail[0] + vz * e_rail[1]
            v_air = v_rail - wind_speed * e_rail[0]          
            D = 0.5 * Cd * air_density * s * v_air * abs(v_air)
            F = T - D - m * c.g * e_rail[1]
            a = F / m
            if v_rail <= 0 and a < 0: # rail holds it until thrust > weight
                a = 0.0
            return [v_rail * e_rail[0], v_rail * e_rail[1],
                    a * e_rail[0], a * e_rail[1], mdot]

        vax, vaz = vx - wind_speed, vz
        Va = np.hypot(vax, vaz)
        ux, uz = (vax / Va, vaz / Va) if Va > 1e-9 else e_rail

        D = 0.5 * Cd * air_density * s * Va**2
        ax = (T - D) * ux / m
        az = (T - D) * uz / m - c.g

        return [vx, vz, ax, az, mdot]

    def rail_exit(t, y, T, mdot, on_rail):
        return y[0] * e_rail[0] + y[1] * e_rail[1] - rail_length
    rail_exit.terminal = True
    rail_exit.direction = 1

    def hit_ground(t, y, T, mdot, on_rail):
        return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    rail = solve_ivp(dynamics, [0, t_burn], y0, args=(Tmax, mdot_burn, True),
                    max_step=0.001, events=rail_exit)
    if rail.status != 1:
        raise RuntimeError("Rocket did not leave the rail before burnout")
    t_exit = rail.t_events[0][0]
    v_exit = np.hypot(*rail.y_events[0][0][2:4])

    burn = solve_ivp(dynamics, [t_exit, t_burn], rail.y_events[0][0],
                    args=(Tmax, mdot_burn, False), max_step=0.01)

    coast = solve_ivp(dynamics, [t_burn, t_burn + 300], burn.y[:, -1],
                    args=(0.0, 0.0, False), max_step=0.05, events=hit_ground)

    time = np.concatenate([rail.t, burn.t[1:], coast.t[1:]])
    x, z, vx, vz, m = np.hstack([rail.y, burn.y[:, 1:], coast.y[:, 1:]])
    speed = np.hypot(vx, vz)
