import constants as c
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from performance import calc_cf

def atmosphere(h):
    P0 = 101325.0 # Pa
    T0 = 288.15 # K
    R = 287 # J/(kg*K)
    L = 0.0065 # K/m

    T = T0 - L * h
    P = P0 * (T / T0) ** (c.g / (L * R))
    rho = P / (R * T)   

    return P, rho

def rocket_2dof(rocket_params, motor_params, prop_params, launch_params, burn_data):            
    d_rocket, m_inert = rocket_params
    id_case, L_case, d_throat, d_exit = motor_params
    c_star, gamma, rho_p, a, n = prop_params
    launch_angle, rail_length, wind_speed = launch_params
    time_arr, Pc_arr, m_prop_arr = burn_data

    Cd = 0.3            
    s = 0.25 * np.pi * d_rocket**2
    At = 0.25 * np.pi * d_throat**2
    Ae = 0.25 * np.pi * d_exit**2

    theta_rail = np.radians(launch_angle)
    e_rail = np.array([np.cos(theta_rail), np.sin(theta_rail)])  # rail direction

    y0 = [0.0, 0.0, 0.0, 0.0] # Initial conditions

    def get_mass(t):
        return m_inert + np.interp(t, time_arr, m_prop_arr)

    def get_Pc(t):
        return np.interp(t, time_arr, Pc_arr)
    
    def dynamics(t, y):
        x, z, vx, vz = y

        Pa, rho = atmosphere(z)
        m = get_mass(t) if t <= time_arr[-1] else m_inert # Vehicle mass
        Pc = get_Pc(t) if t <= time_arr[-1] else 0.0 # Chamber pressure
        cf = calc_cf(Pc, Pa, At, Ae, gamma) # Thrust coefficient
        T = cf * Pc * At # Thrust

        dist = np.hypot(x, z)

        if dist < rail_length:
            v_rail = vx * e_rail[0] + vz * e_rail[1]
            v_air = v_rail - wind_speed * e_rail[0]          
            D = 0.5 * Cd * rho * s * v_air * abs(v_air)
            F = T - D - m * c.g * e_rail[1]
            a = F / m
            return [v_rail * e_rail[0], v_rail * e_rail[1],
                    a * e_rail[0], a * e_rail[1]]

        vax, vaz = vx - wind_speed, vz
        Va = np.hypot(vax, vaz)
        ux, uz = (vax / Va, vaz / Va) if Va > 1e-9 else e_rail

        D = 0.5 * Cd * rho * s * Va**2
        ax = (T - D) * ux / m
        az = (T - D) * uz / m - c.g

        return [vx, vz, ax, az]

    t_burn = time_arr[-1]

    def rail_exit(t, y):
        return np.hypot(y[0], y[1]) - rail_length
    rail_exit.terminal = True
    rail_exit.direction = 1

    def hit_ground(t, y):
        return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    dt = np.min(np.diff(time_arr))

    # On the rail
    rail = solve_ivp(dynamics, [0, t_burn], y0, events=rail_exit, max_step=dt)
    if rail.status != 1:
        raise RuntimeError("Rocket did not leave the rail before burnout")
    t_exit, y_exit = rail.t_events[0][0], rail.y_events[0][0]

    # Burn phase
    burn = solve_ivp(dynamics, [t_exit, t_burn], y_exit, max_step=dt)

    # Coast phase
    coast = solve_ivp(dynamics, [t_burn, t_burn + 600], burn.y[:, -1], events=hit_ground, max_step=dt)

    t = np.concatenate([rail.t, burn.t[1:], coast.t[1:]])
    x, z, vx, vz = np.hstack([rail.y, burn.y[:, 1:], coast.y[:, 1:]])
    return t, x, z, vx, vz
