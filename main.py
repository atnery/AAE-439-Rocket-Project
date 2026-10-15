import constants as c
import numpy as np
from motor_sim import run_regression_simulation

# Rocket parameters
d_rocket = 4 * c.IN2M # m
m_inert_motor = 0.5 * c.LB2KG # kg
m_rocket = 1.95 # kg
m_inert_total = m_inert_motor + m_rocket # kg
rocket_params = [d_rocket, m_inert_total]

# Motor parameters
id_case = 1.212 # in
L_case = 4.6 # in
d_throat = 0.25 # in
d_exit = 0.630 # in
motor_params = [id_case, L_case, d_throat, d_exit]

# Propellant parameters
rho_p = 0.06068 # * c.LBM2KG / c.IN32M3 # kg/m^3
c_star = 5088.0 * 12 #* c.FT2M # m/s
a = 0.0318
n = 0.28
g0 = 386.088 # in/s^2
prop_params = [rho_p, c_star, a, n]

# Launch conditions
launch_angle = 85.0 # deg
rail_length = 2.0 # m
wind_speed = 5.0 # m/s
launch_params = [launch_angle, rail_length, wind_speed]

time_arr, p_arr, m_prop_arr = run_regression_simulation(motor_params, prop_params)
