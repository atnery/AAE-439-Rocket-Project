import constants as c
import numpy as np
from motor_sim import run_regression_simulation
from trajectory import rocket_2dof

# Rocket parameters
d_rocket = 4 * c.IN2M # m
m_inert_motor = 0.5 * c.LBM2KG # kg
m_rocket = 1.95 # kg
m_inert_total = m_inert_motor + m_rocket # kg
rocket_params = [d_rocket, m_inert_total]

# Motor parameters
id_case = 1.212 * c.IN2M # m
L_case = 4.6 * c.IN2M # m
d_throat = 0.25 * c.IN2M # m
d_exit = 0.630 * c.IN2M # m
motor_params = [id_case, L_case, d_throat, d_exit]

# Propellant parameters
rho_p = 0.06068 * c.LBM2KG / c.IN32M3 # kg/m^3
c_star = 5088.0 * c.FT2M # m/s
gamma = 1.3 
n = 0.28
a = 0.0318 * c.IN2M / c.PSI2PA**n
prop_params = [c_star, gamma, rho_p, a, n]

# Launch conditions
launch_angle = 85 # deg
rail_length = 2.0 # m
wind_speed = -5 # m/s
launch_params = [launch_angle, rail_length, wind_speed]

time_arr, p_arr, m_prop_arr = run_regression_simulation(motor_params, prop_params, plot=False, export=False)

burn_data = [time_arr, p_arr, m_prop_arr]

t, x, z, vx, vz = rocket_2dof(rocket_params, motor_params, prop_params, launch_params, burn_data)

speed = np.hypot(vx, vz)
accel = np.hypot(np.gradient(vx, t), np.gradient(vz, t))

i_apo = np.argmax(z)
i_v = np.argmax(speed)
i_a = np.argmax(accel)

print(f"Apogee:           {z[i_apo]:.1f} m ({z[i_apo] * c.M2FT:.0f} ft) at t = {t[i_apo]:.2f} s")
print(f"Max speed:        {speed[i_v]:.1f} m/s at t = {t[i_v]:.2f} s")
print(f"Max acceleration: {accel[i_a]:.1f} m/s^2 ({accel[i_a] / c.g:.1f} g) at t = {t[i_a]:.2f} s")
print(f"Downrange:        {x[-1]:.1f} m at landing (t = {t[-1]:.2f} s)")