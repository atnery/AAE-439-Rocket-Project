import constants as c
import numpy as np
import matplotlib.pyplot as plt
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
wind_speed = -2 # m/s
launch_params = [launch_angle, rail_length, wind_speed]

# Run motor simulation
time_arr, p_arr, m_prop_arr = run_regression_simulation(motor_params, prop_params, plot=False, export=False)
burn_data = [time_arr, p_arr, m_prop_arr]

# Run 2dof
t, x, z, vx, vz, ax, az = rocket_2dof(rocket_params, motor_params, prop_params, launch_params, burn_data)

speed = np.hypot(vx, vz)
accel = np.hypot(ax, az)

i_apo = np.argmax(z)
i_v = np.argmax(speed)
i_a = np.argmax(accel)

print("\n===================== Flight Data ====================")
print(f"Apogee:           {z[i_apo]:.1f} m ({z[i_apo] * c.M2FT:.0f} ft) at t = {t[i_apo]:.2f} s")
print(f"Max speed:        {speed[i_v]:.1f} m/s at t = {t[i_v]:.2f} s")
print(f"Max acceleration: {accel[i_a]:.1f} m/s^2 ({accel[i_a] / c.g:.1f} g) at t = {t[i_a]:.2f} s")
print("\n")

# Plots
t_burn = time_arr[-1]
line = dict(color="#2a6fdb", lw=2)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
ax_traj, ax_alt, ax_spd, ax_acc = axs.flat

# Trajectory
ax_traj.plot(x, z, **line)
ax_traj.plot(x[i_apo], z[i_apo], "o", color="#2a6fdb", ms=8)
ax_traj.annotate(f"Apogee {z[i_apo]:.0f} m", (x[i_apo], z[i_apo]),
                 textcoords="offset points", xytext=(8, -14))
ax_traj.set_xlabel("Downrange [m]")
ax_traj.set_ylabel("Altitude [m]")
ax_traj.set_title("Trajectory")
ax_traj.set_aspect("equal", adjustable="datalim")

# Altitude
ax_alt.plot(t, z, **line)
ax_alt.plot(t[i_apo], z[i_apo], "o", color="#2a6fdb", ms=8)
ax_alt.set_xlabel("Time [s]")
ax_alt.set_ylabel("Altitude [m]")
ax_alt.set_title(f"Altitude (apogee {z[i_apo]:.0f} m at {t[i_apo]:.1f} s)")

# Speed
ax_spd.plot(t, speed, **line)
ax_spd.set_xlabel("Time [s]")
ax_spd.set_ylabel("Speed [m/s]")
ax_spd.set_title(f"Speed (max {speed[i_v]:.1f} m/s)")

# Acceleration
ax_acc.plot(t, accel / c.g, **line)
ax_acc.set_xlabel("Time [s]")
ax_acc.set_ylabel("Acceleration [g]")
ax_acc.set_title(f"Acceleration (max {accel[i_a] / c.g:.1f} g)")

for a in (ax_alt, ax_spd, ax_acc):
    a.axvline(t_burn, color="gray", ls="--", lw=1)
    a.text(t_burn, 0.97, " burnout", transform=a.get_xaxis_transform(),
           va="top", color="gray", fontsize=9)
for a in axs.flat:
    a.grid(alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()