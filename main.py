import csv
from pathlib import Path

import constants as c
import numpy as np
import matplotlib.pyplot as plt
from motor_sim import run_regression_simulation
from performance import calc_cf
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
wind_speed = -6 # m/s
launch_params = [launch_angle, rail_length, wind_speed]

# Run motor simulation
time_arr, p_arr, m_prop_arr, thrust_arr = run_regression_simulation(
    motor_params, prop_params, plot=False, export=False
)
burn_data = [time_arr, p_arr, m_prop_arr]

TWR_LO = thrust_arr[0] / (c.g * (m_inert_total + m_prop_arr[0]))

# Run 2dof
t, x, z, vx, vz, ax, az = rocket_2dof(rocket_params, motor_params, prop_params, launch_params, burn_data)

# Flight computer data uses SI units for height and pressure.
flight_csv = Path(__file__).with_name("2026-11-08-serial-15508-flight-0003.csv")
with flight_csv.open(newline="", encoding="utf-8") as file:
    flight_data = list(csv.DictReader(file, skipinitialspace=True))
flight_time = np.array([float(row["time"]) for row in flight_data])
flight_height = np.array([float(row["height"]) for row in flight_data])
first_nonzero_height = np.flatnonzero(flight_height != 0.0)[0]
flight_time_height = flight_time[first_nonzero_height:] - flight_time[first_nonzero_height]
flight_height = flight_height[first_nonzero_height:]
flight_pc_pa = np.array([float(row["motor_pressure"]) for row in flight_data])
flight_pc_psi = flight_pc_pa * c.PA2PSI
nonzero_pc = np.flatnonzero(flight_pc_pa > 0.0)
first_nonzero_pc, last_nonzero_pc = nonzero_pc[[0, -1]]
flight_time_pc = (
    flight_time[first_nonzero_pc:last_nonzero_pc + 1] - flight_time[first_nonzero_pc]
)
flight_pc_psi = flight_pc_psi[first_nonzero_pc:last_nonzero_pc + 1]

# Estimate flight thrust from measured chamber pressure and the nozzle geometry.
throat_area = 0.25 * np.pi * d_throat**2
exit_area = 0.25 * np.pi * d_exit**2
ambient_pressure = 101325.0
flight_pc_for_thrust = np.maximum(flight_pc_pa, 0.0)
flight_thrust_arr = np.array([
    calc_cf(pc, ambient_pressure, throat_area, exit_area, gamma) * pc * throat_area
    for pc in flight_pc_for_thrust
])
flight_thrust_arr = flight_thrust_arr[first_nonzero_pc:last_nonzero_pc + 1]

speed = np.hypot(vx, vz) # m/s
accel = np.hypot(ax, az) # m/s^2

i_apo = np.argmax(z)
i_v = np.argmax(speed)
i_a = np.argmax(accel)
actual_apogee = np.max(flight_height)
apogee_delta = actual_apogee - z[i_apo]
apogee_delta_pct = 100.0 * apogee_delta / z[i_apo]

print("\n===================== Flight Data ====================")
print(f"Predicted apogee: {z[i_apo]:.1f} m ({z[i_apo] * c.M2FT:.0f} ft) at t = {t[i_apo]:.2f} s")
print(f"Actual apogee:    {actual_apogee:.1f} m ({actual_apogee * c.M2FT:.0f} ft)")
print(
    f"Apogee delta:     {apogee_delta:+.1f} m ({apogee_delta * c.M2FT:+.0f} ft), "
    f"{apogee_delta_pct:+.1f}% of prediction (actual - predicted)"
)
print(f"TWR at liftoff:   {TWR_LO:.2f}")
print(f"Max speed:        {speed[i_v]:.1f} m/s at t = {t[i_v]:.2f} s")
print(f"Max acceleration: {accel[i_a]:.1f} m/s^2 ({accel[i_a] / c.g:.1f} g) at t = {t[i_a]:.2f} s")
print("\n")

# Plots
t_burn = time_arr[-1]
line = dict(color="#2a6fdb", lw=2)

fig, axs = plt.subplots(3, 2, figsize=(12, 11))
ax_traj, ax_alt, ax_spd, ax_acc, ax_pc, ax_thrust = axs.flat

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

# Chamber pressure
p_psi = p_arr * c.PA2PSI
ax_pc.plot(time_arr, p_psi, **line)
ax_pc.set_xlabel("Time [s]")
ax_pc.set_ylabel("Chamber pressure [psi]")
ax_pc.set_title(f"Chamber Pressure (max {np.max(p_psi):.1f} psi)")

# Sea-level thrust
thrust_lbf = thrust_arr / c.LBF2N
ax_thrust.plot(time_arr, thrust_lbf, **line)
ax_thrust.set_xlabel("Time [s]")
ax_thrust.set_ylabel("Sea-Level Thrust [lbf]")
ax_thrust.set_title(f"Sea-Level Thrust (max {np.max(thrust_lbf):.1f} lbf)")

for a in (ax_alt, ax_spd, ax_acc, ax_pc, ax_thrust):
    a.axvline(t_burn, color="gray", ls="--", lw=1)
    a.text(t_burn, 0.97, " burnout", transform=a.get_xaxis_transform(),
           va="top", color="gray", fontsize=9)
for a in axs.flat:
    a.grid(alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()

# Compare the simulated trajectory and chamber pressure with flight data.
fig_compare, (ax_height_compare, ax_pc_compare, ax_thrust_compare) = plt.subplots(
    3, 1, figsize=(10, 11)
)
ax_height_compare.plot(t, z * c.M2FT, color="#2a6fdb", lw=2, label="Simulation")
ax_height_compare.plot(
    flight_time_height, flight_height * c.M2FT, color="#e07a24", lw=1.5,
    label="Flight data"
)
ax_height_compare.set_xlim(0.0, flight_time_height[-1])
ax_height_compare.set_xlabel("Time [s]")
ax_height_compare.set_ylabel("Altitude [ft]")
ax_height_compare.set_title("Altitude vs. Flight Time")
ax_height_compare.legend()

ax_pc_compare.plot(
    time_arr, p_arr * c.PA2PSI, color="#2a6fdb", lw=2, label="Simulation"
)
ax_pc_compare.plot(
    flight_time_pc, flight_pc_psi, color="#e07a24", lw=1.5, label="Flight data"
)
ax_pc_compare.set_xlim(0.0, flight_time_pc[-1])
ax_pc_compare.set_xlabel("Time [s]")
ax_pc_compare.set_ylabel("Chamber pressure, Pc [psi]")
ax_pc_compare.set_title("Chamber Pressure vs. Flight Time")
ax_pc_compare.legend()

ax_thrust_compare.plot(
    time_arr, thrust_arr / c.LBF2N, color="#2a6fdb", lw=2, label="Simulation"
)
ax_thrust_compare.plot(
    flight_time_pc, flight_thrust_arr / c.LBF2N, color="#e07a24", lw=1.5,
    label="Flight data (Pc-derived)"
)
ax_thrust_compare.set_xlim(0.0, flight_time_pc[-1])
ax_thrust_compare.set_xlabel("Time [s]")
ax_thrust_compare.set_ylabel("Thrust [lbf]")
ax_thrust_compare.set_title("Thrust from Measured Chamber Pressure")
ax_thrust_compare.legend()

for a in (ax_height_compare, ax_pc_compare, ax_thrust_compare):
    a.axvline(t_burn, color="gray", ls="--", lw=1)
    a.grid(alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)

fig_compare.tight_layout()
plt.show()