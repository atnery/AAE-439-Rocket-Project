import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.81             
Tmax = 100.0         
t_burn = 5.0         
mdot_burn = -0.1     
Cd = 0.3            
air_density = 1.225  
r = 4 * .0254
s = np.pi * r**2

launch_angle_deg = 85.0  
rail_length = 2.0        
wind_speed = 5.0         

theta_rail = np.radians(launch_angle_deg)
e_rail = np.array([np.cos(theta_rail), np.sin(theta_rail)])  # rail direction

# State: [x (downrange), z (altitude, up), vx, vz, m]
y0 = [0.0, 0.0, 0.0, 0.0, 1.0]

def dynamics(t, y, T, mdot, on_rail):
    x, z, vx, vz, m = y

    if on_rail:
        v_along = vx * e_rail[0] + vz * e_rail[1]
        v_air = v_along - wind_speed * e_rail[0]          
        D = 0.5 * Cd * air_density * s * v_air * abs(v_air)
        F = T - D - m * g * e_rail[1]
        a = F / m
        if v_along <= 0 and a < 0: # rail holds it until thrust > weight
            a = 0.0
        return [v_along * e_rail[0], v_along * e_rail[1],
                a * e_rail[0], a * e_rail[1], mdot]

    vax, vaz = vx - wind_speed, vz
    Va = np.hypot(vax, vaz)
    ux, uz = (vax / Va, vaz / Va) if Va > 1e-9 else e_rail

    D = 0.5 * Cd * air_density * s * Va**2
    ax = (T - D) * ux / m
    az = (T - D) * uz / m - g
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

i_apo = np.argmax(z)
print(f"Rail exit: t = {t_exit:.3f} s, v = {v_exit:.1f} m/s")
print(f"Burnout:   t = {t_burn:.2f} s, alt = {burn.y[1, -1]:.1f} m, v = {np.hypot(*burn.y[2:4, -1]):.1f} m/s")
print(f"Apogee:    t = {time[i_apo]:.2f} s, alt = {z[i_apo]:.1f} m, downrange = {x[i_apo]:.1f} m")
print(f"Max speed: {speed.max():.1f} m/s")
if coast.t_events[0].size:
    yl = coast.y_events[0][0]
    print(f"Landing:   t = {coast.t_events[0][0]:.2f} s, downrange = {yl[0]:.1f} m, "
          f"speed = {np.hypot(yl[2], yl[3]):.1f} m/s")

fig = plt.figure(figsize=(10, 7))
ax_traj = fig.add_subplot(1, 2, 1)
ax_traj.plot(x, z)
ax_traj.plot(x[i_apo], z[i_apo], "o", label="Apogee")
ax_traj.set_xlabel("Downrange [m]"); ax_traj.set_ylabel("Altitude [m]")
ax_traj.set_title(f"Launch {launch_angle_deg:.0f}°, wind {wind_speed:+.1f} m/s")
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.legend(); ax_traj.grid(alpha=0.3)

ax_alt = fig.add_subplot(2, 2, 2)
ax_alt.plot(time, z); ax_alt.set_ylabel("Altitude [m]")
ax_spd = fig.add_subplot(2, 2, 4, sharex=ax_alt)
ax_spd.plot(time, speed); ax_spd.set_ylabel("Speed [m/s]"); ax_spd.set_xlabel("Time [s]")
for a in (ax_alt, ax_spd):
    a.axvline(t_burn, ls="--", c="gray", lw=0.8)
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()