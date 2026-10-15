import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def trajectory(t, states, parameters):
    g = 9.81
    
    # Unpack initial values and parameters
    x, z, theta, u, w, q, m = states
    Tmax, Cd0, CLa, air_density, mdot_i, r, l, kD, da, dm, tvc_angle = parameters

    # Auxiliary equations
    if t < 5:
        T = Tmax
        mdot = mdot_i
    else:
        T = 0
        mdot = 0

    alpha = np.atan2(w, u)
    Cl = CLa * alpha
    Cd = Cd0 + kD * Cl**2 
    V = np.sqrt(u**2 + w**2)
    s = np.pi * r**2
    D = 0.5 * Cd * air_density * s * V**2
    L = 0.5 * Cl * air_density * s * V**2
    J = (m * l**2) / 12
    Jdot = (mdot * l**2) / 12

    # Equations of motion
    f = [
        u*np.cos(theta) + w*np.sin(theta),
        -u*np.sin(theta) + w*np.cos(theta),
        q,
        ((T*np.cos(tvc_angle) - D*np.cos(alpha) + L*np.sin(alpha) - m*g*np.sin(theta)) / m) - q*w,
        ((T*np.sin(tvc_angle) - D*np.sin(alpha) - L*np.cos(alpha) + m*g*np.cos(theta)) / m) + q*u,
        (J**(-1)) * ((T*dm*np.sin(tvc_angle) - D*da*np.sin(alpha) - L*da*np.cos(alpha)) - Jdot*q),
        mdot,   
    ]

    return f

# Parameters
Tmax = 100
Cd0 = 0.01
CLa = 2*np.pi
air_density = 1.225
mdot_i = -0.1
r = 0.1
l = 1
kD = 0.1
da = 0.25
dm = 0.5

# Initial conditions
xi = 0
zi = 0
theta_i = np.pi / 2
ui = 0
wi = 0
qi = 0
mi = 1
tvc_angle_i = 0 

# Pack everything into arrays
initial_values = np.array([xi, zi, theta_i, ui, wi, qi, mi])
parameters = np.array([Tmax, Cd0, CLa, air_density, mdot_i, r, l, kD, da, dm, tvc_angle_i])

# Setup simulation time
t0 = 0
tf = 10
t = [t0, tf]
t_eval = np.linspace(t0, tf, 1000)

# Solve diff eqs
sol = solve_ivp(trajectory, t, initial_values, args=(parameters,), method='RK45', t_eval=t_eval)

# Unpack solution
time = sol.t
x = sol.y[0,:]
z = sol.y[1,:]
theta = sol.y[2,:]
u = sol.y[3,:]
w = sol.y[4,:]
q = sol.y[5,:]
m = sol.y[6,:]