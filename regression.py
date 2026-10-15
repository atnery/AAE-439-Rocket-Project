import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle as MplCircle
import ezdxf
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm

def extract_closed_port_from_dxf(dxf_path, sample_pts_arc=32, sample_pts_spline=100):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    curves = []

    for line in msp.query('LINE'):
        p0 = np.array([line.dxf.start.x, line.dxf.start.y])
        p1 = np.array([line.dxf.end.x, line.dxf.end.y])
        curves.append(np.array([p0, p1]))

    for arc in msp.query('ARC'):
        center = np.array([arc.dxf.center.x, arc.dxf.center.y])
        radius = arc.dxf.radius
        start_angle = np.deg2rad(arc.dxf.start_angle)
        end_angle = np.deg2rad(arc.dxf.end_angle)
        if end_angle <= start_angle:
            end_angle += 2.0 * np.pi
        theta = np.linspace(start_angle, end_angle, sample_pts_arc)
        arc_pts = np.column_stack((center[0] + radius * np.cos(theta),
                                   center[1] + radius * np.sin(theta)))
        curves.append(arc_pts)

    for spline in msp.query('SPLINE'):
        try:
            tool = spline.construction_tool()
            pts = [np.array([p.x, p.y]) for p in tool.approximate(segments=sample_pts_spline)]
            curves.append(np.array(pts))
        except Exception:
            ctrl_pts = [np.array([p[0], p[1]]) for p in spline.control_points]
            curves.append(np.array(ctrl_pts))

    if not curves:
        raise ValueError("No valid entities found in DXF.")

    ordered_pts = list(curves.pop(0))
    while curves:
        curr_tip = ordered_pts[-1]
        best_idx, best_dist, must_reverse = None, float('inf'), False
        for i, seg in enumerate(curves):
            d_start = np.linalg.norm(seg[0] - curr_tip)
            d_end = np.linalg.norm(seg[-1] - curr_tip)
            if d_start < best_dist:
                best_dist, best_idx, must_reverse = d_start, i, False
            if d_end < best_dist:
                best_dist, best_idx, must_reverse = d_end, i, True

        seg = curves.pop(best_idx)
        if must_reverse:
            seg = seg[::-1]
        ordered_pts.extend(seg[1:])

    p_poly = Polygon(np.array(ordered_pts))
    return p_poly if p_poly.is_valid else p_poly.buffer(0)

def get_thrust_coefficient(gamma=1.20, expansion_ratio=6.35, pc_psi=500.0, p_amb_psi=14.7):
    if pc_psi <= p_amb_psi:
        return 0.0

    def area_ratio(M):
        t1 = (gamma + 1.0) / 2.0
        t2 = 1.0 + (gamma - 1.0) / 2.0 * M**2
        return (1.0 / M) * ((t2 / t1) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))

    M_exit = 2.0
    for _ in range(25):
        f = area_ratio(M_exit) - expansion_ratio
        df = (area_ratio(M_exit + 1e-5) - area_ratio(M_exit - 1e-5)) / 2e-5
        M_exit -= f / df
        if abs(f) < 1e-5:
            break

    pe_pc = (1.0 + (gamma - 1.0) / 2.0 * M_exit**2) ** (-gamma / (gamma - 1.0))
    pe_psi = pe_pc * pc_psi

    cf_vac = np.sqrt((2.0 * gamma**2 / (gamma - 1.0)) * 
                     (2.0 / (gamma + 1.0))**((gamma + 1.0) / (gamma - 1.0)) * 
                     (1.0 - (pe_pc)**((gamma - 1.0) / gamma)))
    cf = cf_vac + (pe_psi - p_amb_psi) / pc_psi * expansion_ratio
    return max(cf, 0.0)

def shapely_to_mpl_patch(geom, facecolor='darkorange', edgecolor='black', alpha=0.9):
    if geom.is_empty:
        return None

    polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    vertices, codes = [], []

    for poly in polygons:
        ext_coords = np.asarray(poly.exterior.coords)
        if len(ext_coords) < 3:
            continue
        vertices.extend(ext_coords)
        codes.append(Path.MOVETO)
        codes.extend([Path.LINETO] * (len(ext_coords) - 2))
        codes.append(Path.CLOSEPOLY)

        for interior in poly.interiors:
            int_coords = np.asarray(interior.coords)
            if len(int_coords) < 3:
                continue
            vertices.extend(int_coords)
            codes.append(Path.MOVETO)
            codes.extend([Path.LINETO] * (len(int_coords) - 2))
            codes.append(Path.CLOSEPOLY)

    if not vertices:
        return None

    return PathPatch(Path(vertices, codes), facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=1.2)

def run_regression_simulation(dxf_path, dt=0.005, anim_frames=140):
    d_case_id = 1.212 # inches
    r_case = d_case_id / 2.0 # inches
    L_grain = 4.6 # inches

    d_throat = 0.25 # inches
    d_exit = 0.630 # inches 
    A_throat = np.pi * (d_throat / 2.0)**2 # inches^2
    A_exit = np.pi * (d_exit / 2.0)**2 # inches^2
    expansion_ratio = A_exit / A_throat

    rho_p = 0.06068 # lb/in^3
    c_star = 5088.0 * 12.0 # in/s
    a = 0.0318
    n = 0.28
    g0 = 386.088 # in/s^2
    
    m_inert_motor = 0.5                      # lb
    m_rocket_kg = 1.95                       # kg
    m_rocket_lb = m_rocket_kg * 2.20462262   # converted to lb (~4.30 lb)
    m_inert_total = m_inert_motor + m_rocket_lb

    casing_circle = Point(0, 0).buffer(r_case, quad_segs=32)
    initial_port = extract_closed_port_from_dxf(dxf_path)

    propellant_section = casing_circle.difference(initial_port)
    initial_prop_volume = propellant_section.area * L_grain
    initial_prop_weight = initial_prop_volume * rho_p
    liftoff_total_weight = m_inert_total + initial_prop_weight

    coords = np.array(initial_port.exterior.coords)
    min_core_radius = np.min(np.linalg.norm(coords, axis=1))
    max_web_est = max(r_case - min_core_radius, 1e-3)

    pbar_total = 1000
    pbar = tqdm(total=pbar_total, desc="Burning Propellant", unit="‰",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    times, pressures, thrusts, burn_rates, webs, geom_history = [], [], [], [], [], []
    web_burned, current_time, last_pbar_val = 0.0, 0.0, 0

    while True:
        regressed_port = initial_port.buffer(web_burned, quad_segs=16)
        current_prop = casing_circle.difference(regressed_port)
        
        if current_prop.is_empty or current_prop.area < 1e-4:
            break

        burning_contour = regressed_port.boundary.intersection(casing_circle)
        perim_burning = burning_contour.length
        A_burn = perim_burning * L_grain
        
        if A_burn < 1e-4:
            break

        term = (a * rho_p * c_star * A_burn) / (g0 * A_throat)
        Pc = term ** (1.0 / (1.0 - n))
        rb = a * (Pc ** n)
        cf = get_thrust_coefficient(expansion_ratio=expansion_ratio, pc_psi=Pc)
        thrust = cf * Pc * A_throat

        times.append(current_time)
        pressures.append(Pc)
        thrusts.append(thrust)
        burn_rates.append(rb)
        webs.append(web_burned)
        geom_history.append(current_prop)

        pbar_val = int(min(web_burned / max_web_est, 1.0) * pbar_total)
        inc = pbar_val - last_pbar_val
        if inc > 0:
            pbar.update(inc)
            last_pbar_val = pbar_val

        web_burned += rb * dt
        current_time += dt

    if pbar_total - last_pbar_val > 0:
        pbar.update(pbar_total - last_pbar_val)
    pbar.close()

    time_arr = np.array(times)
    p_arr = np.array(pressures)
    f_arr = np.array(thrusts)

    burn_time = time_arr[-1] if len(time_arr) > 0 else 0.0
    max_pc = np.max(p_arr) if len(p_arr) > 0 else 0.0
    liftoff_thrust = f_arr[0] if len(f_arr) > 0 else 0.0
    twr = liftoff_thrust / liftoff_total_weight if liftoff_total_weight > 0 else 0.0

    print(f"Burn Time:                      {burn_time:.3f} s")
    print(f"Max Chamber Pressure (MEOP):    {max_pc:.2f} psi")
    print(f"Liftoff Thrust:                 {liftoff_thrust:.2f} lbf")
    print(f"Liftoff Total Weight:           {liftoff_total_weight:.2f} lbf ({liftoff_total_weight / 2.20462:.2f} kg)")
    print(f"Liftoff Thrust-to-Weight Ratio: {twr:.2f}")

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.2], height_ratios=[1, 1])

    ax_geom = fig.add_subplot(gs[:, 0])
    ax_pc = fig.add_subplot(gs[0, 1])
    ax_th = fig.add_subplot(gs[1, 1])

    ax_geom.set_aspect('equal')
    ax_geom.set_xlim(-r_case * 1.15, r_case * 1.15)
    ax_geom.set_ylim(-r_case * 1.15, r_case * 1.15)
    ax_geom.set_title("Grain Regression Cross-Section", fontsize=11, fontweight='bold')
    ax_geom.set_xlabel("X (inches)")
    ax_geom.set_ylabel("Y (inches)")
    ax_geom.grid(True, linestyle=':', alpha=0.5)

    casing_artist = MplCircle((0, 0), r_case, color='dimgray', fill=False, linewidth=2, linestyle='--', label='Casing Wall')
    ax_geom.add_artist(casing_artist)

    ax_pc.plot(time_arr, p_arr, color='mistyrose', lw=1.5, label='_nolegend_')
    pc_line, = ax_pc.plot([], [], color='crimson', lw=2.2, label='Chamber Pressure')
    pc_point, = ax_pc.plot([], [], marker='o', color='darkred', markersize=6)
    ax_pc.axhline(800, color='black', linestyle='--', linewidth=1, label='MEOP Limit (800 psi)')
    ax_pc.set_xlim(0, max(burn_time, 0.1) * 1.05)
    ax_pc.set_ylim(0, max(max_pc * 1.15, 850))
    ax_pc.set_ylabel('Chamber Pressure (psi)', fontsize=10)
    ax_pc.grid(True, linestyle=':', alpha=0.6)
    ax_pc.legend(loc='upper right')

    ax_th.plot(time_arr, f_arr, color='lightsteelblue', lw=1.5, label='_nolegend_')
    th_line, = ax_th.plot([], [], color='royalblue', lw=2.2, label='Thrust')
    th_point, = ax_th.plot([], [], marker='o', color='darkblue', markersize=6)
    ax_th.set_xlim(0, max(burn_time, 0.1) * 1.05)
    ax_th.set_ylim(0, max(np.max(f_arr) * 1.15, 10.0))
    ax_th.set_xlabel('Time (s)', fontsize=10)
    ax_th.set_ylabel('Thrust (lbf)', fontsize=10)
    ax_th.grid(True, linestyle=':', alpha=0.6)
    ax_th.legend(loc='upper right')

    telemetry_box = ax_geom.text(0.03, 0.03, '', transform=ax_geom.transAxes, 
                                 fontsize=9, verticalalignment='bottom', family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    total_steps = len(times)
    frame_indices = np.linspace(0, total_steps - 1, min(anim_frames, total_steps), dtype=int)
    current_patch_holder = [None]

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        if current_patch_holder[0] is not None:
            current_patch_holder[0].remove()
            current_patch_holder[0] = None

        prop_geom = geom_history[idx]
        patch = shapely_to_mpl_patch(prop_geom, facecolor='darkorange', edgecolor='black', alpha=0.88)
        if patch is not None:
            ax_geom.add_patch(patch)
            current_patch_holder[0] = patch

        pc_line.set_data(time_arr[:idx+1], p_arr[:idx+1])
        pc_point.set_data([time_arr[idx]], [p_arr[idx]])

        th_line.set_data(time_arr[:idx+1], f_arr[:idx+1])
        th_point.set_data([time_arr[idx]], [f_arr[idx]])

        telemetry_box.set_text(
            f"Time:       {times[idx]:.3f} s\n"
            f"Web Burned: {webs[idx]:.4f} in\n"
            f"Burn Rate:  {burn_rates[idx]:.3f} in/s\n"
            f"Pc:         {pressures[idx]:.1f} psi\n"
            f"Thrust:     {thrusts[idx]:.1f} lbf"
        )
        return [pc_line, pc_point, th_line, th_point, telemetry_box]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=False)
    plt.tight_layout()

    print("Exporting animation to Group10_P_burn.gif...")
    ani.save("Group10_P_burn.gif", dpi=300, writer=PillowWriter(fps=25))
    print("Export complete.")

    plt.show()

if __name__ == "__main__":
    run_regression_simulation("Group10_P.dxf", dt=0.005, anim_frames=140)