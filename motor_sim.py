import constants as c
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle as MplCircle
import ezdxf
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from performance import calc_cf


def extract_closed_port_from_dxf(dxf_path, scale=1.0, sample_pts_arc=32, sample_pts_spline=100):
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

    p_poly = Polygon(np.array(ordered_pts) * scale)
    return p_poly if p_poly.is_valid else p_poly.buffer(0)


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


def run_regression_simulation(motor_params, prop_params, dxf_path="Group10_P.dxf",
                              dxf_scale=c.IN2M, dt=0.005, plot=True, export=True,
                              anim_frames=140, gif_path="Group10_P_burn.gif"):
    
    id_case, L_grain, d_throat, d_exit = motor_params
    c_star, gamma, rho_p, a, n = prop_params

    r_case = id_case / 2.0
    A_throat = 0.25 * np.pi * d_throat**2
    A_exit = 0.25 * np.pi * d_exit**2
    Pa_sl = 101325.0 

    casing_circle = Point(0, 0).buffer(r_case, quad_segs=32)
    initial_port = extract_closed_port_from_dxf(dxf_path, scale=dxf_scale)

    area_tol = 1e-6 * casing_circle.area
    perim_tol = 1e-6 * casing_circle.length * L_grain

    coords = np.array(initial_port.exterior.coords)
    min_core_radius = np.min(np.linalg.norm(coords, axis=1))
    max_web_est = max(r_case - min_core_radius, 1e-6)

    pbar_total = 1000
    pbar = tqdm(total=pbar_total, desc="Burning Propellant", unit="‰",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    times, pressures, burn_rates, webs, m_props, thrusts, geom_history = [], [], [], [], [], [], []
    web_burned, current_time, last_pbar_val = 0.0, 0.0, 0

    while True:
        regressed_port = initial_port.buffer(web_burned, quad_segs=16)
        current_prop = casing_circle.difference(regressed_port)

        if current_prop.is_empty or current_prop.area < area_tol:
            break

        burning_contour = regressed_port.boundary.intersection(casing_circle)
        A_burn = burning_contour.length * L_grain

        if A_burn < perim_tol:
            break

        Pc = (a * rho_p * c_star * A_burn / A_throat) ** (1.0 / (1.0 - n)) 
        rb = a * Pc**n

        times.append(current_time)
        pressures.append(Pc)
        burn_rates.append(rb)
        webs.append(web_burned)
        m_props.append(current_prop.area * L_grain * rho_p)
        thrusts.append(calc_cf(Pc, Pa_sl, A_throat, A_exit, gamma) * Pc * A_throat)
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

    if not times:
        raise RuntimeError("Grain produced no burn steps; check the DXF scale and case diameter.")

    time_arr = np.array(times)
    p_arr = np.array(pressures)
    m_prop_arr = np.array(m_props)
    f_arr = np.array(thrusts)

    burn_time = time_arr[-1]
    max_pc = np.max(p_arr)
    max_thrust = np.max(f_arr)
    impulse = np.sum(0.5 * (f_arr[1:] + f_arr[:-1]) * np.diff(time_arr))
    average_thrust = impulse / burn_time

    print("\n===================== Burn Data ====================")
    print(f"Burn time:                      {burn_time:.3f} s")
    print(f"Max chamber pressure (MEOP):    {max_pc * c.PA2PSI:.2f} psi")
    print(f"Max thrust:                     {max_thrust / c.LBF2N:.1f} lbf")
    print(f"Average thrust:                 {average_thrust / c.LBF2N:.1f} lbf")
    print(f"Propellant mass:                {m_prop_arr[0] / c.LBM2KG:.3f} lbm")
    print(f"Total impulse (sea level):      {impulse:.1f} N*s")

    if plot or export:
        animate_burn(time_arr, p_arr, f_arr, burn_rates, webs, geom_history, r_case,
                     anim_frames=anim_frames, gif_path=gif_path, plot=plot, export=export)

    return time_arr, p_arr, m_prop_arr, f_arr


def animate_burn(time_arr, p_arr, f_arr, burn_rates, webs, geom_history, r_case,
                 anim_frames=140, gif_path="Group10_P_burn.gif", plot=True, export=True):
    p_psi = p_arr * c.PA2PSI
    f_lbf = f_arr / c.LBF2N
    r_in = r_case * c.M2IN
    burn_time = time_arr[-1]
    max_pc = np.max(p_psi)

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.2], height_ratios=[1, 1])

    ax_geom = fig.add_subplot(gs[:, 0])
    ax_pc = fig.add_subplot(gs[0, 1])
    ax_th = fig.add_subplot(gs[1, 1])

    ax_geom.set_aspect('equal')
    ax_geom.set_xlim(-r_in * 1.15, r_in * 1.15)
    ax_geom.set_ylim(-r_in * 1.15, r_in * 1.15)
    ax_geom.set_title("Grain Regression Cross-Section", fontsize=11, fontweight='bold')
    ax_geom.set_xlabel("X (inches)")
    ax_geom.set_ylabel("Y (inches)")
    ax_geom.grid(True, linestyle=':', alpha=0.5)

    casing_artist = MplCircle((0, 0), r_in, color='dimgray', fill=False, linewidth=2, linestyle='--', label='Casing Wall')
    ax_geom.add_artist(casing_artist)

    ax_pc.plot(time_arr, p_psi, color='mistyrose', lw=1.5, label='_nolegend_')
    pc_line, = ax_pc.plot([], [], color='crimson', lw=2.2, label='Chamber Pressure')
    pc_point, = ax_pc.plot([], [], marker='o', color='darkred', markersize=6)
    ax_pc.axhline(800, color='black', linestyle='--', linewidth=1, label='MEOP Limit (800 psi)')
    ax_pc.set_xlim(0, max(burn_time, 0.1) * 1.05)
    ax_pc.set_ylim(0, max(max_pc * 1.15, 850))
    ax_pc.set_ylabel('Chamber Pressure (psi)', fontsize=10)
    ax_pc.grid(True, linestyle=':', alpha=0.6)
    ax_pc.legend(loc='upper right')

    ax_th.plot(time_arr, f_lbf, color='lightsteelblue', lw=1.5, label='_nolegend_')
    th_line, = ax_th.plot([], [], color='royalblue', lw=2.2, label='Thrust (sea level)')
    th_point, = ax_th.plot([], [], marker='o', color='darkblue', markersize=6)
    ax_th.set_xlim(0, max(burn_time, 0.1) * 1.05)
    ax_th.set_ylim(0, max(np.max(f_lbf) * 1.15, 10.0))
    ax_th.set_xlabel('Time (s)', fontsize=10)
    ax_th.set_ylabel('Thrust (lbf)', fontsize=10)
    ax_th.grid(True, linestyle=':', alpha=0.6)
    ax_th.legend(loc='upper right')

    telemetry_box = ax_geom.text(0.03, 0.03, '', transform=ax_geom.transAxes,
                                 fontsize=9, verticalalignment='bottom', family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    total_steps = len(time_arr)
    frame_indices = np.linspace(0, total_steps - 1, min(anim_frames, total_steps), dtype=int)
    current_patch_holder = [None]

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        if current_patch_holder[0] is not None:
            current_patch_holder[0].remove()
            current_patch_holder[0] = None

        geom_in = geom_history[idx]
        patch = shapely_to_mpl_patch(geom_in, facecolor='darkorange', edgecolor='black', alpha=0.88)
        if patch is not None:
            patch.set_transform(plt.matplotlib.transforms.Affine2D().scale(c.M2IN) + ax_geom.transData)
            ax_geom.add_patch(patch)
            current_patch_holder[0] = patch

        pc_line.set_data(time_arr[:idx+1], p_psi[:idx+1])
        pc_point.set_data([time_arr[idx]], [p_psi[idx]])

        th_line.set_data(time_arr[:idx+1], f_lbf[:idx+1])
        th_point.set_data([time_arr[idx]], [f_lbf[idx]])

        telemetry_box.set_text(
            f"Time:       {time_arr[idx]:.3f} s\n"
            f"Web Burned: {webs[idx] * c.M2IN:.4f} in\n"
            f"Burn Rate:  {burn_rates[idx] * c.M2IN:.3f} in/s\n"
            f"Pc:         {p_psi[idx]:.1f} psi\n"
            f"Thrust:     {f_lbf[idx]:.1f} lbf"
        )
        return [pc_line, pc_point, th_line, th_point, telemetry_box]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=False)
    plt.tight_layout()

    if export:
        print(f"Exporting animation to {gif_path}...")
        ani.save(gif_path, dpi=300, writer=PillowWriter(fps=25))
        print("Export complete.")

    if plot:
        plt.show()
    else:
        plt.close(fig)