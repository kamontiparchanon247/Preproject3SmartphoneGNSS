

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, sosfiltfilt


# GROUND TRUTH 

GROUND_TRUTH_LAT = 13.72728500
GROUND_TRUTH_LON = 100.77642400
GROUND_TRUTH_ALT = -15.90


# COORDINATE CONVERSION


def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0):
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    a  = 6378137.0
    e2 = 0.00669437999014
    N  = a / np.sqrt(1 - e2 * np.sin(lat0_rad)**2)
    x0 = (N + alt0) * np.cos(lat0_rad) * np.cos(lon0_rad)
    y0 = (N + alt0) * np.cos(lat0_rad) * np.sin(lon0_rad)
    z0 = (N * (1 - e2) + alt0) * np.sin(lat0_rad)
    dx = x_ecef - x0; dy = y_ecef - y0; dz = z_ecef - z0
    sin_lat = np.sin(lat0_rad); cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad); cos_lon = np.cos(lon0_rad)
    e =  -sin_lon*dx + cos_lon*dy
    n =  -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz
    u =   cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz
    return e, n, u


def latlon_to_enu(lat_t, lon_t, alt_t, lat0, lon0, alt0):
    lat_r = np.radians(lat_t); lon_r = np.radians(lon_t)
    a  = 6378137.0; e2 = 0.00669437999014
    N  = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
    xe = (N + alt_t) * np.cos(lat_r) * np.cos(lon_r)
    ye = (N + alt_t) * np.cos(lat_r) * np.sin(lon_r)
    ze = (N*(1-e2) + alt_t) * np.sin(lat_r)
    return ecef_to_enu(xe, ye, ze, lat0, lon0, alt0)



# LPF WITH LINEAR EXTRAPOLATION


def apply_lpf_extended(signal, fs=1.0, cutoff_hz=None, order=4):
    n = len(signal)
    if n < 4:
        return np.array(signal, dtype=float)
    if cutoff_hz is None:
        cutoff_hz = fs / 5.0
    nyq       = fs / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.99)
    sos      = butter(order, cutoff_hz / nyq, btype='low', analog=False, output='sos')
    settling = int(np.ceil(order * (fs / cutoff_hz)))
    ext      = max(100, 3 * settling)
    ext      = min(ext, n - 1)
    x = np.asarray(signal, dtype=float)
    slope_head = x[1] - x[0]
    head       = x[0] - slope_head * np.arange(ext, 0, -1)
    slope_tail = x[-1] - x[-2]
    tail       = x[-1] + slope_tail * np.arange(1, ext + 1)
    x_ext = np.concatenate([head, x, tail])
    y_ext = sosfiltfilt(sos, x_ext)
    return y_ext[ext: ext + n]


def run_lpf(df, fs=1.0, cutoff_hz=None, order=4):
    print(f"\n[LPF]  Running Butterworth order={order}, "
          f"cutoff={'auto (fs/5)' if cutoff_hz is None else f'{cutoff_hz:.4f} Hz'}")
    for col, out_col in [('x', 'lpf_x'), ('y', 'lpf_y'), ('z', 'lpf_z')]:
        df[out_col] = apply_lpf_extended(df[col].values, fs=fs,
                                         cutoff_hz=cutoff_hz, order=order)
    print(f"[LPF]  Done")
    return df



# DATA LOADING


def load_wls_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} rows | Columns: {df.columns.tolist()}")
    if 'X_ECEF' in df.columns:
        lat0 = df['Latitude'].iloc[0]
        lon0 = df['Longitude'].iloc[0]
        alt0 = df['Altitude'].iloc[0]
        e, n, u = ecef_to_enu(df['X_ECEF'].values, df['Y_ECEF'].values,
                               df['Z_ECEF'].values, lat0, lon0, alt0)
        df['x'] = e; df['y'] = n; df['z'] = u
    elif 'Latitude' in df.columns:
        lat0 = df['Latitude'].iloc[0]
        lon0 = df['Longitude'].iloc[0]
        df['x'] = (df['Longitude'] - lon0) * 111000 * np.cos(np.radians(lat0))
        df['y'] = (df['Latitude']  - lat0) * 111000
        df['z'] = df['Altitude'] - df['Altitude'].iloc[0]
    elif 'x' not in df.columns:
        raise ValueError("No position columns found!")
    if 'Epoch' in df.columns and 'timestamp' not in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df['Epoch']):
                df['timestamp'] = df['Epoch'] - df['Epoch'].iloc[0]
            else:
                df['Epoch'] = pd.to_datetime(df['Epoch'])
                df['timestamp'] = (df['Epoch'] - df['Epoch'].iloc[0]).dt.total_seconds()
        except:
            df['timestamp'] = np.arange(len(df), dtype=float)
    elif 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df), dtype=float)
    return df


def add_ground_truth(df):
    print(f"\n Ground Truth: ({GROUND_TRUTH_LAT}, {GROUND_TRUTH_LON}, alt={GROUND_TRUTH_ALT}m)")
    lat0 = df['Latitude'].iloc[0]
    lon0 = df['Longitude'].iloc[0]
    alt0 = df['Altitude'].iloc[0] if 'Altitude' in df.columns else 0.0
    e_true, n_true, u_true = latlon_to_enu(
        GROUND_TRUTH_LAT, GROUND_TRUTH_LON, GROUND_TRUTH_ALT, lat0, lon0, alt0)
    df['x_true'] = e_true
    df['y_true'] = n_true
    df['z_true'] = u_true
    return df


# RMS HELPER


def _rms(series):
    return float(np.sqrt((np.array(series)**2).mean()))


# COMPUTE ERRORS


def compute_errors(df):
    for prefix, xc, yc, zc in [
        ('raw', 'x',     'y',     'z'),
        ('lpf', 'lpf_x', 'lpf_y', 'lpf_z'),
    ]:
        df[f'err_{prefix}_x']  = df[xc] - df['x_true']
        df[f'err_{prefix}_y']  = df[yc] - df['y_true']
        df[f'err_{prefix}_z']  = df[zc] - df['z_true']
        df[f'err_{prefix}_2d'] = np.sqrt(df[f'err_{prefix}_x']**2 + df[f'err_{prefix}_y']**2)
        df[f'err_{prefix}_3d'] = np.sqrt(df[f'err_{prefix}_x']**2 + df[f'err_{prefix}_y']**2 + df[f'err_{prefix}_z']**2)
    return df



# PRINT STATS


def _print_stats(df):
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY  (Raw WLS  vs  WLS + LPF)")
    print(f"{'='*70}")
    axes = [('X (East)', 'x'), ('Y (North)', 'y'), ('Z (Up)', 'z')]
    print(f"\n  {'Axis':<12} {'Raw RMS':>10} {'LPF RMS':>10} {'Improve':>10}  Status")
    print(f"  {'-'*65}")
    for label, ax in axes:
        r   = _rms(df[f'err_raw_{ax}'])
        l   = _rms(df[f'err_lpf_{ax}'])
        imp = (r - l) / r * 100
        status = " BELOW 10m!" if l < 10 else f"⚠  need {l-10:.2f}m more"
        print(f"  {label:<12} {r:>9.3f}m {l:>9.3f}m  {'▼' if imp>0 else '▲'}{abs(imp):>8.1f}%  {status}")
    r2d = _rms(df['err_raw_2d']); l2d = _rms(df['err_lpf_2d'])
    r3d = _rms(df['err_raw_3d']); l3d = _rms(df['err_lpf_3d'])
    print(f"\n  {'2D (Horiz)':<12} {r2d:>9.3f}m {l2d:>9.3f}m  {'▼' if l2d<r2d else '▲'}{abs((r2d-l2d)/r2d*100):>8.1f}%")
    print(f"  {'3D (Full)':<12} {r3d:>9.3f}m {l3d:>9.3f}m  {'▼' if l3d<r3d else '▲'}{abs((r3d-l3d)/r3d*100):>8.1f}%")
    print(f"{'='*70}\n")



# PLOT FUNCTIONS — 


def _error_plot(df, raw_col, lpf_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    t       = df['timestamp'].values
    raw_err = df[raw_col].values
    lpf_err = df[lpf_col].values
    rms_r   = _rms(df[raw_col])
    rms_l   = _rms(df[lpf_col])
    improve = (rms_r - rms_l) / rms_r * 100
    ax.plot(t, raw_err, 'r-', alpha=0.45, linewidth=1.2, label=f'Raw WLS  (RMS = {rms_r:.2f} m)')
    ax.plot(t, lpf_err, color='#1565C0', linewidth=2.2, label=f'WLS + LPF  (RMS = {rms_l:.2f} m)')
    ax.axhline(0,    color='black',   linestyle='--', linewidth=0.7, alpha=0.4)
    ax.axhline( 10,  color='#2E7D32', linestyle='--', linewidth=1.8, alpha=0.85, label='±10 m target')
    ax.axhline(-10,  color='#2E7D32', linestyle='--', linewidth=1.8, alpha=0.85)
    color = '#2E7D32' if rms_l < 10 else '#C62828'
    ax.set_title(f'{title}  →  LPF RMS = {rms_l:.2f} m  ({"▼" if improve>0 else "▲"}{abs(improve):.1f}%)',
                 fontsize=13, fontweight='bold', color=color, pad=12)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    plt.show()   


def plot_2d_trajectory(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df['x'],     df['y'],     'r.', alpha=0.35, markersize=3, label='Raw GPS')
    ax.plot(df['lpf_x'], df['lpf_y'], color='#1565C0', linewidth=2.2, label='WLS + LPF')
    ax.plot(df['x_true'].iloc[0], df['y_true'].iloc[0], 'g*', markersize=18,
            zorder=10, markeredgecolor='darkgreen', markeredgewidth=0.8, label='Ground Truth')
    rms_2d = _rms(df['err_lpf_2d'])
    ax.set_title(f'2D Trajectory  (LPF 2D RMS = {rms_2d:.2f} m)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('East [m]', fontsize=11)
    ax.set_ylabel('North [m]', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()   


def plot_rms_bar(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    lbls    = ['X\n(East)', 'Y\n(North)', 'Z\n(Up)', '2D\n(Horiz)', '3D\n(Full)']
    raw_rms = [_rms(df[f'err_raw_{a}']) for a in ['x','y','z']] + \
              [_rms(df['err_raw_2d']), _rms(df['err_raw_3d'])]
    lpf_rms = [_rms(df[f'err_lpf_{a}']) for a in ['x','y','z']] + \
              [_rms(df['err_lpf_2d']), _rms(df['err_lpf_3d'])]
    xp = np.arange(len(lbls)); w = 0.38
    ax.bar(xp - w/2, raw_rms, w, label='Raw WLS',   color='#EF5350', alpha=0.85)
    bars = ax.bar(xp + w/2, lpf_rms, w, label='WLS + LPF', color='#1565C0', alpha=0.85)
    for bar, val in zip(bars, lpf_rms):
        if val < 10:
            bar.set_color('#2E7D32'); bar.set_alpha(0.9)
    ax.axhline(10, color='#2E7D32', linestyle='--', linewidth=2.2, label='10 m target', alpha=0.9)
    for i, (r, l) in enumerate(zip(raw_rms, lpf_rms)):
        imp = (r - l) / r * 100
        ax.text(i, max(r, l) + 0.8, f'{"▼" if imp>0 else "▲"}{abs(imp):.0f}%',
                ha='center', fontsize=10, fontweight='bold',
                color='#1A237E' if imp > 0 else '#C62828')
    ax.set_xticks(xp); ax.set_xticklabels(lbls, fontsize=11)
    ax.set_ylabel('RMS Error [m]', fontsize=11)
    ax.set_title('RMS Error  Raw WLS vs WLS+LPF', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(raw_rms) * 1.18)
    plt.tight_layout()
    plt.show()  


def plot_all(df):
    print("\n Showing 5 plots...")
    _error_plot(df, 'err_raw_x', 'err_lpf_x', 'X (East)',  'X (East) Error [m]')
    _error_plot(df, 'err_raw_y', 'err_lpf_y', 'Y (North)', 'Y (North) Error [m]')
    _error_plot(df, 'err_raw_z', 'err_lpf_z', 'Z (Up)',    'Z (Up) Error [m]')
    plot_2d_trajectory(df)
    plot_rms_bar(df)



# MAIN PIPELINE


def run_pipeline(input_file, cutoff_hz=None, lpf_order=4):
    print("\n" + "="*70)
    print("  LPF PIPELINE  (Butterworth, linear-extrapolation padded)")
    print("="*70)
    df = load_wls_data(input_file)
    df = add_ground_truth(df)
    dt = df['timestamp'].diff().median() if len(df) > 1 else 1.0
    fs = 1.0 / dt if dt > 0 else 1.0
    print(f"\n   fs={fs:.4f} Hz  dt={dt:.4f} s  |  epochs={len(df)}")
    df = run_lpf(df, fs=fs, cutoff_hz=cutoff_hz, order=lpf_order)
    df = compute_errors(df)
    _print_stats(df)
    plot_all(df)
    return df




if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\user\Downloads\gnss_results_FINAL.xlsx"

    try:
        df_results = run_pipeline(
            input_file=INPUT_FILE,
            cutoff_hz=None,   
            lpf_order=4,
        )
    except FileNotFoundError:
        print(f"\n File not found: {INPUT_FILE}")
    except Exception as e:
        import traceback
        print(f"\n Error: {e}")
        traceback.print_exc()