
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from collections import deque



# COORDINATE CONVERSION


def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0):
    lat0_rad = np.radians(lat0); lon0_rad = np.radians(lon0)
    a = 6378137.0; e2 = 0.00669437999014
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

def latlon_alt_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    a = 6378137.0; e2 = 0.00669437999014
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return x, y, z



#  9-STATE ADAPTIVE KALMAN FILTER


class AdaptiveKalmanFilter9State:
   

    def __init__(self, dt=1.0, sigma_a=0.5, alpha=0.05, window=20,
                 use_quality_weight=True,
                 q_scale_x=0.2,  
                 q_scale_y=1.0,
                 q_scale_z=1.0,
                 R_min=1.0,      
                 R_max=500.0):   

        self.dt    = dt
        self.n     = 9
        self.alpha = alpha
        self.window = window
        self.use_quality_weight = use_quality_weight
        self.R_min = R_min
        self.R_max = R_max
        
        # Per-axis Q scaling
        self.q_scale = np.array([q_scale_x, q_scale_y, q_scale_z])

        dt2 = 0.5 * dt**2
        self.F = np.array([
            [1,0,0, dt,0,0, dt2,0,  0  ],
            [0,1,0, 0,dt,0, 0,  dt2,0  ],
            [0,0,1, 0,0,dt, 0,  0,  dt2],
            [0,0,0, 1,0,0,  dt, 0,  0  ],
            [0,0,0, 0,1,0,  0,  dt, 0  ],
            [0,0,0, 0,0,1,  0,  0,  dt ],
            [0,0,0, 0,0,0,  1,  0,  0  ],
            [0,0,0, 0,0,0,  0,  1,  0  ],
            [0,0,0, 0,0,0,  0,  0,  1  ]
        ], dtype=float)

        self.H = np.zeros((3, 9))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1

        # Build base Q, then scale per axis
        self.Q_base = self._build_Q_singer(dt, sigma_a)
        self.Q = self._scale_Q_per_axis(self.Q_base, self.q_scale)
        
        self.R = np.diag([50.0, 50.0, 100.0])

        self.x = np.zeros((9, 1))
        self.P = np.diag([100.,100.,200.,4.,4.,4.,1.,1.,1.])
        self.innov_window = deque(maxlen=window)

        self.history = {
            'x':[], 'y':[], 'z':[],
            'Vx':[], 'Vy':[], 'Vz':[],
            'ax':[], 'ay':[], 'az':[],
            'R_x':[], 'R_y':[], 'R_z':[],
            'innov_mag':[], 'quality_scale':[]
        }

    def _build_Q_singer(self, dt, sigma_a):
        dt2=dt**2; dt3=dt**3; dt4=dt**4; dt5=dt**5
        Q1 = sigma_a**2 * np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  dt   ]
        ])
        Q = np.zeros((9,9))
        for i in [0,3,6]:
            Q[i:i+3, i:i+3] = Q1
        return Q

    def _scale_Q_per_axis(self, Q_base, scales):
        
        Q = Q_base.copy()
        # Position, velocity, acceleration indices for each axis
        idx_x = [0, 3, 6]  
        idx_y = [1, 4, 7]  
        idx_z = [2, 5, 8]  
        for idx, scale in zip([idx_x, idx_y, idx_z], scales):
            for i in idx:
                for j in idx:
                    Q[i, j] *= scale
        return Q

    def initialize(self, x0, y0, z0):
        self.x = np.array([[x0],[y0],[z0],[0],[0],[0],[0],[0],[0]], dtype=float)
        self.P = np.diag([100.,100.,100.,4.,4.,4.,1.,1.,1.])
        self._store_state(1.0)

    def step(self, z_meas, quality_scale=1.0):
        if z_meas.ndim == 1:
            z_meas = z_meas.reshape(-1, 1)
        
        # PREDICT
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # INNOVATION
        y_tilde = z_meas - self.H @ x_pred
        self.innov_window.append(y_tilde.flatten())
        
        # ADAPTIVE R UPDATE 
        if len(self.innov_window) >= 5:
            innov_arr = np.array(self.innov_window)
            C_innov = (innov_arr.T @ innov_arr) / len(innov_arr)
            R_new = C_innov - self.H @ P_pred @ self.H.T
            R_new = np.diag(np.clip(np.diag(R_new), self.R_min, self.R_max))
            self.R = (1 - self.alpha) * self.R + self.alpha * R_new
        
        # Apply quality weighting
        R_used = self.R * quality_scale
        
        # KALMAN UPDATE
        S = self.H @ P_pred @ self.H.T + R_used
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_tilde
        self.P = (np.eye(self.n) - K @ self.H) @ P_pred
        
        self._store_state(quality_scale)
        return self.x.flatten()

    def _store_state(self, quality_scale):
        s = self.x.flatten()
        self.history['x'].append(s[0]);  self.history['y'].append(s[1])
        self.history['z'].append(s[2]);  self.history['Vx'].append(s[3])
        self.history['Vy'].append(s[4]); self.history['Vz'].append(s[5])
        self.history['ax'].append(s[6]); self.history['ay'].append(s[7])
        self.history['az'].append(s[8])
        self.history['R_x'].append(self.R[0,0])
        self.history['R_y'].append(self.R[1,1])
        self.history['R_z'].append(self.R[2,2])
        self.history['quality_scale'].append(quality_scale)
        if len(self.innov_window) > 0:
            self.history['innov_mag'].append(float(np.linalg.norm(self.innov_window[-1])))
        else:
            self.history['innov_mag'].append(0.0)

    def get_history_dataframe(self):
        return pd.DataFrame(self.history)


# LOAD AND MERGE 


def load_and_merge(smartphone_file, rtk_file, time_tolerance_sec=2.0):
    print("="*70)
    print("  LOADING DATA")
    print("="*70)
    sp = pd.read_excel(smartphone_file)
    print(f"\n  Smartphone: {len(sp)} rows | Columns: {sp.columns.tolist()}")
    sp['UTC_Time'] = pd.to_datetime(sp['UTC_Time'], errors='coerce')
    sp = sp.dropna(subset=['UTC_Time']).reset_index(drop=True)
    rtk = pd.read_excel(rtk_file)
    print(f"  RTK:        {len(rtk)} rows | Columns: {rtk.columns.tolist()}")
    rtk['UTC_Time'] = pd.to_datetime(rtk['UTC_Time'], errors='coerce')
    rtk = rtk.dropna(subset=['UTC_Time']).reset_index(drop=True)
    if 'Fix_Quality' in rtk.columns:
        rtk_good = rtk[rtk['Fix_Quality'].isin([4, 5])].copy()
        print(f"\n  RTK fixed/float epochs: {len(rtk_good)} / {len(rtk)}")
        if len(rtk_good) == 0:
            print("  ⚠ No RTK fixed epochs found! Using all RTK data.")
            rtk_good = rtk.copy()
    else:
        rtk_good = rtk.copy()
    lat0 = sp['Latitude'].iloc[0]; lon0 = sp['Longitude'].iloc[0]; alt0 = sp['Altitude'].iloc[0]
    print(f"\n  Reference: Lat={lat0:.7f}°  Lon={lon0:.7f}°  Alt={alt0:.2f}m")
    if 'X_ECEF' in sp.columns:
        e, n, u = ecef_to_enu(sp['X_ECEF'].values, sp['Y_ECEF'].values, sp['Z_ECEF'].values, lat0, lon0, alt0)
    else:
        e = (sp['Longitude'] - lon0) * 111000 * np.cos(np.radians(lat0))
        n = (sp['Latitude']  - lat0) * 111000
        u = sp['Altitude'] - alt0
    sp['x'] = e; sp['y'] = n; sp['z'] = u
    rtk_x_ecef, rtk_y_ecef, rtk_z_ecef = latlon_alt_to_ecef(
        rtk_good['Latitude'].values, rtk_good['Longitude'].values, rtk_good['Altitude_m'].values)
    re, rn, ru = ecef_to_enu(rtk_x_ecef, rtk_y_ecef, rtk_z_ecef, lat0, lon0, alt0)
    rtk_good = rtk_good.copy()
    rtk_good['x_rtk'] = re; rtk_good['y_rtk'] = rn; rtk_good['z_rtk'] = ru
    print(f"\n  Matching epochs (tolerance={time_tolerance_sec}s)...")
    sp_times  = sp['UTC_Time'].values.astype('int64') / 1e9
    rtk_times = rtk_good['UTC_Time'].values.astype('int64') / 1e9
    matched_x_rtk = np.full(len(sp), np.nan); matched_y_rtk = np.full(len(sp), np.nan)
    matched_z_rtk = np.full(len(sp), np.nan); matched_fq    = np.full(len(sp), np.nan)
    for i, t_sp in enumerate(sp_times):
        diffs = np.abs(rtk_times - t_sp); j = np.argmin(diffs)
        if diffs[j] <= time_tolerance_sec:
            matched_x_rtk[i] = rtk_good['x_rtk'].iloc[j]
            matched_y_rtk[i] = rtk_good['y_rtk'].iloc[j]
            matched_z_rtk[i] = rtk_good['z_rtk'].iloc[j]
            if 'Fix_Quality' in rtk_good.columns:
                matched_fq[i] = rtk_good['Fix_Quality'].iloc[j]
    sp['x_true'] = matched_x_rtk; sp['y_true'] = matched_y_rtk
    sp['z_true'] = matched_z_rtk; sp['rtk_fix_quality'] = matched_fq
    matched = (~np.isnan(matched_x_rtk)).sum()
    print(f"  Matched {matched} / {len(sp)} epochs to RTK")
    sp['timestamp'] = (sp['UTC_Time'] - sp['UTC_Time'].iloc[0]).dt.total_seconds()
    print(f"  Duration: {sp['timestamp'].iloc[-1]:.1f}s | Epochs: {len(sp)} | Mean dt: {sp['timestamp'].diff().median():.2f}s")
    return sp, lat0, lon0, alt0


def compute_quality_scale(df):
    scale = np.ones(len(df))
    if 'HDOP' in df.columns:
        hdop = np.clip(df['HDOP'].values.astype(float), 0.5, 10.0)
        scale *= (hdop / hdop.mean())
    if 'NumSatellites' in df.columns:
        nsats = np.clip(df['NumSatellites'].values.astype(float), 1, 30)
        scale *= (nsats.mean() / nsats)
    if 'RMS_Residual' in df.columns:
        rms_res = np.clip(df['RMS_Residual'].values.astype(float), 0.01, 100)
        scale *= (rms_res / rms_res.mean())
    scale = scale / scale.mean()
    return np.clip(scale, 0.2, 5.0)


def run_adaptive_kalman(smartphone_file, rtk_file, output_file,
                        sigma_a=0.5, alpha=0.05, window=20, time_tolerance_sec=2.0,
                        q_scale_x=0.2, q_scale_y=1.0, q_scale_z=1.0):
    df, lat0, lon0, alt0 = load_and_merge(smartphone_file, rtk_file, time_tolerance_sec)
    dt = df['timestamp'].diff().median()
    if np.isnan(dt) or dt <= 0: dt = 1.0
    print(f"\n  dt={dt:.3f}s | sigma_a={sigma_a} | alpha={alpha} | window={window}")
    print(f"  Q scaling: X={q_scale_x} Y={q_scale_y} Z={q_scale_z}")
    quality_scale = compute_quality_scale(df)
    print("\n" + "="*70)
    print("  RUNNING FIXED 9-STATE ADAPTIVE KALMAN FILTER")
    print("="*70)
    kf = AdaptiveKalmanFilter9State(dt=dt, sigma_a=sigma_a, alpha=alpha,
                                     window=window, use_quality_weight=True,
                                     q_scale_x=q_scale_x, q_scale_y=q_scale_y, q_scale_z=q_scale_z)
    kf.initialize(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0])
    print(f"\n  Processing {len(df)} epochs...")
    for i in range(1, len(df)):
        z = np.array([df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]])
        kf.step(z, quality_scale=quality_scale[i])
        if (i+1) % 200 == 0:
            print(f"  ... {i+1}/{len(df)}")
    print("  ✓ Done")
    df_kf = kf.get_history_dataframe()
    n = len(df_kf)
    has_truth = ~np.isnan(df['x_true'].values[:n])
    if has_truth.sum() >= 3:
        bias_x = float(np.nanmean(df_kf['x'].values[has_truth] - df['x_true'].values[:n][has_truth]))
        bias_y = float(np.nanmean(df_kf['y'].values[has_truth] - df['y_true'].values[:n][has_truth]))
        bias_z = float(np.nanmean(df_kf['z'].values[has_truth] - df['z_true'].values[:n][has_truth]))
        print(f"\n  Bias → X={bias_x:+.3f}m  Y={bias_y:+.3f}m  Z={bias_z:+.3f}m")
    else:
        bias_x = bias_y = bias_z = 0.0
        print("  ⚠ Not enough matched epochs — bias correction skipped")
    df_out = pd.DataFrame()
    df_out['UTC_Time']  = df['UTC_Time'].values[:n]
    df_out['timestamp'] = df['timestamp'].values[:n]
    df_out['kf_x']  = df_kf['x'].values - bias_x
    df_out['kf_y']  = df_kf['y'].values - bias_y
    df_out['kf_z']  = df_kf['z'].values - bias_z
    df_out['kf_Vx'] = df_kf['Vx'].values
    df_out['kf_Vy'] = df_kf['Vy'].values
    df_out['kf_Vz'] = df_kf['Vz'].values
    df_out['kf_ax'] = df_kf['ax'].values
    df_out['kf_ay'] = df_kf['ay'].values
    df_out['kf_az'] = df_kf['az'].values
    df_out['R_x']   = df_kf['R_x'].values
    df_out['R_y']   = df_kf['R_y'].values
    df_out['R_z']   = df_kf['R_z'].values
    df_out['raw_x'] = df['x'].values[:n]
    df_out['raw_y'] = df['y'].values[:n]
    df_out['raw_z'] = df['z'].values[:n]
    df_out['x_true'] = df['x_true'].values[:n]
    df_out['y_true'] = df['y_true'].values[:n]
    df_out['z_true'] = df['z_true'].values[:n]
    df_out['rtk_fix_quality'] = df['rtk_fix_quality'].values[:n]
    df_out['quality_scale']   = quality_scale[:n]
    for col in ['HDOP','VDOP','PDOP','NumSatellites','MeanCNo','RMS_Residual','Tropo_m','MeanElevation']:
        if col in df.columns:
            df_out[col] = df[col].values[:n]
    has_truth_out = ~np.isnan(df_out['x_true'].values)
    df_out['err_raw_x']  = df_out['raw_x'] - df_out['x_true']
    df_out['err_raw_y']  = df_out['raw_y'] - df_out['y_true']
    df_out['err_raw_z']  = df_out['raw_z'] - df_out['z_true']
    df_out['err_raw_2d'] = np.sqrt(df_out['err_raw_x']**2 + df_out['err_raw_y']**2)
    df_out['err_raw_3d'] = np.sqrt(df_out['err_raw_x']**2 + df_out['err_raw_y']**2 + df_out['err_raw_z']**2)
    df_out['err_kf_x']  = df_out['kf_x'] - df_out['x_true']
    df_out['err_kf_y']  = df_out['kf_y'] - df_out['y_true']
    df_out['err_kf_z']  = df_out['kf_z'] - df_out['z_true']
    df_out['err_kf_2d'] = np.sqrt(df_out['err_kf_x']**2 + df_out['err_kf_y']**2)
    df_out['err_kf_3d'] = np.sqrt(df_out['err_kf_x']**2 + df_out['err_kf_y']**2 + df_out['err_kf_z']**2)
    if not output_file.lower().endswith('.xlsx'):
        output_file += '.xlsx'
    df_out.to_excel(output_file, index=False)
    print(f"  Saved → {output_file}")
    _print_rms_summary(df_out, has_truth_out, (bias_x, bias_y, bias_z))
    return df_out, kf


def _rms(arr, mask=None):
    if mask is not None: arr = arr[mask]
    arr = arr[~np.isnan(arr)]
    return float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else np.nan

def _print_rms_summary(df, has_truth, biases=(0,0,0)):
    m = has_truth; bx, by, bz = biases
    print(f"\n{'='*70}")
    print(f"  RMS ERROR vs RTK | Bias X={bx:+.2f}m  Y={by:+.2f}m  Z={bz:+.2f}m")
    print(f"  Matched epochs: {m.sum()} / {len(df)}")
    print(f"{'='*70}")
    for label, ax in [('X (East)','x'),('Y (North)','y'),('Z (Up)','z')]:
        r_raw = _rms(df[f'err_raw_{ax}'].values, m)
        r_kf  = _rms(df[f'err_kf_{ax}'].values,  m)
        imp   = (r_raw - r_kf) / r_raw * 100 if r_raw > 0 else 0
        status = '✓' if imp > 0 else '✗ WORSE'
        print(f"  {label:<12} Raw={r_raw:.3f}m  KF={r_kf:.3f}m  {imp:+.1f}%  {status}")
    r2d_raw = _rms(df['err_raw_2d'].values, m); r2d_kf = _rms(df['err_kf_2d'].values, m)
    r3d_raw = _rms(df['err_raw_3d'].values, m); r3d_kf = _rms(df['err_kf_3d'].values, m)
    print(f"  {'2D':<12} Raw={r2d_raw:.3f}m  KF={r2d_kf:.3f}m  {(r2d_raw-r2d_kf)/r2d_raw*100:+.1f}%")
    print(f"  {'3D':<12} Raw={r3d_raw:.3f}m  KF={r3d_kf:.3f}m  {(r3d_raw-r3d_kf)/r3d_raw*100:+.1f}%")
    print(f"{'='*70}\n")



# ISOLATED PLOTS (10 plots)


_S = {
    "raw" : "#E05252",
    "kf"  : "#2563EB",
    "rtk" : "#16A34A",
    "rc"  : ["#E05252", "#16A34A", "#2563EB"],
    "bg"  : "#F9FAFB",
    "ga"  : 0.25,
    "dpi" : 200,
}

def _fig(title, figsize=(9,5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_S["bg"]); ax.set_facecolor(_S["bg"])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, alpha=_S["ga"])
    return fig, ax

def _save_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=_S["dpi"], bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")

# Plot 1–3: Error per axis 
def _plot_error(df, key, label, has_truth, out_dir, prefix):
    t = df["timestamp"].values
    fig, ax = _fig(f"{label} Error vs RTK Ground Truth")
    ax.plot(t, df[f"err_raw_{key}"], color=_S["raw"], alpha=0.45, lw=1.2,
            label=f"Raw   RMS={_rms(df[f'err_raw_{key}'].values, has_truth):.3f} m")
    ax.plot(t, df[f"err_kf_{key}"],  color=_S["kf"],  lw=2.0,
            label=f"KF    RMS={_rms(df[f'err_kf_{key}'].values, has_truth):.3f} m")
    ax.axhline(0, color="#555", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Error [m]"); ax.legend(fontsize=9)
    _save_show(fig, os.path.join(out_dir, f"{prefix}error_{key}.png"))

#  Plot 4: Adaptive R 
def _plot_R(df, out_dir, prefix):
    t = df["timestamp"].values
    fig, ax = _fig("Adaptive R — Measurement Noise Std Dev")
    for lbl, key, col in zip(["σ_x","σ_y","σ_z"], ["R_x","R_y","R_z"], _S["rc"]):
        ax.plot(t, np.sqrt(df[key]), color=col, lw=1.8, label=f"{lbl} = √{key}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("√R  [m]"); ax.legend(fontsize=9)
    _save_show(fig, os.path.join(out_dir, f"{prefix}adaptive_R.png"))

# Plot 5: 2D Trajectory 
def _plot_2d(df, has_truth, out_dir, prefix):
    fig, ax = _fig("2D Trajectory (East – North)", figsize=(7,7))
    ax.plot(df["raw_x"], df["raw_y"], ".", color=_S["raw"], alpha=0.35, ms=3, label="Raw")
    ax.plot(df["kf_x"],  df["kf_y"],  "-", color=_S["kf"],  lw=2.2, label="KF")
    if has_truth.any():
        ax.plot(df["x_true"][has_truth], df["y_true"][has_truth],
                ".", color=_S["rtk"], ms=5, alpha=0.7, label="RTK truth")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.axis("equal"); ax.legend(fontsize=9)
    _save_show(fig, os.path.join(out_dir, f"{prefix}2d_trajectory.png"))

#  Plot 6: RMS bar chart 
def _plot_rms_bar(df, has_truth, out_dir, prefix):
    lbls    = ["X","Y","Z","2D","3D"]
    raw_rms = [_rms(df[f"err_raw_{a}"].values, has_truth) for a in ["x","y","z"]] + \
              [_rms(df["err_raw_2d"].values, has_truth), _rms(df["err_raw_3d"].values, has_truth)]
    kf_rms  = [_rms(df[f"err_kf_{a}"].values, has_truth) for a in ["x","y","z"]] + \
              [_rms(df["err_kf_2d"].values, has_truth), _rms(df["err_kf_3d"].values, has_truth)]
    fig, ax = _fig("RMS Error vs RTK  (blue = improvement)")
    xp = np.arange(len(lbls)); w = 0.35
    ax.bar(xp-w/2, raw_rms, w, color=_S["raw"], alpha=0.75, label="Raw")
    bars = ax.bar(xp+w/2, kf_rms, w, alpha=0.85, label="KF")
    for bar,(r,k) in zip(bars, zip(raw_rms, kf_rms)):
        bar.set_color(_S["kf"] if k<=r else _S["raw"])
    for i,(r,k) in enumerate(zip(raw_rms, kf_rms)):
        if r and r>0:
            imp=(r-k)/r*100
            ax.text(i, max(r,k)+0.3, f"{imp:+.0f}%", ha="center",
                    fontsize=9, fontweight="bold", color="navy" if imp>=0 else "darkred")
    ax.axhline(10, color=_S["rtk"], ls="--", lw=1.5, label="10 m target")
    ax.set_xticks(xp); ax.set_xticklabels(lbls)
    ax.set_ylabel("RMS Error [m]"); ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save_show(fig, os.path.join(out_dir, f"{prefix}rms_bar.png"))

#  Plot 7: Quality scale 
def _plot_quality(df, out_dir, prefix):
    t = df["timestamp"].values
    fig, ax = _fig("GNSS Quality Scale per Epoch")
    ax.plot(t, df["quality_scale"], color="#9333EA", lw=1.4, alpha=0.85)
    ax.axhline(1.0, color="#555", ls="--", lw=0.8, alpha=0.6, label="Scale = 1")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Quality Scale"); ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save_show(fig, os.path.join(out_dir, f"{prefix}quality_scale.png"))

#  Plot 8: 3D error histogram 
def _plot_hist(df, has_truth, out_dir, prefix):
    e_raw = df["err_raw_3d"].values[has_truth]; e_raw = e_raw[~np.isnan(e_raw)]
    e_kf  = df["err_kf_3d"].values[has_truth];  e_kf  = e_kf[~np.isnan(e_kf)]
    fig, ax = _fig("3D Error Distribution vs RTK")
    if len(e_raw) > 0:
        bins = np.linspace(0, max(np.nanmax(e_raw), np.nanmax(e_kf)), 40)
        ax.hist(e_raw, bins=bins, alpha=0.5,  color=_S["raw"], density=True, label="Raw")
        ax.hist(e_kf,  bins=bins, alpha=0.75, color=_S["kf"],  density=True, label="KF")
    ax.set_xlabel("3D Error [m]"); ax.set_ylabel("Density"); ax.legend(fontsize=9)
    _save_show(fig, os.path.join(out_dir, f"{prefix}error_histogram.png"))

#  Plot 9: Speed |V| over time 
def _plot_speed(df, out_dir, prefix):
    t        = df["timestamp"].values
    speed    = np.sqrt(df["kf_Vx"]**2 + df["kf_Vy"]**2 + df["kf_Vz"]**2)
    speed_2d = np.sqrt(df["kf_Vx"]**2 + df["kf_Vy"]**2)
    fig, ax  = _fig("Speed |V| over Time")
    ax.plot(t, speed,    color=_S["kf"],  lw=2.2, label="|V| 3D  = √(Vx²+Vy²+Vz²)")
    ax.plot(t, speed_2d, color="#9333EA", lw=1.5, ls="--", label="|V| 2D horizontal = √(Vx²+Vy²)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Speed  [m/s]"); ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save_show(fig, os.path.join(out_dir, f"{prefix}speed.png"))

# Plot 10: Acceleration magnitude |a| over time 
def _plot_accel(df, out_dir, prefix):
    t      = df["timestamp"].values
    a_mag  = np.sqrt(df["kf_ax"]**2 + df["kf_ay"]**2 + df["kf_az"]**2)
    fig, ax = _fig("Acceleration over Time")
    ax.plot(t, df["kf_ax"], color="#E05252", lw=1.5, label="ax (East)")
    ax.plot(t, df["kf_ay"], color="#16A34A", lw=1.5, label="ay (North)")
    ax.plot(t, df["kf_az"], color="#2563EB", lw=1.5, label="az (Up)")
    ax.plot(t, a_mag,        color="#F97316", lw=2.2, ls="--", label="|a| magnitude")
    ax.axhline(0, color="#555", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Acceleration  [m/s²]"); ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save_show(fig, os.path.join(out_dir, prefix + "acceleration.png"))

#  Master function 
def plot_results_isolated(df, out_dir="kf_plots", prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    has_truth = ~np.isnan(df["x_true"].values)
    print("=" * 60)
    print("  Saving 10 isolated plots -> " + os.path.abspath(out_dir))
    print("  Close each window to continue to the next plot")
    print("=" * 60)
    for key, label in [("x","X (East)"),("y","Y (North)"),("z","Z (Up)")]:
        _plot_error(df, key, label, has_truth, out_dir, prefix)
    _plot_R(df, out_dir, prefix)
    _plot_2d(df, has_truth, out_dir, prefix)
    _plot_rms_bar(df, has_truth, out_dir, prefix)
    _plot_quality(df, out_dir, prefix)
    _plot_hist(df, has_truth, out_dir, prefix)
    _plot_speed(df, out_dir, prefix)
    _plot_accel(df, out_dir, prefix)
    print("All 10 plots saved to: " + os.path.abspath(out_dir))


# MAIN  


if __name__ == "__main__":

    SMARTPHONE_FILE = r"C:\project\gnss_outputwith_RTK.xlsx"
    RTK_FILE        = r"C:\Project\epoch_RTKแบบเคลื่อนที่.xlsx"
    OUTPUT_FILE     = "kf_results.xlsx"

    SIGMA_A        = 0.5
    ALPHA          = 0.05
    WINDOW         = 20
    TIME_TOLERANCE = 2.0
    
    # ← KEY FIX: Lower Q for X (trust measurements more)
    Q_SCALE_X = 0.2    # 0.2 = trust X measurements 5x more than model
    Q_SCALE_Y = 1.0    # 1.0 = normal (Y is working fine)
    Q_SCALE_Z = 1.0    # 1.0 = normal (Z is working fine)
    
    # Plot settings
    PLOT_DIR    = "kf_plots"
    PLOT_PREFIX = "fixed_"

    try:
        df_results, kf = run_adaptive_kalman(
            smartphone_file    = SMARTPHONE_FILE,
            rtk_file           = RTK_FILE,
            output_file        = OUTPUT_FILE,
            sigma_a            = SIGMA_A,
            alpha              = ALPHA,
            window             = WINDOW,
            time_tolerance_sec = TIME_TOLERANCE,
            q_scale_x          = Q_SCALE_X,
            q_scale_y          = Q_SCALE_Y,
            q_scale_z          = Q_SCALE_Z
        )

        print("\nGenerating isolated plots...")
        plot_results_isolated(df_results, out_dir=PLOT_DIR, prefix=PLOT_PREFIX)

        print(f"\n{'='*70}")
        print(f"   Complete!")
        print(f"  Excel: {OUTPUT_FILE}")
        print(f"  Plots: {PLOT_DIR}/")
        print(f"  Try different Q_SCALE_X values if X still not improved:")
        print(f"    Q_SCALE_X=0.1 → trust X measurements 10x more (aggressive)")
        print(f"    Q_SCALE_X=0.5 → moderate")
        print(f"    Q_SCALE_X=2.0 → trust X model more (if measurements noisy)")
        print(f"{'='*70}\n")

    except FileNotFoundError as e:
        print(f"\n File not found: {e}")
    except Exception as e:
        import traceback
        print(f"\n Error: {e}")
        traceback.print_exc()