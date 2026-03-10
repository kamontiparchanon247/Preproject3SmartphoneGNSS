
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt



INPUT_FILE = r"C:\project\gnss_outputwith_RTK.xlsx"
GT_FILE    = r"C:\Project\epoch_RTKแบบเคลื่อนที่.xlsx"

LAT0 = 13.7271932
LON0 = 100.7772124
ALT0 = 16.01

TIME_TOLERANCE = 2.0   



# 1. LOAD DATA


def load_data(filepath):
    df = pd.read_excel(filepath)
    df["UTC_Time"] = pd.to_datetime(df["UTC_Time"], errors="coerce")
    df = df.dropna(subset=["UTC_Time"])
    df = df.sort_values("UTC_Time").reset_index(drop=True)
    return df


def compute_sampling_rate(df):
    dt = df["UTC_Time"].diff().dt.total_seconds().dropna()
    median_dt = dt.median()
    fs = 1.0 / median_dt
    print(f"[Data]  Sampling rate: {fs:.4f} Hz  (median dt = {median_dt:.4f} s)")
    return fs



# 2. COORDINATE CONVERSION


WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    alt = np.asarray(alt_m, dtype=float)
    N   = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    X   = (N + alt) * np.cos(lat) * np.cos(lon)
    Y   = (N + alt) * np.cos(lat) * np.sin(lon)
    Z   = (N * (1 - WGS84_E2) + alt) * np.sin(lat)
    return X, Y, Z


def ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0):
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    N0     = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat0_r) ** 2)
    x0     = (N0 + alt0) * np.cos(lat0_r) * np.cos(lon0_r)
    y0     = (N0 + alt0) * np.cos(lat0_r) * np.sin(lon0_r)
    z0     = (N0 * (1 - WGS84_E2) + alt0) * np.sin(lat0_r)
    dx = np.asarray(x_ecef) - x0
    dy = np.asarray(y_ecef) - y0
    dz = np.asarray(z_ecef) - z0
    sin_lat = np.sin(lat0_r); cos_lat = np.cos(lat0_r)
    sin_lon = np.sin(lon0_r); cos_lon = np.cos(lon0_r)
    e =  -sin_lon * dx + cos_lon * dy
    n =  -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u =   cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return e, n, u



# 3. BUTTERWORTH LPF  ← KEY FIX HERE


def design_butterworth_lpf(fs, cutoff_hz=None, order=4):
    if cutoff_hz is None:
        cutoff_hz = fs / 5.0
        print(f"[LPF]  Auto cutoff: {cutoff_hz:.4f} Hz  (fs/5 rule)")
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(f"Cutoff {cutoff_hz:.4f} Hz must be < Nyquist {nyq:.4f} Hz")
    sos = butter(order, cutoff_hz / nyq, btype='low', analog=False, output='sos')
    print(f"[LPF]  Butterworth order={order}, cutoff={cutoff_hz:.4f} Hz, fs={fs:.4f} Hz")
    return sos


def apply_lpf_extended(sos, data, fs, cutoff_hz, order):

   
    settling = int(np.ceil(order * (fs / cutoff_hz)))
    ext      = max(100, 3 * settling)
    n        = len(data) if data.ndim == 1 else data.shape[0]
    print(f"[LPF]  Extension = {ext} samples each side  (data length={n})")

    def _extend_and_filter(x):
        # Linear extrapolation at START (step back from x[0])
        slope_head = x[1] - x[0]
        head = x[0] - slope_head * np.arange(ext, 0, -1)

        # Linear extrapolation at END (step forward from x[-1])
        slope_tail = x[-1] - x[-2]
        tail = x[-1] + slope_tail * np.arange(1, ext + 1)

        x_ext = np.concatenate([head, x, tail])   
        y_ext = sosfiltfilt(sos, x_ext)
        return y_ext[ext: ext + n]                 

    if data.ndim == 1:
        return _extend_and_filter(data)
    return np.column_stack([_extend_and_filter(data[:, k])
                            for k in range(data.shape[1])])



# 4. LOAD & MERGE


def load_and_merge(wls_enu, df_wls, df_gt):
    if "Fix_Quality" in df_gt.columns:
        rtk_good = df_gt[df_gt["Fix_Quality"].isin([4, 5])].copy()
        print(f"[GT]    RTK fixed/float epochs: {len(rtk_good)} / {len(df_gt)}")
        if len(rtk_good) == 0:
            print("[GT]     No Fix_Quality 4/5 found — using all RTK epochs")
            rtk_good = df_gt.copy()
    else:
        print("[GT]     No Fix_Quality column — using all RTK epochs")
        rtk_good = df_gt.copy()

    rx, ry, rz = lla_to_ecef(rtk_good["Latitude"].values,
                              rtk_good["Longitude"].values,
                              rtk_good["Altitude_m"].values)
    re, rn, ru = ecef_to_enu(rx, ry, rz, LAT0, LON0, ALT0)
    rtk_enu    = np.column_stack([re, rn, ru])
    rtk_times  = rtk_good["UTC_Time"].values.astype("int64") / 1e9

    wls_times   = df_wls["UTC_Time"].values.astype("int64") / 1e9
    M           = len(df_wls)
    gt_enu_full = np.full((M, 3), np.nan)

    for i, t_wls in enumerate(wls_times):
        diffs = np.abs(rtk_times - t_wls)
        j     = np.argmin(diffs)
        if diffs[j] <= TIME_TOLERANCE:
            gt_enu_full[i] = rtk_enu[j]

    has_truth       = ~np.isnan(gt_enu_full[:, 0])
    matched_wls_enu = wls_enu[has_truth]
    matched_gt_enu  = gt_enu_full[has_truth]

    print(f"[Align] Matched {has_truth.sum()} / {M} WLS epochs to RTK")
    if has_truth.sum() < M * 0.5:
        print("[Align]  Less than 50% matched — check time formats")

    return matched_wls_enu, matched_gt_enu, has_truth, gt_enu_full, rtk_enu



# 5. RMS ERROR


def _rms(arr):
    arr = arr[~np.isnan(arr)]
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else np.nan


def compute_rms_enu(estimated_enu, gt_enu):
    err_e  = estimated_enu[:, 0] - gt_enu[:, 0]
    err_n  = estimated_enu[:, 1] - gt_enu[:, 1]
    err_u  = estimated_enu[:, 2] - gt_enu[:, 2]
    err_2d = np.sqrt(err_e**2 + err_n**2)
    err_3d = np.sqrt(err_e**2 + err_n**2 + err_u**2)
    return {
        "rms_e"  : _rms(err_e),
        "rms_n"  : _rms(err_n),
        "rms_u"  : _rms(err_u),
        "rms_2d" : _rms(err_2d),
        "rms_3d" : _rms(err_3d),
        "mae_e"  : float(np.mean(np.abs(err_e))),
        "mae_n"  : float(np.mean(np.abs(err_n))),
        "mae_u"  : float(np.mean(np.abs(err_u))),
        "err_e"  : err_e,
        "err_n"  : err_n,
        "err_u"  : err_u,
        "err_2d" : err_2d,
        "err_3d" : err_3d,
    }



# 6. PRINT RMS TABLE


def print_rms_table(m_wls, m_lpf, n_matched, n_total):
    print("\n" + "=" * 72)
    print("  RMS ERROR vs RTK GROUND TRUTH  (ENU frame, metres)")
    print(f"  Matched epochs: {n_matched} / {n_total}  (Fix_Quality 4/5 only)")
    print("=" * 72)
    print(f"  {'Axis':<14} {'WLS RMS':>10} {'LPF RMS':>10} {'Improve':>10}"
          f"  {'WLS MAE':>10} {'LPF MAE':>10}")
    print("  " + "-" * 68)
    axes = [("E (East)",   "e"),
            ("N (North)",  "n"),
            ("U (Up)",     "u"),
            ("2D (Horiz)", "2d"),
            ("3D (Full)",  "3d")]
    for label, ax in axes:
        rms_wls = m_wls[f"rms_{ax}"]
        rms_lpf = m_lpf[f"rms_{ax}"]
        improve = (1 - rms_lpf / rms_wls) * 100 if rms_wls > 0 else 0.0
        arrow   = "▼" if improve > 0 else "▲"
        mae_str = ""
        if ax in ("e", "n", "u"):
            mae_str = f"  {m_wls[f'mae_{ax}']:>10.4f} {m_lpf[f'mae_{ax}']:>10.4f}"
        print(f"  {label:<14} {rms_wls:>10.4f} {rms_lpf:>10.4f} "
              f"{arrow}{abs(improve):>8.1f}%{mae_str}")
    print("=" * 72)
    print("  ▼ = LPF reduced error   ▲ = LPF increased error")
    print("=" * 72 + "\n")


# 7. SAVE OUTPUT


def save_output(df, lpf_xyz):
    df_out = df.copy()
    df_out["X_ECEF_LPF"] = lpf_xyz[:, 0]
    df_out["Y_ECEF_LPF"] = lpf_xyz[:, 1]
    df_out["Z_ECEF_LPF"] = lpf_xyz[:, 2]
    out_path = INPUT_FILE.replace(".xlsx", "_lpf_output.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"[Save]  Saved → {out_path}")



# 8. PLOTS


def plot_xyz_signal(df, lpf_xyz):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("WLS Input vs Butterworth LPF — ECEF Coordinates",
                 fontsize=13, fontweight='bold')
    labels = ["X_ECEF [m]", "Y_ECEF [m]", "Z_ECEF [m]"]
    cols   = ["X_ECEF",     "Y_ECEF",     "Z_ECEF"]
    for i, ax in enumerate(axes):
        ax.plot(df[cols[i]].values, color="#e74c3c", lw=0.8,
                alpha=0.65, label="WLS input")
        ax.plot(lpf_xyz[:, i], color="#3498db", lw=1.8, label="WLS + LPF")
        ax.set_ylabel(labels[i]); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out = INPUT_FILE.replace(".xlsx", "_lpf_xyz.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[Plot]  Saved → {out}")
    plt.show()


def plot_trajectory_closed(wls_enu, lpf_enu, rtk_enu=None):
    def close_loop(arr):
        return np.vstack([arr, arr[0:1]])

    wls_c = close_loop(wls_enu)
    lpf_c = close_loop(lpf_enu)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Trajectory — ENU Frame (Closed Loop: Start = Stop)",
                 fontsize=13, fontweight='bold')

    ax.plot(wls_c[:, 0], wls_c[:, 1],
            color="#e74c3c", lw=1.0, alpha=0.6, label="WLS (raw)")
    ax.plot(lpf_c[:, 0], lpf_c[:, 1],
            color="#3498db", lw=1.8, label="WLS + LPF")

    if rtk_enu is not None and len(rtk_enu) > 1:
        rtk_c = close_loop(rtk_enu)
        ax.plot(rtk_c[:, 0], rtk_c[:, 1],
                color="#2ecc71", lw=2.0, ls="--", label="RTK GT")

    start_E, start_N = wls_enu[0, 0], wls_enu[0, 1]
    ax.scatter(start_E, start_N, s=120, zorder=5,
               color="gold", edgecolors="black", linewidths=1.5,
               label=f"Start = Stop  ({start_E:.2f}, {start_N:.2f}) m")

    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    out = INPUT_FILE.replace(".xlsx", "_trajectory_closed.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[Plot]  Saved → {out}")
    plt.show()


def plot_error(m_wls, m_lpf):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("RMS Error per Axis (ENU) — WLS vs WLS+LPF vs RTK",
                 fontsize=13, fontweight='bold')
    C_wls = "#e74c3c"; C_lpf = "#3498db"
    keys   = ["e",        "n",        "u",      "3d"         ]
    labels = ["E (East)", "N (North)", "U (Up)", "3D Combined"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[i // 2][i % 2]
        t  = np.arange(len(m_wls[f"err_{key}"]))
        ax.plot(t, np.abs(m_wls[f"err_{key}"]), color=C_wls, lw=0.8, alpha=0.7,
                label=f"WLS   RMS={m_wls[f'rms_{key}']:.4f} m")
        ax.plot(t, np.abs(m_lpf[f"err_{key}"]), color=C_lpf, lw=1.5,
                label=f"LPF   RMS={m_lpf[f'rms_{key}']:.4f} m")
        improve = (1 - m_lpf[f'rms_{key}'] / m_wls[f'rms_{key}']) * 100
        ax.set_title(f"{label}  (improvement: {improve:+.1f}%)", fontweight='bold')
        ax.set_xlabel("Matched Epoch"); ax.set_ylabel("Absolute Error [m]")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = INPUT_FILE.replace(".xlsx", "_lpf_error_enu.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[Plot]  Saved → {out}")
    plt.show()


def plot_rms_bar(m_wls, m_lpf):
    labels  = ["E (East)", "N (North)", "U (Up)", "2D", "3D"]
    rms_wls = [m_wls["rms_e"], m_wls["rms_n"], m_wls["rms_u"],
               m_wls["rms_2d"], m_wls["rms_3d"]]
    rms_lpf = [m_lpf["rms_e"], m_lpf["rms_n"], m_lpf["rms_u"],
               m_lpf["rms_2d"], m_lpf["rms_3d"]]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, rms_wls, w, label="WLS (before LPF)",
                   color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + w/2, rms_lpf, w, label="WLS + LPF (after)",
                   color="#3498db", alpha=0.8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
    for i, (r_w, r_l) in enumerate(zip(rms_wls, rms_lpf)):
        imp   = (1 - r_l / r_w) * 100
        color = "green" if imp > 0 else "red"
        ax.text(i, max(r_w, r_l) + 0.5, f"{imp:+.1f}%",
                ha='center', fontsize=9, fontweight='bold', color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMS Error [m]")
    ax.set_title("RMS Error Before vs After LPF  (ENU, vs RTK Fixed/Float only)",
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = INPUT_FILE.replace(".xlsx", "_lpf_rms_bar_enu.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[Plot]  Saved → {out}")
    plt.show()



# 9. MAIN PIPELINE


def run_pipeline(cutoff_hz=None, lpf_order=4):
    print("─" * 72)
    print("  GPS Pipeline: WLS → LPF → RMS (ENU) vs RTK")
    print(f"  Reference: Lat={LAT0}° Lon={LON0}° Alt={ALT0}m")
    print("─" * 72)

    df = load_data(INPUT_FILE)
    print(f"[Data]  {len(df)} epochs | columns: {list(df.columns)}")

    fs  = compute_sampling_rate(df)
    xyz = df[["X_ECEF", "Y_ECEF", "Z_ECEF"]].values
    sos = design_butterworth_lpf(fs, cutoff_hz, lpf_order)

    # Actual cutoff for extension length calculation
    actual_cutoff = cutoff_hz if cutoff_hz is not None else fs / 5.0

    # KEY FIX: use extended filter instead of plain sosfiltfilt 
    lpf_xyz = apply_lpf_extended(sos, xyz, fs=fs,
                                 cutoff_hz=actual_cutoff,
                                 order=lpf_order)

    we, wn, wu = ecef_to_enu(xyz[:, 0], xyz[:, 1], xyz[:, 2], LAT0, LON0, ALT0)
    wls_enu    = np.column_stack([we, wn, wu])

    le, ln, lu = ecef_to_enu(lpf_xyz[:, 0], lpf_xyz[:, 1], lpf_xyz[:, 2],
                              LAT0, LON0, ALT0)
    lpf_enu    = np.column_stack([le, ln, lu])

    save_output(df, lpf_xyz)
    plot_xyz_signal(df, lpf_xyz)

    rtk_enu_for_traj = None
    try:
        df_gt = load_data(GT_FILE)
        print(f"[GT]    {len(df_gt)} epochs from '{GT_FILE}'")

        wls_matched, gt_matched, has_truth, _, rtk_enu_all = \
            load_and_merge(wls_enu, df, df_gt)

        rtk_enu_for_traj = rtk_enu_all
        lpf_matched      = lpf_enu[has_truth]

        metrics_wls = compute_rms_enu(wls_matched, gt_matched)
        metrics_lpf = compute_rms_enu(lpf_matched, gt_matched)

        print_rms_table(metrics_wls, metrics_lpf,
                        n_matched=has_truth.sum(), n_total=len(df))
        plot_error(metrics_wls, metrics_lpf)
        plot_rms_bar(metrics_wls, metrics_lpf)

    except FileNotFoundError:
        print(f"\n[GT]  '{GT_FILE}' not found — skipping RMS comparison.")

    plot_trajectory_closed(wls_enu, lpf_enu, rtk_enu=rtk_enu_for_traj)


if __name__ == "__main__":
    run_pipeline(
        cutoff_hz=None,   
        lpf_order=4,
    )
