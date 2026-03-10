import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants

LIGHTSPEED = 299792458.0  
EARTH_RADIUS = 6371000.0  
OMEGA_E = 7.2921151467e-5

def convert_utc_to_thai_time(utc_time_millis):
    if pd.isna(utc_time_millis):
        return None
    utc_time_sec = utc_time_millis / 1000.0
    dt_utc = datetime.utcfromtimestamp(utc_time_sec)
    dt_thai = dt_utc + timedelta(hours=7)
    return dt_thai

def calculate_elevation_angle(sat_pos, rcv_pos):
    los_vector = sat_pos - rcv_pos
    up_vector = rcv_pos / np.linalg.norm(rcv_pos)
    cos_angle = np.dot(los_vector, up_vector) / (np.linalg.norm(los_vector) * np.linalg.norm(up_vector))
    elevation = np.pi/2 - np.arccos(np.clip(cos_angle, -1, 1))
    return elevation

def ecef_to_lla(x, y, z):
    if np.isnan(x) or np.isnan(y) or np.isnan(z):
        return np.nan, np.nan, np.nan
    a = 6378137.0
    e2 = 6.6943799901377997e-3
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    return lat, lon, alt

def lla_to_ecef(lat, lon, alt):
    a = 6378137.0
    e2 = 6.6943799901377997e-3
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])


# TROPOSPHERIC MODEL


def tropospheric_saastamoinen(elevation, height=0, pressure_mbar=1013.25, 
                               temperature_K=293.15, humidity_percent=50):
    if elevation < np.radians(5):
        return 0.0
    
    lat_rad = np.radians(13.727)
    
    if height > 0:
        T = temperature_K - 0.0065 * height
        P = pressure_mbar * np.exp(-height / 8435.0)
    else:
        T = temperature_K
        P = pressure_mbar
    
    T_celsius = T - 273.15
    e_s = 6.1094 * np.exp((17.625 * T_celsius) / (T_celsius + 243.04))
    e = (humidity_percent / 100.0) * e_s
    
    zenith_dry = (0.002277 * P) / (1 - 0.00266 * np.cos(2 * lat_rad) - 0.00028 * height / 1000.0)
    zenith_wet = 0.002277 * ((1255.0 / T) + 0.05) * e
    
    sin_E = np.sin(elevation)
    mapping_dry = 1.001 / np.sqrt(0.002001 + sin_E**2)
    mapping_wet = mapping_dry
    
    tropo_delay = zenith_dry * mapping_dry + zenith_wet * mapping_wet
    tropo_delay = np.clip(tropo_delay, 0, 30)
    
    return tropo_delay

# DATA PREPARATION

def prepare_data_enhanced(df):
    df_clean = df.copy()
    print(f"\nInitial: {len(df_clean)} observations")
    
    df_clean = df_clean[df_clean['PseudorangeCorrected_m'].notna()]
    df_clean = df_clean[(df_clean['SvPositionEcefXMeters'].notna()) & 
                        (df_clean['SvPositionEcefYMeters'].notna()) & 
                        (df_clean['SvPositionEcefZMeters'].notna())]
    print(f"After basic checks: {len(df_clean)}")
    
    ranges = np.sqrt(df_clean['SvPositionEcefXMeters']**2 + df_clean['SvPositionEcefYMeters']**2 + df_clean['SvPositionEcefZMeters']**2)
    df_clean = df_clean[(ranges > 2e7) & (ranges < 3e7)]
    
    df_clean = df_clean[(df_clean['PseudorangeCorrected_m'] > 1e7) & (df_clean['PseudorangeCorrected_m'] < 3e7)]
    
    df_clean = df_clean[(df_clean['Cn0DbHz'].notna()) & (df_clean['Cn0DbHz'] >= 26)]
    print(f"After C/N0 >= 26: {len(df_clean)}")
    
    if 'SnrInDb' in df_clean.columns:
        snr_mask = df_clean['SnrInDb'].isna() | (df_clean['SnrInDb'] >= 10)
        df_clean = df_clean[snr_mask]
        print(f"After SNR check: {len(df_clean)}")
    
    if 'State' in df_clean.columns:
        code_lock = (df_clean['State'].astype(int) & 1) == 1
        tow_decoded = (df_clean['State'].astype(int) & 8) == 8
        df_clean = df_clean[code_lock & tow_decoded]
        print(f"After state flags: {len(df_clean)}")
    
    if 'MultipathIndicator' in df_clean.columns:
        print(f"\nMultipath distribution:")
        mp_dist = df_clean['MultipathIndicator'].value_counts().sort_index()
        for val, count in mp_dist.items():
            print(f"  MP={val}: {count} ({100*count/len(df_clean):.1f}%)")
    
    if 'AccumulatedDeltaRangeUncertaintyMeters' in df_clean.columns:
        print(f"\nADR Quality:")
        adr_valid = df_clean['AccumulatedDeltaRangeUncertaintyMeters'].notna()
        if adr_valid.any():
            adr_unc = df_clean.loc[adr_valid, 'AccumulatedDeltaRangeUncertaintyMeters']
            print(f"  Available: {adr_valid.sum()} ({100*adr_valid.sum()/len(df_clean):.1f}%)")
            print(f"  Mean uncertainty: {adr_unc.mean():.3f} m")
            print(f"  Good (<0.5m): {(adr_unc < 0.5).sum()} ({100*(adr_unc < 0.5).sum()/len(adr_unc):.1f}%)")
    
    df_clean['Epoch'] = pd.factorize(df_clean['utcTimeMillis'])[0]
    
    epoch_counts = df_clean.groupby('Epoch').size()
    valid_epochs = epoch_counts[epoch_counts >= 6].index
    df_clean = df_clean[df_clean['Epoch'].isin(valid_epochs)]
    
    print(f"\nFinal: {len(df_clean)} obs, {df_clean['Epoch'].nunique()} epochs")
    if df_clean['Epoch'].nunique() > 0:
        counts = epoch_counts[epoch_counts >= 6]
        print(f"Sats per epoch: {counts.min():.0f}-{counts.max():.0f} (mean: {counts.mean():.1f})")
    
    return df_clean


#  POSITIONING


def get_initial_guess(xs):
    sat_center = np.mean(xs, axis=0)
    sat_distance = np.linalg.norm(sat_center)
    x0 = sat_center * (EARTH_RADIUS / sat_distance)
    return x0

def robust_single_diff_wls(xs, measured_pr, weights, x0, rcv_height=0, max_iterations=30):
    n_sats = len(xs)
    if n_sats < 6:
        raise ValueError("Need >=6 satellites")
    
    elevations = np.array([calculate_elevation_angle(xs[i], x0) for i in range(n_sats)])
    ref_idx = np.argmax(elevations)
    
    mask = np.arange(n_sats) != ref_idx
    xs_diff = xs[mask]
    
    pr_ref = measured_pr[ref_idx]
    pr_diff = measured_pr[mask] - pr_ref
    
    weights_diff = weights[mask] * weights[ref_idx]
    weights_diff = weights_diff / np.mean(weights_diff)
    
    pressure_mbar = 1013.25
    temperature_K = 306.15
    humidity_percent = 75
    
    dx = 100 * np.ones(3)
    iterations = 0
    W = np.diag(weights_diff)
    
    while np.linalg.norm(dx) > 1e-5 and iterations < max_iterations:
        r_ref = np.linalg.norm(xs[ref_idx] - x0)
        r_diff = np.linalg.norm(xs_diff - x0, axis=1)
        
        elev_ref = calculate_elevation_angle(xs[ref_idx], x0)
        tropo_ref = tropospheric_saastamoinen(elev_ref, rcv_height, 
                                              pressure_mbar, temperature_K, humidity_percent)
        
        tropo_diff = np.zeros(len(xs_diff))
        for i in range(len(xs_diff)):
            elev_i = calculate_elevation_angle(xs_diff[i], x0)
            tropo_diff[i] = tropospheric_saastamoinen(elev_i, rcv_height,
                                                      pressure_mbar, temperature_K, humidity_percent)
        
        pr_ref_corr = pr_ref + tropo_ref
        pr_diff_corr = pr_diff + (tropo_diff - tropo_ref)
        
        rho_diff = r_diff - r_ref
        deltaP = pr_diff_corr - rho_diff
        
        los_ref = (xs[ref_idx] - x0) / r_ref
        G = np.zeros((len(xs_diff), 3))
        for i in range(len(xs_diff)):
            los_i = (xs_diff[i] - x0) / r_diff[i]
            G[i, :] = los_ref - los_i
        
        GTW = G.T @ W
        GTWG = GTW @ G
        GTWG += 1e-8 * np.eye(3)
        GTWdeltaP = GTW @ deltaP
        
        try:
            dx = np.linalg.solve(GTWG, GTWdeltaP)
        except:
            dx = np.linalg.lstsq(GTWG, GTWdeltaP, rcond=None)[0]
        
        if iterations > 20:
            dx *= 0.5
        
        x0 = x0 + dx
        iterations += 1
    
    r_ref = np.linalg.norm(xs[ref_idx] - x0)
    r_diff = np.linalg.norm(xs_diff - x0, axis=1)
    rho_diff = r_diff - r_ref
    
    elev_ref = calculate_elevation_angle(xs[ref_idx], x0)
    tropo_ref = tropospheric_saastamoinen(elev_ref, rcv_height, pressure_mbar, temperature_K, humidity_percent)
    
    tropo_diff = np.zeros(len(xs_diff))
    for i in range(len(xs_diff)):
        elev_i = calculate_elevation_angle(xs_diff[i], x0)
        tropo_diff[i] = tropospheric_saastamoinen(elev_i, rcv_height, pressure_mbar, temperature_K, humidity_percent)
    
    pr_ref_corr = pr_ref + tropo_ref
    pr_diff_corr = pr_diff + (tropo_diff - tropo_ref)
    
    residuals = pr_diff_corr - rho_diff
    rms_residual = np.sqrt(np.mean(residuals**2))
    
    try:
        Q = np.linalg.inv(G.T @ W @ G)
        hdop = np.sqrt(Q[0,0] + Q[1,1])
        vdop = np.sqrt(Q[2,2])
        pdop = np.sqrt(np.trace(Q))
    except:
        hdop = np.nan
        vdop = np.nan
        pdop = np.nan
    
    avg_tropo = np.mean(tropo_diff)
    
    return x0, rms_residual, iterations, pdop, hdop, vdop, residuals, ref_idx, avg_tropo

# MAIN PROCESSING

def process_enhanced(df_clean, verbose=False):
    if 'utcTimeMillis' in df_clean.columns:
        df_clean['ThaiTime'] = df_clean['utcTimeMillis'].apply(convert_utc_to_thai_time)
    
    epochs = sorted(df_clean['Epoch'].unique())
    results = []
    
    print(f"\nProcessing {len(epochs)} epochs...")
    
    for idx, epoch in enumerate(epochs):
        epoch_data = df_clean[df_clean['Epoch'] == epoch].copy()
        
        if verbose and idx < 3:
            print(f"\nEpoch {epoch} ({idx+1}/{len(epochs)}): {len(epoch_data)} sats")
        
        xs = epoch_data[['SvPositionEcefXMeters', 'SvPositionEcefYMeters', 'SvPositionEcefZMeters']].values.astype(float)
        measured_pr = epoch_data['PseudorangeCorrected_m'].values.astype(float)
        
        valid_mask = ~(np.isnan(xs).any(axis=1) | np.isnan(measured_pr))
        xs_clean = xs[valid_mask]
        measured_pr_clean = measured_pr[valid_mask]
        epoch_data_clean = epoch_data[valid_mask].copy()
        
        if len(xs_clean) < 6:
            continue
        
        x0 = get_initial_guess(xs_clean)
        
        elevation_angles = np.array([calculate_elevation_angle(xs_clean[i], x0)
                                     for i in range(len(xs_clean))])
        
        w_elev = np.sin(elevation_angles)**4
        
        cn0 = epoch_data_clean['Cn0DbHz'].values
        cn0_norm = np.clip((cn0 - 28) / 15, 0, 1)
        w_cn0 = np.exp(2.5 * cn0_norm)
        
        if 'SnrInDb' in epoch_data_clean.columns:
            snr = epoch_data_clean['SnrInDb'].values
            w_snr = np.ones(len(snr))
            valid_snr = ~np.isnan(snr)
            if np.any(valid_snr):
                snr_norm = np.clip((snr[valid_snr] - 15) / 20, 0, 1)
                w_snr[valid_snr] = np.exp(snr_norm)
        else:
            w_snr = np.ones(len(xs_clean))
        
        if 'MultipathIndicator' in epoch_data_clean.columns:
            mp = epoch_data_clean['MultipathIndicator'].values
            w_mp = np.ones(len(mp))
            w_mp[mp == 0] = 10.0
            w_mp[mp == 1] = 3.0
            w_mp[mp == 2] = 0.1
        else:
            w_mp = np.ones(len(xs_clean))
        
        if 'AccumulatedDeltaRangeUncertaintyMeters' in epoch_data_clean.columns:
            adr_unc = epoch_data_clean['AccumulatedDeltaRangeUncertaintyMeters'].values
            w_adr = np.ones(len(adr_unc))
            valid_adr = ~np.isnan(adr_unc)
            if np.any(valid_adr):
                w_adr[valid_adr & (adr_unc < 0.5)] = 2.0
                w_adr[valid_adr & (adr_unc >= 0.5)] = 0.5
        else:
            w_adr = np.ones(len(xs_clean))
        
        weights = w_elev * w_cn0 * w_snr * w_mp * w_adr
        weights = weights / np.mean(weights)
        weights = np.maximum(weights, 0.01)
        
        try:
            pos_ecef, rms_res, iters, pdop, hdop, vdop, residuals, ref_idx, avg_tropo = \
                robust_single_diff_wls(xs_clean, measured_pr_clean, weights, x0, rcv_height=0)
            
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            
            if mad > 1e-6:
                z_scores = 0.6745 * (residuals - median_res) / mad
                outlier_mask = np.abs(z_scores) < 2.5
                
                full_mask = np.ones(len(xs_clean), dtype=bool)
                full_mask[ref_idx] = True
                non_ref_indices = np.where(np.arange(len(xs_clean)) != ref_idx)[0]
                full_mask[non_ref_indices] = outlier_mask
                
                n_outliers = np.sum(~full_mask)
                
                if n_outliers > 0 and np.sum(full_mask) >= 6:
                    xs_final = xs_clean[full_mask]
                    pr_final = measured_pr_clean[full_mask]
                    weights_final = weights[full_mask]
                    weights_final = weights_final / np.mean(weights_final)
                    
                    x0 = get_initial_guess(xs_final)
                    
                    pos_ecef, rms_res, iters, pdop, hdop, vdop, _, _, avg_tropo = \
                        robust_single_diff_wls(xs_final, pr_final, weights_final, x0, rcv_height=0)
                    
                    xs_clean = xs_final
                    elevation_angles = np.array([calculate_elevation_angle(xs_clean[i], pos_ecef)
                                               for i in range(len(xs_clean))])
                    cn0 = cn0[full_mask]
                    
                    if verbose and idx < 3:
                        print(f"  Removed {n_outliers} outliers")
            
            if pdop > 3.5 or rms_res > 20:
                if verbose and idx < 3:
                    print(f"  Rejected: PDOP={pdop:.2f}, RMS={rms_res:.2f}m")
                continue
            
            lat_rad, lon_rad, alt = ecef_to_lla(pos_ecef[0], pos_ecef[1], pos_ecef[2])
            lat = np.degrees(lat_rad)
            lon = np.degrees(lon_rad)
            
            if np.isnan(lat) or np.isnan(lon) or np.isnan(alt):
                continue
            
            if abs(lat) > 90 or abs(lon) > 180 or alt < -500 or alt > 10000:
                continue
            
            result = {
                'Epoch': epoch,
                'X_ECEF': pos_ecef[0],
                'Y_ECEF': pos_ecef[1],
                'Z_ECEF': pos_ecef[2],
                'Latitude': lat,
                'Longitude': lon,
                'Altitude': alt,
                'RMS_Residual': rms_res,
                'PDOP': pdop,
                'HDOP': hdop,
                'VDOP': vdop,
                'NumSatellites': len(xs_clean),
                'MeanCNo': np.mean(cn0),
                'MeanElevation': np.degrees(np.mean(elevation_angles)),
                'Iterations': iters,
                'Tropo_m': avg_tropo
            }
            
            results.append(result)
            
            if verbose and idx < 3:
                print(f"  {lat:.6f}, {lon:.6f}, {alt:.1f}m")
                print(f"    RMS={rms_res:.2f}m, HDOP={hdop:.2f}, Tropo={avg_tropo:.2f}m")
        
        except Exception as e:
            if verbose and idx < 3:
                print(f"  Error: {e}")
            continue
    
    if len(results) == 0:
        raise ValueError("No epochs processed!")
    
    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results)}/{len(epochs)} epochs ({len(results)/len(epochs)*100:.1f}%)")
    
    return results_df

def calculate_errors(results_df, ref_lat=13.727285, ref_lon=100.776424, ref_alt=-15.9):
    ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    
    results_df['Error_3D'] = np.sqrt(
        (results_df['X_ECEF'] - ref_ecef[0])**2 +
        (results_df['Y_ECEF'] - ref_ecef[1])**2 +
        (results_df['Z_ECEF'] - ref_ecef[2])**2
    )
    
    R_earth = 6371000
    results_df['Error_Lat'] = (results_df['Latitude'] - ref_lat) * (np.pi/180) * R_earth
    results_df['Error_Lon'] = (results_df['Longitude'] - ref_lon) * (np.pi/180) * R_earth * np.cos(np.radians(ref_lat))
    results_df['Error_2D'] = np.sqrt(results_df['Error_Lat']**2 + results_df['Error_Lon']**2)
    results_df['Error_Alt'] = results_df['Altitude'] - ref_alt
    
    return results_df


# MAP PLOTTING  


def plot_gnss_map(results_df, ref_lat=13.727285, ref_lon=100.776424, output_file='gnss_map_results.html'):
    import folium
    import webbrowser
    import os

    df = results_df[['Latitude', 'Longitude', 'Altitude', 'Error_2D']].copy()
    df.columns = ['latitude', 'longitude', 'altitude', 'distance_from_ref']

    mean_err  = df['distance_from_ref'].mean()
    med_err   = df['distance_from_ref'].median()
    max_err   = df['distance_from_ref'].max()
    min_err   = df['distance_from_ref'].min()

    print(f"\nMap Statistics:")
    print(f"  Total epochs : {len(df)}")
    print(f"  Error mean   : {mean_err:.1f} m")
    print(f"  Error median : {med_err:.1f} m")
    print(f"  Error max    : {max_err:.1f} m")
    print(f"  Error min    : {min_err:.1f} m")

    # สร้างแผนที่ 
    m = folium.Map(location=[ref_lat, ref_lon], zoom_start=17, tiles='OpenStreetMap')

    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False, control=True
    ).add_to(m)

  
    # สร้าง circle 
    step = 5   
    max_circle = max(int(np.ceil(max_err / step)) * step, 15)
    circle_radii = list(range(step, max_circle + step, step))

    circle_colors = ['#e74c3c', '#e67e22', '#f1c40f', "#139549", '#3498db',
                     '#9b59b6', '#1abc9c', '#e74c3c', '#e67e22', '#f1c40f']

    for i, r in enumerate(circle_radii):
        color = circle_colors[i % len(circle_colors)]
        folium.Circle(
            location=[ref_lat, ref_lon],
            radius=r,
            color=color,
            fill=False,
            weight=1.5,
            opacity=0.6,
            tooltip=f'{r} m'
        ).add_to(m)
        #  วงกลม
        folium.Marker(
            location=[ref_lat + r / 111320, ref_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px; color:{color}; font-weight:bold; white-space:nowrap;">{r} m</div>',
                icon_size=(40, 12),
                icon_anchor=(0, 6)
            )
        ).add_to(m)

    # จุดอ้างอิง 
    folium.Marker(
        location=[ref_lat, ref_lon],
        popup=f'<b>Reference Point</b><br>Lat: {ref_lat:.6f}<br>Lon: {ref_lon:.6f}',
        tooltip='Reference Point',
        icon=folium.Icon(color='orange', icon='star', prefix='fa')
    ).add_to(m)

    # Plot epoch 
    # สี
    for i, row in df.iterrows():
        dist = row['distance_from_ref']
        if dist < 10:
            color = 'green';     fillcolor = 'lightgreen'
        elif dist < 20:
            color = 'blue';      fillcolor = 'lightblue'
        elif dist < 50:
            color = 'orange';    fillcolor = 'yellow'
        else:
            color = 'red';       fillcolor = 'pink'

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=(f'<b>Epoch #{i+1}</b><br>'
                   f'Lat: {row["latitude"]:.6f}<br>'
                   f'Lon: {row["longitude"]:.6f}<br>'
                   f'Alt: {row["altitude"]:.1f} m<br>'
                   f'Error 2D: {dist:.1f} m'),
            tooltip=f'Epoch {i+1} | err={dist:.1f} m',
            color=color,
            fill=True,
            fillColor=fillcolor,
            fillOpacity=0.75,
            weight=2
        ).add_to(m)

    # เส้นเชื่อม 
    coords = df[['latitude', 'longitude']].values.tolist()
    folium.PolyLine(coords, color='blue', weight=1.5, opacity=0.3, tooltip='Path').add_to(m)

  
    legend_html = f'''
    <div style="position:fixed; top:10px; right:10px; width:230px;
                background:white; border:2px solid grey; z-index:9999;
                padding:12px; border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
      <h4 style="margin:0 0 8px 0; border-bottom:2px solid #333;">GNSS WLS Results</h4>
      <p style="margin:4px 0;"><span style="color:orange; font-size:16px;">&#9733;</span> Reference Point</p>
      <p style="margin:4px 0;"><span style="color:green;  font-size:14px;">&#9679;</span> Error &lt; 10 m</p>
      <p style="margin:4px 0;"><span style="color:blue;   font-size:14px;">&#9679;</span> Error 10&#8211;20 m</p>
      <p style="margin:4px 0;"><span style="color:orange; font-size:14px;">&#9679;</span> Error 20&#8211;50 m</p>
      <p style="margin:4px 0;"><span style="color:red;    font-size:14px;">&#9679;</span> Error &gt; 50 m</p>
      <hr style="margin:8px 0;">
      <p style="margin:3px 0; font-size:11px;"><b>Statistics:</b></p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Total  : {len(df)} epochs</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Mean   : {mean_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Median : {med_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Max    : {max_err:.1f} m</p>
      <p style="margin:2px 0 2px 8px; font-size:10px;">Min    : {min_err:.1f} m</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"\nMap saved: {output_file}")
    print(f"File size: {os.path.getsize(output_file)/1024:.1f} KB")

    webbrowser.open('file://' + os.path.abspath(output_file))


# MAIN


if __name__ == "__main__":
    input_file  = r"C:\project\gnss_pseudorange_sppแบบมือถือนิ่ง.xlsx"
    output_xlsx = r"C:\Users\user\Downloads\gnss_results_FINAL.xlsx"
    output_map  = r"C:\project\gnss_map_results.html"

    try:
        print(f"\nReading: {input_file}")
        df_raw = pd.read_excel(input_file)
        print(f"Loaded: {len(df_raw)} observations")

        df_clean = prepare_data_enhanced(df_raw)

        if len(df_clean) == 0:
            print("\nNo data after filtering")
        else:
            results = process_enhanced(df_clean, verbose=True)
            results = calculate_errors(results)
            results.to_excel(output_xlsx, index=False)

            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)

            print(f"\nEpochs: {len(results)}")
            print(f"\nPosition:")
            print(f"  Ref:  {13.727285:.6f}N, {100.776424:.6f}E, {-15.9:.1f}m")
            print(f"  Mean: {results['Latitude'].mean():.6f}N, {results['Longitude'].mean():.6f}E, {results['Altitude'].mean():.1f}m")

            mean_2d = results['Error_2D'].mean()
            print(f"\n2D ERROR:")
            print(f"  Mean   : {mean_2d:.2f} m")
            print(f"  Median : {results['Error_2D'].median():.2f} m")
            print(f"  Std    : {results['Error_2D'].std():.2f} m")
            print(f"  68%    : {np.percentile(results['Error_2D'], 68):.2f} m")
            print(f"  95%    : {np.percentile(results['Error_2D'], 95):.2f} m")

            print(f"\n3D & ALT:")
            print(f"  3D  : {results['Error_3D'].mean():.2f} +/- {results['Error_3D'].std():.2f} m")
            print(f"  Alt : {results['Error_Alt'].mean():.2f} +/- {results['Error_Alt'].std():.2f} m")

            print(f"\nTropospheric (Saastamoinen):")
            print(f"  Mean  : {results['Tropo_m'].mean():.2f} m")
            print(f"  Range : {results['Tropo_m'].min():.2f} - {results['Tropo_m'].max():.2f} m")

            print(f"\nQuality:")
            print(f"  RMS  : {results['RMS_Residual'].mean():.2f} m")
            print(f"  HDOP : {results['HDOP'].mean():.2f}")
            print(f"  PDOP : {results['PDOP'].mean():.2f}")
            print(f"  Sats : {results['NumSatellites'].mean():.1f} +/- {results['NumSatellites'].std():.1f}")
            print(f"  C/N0 : {results['MeanCNo'].mean():.1f} dB-Hz")

            print(f"\nSaved: {output_xlsx}")

           
            plot_gnss_map(results, ref_lat=13.727285, ref_lon=100.776424, output_file=output_map)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()