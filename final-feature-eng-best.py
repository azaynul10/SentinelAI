import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

def sample_entropy(ts, m=2, r=0.2):
    """
    Compute sample entropy of time series ts.
    m: embedding dimension, r: threshold as fraction of std.
    """
    N = len(ts)
    if N <= m + 1:
       return 0.0
    r_val = r * np.std(ts)
    def _count_matches(m_dim):
        x = np.array([ts[i:i+m_dim] for i in range(N - m_dim + 1)])
        count = 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            count += np.sum(dist <= r_val) - 1
        return count
    B = _count_matches(m)
    A = _count_matches(m + 1)
    return float(-np.log((A + 1e-8) / (B + 1e-8)))

def extract_advanced_features(df):
    # Add log-transformed signals
    df['log_com_speed']         = np.log1p(np.abs(df['com_speed'].values))
    df['log_upper_body_angle']  = np.log1p(np.abs(df['torso_angle_deg'].values))

    features = {}
    t        = df['timestamp'].values
    duration = t.max() + 1e-8
    mid_time = duration / 2.0

    # 1. Peak dynamics & timing
    dyn_cols = [
        'com_speed', 'com_acc', 'com_jerk',
        'angular_velocity_deg_s',
        'log_upper_body_angle',
        'log_com_speed'
    ]
    for col in dyn_cols:
        x   = np.abs(df[col].values)
        idx = x.argmax()
        features[f'peak_{col}']   = float(x[idx])
        features[f't_peak_{col}'] = float(t[idx] / duration)

    # 2. 2 Phase-based mean deltas
    first  = df[df['timestamp'] <= mid_time]
    second = df[df['timestamp'] >  mid_time]
    skip  = {'head_hip_gap_norm'}  # allow angular_velocity
    for col in df.columns:
        if col in skip or col == 'timestamp':
            continue
        delta = float(second[col].mean() - first[col].mean())
        features[f'delta_mean_{col}'] = delta

    # 3. Precompute derivatives for core signals (3 segment)
    n   = len(df) 
    seg_size = max(n // 3, 1) 
    deriv = {} #Derivative storing dictionary
    for col in ['com_speed', 'com_acc', 'angular_velocity_deg_s']:
        vals = df[col].values
        d    = np.zeros_like(vals)
        d[:-1] = (vals[1:] - vals[:-1]) / (t[1:] - t[:-1] + 1e-8)
        d[-1]  = d[-2]
        deriv[col] = d

    # 4. New: segment-level derivatives & geometric summaries 3 seg 
    for s in range(3):
        start = s * seg_size
        end   = min((s+1) * seg_size, n)
        idx   = start
        for col in ['com_speed', 'com_acc', 'angular_velocity_deg_s']:
            # instantaneous derivative at segment start
            features[f'd1_{col}_seg{s+1}'] = float(deriv[col][idx])
            # geometric summaries
            seg_vals = df[col].values[start:end]
            features[f'mean_{col}_seg{s+1}']  = float(np.mean(seg_vals))
            features[f'std_{col}_seg{s+1}']   = float(np.std(seg_vals))
            features[f'range_{col}_seg{s+1}'] = float(np.max(seg_vals) - np.min(seg_vals))

    # 5. Three-segment slopes
    slope_cols = ['torso_angle_deg','com_speed','log_upper_body_angle','log_com_speed']
    for col in slope_cols:
        vals = df[col].values
        for i in range(3):
            start = i * seg_size
            end   = min((i + 1) * seg_size, n)
            seg_vals  = vals[start:end]
            seg_times = t[start:end].reshape(-1,1)
            if len(seg_vals) < 2:
                slope = 0.0
            else:
                lr    = LinearRegression().fit(seg_times, seg_vals)
                slope = float(lr.coef_[0])
            features[f'slope_{col}_seg{i+1}'] = slope

    # 6. Inter-peak intervals in com_acc
    acc = df['com_acc'].values
    peaks, _ = find_peaks(acc, height=np.median(acc) + np.std(acc))
    if len(peaks) > 1:
        intervals = np.diff(t[peaks])
        features['mean_interval_acc'] = float(np.mean(intervals))
        features['std_interval_acc']  = float(np.std(intervals))
    else:
        features['mean_interval_acc'] = 0.0
        features['std_interval_acc']  = 0.0

    # 7. CUSUM change-point on angular_velocity
    ang   = df['angular_velocity_deg_s'].values
    csum  = np.cumsum(ang - np.mean(ang))
    cp_i  = np.argmax(np.abs(csum))
    features['cp_magnitude_ang'] = float(csum[cp_i])
    features['cp_time_ang']      = float(t[cp_i] / duration)

    # 8. Wavelet entropy for speed series
    for col in ['com_speed','log_com_speed']:
        coeffs = pywt.wavedec(df[col].values, 'db4', level=3)
        for lvl, detail in enumerate(coeffs[1:], start=1):
            total = np.sum(np.abs(detail)) + 1e-8
            probs = np.abs(detail) / total
            ent   = float(-np.sum(probs * np.log(probs + 1e-8)))
            features[f'wavelet_entropy_{col}_d{lvl}'] = ent

    # 9. Sample entropy of speed
    features['sampen_com_speed']     = sample_entropy(df['com_speed'].values)
    features['sampen_log_com_speed'] = sample_entropy(df['log_com_speed'].values)

    # 10. Spatial-temporal posture coupling (safe correlation)
    hhg = df['head_hip_gap_norm'].values
    lr  = LinearRegression().fit(t.reshape(-1,1), hhg)
    features['slope_head_hip'] = float(lr.coef_[0])

    ta = df['torso_angle_deg'].values - np.mean(df['torso_angle_deg'].values)
    h  = hhg - np.mean(hhg)
    std_ta = np.std(ta)
    std_h  = np.std(h)
    if std_ta < 1e-8 or std_h < 1e-8:
        corr = 0.0
    else:
        corr = float(np.dot(ta, h) / (std_ta * std_h * len(ta)))
    features['corr_torso_hhg'] = corr
    
    return features

# Build master CSV
# folders = [
#     (r'E:\Fall-Detection research latest\outputs_2025-10-03\2025-10-03_perframe_fall',     1),
#     (r'E:\Fall-Detection research latest\outputs_2025-10-03\2025-10-03_perframe_not_fall', 0),
# ]
# rows = []
# for folder, label in folders:
#     for fname in os.listdir(folder):
#         if not fname.lower().endswith('.csv'):
#             continue
#         df_clip = pd.read_csv(os.path.join(folder, fname))
#         feats    = extract_advanced_features(df_clip)
#         feats['video'] = os.path.splitext(fname)[0]
#         feats['label'] = label
#         rows.append(feats)

# master_df = pd.DataFrame(rows) 
# master_df.to_csv('master_training_pruned_and_enhanced.csv', index=False)
# print(f"master_training_pruned_and_enhanced.csv created with {len(master_df)} rows")



