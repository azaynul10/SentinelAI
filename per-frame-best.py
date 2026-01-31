# test.py

import warnings
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# MediaPipe pose
mp_pose = mp.solutions.pose

# Landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Helpers (winsor, ema, etc.)
def _mid(a, b): return (a + b) / 2.0
def _norm(v, eps=1e-8): ##Normalizing function
    n = np.linalg.norm(v)
    return v / (n + eps), n
def _angle_to_vertical_deg(v):
    vy = np.array([0.0, -1.0], dtype=np.float32) ##Vertical AXIS vector
    vu, _ = _norm(v.astype(np.float32)) 
    c = np.clip(float(vu @ vy), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
def _angle3(a, b, c):
    va = a - b; vc = c - b
    va_u, _ = _norm(va); vc_u, _ = _norm(vc)
    cval = np.clip(float(va_u @ vc_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(cval)))


def _ema(x, α=0.25): ##Moving Averages
    y = np.empty_like(x, dtype=np.float32)
    if not len(x): return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = α * x[i] + (1 - α) * y[i - 1]
    return y



def _winsor_mad(x, k=5.0): ###Winsorizing function & outlier removal
    x = np.asarray(x, np.float32); m = np.isfinite(x)
    if m.sum() < 3: return x
    med = np.median(x[m]); mad = np.median(np.abs(x[m] - med)) + 1e-6
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    y = x.copy(); y[m & (y < lo)] = lo; y[m & (y > hi)] = hi
    return y


def _interp_nans(x):
    x = np.asarray(x, np.float32); idx = np.arange(len(x)); m = np.isfinite(x)
    if not m.any(): return np.zeros_like(x)
    x[~m] = np.interp(idx[~m], idx[m], x[m]); return x


def _bbox(pts):
    xs, ys = pts[:,0], pts[:,1]; return float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max())

# Helper class for in-memory frames
class MockVideoCapture:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.count = len(frames)
        self.width = frames[0].shape[1] if frames else 0
        self.height = frames[0].shape[0] if frames else 0

    def isOpened(self): return True
    def release(self): pass
    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.idx = int(val)
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT: return self.count
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.height
        return 0
    def read(self):
        if self.idx < self.count:
            f = self.frames[self.idx]
            self.idx += 1
            return True, f
        return False, None

# Refactored to separate extraction from MediaPipe execution
def extract_from_landmarks(landmarks_list, width, height, fps=30.0):
    """
    Extracts features from a list of MediaPipe pose landmarks.
    landmarks_list: List of mp_pose.PoseLandmark objects (or None for failed frames)
    """
    n = len(landmarks_list)
    idxs = np.arange(n) # Use all frames provided
    
    W = width
    H = height

    ts, A, AR, KA, HH, COM = [], [], [], [], [], []
    
    for i, res_landmarks in enumerate(landmarks_list):
        ts.append(i / fps)
         
        # Handle failed detection (None)
        if res_landmarks is None:
            A.append(np.nan); AR.append(np.nan); KA.append(np.nan)
            HH.append(np.nan); COM.append([np.nan,np.nan]); continue

        # Convert landmarks to pixel coordinates
        pts = np.array([[lm.x*W, lm.y*H] for lm in res_landmarks.landmark], np.float32)
        
        # Verify we have enough points (33 for full body)
        if len(pts) < 33:
             A.append(np.nan); AR.append(np.nan); KA.append(np.nan)
             HH.append(np.nan); COM.append([np.nan,np.nan]); continue

        head = pts[NOSE]; lsh, rsh = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
        lhp, rhp = pts[LEFT_HIP], pts[RIGHT_HIP]
        lkn, rkn = pts[LEFT_KNEE], pts[RIGHT_KNEE]
        lan, ran = pts[LEFT_ANKLE], pts[RIGHT_ANKLE]

        scale = np.median([np.linalg.norm(rsh-lsh), np.linalg.norm(rhp-lhp),1e-3])
        mid_sh = _mid(lsh, rsh); mid_hp = _mid(lhp, rhp)
        A.append(_angle_to_vertical_deg(mid_hp-mid_sh))

        xmin,ymin,xmax,ymax = _bbox(pts)
        AR.append((ymax-ymin)/(xmax-xmin+1e-6))

        k1 = _angle3(lhp, lkn, lan); k2 = _angle3(rhp, rkn, ran)
        KA.append(np.nanmean([k1,k2]))

        HH.append((head[1]-mid_hp[1])/(scale+1e-6))

        major = np.vstack([lsh,rsh,lhp,rhp,lkn,rkn,lan,ran])
        COM.append(np.nanmean(major,axis=0))

    ts = np.asarray(ts,np.float32)
    A = _interp_nans(_winsor_mad(np.asarray(A))); AR = _interp_nans(_winsor_mad(np.asarray(AR)))
    KA = _interp_nans(_winsor_mad(np.asarray(KA))); HH = _interp_nans(_winsor_mad(np.asarray(HH)))
    COM = np.stack(COM)
    for d in (0,1): COM[:,d]=_interp_nans(_winsor_mad(COM[:,d]))

    A = _ema(A,α=0.25); AR = _ema(AR,α=0.25); KA = _ema(KA,α=0.25); HH = _ema(HH,α=0.25)
    COM = np.stack([_ema(COM[:,0],α=0.25), _ema(COM[:,1],α=0.25)], axis=1)

    dt = np.gradient(ts)+1e-6
    dA = np.gradient(A)/dt
    dC = (np.vstack([np.gradient(COM[:,0]),np.gradient(COM[:,1])]).T)/dt[:,None]
    v = np.linalg.norm(dC,axis=1)
    a = np.gradient(v)/dt; j = np.gradient(a)/dt
    angmag = np.abs(dA); logup = np.log1p(np.abs(A))

    return pd.DataFrame({
        "timestamp":ts,
        "torso_angle_deg":A,
        "angular_velocity_deg_s":dA,
        "angular_velocity":angmag,
        "aspect_ratio":AR,
        "knee_angle":KA,
        "head_hip_gap_norm":HH,
        "com_speed":v,
        "com_acc":a,
        "com_jerk":j,
        "log_upper_body_angle":logup
    })

# Per-frame extraction (Legacy Wrapper)
def extract_per_frame(path_or_frames, fps=30.0, n=90, α=0.25):
    """
    Main entry point. 
    If path_or_frames is a string: Open video, run MP, get landmarks.
    If path_or_frames is a list of FRAMES (old way): Run MP on them.
    If path_or_frames is a list of LANDMARKS (new way): Pass strictly to extractor.
    """
    
    # 1. Check if input is already Landmarks (New Optimization)
    # We assume if it's a list and the first element is NOT numpy array (frame), it's a landmark object or None
    if isinstance(path_or_frames, list) and len(path_or_frames) > 0:
        first_item = path_or_frames[0]
        # Frame is usually numpy array (uint8). Landmark is class or None.
        if hasattr(first_item, 'landmark') or first_item is None:
             # It is a list of landmarks!
             # We need W, H. We can assume defaults or pass them. 
             # In this legacy wrapper, we might not know W/H if only landmarks are passed. 
             # However, for the NEW flow, we will call extract_from_landmarks DIRECTLY.
             # This block handles the case if someone calls this legacy function with landmarks.
             return extract_from_landmarks(path_or_frames, 1920, 1080, fps) # Assume HD

    # 2. Legacy: Video Path or List of Frames
    if isinstance(path_or_frames, (list, tuple, np.ndarray)) and len(path_or_frames) > 0 and isinstance(path_or_frames[0], np.ndarray):
        cap = MockVideoCapture(path_or_frames)
        total = len(path_or_frames)
    else:
        cap = cv2.VideoCapture(str(path_or_frames))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n)
    
    idxs = np.linspace(0, total - 1, n, dtype=int)
    
    # WARNING: This creates a NEW MediaPipe instance. 
    # Use extract_from_landmarks() to avoid this overhead if you already have landmarks.
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    landmarks_buffer = []

    for f in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            landmarks_buffer.append(None)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
             landmarks_buffer.append(res.pose_landmarks)
        else:
             landmarks_buffer.append(None)

    cap.release()
    pose.close()
    
    return extract_from_landmarks(landmarks_buffer, W, H, fps)


def main():
    FALL_DIR = Path(r"E:\Fall-Detection research latest\Fall-Detection-Vdo-master\Video Files\fall\Second Counting Videos Fall\all_3_second_videos_fall")
    NOTFALL_DIR = Path(r"E:\Fall-Detection research latest\Fall-Detection-Vdo-master\Video Files\not_fall\Second Counting Videos\all 3 second videos not fall")
    today = datetime.now().strftime("%Y-%m-%d")
    outbase = Path(f"outputs_{today}")
    pf_fall = outbase/f"{today}_perframe_fall"; pf_nf = outbase/f"{today}_perframe_not_fall"
    outbase.mkdir(exist_ok=True,parents=True); pf_fall.mkdir(exist_ok=True,parents=True); pf_nf.mkdir(exist_ok=True,parents=True)

    # Discover all video files
    vids = [(p,1) for p in FALL_DIR.rglob("*") if p.suffix.lower() in [".mp4",".avi"]] \
         + [(p,0) for p in NOTFALL_DIR.rglob("*") if p.suffix.lower() in [".mp4",".avi"]]
    
    print(f"Total videos found: {len(vids)}")
    
    # Check which videos already have CSV files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, (p, label) in enumerate(vids):
        # Determine output CSV path
        output_csv = (pf_fall if label else pf_nf) / f"{p.stem}.csv"
        
        # Check if CSV already exists
        if output_csv.exists():
            processed_count += 1
            print(f"[{i+1}/{len(vids)}] ALREADY PROCESSED: {p.name} -> {output_csv.name}")
            continue
            
        # Process the video
        try:
            print(f"[{i+1}/{len(vids)}] PROCESSING: {p.name} -> {output_csv.name}")
            df_pf = extract_per_frame(p)
            df_pf.to_csv(output_csv, index=False)
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            warnings.warn(f"{p.name} skipped: {e}")
            print(f"[{i+1}/{len(vids)}] ERROR: {p.name} - {e}")

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total videos: {len(vids)}")
    print(f"Already processed (skipped): {skipped_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Per-frame CSVs saved in: {pf_fall} and {pf_nf}")

if __name__=="__main__":
    main()