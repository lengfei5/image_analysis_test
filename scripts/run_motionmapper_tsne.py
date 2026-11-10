#!/usr/bin/env python3
"""
Run MotionMapper (motionmapperpy) wavelet analysis + t-SNE nonlinear embedding
on your own pose-estimation data.

Pipeline
--------
1. Load your pose data (an xarray .nc file as produced by `movement`/VAME-style
   egocentric alignment, OR a DeepLabCut CSV, OR a plain .npy/.csv matrix).
2. Flatten it into a (frames x features) "projections" matrix and save it as a
   `*_pcaModes.mat` file inside the MotionMapper project `Projections/` folder.
3. Morlet WAVELET decomposition of every feature over a bank of frequencies
   (this is what `waveletDecomp = True` turns on).
4. Sub-sampled t-SNE to learn a training embedding, then re-embed every frame
   (`findEmbeddings`) -> 2-D t-SNE coordinates per frame.
5. Watershed segmentation of the density map -> discrete behavioural regions.

Usage
-----
    python run_motionmapper_tsne.py --input /path/to/pose.nc --fps 30
    python run_motionmapper_tsne.py --input /path/to/dlc.csv --fps 60 --project ./data/MyProject

Outputs (inside <project>/)
    Projections/<name>_pcaModes.mat        your data as MotionMapper projections
    TSNE/training_embedding.mat            learned training t-SNE map
    Projections/<name>_pcaModes_zVals.mat  per-frame 2-D t-SNE coordinates  <-- main result
    <project>/*.png                        density map + watershed region plots
    <name>_tsne_embedding.csv              tidy CSV: frame, tsne_1, tsne_2
"""

import argparse
import glob
import os
import pickle
from datetime import datetime

import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import h5py

import motionmapperpy as mmpy


# --------------------------------------------------------------------------- #
# 1. Load your pose data into a (frames, features) matrix
# --------------------------------------------------------------------------- #
def load_pose_matrix(path):
    """Return (data, feature_names). data is (n_frames, n_features) float array."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".nc":
        # xarray dataset (e.g. from `movement` / VAME egocentric export).
        import xarray as xr
        ds = xr.open_dataset(path)
        # pick the first data variable (usually "position")
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        print(f"Loaded variable '{var_name}' with dims {da.dims} and shape {da.shape}")

        # Collapse everything that is not 'time' into a flat feature axis.
        # Typical dims: (time, space, keypoints[, individuals])
        da = da.transpose("time", ...)                 # time first
        flat = da.stack(feature=[d for d in da.dims if d != "time"])
        data = np.asarray(flat.values, dtype=np.float64)   # (time, feature)

        # Build readable feature names like "nose_x", "nose_y"
        names = []
        for combo in flat.coords["feature"].values:
            names.append("_".join(str(c) for c in np.atleast_1d(combo)))
        ds.close()
        return data, names

    if ext == ".npy":
        data = np.load(path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f".npy must be 2-D (frames x features); got {data.shape}")
        return data, [f"feat_{i}" for i in range(data.shape[1])]

    if ext in (".csv", ".h5", ".hdf5"):
        import pandas as pd
        if ext == ".csv":
            # DeepLabCut CSV has 3 header rows (scorer, bodyparts, coords)
            try:
                df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
            except Exception:
                df = pd.read_csv(path, header=[1, 2], index_col=0)
        else:
            df = pd.read_hdf(path)
        # drop likelihood columns, keep x/y
        keep = [c for c in df.columns if str(c[-1]).lower() in ("x", "y")]
        df = df[keep]
        data = df.to_numpy(dtype=np.float64)
        names = ["_".join(str(x) for x in c[-2:]) for c in df.columns]
        return data, names

    raise ValueError(f"Unsupported input extension: {ext}")


def clean_matrix(data):
    """Interpolate NaNs column-wise so the wavelet transform doesn't choke."""
    data = data.astype(np.float64).copy()
    n = np.arange(data.shape[0])
    for j in range(data.shape[1]):
        col = data[:, j]
        bad = np.isnan(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.interp(n[bad], n[~bad], col[~bad])
        data[:, j] = col
    return data


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="MotionMapper wavelet + t-SNE on pose data")
    ap.add_argument("--input", required=True, help="pose file: .nc / .npy / .csv / .h5")
    ap.add_argument("--project", default="./data/MyProject", help="MotionMapper project folder")
    ap.add_argument("--fps", type=float, default=30.0, help="sampling frequency (frames/sec)")
    ap.add_argument("--minF", type=float, default=0.5, help="min wavelet frequency (Hz)")
    ap.add_argument("--maxF", type=float, default=None, help="max wavelet frequency (Hz); default = fps/2 (Nyquist)")
    ap.add_argument("--numPeriods", type=int, default=25, help="number of wavelet frequencies")
    ap.add_argument("--perplexity", type=int, default=32, help="t-SNE perplexity")
    ap.add_argument("--trainingSetSize", type=int, default=3000, help="representative training points")
    ap.add_argument("--training_numPoints", type=int, default=1000, help="points per mini-tSNE")
    ap.add_argument("--num_regions", type=int, default=50, help="target watershed regions")
    args = ap.parse_args()

    maxF = args.maxF if args.maxF is not None else args.fps / 2.0  # Nyquist

    # ---- 1. create project + convert data to projections -------------------- #
    projectPath = args.project
    mmpy.createProjectDirectory(projectPath)

    data, feat_names = load_pose_matrix(args.input)
    data = clean_matrix(data)
    print(f"Pose matrix: {data.shape[0]} frames x {data.shape[1]} features")
    print("Features:", feat_names)

    name = os.path.splitext(os.path.basename(args.input))[0]
    proj_file = os.path.join(projectPath, "Projections", f"{name}_pcaModes.mat")
    # MotionMapper expects variable name 'projections', shape (frames, features)
    savemat(proj_file, {"projections": data})
    print("Saved projections ->", proj_file)

    # ---- 2. run parameters -------------------------------------------------- #
    parameters = mmpy.setRunParameters()
    parameters.projectPath = projectPath
    parameters.method = "TSNE"            # <-- t-SNE (not UMAP)
    parameters.waveletDecomp = True       # <-- wavelet analysis ON
    parameters.minF = args.minF
    parameters.maxF = maxF
    parameters.samplingFreq = args.fps
    parameters.numPeriods = args.numPeriods
    parameters.perplexity = args.perplexity
    parameters.numProcessors = -1
    parameters.useGPU = -1                # set to GPU id if cupy installed
    parameters.training_numPoints = args.training_numPoints
    parameters.trainingSetSize = args.trainingSetSize
    parameters.embedding_batchSize = 30000

    m = loadmat(proj_file, variable_names=["projections"])["projections"]
    parameters.pcaModes = m.shape[1]
    parameters.numProjections = m.shape[1]
    del m

    # ---- 3 & 4. sub-sampled t-SNE on wavelet features ----------------------- #
    tsnefolder = parameters.projectPath + "/TSNE/"   # wavelet + TSNE -> /TSNE/
    print(datetime.now().strftime("%m-%d-%Y_%H-%M"), "tSNE started")

    if not os.path.exists(tsnefolder + "training_embedding.mat"):
        print("Running sub-sampled mini-tSNE (this computes the wavelets first)...")
        mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
        print("Training embedding done.")

    with h5py.File(tsnefolder + "training_data.mat", "r") as hf:
        trainingSetData = hf["trainingSetData"][:].T
    with h5py.File(tsnefolder + "training_embedding.mat", "r") as hf:
        trainingEmbedding = hf["trainingEmbedding"][:].T

    zValstr = "zVals"  # TSNE + waveletDecomp
    projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")

    for i, pf in enumerate(projectionFiles):
        print(f"Embedding {i+1}/{len(projectionFiles)}: {pf}")
        out = pf[:-4] + f"_{zValstr}.mat"
        if os.path.exists(out):
            print("  already done, skipping")
            continue
        projections = loadmat(pf)["projections"]
        zValues, outputStatistics = mmpy.findEmbeddings(
            projections, trainingSetData, trainingEmbedding, parameters
        )
        hdf5storage.write(data={"zValues": zValues}, path="/", truncate_existing=True,
                          filename=out, store_python_metadata=False, matlab_compatible=True)
        with open(pf[:-4] + f"_{zValstr}_outputStatistics.pkl", "wb") as hf:
            pickle.dump(outputStatistics, hf)

        # tidy CSV of per-frame t-SNE coordinates
        csv_out = pf[:-4] + "_tsne_embedding.csv"
        np.savetxt(csv_out, np.column_stack([np.arange(len(zValues)), zValues]),
                   delimiter=",", header="frame,tsne_1,tsne_2", comments="")
        print("  saved:", out, "and", csv_out)

    # ---- 5. watershed behavioural regions ----------------------------------- #
    print("Finding watershed regions...")
    mmpy.findWatershedRegions(parameters, num_regions=args.num_regions, startsigma=0.3,
                              pThreshold=[0.33, 0.67], saveplot=True,
                              endident="*_pcaModes.mat")
    print("All done. See", projectPath)


if __name__ == "__main__":
    main()
