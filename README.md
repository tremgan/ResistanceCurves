# ResistanceCurves

Plots the resistance curve of a plate-loaded machine from a video: detects the plates frame by frame and computes torque at the pivot, relative to peak.

**App:** https://tremgan.github.io/ResistanceCurves/

- `app/` – static web app, runs inference in the browser; deployed to GitHub Pages on push to `main`
- `train/` – trains the plate detector and exports ONNX ([train/README.md](train/README.md))
