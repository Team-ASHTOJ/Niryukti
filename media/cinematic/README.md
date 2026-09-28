# Niryukti cinematic opening and closing — motion edition

A complete rebuild of the original title cards: projected 3D geometry, bronze facets, orbiting light packets, kinetic typography, scanning sparse matrices, particle assembly and a moving optimization surface. The palette and fonts follow the dashboard: graphite, ivory and champagne gold.

## Watch and use

Open **`preview.html`** to watch both clips locally, jump between the intro scenes and download the videos. No server or internet connection is needed.

- **`niryukti-intro-1080p.mp4`** — 24 seconds, 1920 × 1080, 60 fps, H.264.
- **`niryukti-outro-1080p.mp4`** — 12 seconds, matching settings.
- **`storyboard.jpg`** — ten representative frames showing the visual progression.
- **`render.py`** — editable deterministic animation source.
- **`*-preview.png`** — full-resolution scene stills.

Both videos are silent, leaving the voiceover and music to your final edit. The clips replace the earlier 12-second/8-second version.

## Intro sequence

| Time | What happens | Narration opportunity |
| --- | --- | --- |
| 0–1.5 s | A quiet graphite field wakes up; light traces and geometry expand from the centre. | A brief music lead-in. |
| 1.5–4.8 s | A rotating bronze polytope, orbiting light trails and perspective grid create the opening hook. A camera push prepares the team reveal. | Optional short opening line. |
| 4.8–9.9 s | **TEAM ASHTOJ** appears with staggered letter motion and opening gold rules. The geometry keeps moving behind it. | “Hello, we are Team Ashtoj…” Add your own introduction here. |
| 10.1–13.1 s | The team card fades out. **PRESENTS** appears as its own scene inside a smaller moving orbital form, then fades. | Pause or “…and we present…” |
| 13.1–16.7 s | Hundreds of particles converge into the Niryukti coordinate mark. Its gold path resolves and the project name appears letter by letter. | “…Niryukti.” |
| 16.7–22.4 s | Full title, “Independent sparse optimization engine”, and “Model / Solve / Verify”. A convex surface, moving trajectory, sparse-matrix scans and orbiting light keep the frame alive. | “Our independent optimization engine, built around transparent mathematics and verifiable results.” |
| 22.5–24 s | A diagonal gold ribbon passes through, followed by a fade to black. | Begin the handoff to your dashboard walkthrough. |

Team name, Presents and project title have separate entrances and exits. No award or competition-history claim is baked into the animation; use your own accurate voiceover introduction.

## Outro sequence

| Time | What happens |
| --- | --- |
| 0–2.5 s | The coordinate mark draws in and Niryukti assembles above a moving optimization surface. |
| 2–4 s | “Built to optimize. Designed to be inspected.” appears. |
| 4.1–6 s | Gold rule and **TEAM ASHTOJ** arrive. |
| 5.4–10.4 s | Installation commands appear: `pip install niryukti / npm install niryukti`. |
| 10.4–12 s | Type, mark and geometry fade down to black. |

Suggested narration: “This is Niryukti: independently engineered, openly inspectable, and available for you to try. Built by Team Ashtoj. Thank you.”

## Editing notes

1. Put the intro before your dashboard capture and the outro after the final explanation.
2. Use a 1920 × 1080 timeline. The 60 fps clips also work on a 30 fps timeline.
3. For the intro handoff, cross-dissolve its last 0.5 seconds into the recording. A direct cut after the fade works too. The actual dashboard footage is not embedded in the clip.
4. Keep narration comfortably above background music. A restrained electronic/ambient score can build during the first geometric reveal, soften under the team introduction, and land a gentle accent on the Niryukti reveal around 15–17 seconds.
5. All motion is editorial artwork. It does not represent measured benchmark progress, recorded solver iterations or actual certificates.

## Re-render or customize

Dependencies: Python, NumPy, Pillow, OpenCV and FFmpeg. The included project fonts have their original OFL licences under `dashboard/static/fonts/`. No stock footage, external art or Blender installation is required.

```bash
python3 -m pip install Pillow numpy opencv-python-headless

# Storyboard and full-resolution stills
python3 media/cinematic/render.py --preview

# Both clips, portable CPU encoding
python3 media/cinematic/render.py

# Faster encoding with an NVIDIA GPU and NVENC-enabled FFmpeg
python3 media/cinematic/render.py --encoder nvenc

# Other sizes or frame rates; filenames follow the resulting height
python3 media/cinematic/render.py --width 3840 --fps 60
python3 media/cinematic/render.py --width 1280 --fps 30 --kind intro
```

The renderer computes real perspective projections of procedural 3D geometry and composites typography and bloom into each frame. NVENC accelerates video encoding; the animation frame rendering runs on CPU. The default export is 1080p60. Source output is deterministic for a given configuration; outputs are overwritten when rerendered.

Adjust the scene times and typography in `Film.frame`, the geometry in `orbit`, `polytope`, `surface` and `assembly`, and the clip lengths in `DURATIONS`.

For the full project narration, see [the demo walkthrough](../../docs/complete_demo_walkthrough.md).

## Export checks

The delivered intro contains 1,440 frames and the outro 720 frames, both H.264 at 1920 × 1080 and 60 fps. Both decoded completely without FFmpeg errors. The local preview page was checked in Chrome for playback, scene seeking, switching clips, the download target and mobile horizontal overflow.
