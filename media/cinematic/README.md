# Niryukti cinematic opening and closing

Ready-to-edit silent videos, matching the dashboard's graphite, ivory and champagne palette. Animation uses local project fonts, a reconstruction of the existing coordinate-frame mark, sparse moving nodes and convergence contours. No Blender installation or external stock assets are needed.

## Deliverables

- `niryukti-intro-1080p.mp4`: 12 seconds, 1920 × 1080, 30 fps, H.264.
- `niryukti-outro-1080p.mp4`: 8 seconds, same settings.
- PNG previews: team card, project reveal and closing card.
- `render.py`: editable, deterministic source; typography, colours, timing and copy are all local.

Both clips deliberately have no audio so your voiceover and background music can run continuously through your final edit.

## Opening timeline and suggested voiceover

| Time | Screen | Suggested narration |
| --- | --- | --- |
| 0–1.3 s | Sparse computational field emerges from black | Let the music start; leave a short breath. |
| 1.3–3.4 s | TEAM ASHTOJ / PRESENTS | “Team Ashtoj presents…” |
| 4–6.3 s | Coordinate mark appears; its gold optimization path draws | “…Niryukti.” |
| 6.3–10.5 s | NIRYUKTI / INDEPENDENT SPARSE OPTIMIZATION ENGINE / Mathematics in motion. | “An independent optimization engine, built for transparent mathematical decision-making.” |
| 10.5–12 s | Gold transition rule; fade to black | Transition into your recorded dashboard. |

This is an editorial animation, not a depiction of measured solver iterations or benchmark performance.

## Closing voiceover

The outro reveals Niryukti, “Built to optimize. Designed to be inspected.”, Team Ashtoj and the pip/npm installation commands.

Suggested voiceover: “This is Niryukti: independently engineered, openly inspectable, and available for you to try. Built by Team Ashtoj. Thank you.”

## Editing

1. Place the intro before the dashboard recording and the outro after your final explanation.
2. Use a 1920 × 1080, 30 fps timeline. If the recording is 60 fps, the clips can also be used on that timeline.
3. For a smooth handoff, overlap the final 0.4–0.6 seconds of the intro with the recording using a cross dissolve. The default final fade also supports a direct cut from black.
4. Keep voiceover clear above the music. Choose a restrained ambient/piano/electronic track you have permission to use.
5. The title cards keep important text comfortably within frame margins; avoid additional subtitles covering the main title.

## Re-render

Requires Python, Pillow, NumPy and FFmpeg with libx264. Uses the existing project font files and their included OFL licences.

```bash
python3 -m pip install Pillow numpy
python3 media/cinematic/render.py --preview
python3 media/cinematic/render.py
```

For 4K output, use `--width 3840`. For 60 fps, use `--fps 60`. Output filenames retain their default `1080p` suffix, so rename them appropriately when changing resolution. Rendering overwrites the local outputs.

These assets complement the full speaking plan in `docs/complete_demo_walkthrough.md`.
