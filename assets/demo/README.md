# FactoryLens Demo Data

This folder is a small, fixed input pack for local demos and screen recordings.
It is intentionally lightweight and does not replace the full MVTec dataset.

## Images

| File | Defect type | Demo use |
|---|---|---|
| `images/hazelnut_crack_005.png` | crack | Run image analysis and show a long visible shell split. |
| `images/hazelnut_cut_000.png` | cut | Run image analysis and show a short clean surface cut. |
| `images/hazelnut_hole_004.png` | hole | Run image analysis and show a clear round hole. |
| `images/hazelnut_print_000.png` | print | Run image analysis and show a printed or transfer-like surface mark. |

The images are resized copies from the local MVTec hazelnut test split so the
demo pack stays small enough to commit.

## Logs

`hazelnut_demo_logs.csv` contains one passing unit and four defect units:

- `HZ-GOOD-000-001`
- `HZ-CRACK-000-016`
- `HZ-CUT-000-017`
- `HZ-HOLE-000-018`
- `HZ-PRINT-000-019`

Use it when demoing log upload, filtering, and root-cause lookup around a small
set of known unit IDs.

## Known Issues

`known_issues/` contains two representative Markdown docs copied from the main
known-issue corpus:

- `KI-002-crack-compression.md`
- `KI-007-print-transfer-mark.md`

Use these as compact SOP/context examples when recording the retrieval or report
generation flow.
