# Input data

Not included in this repository. See the Data section of the main README.

Expected files:

| File | Description |
|---|---|
| Video | Fixed-camera footage where the road surface is approximately planar |
| Tracking JSON | Per-frame track IDs and bounding boxes |
| Reference sizes | Spreadsheet with `track_id`, `gt_width_m`, `gt_length_m` for evaluation |

Tracking JSON format:

```json
[
  {
    "id": 1,
    "timestamps": [0, 1, 2],
    "boxes": [[x1, y1, x2, y2], "..."]
  }
]
```

Alternative keys accepted: `track_id` / `tid` for `id`, `frames` for `timestamps`,
`bboxes` for `boxes`.
