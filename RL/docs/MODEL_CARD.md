# Model Card

## Selected Local Checkpoint

```text
fishbot_mujoco_sac/runs_dyn/sac_20260825_010443/best_model.zip
```

This SAC checkpoint is approximately 324 MB and is intentionally not committed
to Git. GitHub rejects normal files above 100 MB. Publish it as a GitHub Release
asset or track it with Git LFS.

## Policy Inputs

- 72 lidar range rays normalized by max range.
- Remaining path distance.
- Local guide angle.
- Final goal angle.
- Forward velocity.
- Yaw rate.
- Minimum lidar range.

## Policy Output

Two continuous normalized commands:

```text
[linear_velocity_command, angular_velocity_command]
```

## Validation Result

The selected SAC checkpoint alone is not fully robust on the tested continuous
multi-goal seeds. The public demo uses SAC plus the heading guard in
`scripts/view_multigoal_dyn.py`.

| Setting | Result |
| --- | --- |
| SAC only, seeds 2/7/8/10 | seed 8 and 10 pass; seed 2 and 7 collide |
| SAC + heading guard, seeds 2/7/8/10 | all pass 8/8 goals, 0 collision |
| SAC + heading guard, GUI seed 2 | 10/10 goals, 0 collision, 0 timeout |

## Intended Use

Simulation research and reproducible navigation demos in MuJoCo. This is not a
drop-in real-robot controller without sensor calibration, velocity limiting,
watchdogs, and emergency stop handling.
