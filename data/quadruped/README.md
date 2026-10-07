# Quadruped proxy table

Place the simulator export at `data/quadruped/fixed_twist_combined.csv`.

Required columns:

- `fixed_command_id`
- `ref_vx`, `ref_vy`, `ref_wz`
- `tracking_err_vx`, `tracking_err_vy`, `tracking_err_wz`

`t` is optional and is used to order rows inside each command. Hardware trials that ship with this repo are under `experiments_quadruped/sample_trials/`.
