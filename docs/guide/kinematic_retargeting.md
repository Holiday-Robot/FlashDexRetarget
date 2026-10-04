# Kinematic retargeting: human demos to `motion.pt`

Three releases are supported: TACO, OakInk2 and HOT3D (the segments in which both hands handle one object).
One script per release turns it into the `motion.pt` training reads:

```sh
ROBOTS="xhand sharpa" bash scripts/retarget/taco.sh <raw_dataset_path> <output_dataset_path> [overrides]   # oakink2.sh | hot3d_1obj.sh
```

`<raw_dataset_path>` is the downloaded release and `<output_dataset_path>` the dataset to write. Each script
imports the release once into `<output_dataset_path>` (`human_demo/`, `objects/`; skipped when it already
has clips, `REIMPORT=1` to redo), then retargets it to each robot of `ROBOTS` (default xhand) in
`<output_dataset_path>/<robot>/`; train with `MOTION=<output_dataset_path>/<robot>/motion.pt`. The overrides
go to `kinematic_retargeting.yaml`. The scripts run two commands, which can also be called directly:

```sh
bash scripts/retarget/import_human_demo.sh source=taco raw=<raw_dataset_path> dataset=<output_dataset_path>   # taco | oakink2 | hot3d
bash scripts/retarget/kinematic_retargeting.sh dataset=<output_dataset_path> robot=xhand
```

| | step | does | writes (under the dataset) |
|---|---|---|---|
| `import_human_demo.sh` | import | MANO reference + object poses, z-up, `fps` Hz | `human_demo/<clip>.npz`, `objects/<id>/visual.obj` |
| `kinematic_retargeting.sh` | preprocess | support disks, reject rules | `report.csv` |
| | convex | convex decomposition (CuACD or CoACD) | `objects/<id>/convex/*.obj` |
| | retarget | MuJoCo physics IK | `retargeted/<clip>.npz` |
| | pack | kept clips into one file | `motion.pt` |
| | usd, sdf | Isaac Sim object assets | `objects/<id>/.isaac_usd/`, `objects/<id>/.sdf_cache/` |

Settings live in `config/retarget/import_human_demo.yaml` and `config/retarget/kinematic_retargeting.yaml`;
any key is overridden as `key=value`, and each run's config is kept under `.hydra/` next to the outputs.
`workers=` sets the CPU parallelism (default: half the cores), `steps.<name>=false` skips a step (e.g.
`steps.retarget=false steps.pack=false` to review `report.csv` first).

## Dataset layout

```
<dataset>/
├── motion.pt          every kept clip, packed; training reads this and objects/
├── human_demo/        imported clips, one <clip>.npz each
├── retargeted/        IK results (a cache: redone only when the support disks change or retarget.force=true)
├── report.csv         every clip's reject verdict, IK error and quality flags
└── objects/<id>/
    ├── visual.obj     object mesh in metres
    ├── convex/*.obj   convex parts: the shapes MuJoCo and PhysX collide with
    ├── .isaac_usd/    object USD for Isaac Sim
    └── .sdf_cache/    object SDF grids
```

## 1. Import

`source=` picks a loader in `src/retarget/human_demo/sources/`; each turns its release into MANO hands and
object poses, then `base.py` handles every source the same way.

| source | raw layout | which object each hand holds |
|---|---|---|
| `taco` | `Hand_Poses/`, `Object_Poses/`, `object_models/` | tool and target go to the nearer hand |
| `oakink2` | `anno_preview/`, `program/`, `object_repair/` | the object the hand touches most; stages where one hand has several objects are skipped |
| `hot3d` | `dataset/<recording>/`, `_assets/` | recordings are cut into segments where both hands hold one object (`source.segment`) |

Per clip: MANO forward pass with `assets/mano/MANO_{RIGHT,LEFT}.pkl`, rotation into a z-up world, table height
(`source.table_z` for HOT3D), the wrist and fingertip reference the rewards compare against, and resampling to
`fps` (60 Hz). A clip is skipped when a hand has no object or an object flips more than `max_obj_rot_step_deg`
in one frame.

### Clip format (`human_demo/<clip>.npz`)

All arrays at `fps`, z-up world, metres; `s` is `right` or `left`:

| key | shape | content |
|---|---|---|
| `fps` | () | frame rate (the same for every clip of a dataset) |
| `mano_s_wrist_pos`, `mano_s_wrist_rot` | (T, 3), (T, 3, 3) | MANO wrist |
| `mano_s_joints` | (T, 20, 3) | MANO finger joints and tips |
| `obj_s_pose`, `obj_s_id` | (T, 4, 4), () | object pose and its folder under `objects/` |
| `obj_s_scale`, `obj_s_mass` | () | optional mesh scale and mass |

## 2. Preprocess

Writes only `report.csv`.

**Support disks.** Tables and shelves become kinematic disks `[x, y, z_top, r, h]`, fitted where objects rest
still (`desk.*`). `desk.mode=keep` keeps a clip's own disks, `desk.mode=none` uses the floor only.

**Reject rules** (`filters.*`, `filters.enable=false` keeps every clip):

| rule | rejects a clip when |
|---|---|
| `obj_offdisk_untouched` | an untouched object floats where nothing supports it |
| `wrist_acc_spike` | the wrist accelerates faster than 60 m/s² |
| `obj_below_desk` | an untouched object starts or ends below the desk |
| `too_short` | the clip is shorter than 2.5 s |
| `first_unsupported` | an object topples in the first frames |

## 3. Convex parts

`convex=` picks the backend (`config/retarget/convex/`): `coacd` (default, CPU) or `cuacd` (GPU), at most 32
parts per object. Objects that already have parts are skipped (`convex.force=true` redoes them).

## 4. Retarget

`src/retarget/robot_hand/` solves a physics IK in MuJoCo: the robot's palm and fingertip sites follow the MANO
wrist and fingertips while the hands collide with the objects and support disks. Each clip runs `attempts`
times from different starts, and the run with the lowest fingertip error plus `pen_weight` x hand-object
penetration is kept. Clips are kept but flagged in `report.csv` (`flags`) on:

| flag | when |
|---|---|
| `palm_acc_spike` | palm acceleration > 60 m/s² |
| `hand_far_below_desk` | the hand goes > 8 cm below the desk beside it |
| `tip_tracking` | > 35 % of fingertip samples are > 3 cm from MANO |
| `hand_obj_penetration` | hand-object penetration > 2 cm |
| `hand_desk_penetration` | the hand is > 2 cm inside a disk for 5+ frames |

## 5. Pack, USD, SDF

- `pack`: concatenates the kept clips into `motion.pt`, with velocities and fingertip contact labels.
- `usd`: object USDs for Isaac Sim.
- `sdf`: object SDF grids (`sdf.grid_n`, `sdf.extent`), which must match `commands.object.sdf` of training.

### `motion.pt` contents

Per-frame arrays are concatenated over clips; clip `i` covers frames `length_starts[i]` to
`length_starts[i] + motion_num_frames[i]`.

| key | content |
|---|---|
| `joint_pos`, `joint_vel` | robot joint positions / velocities |
| `mano_s_wrist_*`, `mano_s_joints*` | MANO wrist and joints |
| `obj_s_pos`, `_rotmat`, `_vel`, `_angvel` | object trajectory |
| `tips_distance_s`, `contact_contact_s`, `contact_contact_pos_full_s` | fingertip-object distance, contact label and point |
| `motion_num_frames`, `length_starts`, `motion_filename` | per clip: length, first frame, name |
| `motion_object_slot_s`, `s_object_mesh_dirs`, `s_object_mesh_scales` | which object each clip uses |
| `support_disks` | optional per-clip table disks |
