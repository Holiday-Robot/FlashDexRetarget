# Sharpa Wave hand

Meshes, URDF and MJCF derived from the [Sharpa Wave hand repository](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
(Apache License 2.0, see `LICENSE`). Changes from the original files:

1. Sites added for retargeting and tracking (`track_hand_*`, `contact_*`, palm).
2. Collision meshes replaced by primitive capsules to cut simulation time.
3. Thumb and palm meshes adjusted to avoid self-collisions.
4. Left and right hands combined into `bimanual.xml` / `urdf/sharpa_bimanual.urdf`.
