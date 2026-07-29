# Big-network dexterity (2026-07-30)

Capacity ablation settled it: at identical settings (2.6 rad relative goals,
deterministic, 128 envs) the 3x512 actor (78_carry_bignet_s33, trained from
scratch, 200k steps) delivers gpe 5.38 / drop 49% vs the 2x128 baselines
(73: 2.95 / 57%, 77 antidrop: 2.69 / 58%).  +82% goal throughput, -8pp drops.
Clips at the hardest setting; green flash = reach, red = 5s timeout skip.
