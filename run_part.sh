#!/usr/bin/env bash
set -u
variant=$1; seed=$2
OMNI="--cmd_profile omni --track_sigma 0.15 --gait_clock --gait_rewards --slip_coef 0.1"
COMMON="--init runs/omni_scratch/latest.pt --focus_prob 0.5 --style_teacher runs/omni_teacher_front/latest.pt \
  --rear_teacher runs/omni_teacher_rear/latest.pt --bc_coef 1.0 --steps 3e6 --workers 1 --lr 1e-4"
if [ "$variant" = anchored ]; then
  EXTRA="--anchor_teacher runs/omni_scratch/latest.pt --anchor_coef 0.2"
else
  EXTRA=""
fi
python3 ppo.py --run p5_${variant}_s$seed --seed $seed $OMNI $COMMON $EXTRA
