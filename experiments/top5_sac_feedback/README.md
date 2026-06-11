# Top-5 SAC Feedback Experiment

This experiment compares two Space Invaders SAC students:

- `Baseline`: standard Discrete SAC from `training/space_invaders/sac_atari_v2.py`
- `Guided`: the same SAC student plus one end-of-episode actor correction on the top-5 states where the pre-trained D3QN teacher sees the largest action gap

Run from the project root:

```bash
python experiments/top5_sac_feedback/compare_sac_student.py --total-frames 200000
```

By default this experiment now matches the teacher's training environment:

- `FRAME_SKIP=15`
- `TERMINAL_ON_LIFE_LOSS=True`

Quick smoke test:

```bash
python experiments/top5_sac_feedback/compare_sac_student.py --guided-only --total-frames 4000 --eval-freq 4000 --learning-start 50 --feedback-start 50 --no-plot
```

Outputs are saved to:

```text
experiments/top5_sac_feedback/outputs/
```

Expected files:

- `eval_history.json`
- `eval_history.csv`
- `sac_student_comparison.png`

## D3QN Student Variant

The D3QN variant uses the same teacher top-k moment idea, but the feedback is
now buffered and replayed during regular D3QN updates. At episode end, high-gap
teacher moments are stored in a feedback buffer. Each D3QN train step can then
add a KL distillation loss from teacher Q-value preferences to student Q-values.

It imports the teacher-training D3QN implementation from `experiments/d3qn`, so
the student uses the same environment wrapper and default `FRAME_SKIP=15`,
`TERMINAL_ON_LIFE_LOSS=True` setup.

Run:

```bash
python experiments/top5_sac_feedback/compare_d3qn_student.py --total-frames 1000000
```

Recommended run without overwriting previous outputs:

```bash
python experiments/top5_sac_feedback/compare_d3qn_student.py \
  --total-frames 300000000 \
  --eval-freq 1000000 \
  --eval-episodes 5 \
  --top-k 20 \
  --feedback-weight 0.3 \
  --feedback-buffer-size 50000 \
  --feedback-batch-size 16 \
  --distill-temperature 2.0 \
  --checkpoint-freq 5000000 \
  --keep-checkpoints 2 \
  --output-dir experiments/top5_sac_feedback/outputs_d3qn_kl_300m
```

The D3QN comparison runs `Guided` first, then `Baseline`. Histories and the plot
are refreshed after each evaluation, and rolling checkpoints are stored under
`d3qn_checkpoints/{guided,baseline}/` with only the latest two checkpoints kept
per run label.

Resume examples:

```bash
python experiments/top5_sac_feedback/compare_d3qn_student.py \
  --guided-only \
  --total-frames 300000000 \
  --resume-guided experiments/top5_sac_feedback/outputs_d3qn_kl_300m/d3qn_checkpoints/guided/guided_frame_000100000005.pth \
  --output-dir experiments/top5_sac_feedback/outputs_d3qn_kl_300m
```

Quick smoke test:

```bash
python experiments/top5_sac_feedback/compare_d3qn_student.py --guided-only --total-frames 1200 --eval-freq 1200 --learning-start 50 --feedback-start 50 --replay-capacity 1000 --no-plot
```

D3QN outputs:

- `d3qn_eval_history.json`
- `d3qn_eval_history.csv`
- `d3qn_student_comparison.png`
- `d3qn_checkpoints/guided/*.pth`
- `d3qn_checkpoints/baseline/*.pth`
