# unitree_rl_lab overrides

`/home/nvidia/unitree_rl_lab`(upstream `git clone`) 위에서 우리가 수정/추가한 파일들의 백업.
디렉토리 구조는 원본 그대로 유지 (이 폴더 기준 상대경로 = `/home/nvidia/unitree_rl_lab/` 기준 상대경로).

복사 시점: 2026-05-22.
원본 레포 HEAD: `4960b84` (`doc: add comment to generate npz file befroe traning mimic task`).

## 파일 목록과 원본 위치

### Modified (upstream 파일을 수정)

| 이 폴더 안 경로 | 원본 위치 | 변경 요약 |
|---|---|---|
| `scripts/rsl_rl/play.py` | `/home/nvidia/unitree_rl_lab/scripts/rsl_rl/play.py` | `import posefork.tasks` 추가, ModeConditionedActorCritic처럼 `.actor`/`.student`가 없는 policy는 jit/onnx export 스킵 |
| `scripts/rsl_rl/train.py` | `/home/nvidia/unitree_rl_lab/scripts/rsl_rl/train.py` | `import posefork.tasks` 추가, `--task` argparse `choices` 제약 제거 |
| `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` | `/home/nvidia/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` | `UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR` placeholder → `os.environ.get(..., "/home/nvidia/...")` |

### Added (untracked, 우리가 새로 만든 것)

| 이 폴더 안 경로 | 원본 위치 | 목적 |
|---|---|---|
| `scripts/probe_teacher_diversity.py` | `/home/nvidia/unitree_rl_lab/scripts/probe_teacher_diversity.py` | teacher 분포 probe |
| `scripts/rsl_rl/visualize_arm_reach.py` | `/home/nvidia/unitree_rl_lab/scripts/rsl_rl/visualize_arm_reach.py` | arm reach 시각화 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/manipulation/` (디렉토리 전체) | 동일 경로 | g1_upper_body arm_reach task 등록 (env_cfg, mdp, agents) |

`__pycache__/` 는 복사에서 제외.

## 복원 방법

upstream을 새로 clone한 뒤 이 폴더 내용을 그대로 덮어쓰기:

```bash
rsync -av ~/teacher-without-a-human/code/posefork/unitree_rl_lab_overrides/ ~/unitree_rl_lab/
```
