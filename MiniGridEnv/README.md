<p align="center">
  <img src="MiniGridEnv/Media/EmptyEnv.gif" width="500">
</p>

# MiniGrid Empty Environment

This environment is an **empty room** where the agent must reach the **green goal square** to obtain a sparse reward.  
It is useful for validating RL algorithms in **small rooms** and testing **exploration in sparse-reward settings** in **large rooms**.  

---

## Action Space
| Action  | Meaning       |
|---------|---------------|
| 0       | Turn left     |
| 1       | Turn right    |
| 2       | Move forward  |
| 3       | Pickup (unused) |
| 4       | Drop (unused)   |
| 5       | Toggle (unused) |
| 6       | Done (unused)   |

- The action space is `Discrete(7)`.  

---

## Observation Space
`Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace)`

- **direction**: The agent’s facing direction (0–3).  
- **image**: A `(7 × 7 × 3)` RGB observation of the visible environment.  
- **mission**: A text string describing the task.  

### Encoding
- Each grid tile is represented as a tuple: `(OBJECT_IDX, COLOR_IDX, STATE)`  
- **Objects** and **colors** are mapped in `minigrid/core/constants.py`.  
- **STATE** (for doors):  
  - `0 = open`  
  - `1 = closed`  
  - `2 = locked`  

---

## Description
- The agent starts at a random location in the room.  
- The mission is always **"get to the green goal square"**.  
- The goal is placed randomly.  
- The environment is empty (no walls or obstacles other than borders).  

---

## Starting State
- The agent’s starting position and direction are **randomized** at the beginning of each episode.  

---

## Rewards
| Event            | Reward |
|------------------|--------|
| Reach goal       | `1 - 0.9 × (step_count / max_steps)` |
| Timeout/failure  | 0      |

---

## Episode End
**Termination occurs if:**
- The agent reaches the goal.  
- The step limit (`max_steps`) is reached.  

---
