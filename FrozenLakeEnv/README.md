<p align="center">
  <img src="https://raw.githubusercontent.com/tahamm786/Autonomous-Car-Driving-using-Deep-RL/main/FrozenLakeEnv/Media/frozen_lake.gif" width="500">
</p>

# Frozen Lake Environment

This environment is part of the **Toy Text environments** which contains general information about the environment.

---

## Action Space
| Action | Meaning      |
|--------|-------------|
| 0      | Move left   |
| 1      | Move down   |
| 2      | Move right  |
| 3      | Move up     |

- The action shape is `(1,)` in the range `{0, 3}` indicating which direction to move the player.

---

## Observation Space
- The observation is an integer representing the player’s current position:  
  `current_row * ncols + current_col` (where both `row` and `col` start at 0).

- Example: For a 4x4 map, the goal position `[3,3]` is calculated as `3 * 4 + 3 = 15`.

- The number of possible observations depends on the size of the map.

---

## Description
- The game starts with the player at location `[0,0]` with the goal at `[3,3]` (for a 4x4 environment).  
- Holes in the ice are distributed in fixed locations (pre-determined map) or random locations (random map).  
- The player continues moving until they **reach the goal** or **fall into a hole**.  
- The lake is slippery (unless disabled) — the player may move perpendicular to the intended direction (`is_slippery`).  
- Randomly generated worlds will **always have a path** to the goal.  

---

## Starting State
- The episode always starts with the player in state `[0]` (location `[0,0]`).  

---

## Rewards
| Event        | Reward |
|--------------|--------|
| Reach goal   | +1     |
| Reach hole   | 0      |
| Reach frozen | 0      |

---

## Episode End
**Termination occurs if:**
- The player moves into a hole.  
- The player reaches the goal at `(max(nrow) * max(ncol) - 1)`.  

**Truncation occurs if (with time limit wrapper):**
- The episode length exceeds:  
  - **100 steps** for `FrozenLake-v1 (4x4)`  
  - **200 steps** for `FrozenLake8x8-v1`  

---
