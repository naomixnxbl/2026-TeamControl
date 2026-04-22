# TurtleRabbit — Future Improvements Roadmap

Everything below is **software/AI/decision-making only** — no hardware or network protocol changes.

---

## 1. Ball Physics & Prediction

### Current state
- Linear friction model (`f = 1 - 0.4*dt`) applied per Euler step
- Wall bounces with fixed restitution (0.7)
- No spin, no rolling vs sliding distinction
- Ball prediction code is **duplicated** across striker.py, goalie.py, team.py, and Trajectory.py

### Improvements
- **Quadratic drag model**: Real balls experience `F = -0.5 * rho * A * Cd * v^2`, not linear friction. Massive accuracy gain at high speeds.
- **Spin / Magnus effect**: Track ball angular velocity. A spinning ball curves — this matters for predicting deflections and wall-bounce angles.
- **Rolling vs sliding friction**: Below ~1 m/s balls roll (lower friction); above they slide. Two-regime model.
- **RK4 integration**: Replace Euler stepping with Runge-Kutta 4th order. Removes timestep-dependent drift (~1% error per step currently).
- **Consolidate into one module**: Single `ball_physics.py` with `predict(ball, velocity, dt)` used everywhere. Kills the duplication bug risk.
- **Confidence decay**: Predictions further in time should carry an uncertainty radius that grows. Robots can use this to decide how aggressively to commit.

---

## 2. Opponent Modeling

### Current state
- Goalie detection only (is a robot near opponent goal line?)
- Threat scoring: `distance_to_ball + distance_to_goal`
- No formation recognition, no behavior prediction, no pattern tracking

### Improvements
- **Formation classifier**: Track opponent robot positions over N frames. Cluster into known formations (2-3-1, 3-2-1, etc.). Adjust our formation in response.
- **Attacker trajectory prediction**: For each opponent, extrapolate their heading + speed forward 0.5–1.0s. Predict where they'll be, not where they are. Massively improves marking.
- **Pass interception probability**: Given opponent ball carrier and potential receivers, estimate which passes are likely and position defenders to intercept.
- **Pressing trigger detection**: Detect when opponents are pressing us (multiple robots advancing fast). Trigger long clearance or quick switch of play.
- **Behavioral fingerprinting**: Over the course of a match, build a model of each opponent robot's tendencies (always goes left, always shoots near-post, etc.). Adjust goalie positioning and defender marking.
- **Opponent possession prediction**: Don't just track "who has the ball now" — predict who will have it in 0.5s based on trajectories. Allows preemptive positioning.

---

## 3. Passing & Ball Distribution

### Current state
- Striker: shoot or dribble, no passing
- Team attacker: shoot, pass, or dribble with scoring thresholds
- Through-ball passing exists but only forward
- Pass safety is geometric only (lane clearance check)
- No one-touch passes, no wall passes, no diagonal switches

### Improvements
- **Striker passing**: When shot score is low and a teammate is open, pass instead of dribbling into pressure. The striker currently never passes.
- **One-touch redirect**: When ball is incoming at speed and a teammate is in a good position, redirect with first touch instead of trapping. Already partially implemented in BT striker but not in the main striker.
- **Diagonal long balls**: Switch play from one side to the other with a long diagonal pass. Bypasses compact defenses.
- **Wall passes (give-and-go)**: Pass to teammate, sprint forward, receive return pass. Requires coordination protocol between two robots.
- **Pass-into-space**: Instead of passing to where a teammate IS, pass to where they WILL BE. Use teammate velocity to lead the pass. Team.py has through-balls but they don't account for teammate movement.
- **Back-pass to goalie**: When under extreme pressure, pass back to goalie who can redistribute. Currently never happens.
- **Pass speed selection**: Hard passes through traffic, soft passes to feet. Currently all passes are the same speed.
- **Receiver preparation**: The robot about to receive a pass should pre-orient toward the next target (goal or next pass) before the ball arrives.

---

## 4. Set Pieces

### Current state
- **No set piece handling at all**. The system treats RUNNING state identically regardless of what triggered it. Kickoffs, free kicks, goal kicks, throw-ins, corners, penalties — all handled as generic "ball is here, go get it."

### Improvements
- **Kickoff plays**: Pre-planned 2-3 move sequences from kickoff position. E.g., short pass → through ball → shot.
- **Free kick routines**: When awarded a free kick, position robots in rehearsed formations before kicking. Dummy runners to pull defenders.
- **Corner kicks**: Deliver ball to near/far post with robots positioned to attack. Simple but effective.
- **Goal kicks**: Goalie distributes to specific safe positions rather than blind clearance.
- **Penalty kicks**: Goalie uses opponent history to choose dive direction. Shooter picks corner based on goalie lean.
- **Throw-in equivalent**: Quick restart plays to exploit opponents not yet set.
- **Defensive set pieces**: When opponent has a free kick/corner, position defensively at goal line and mark dangerous positions.

---

## 5. Defensive Strategy

### Current state
- Defenders mark opponents by interpolating between opponent and own goal
- Coordinated pressing exists (2-robot) but only in opponent half
- No zonal defense, no offside trap, no defensive line concept
- Goalie never leaves penalty box

### Improvements
- **Defensive line**: All defenders hold a consistent horizontal line. Line shifts based on ball position. Compresses space.
- **Offside trap**: Defenders step up in unison when opponent is about to pass, catching receiver offside (if rules apply) or at minimum forcing them deep.
- **Zonal defense**: Instead of man-marking, defenders cover zones. More resilient to overlapping runs and player switches.
- **Counter-press**: Immediately after losing the ball, the nearest 2-3 robots swarm the new ball carrier for 3-5 seconds. If unsuccessful, fall back to shape.
- **Goalie sweeper**: When ball is >3m from goal and no shot threat, goalie advances to the edge of the penalty area to act as an extra defender. Intercepts through-balls.
- **Defensive compactness metric**: Track the spread of our defensive shape. If spread > threshold, prioritize getting compact over pressing.
- **Covering runs**: When one defender steps out to press, another slides over to cover the gap.
- **Shot blocking**: Non-goalie robots position themselves between ball and goal when opponent is shooting from distance.

---

## 6. Attacking Intelligence

### Current state
- Shoot when shot score > threshold, pass when pass score > threshold, otherwise dribble
- 5 hardcoded aim points in goal mouth
- No feints, no skill moves, no positional rotation
- Support positions from static offset arrays

### Improvements
- **Expected goals (xG) model**: Instead of 5 fixed aim points, evaluate the probability of scoring from any position/angle. Weight by goalie position, defender positions, shot distance, shot angle.
- **Dribble feints**: When 1v1 with a defender, fake one direction then go the other. Even a simple "move left 100ms then cut right" is effective.
- **Positional rotation**: Attackers swap positions dynamically to confuse markers. If striker drops deep, a midfielder pushes forward to fill the space.
- **Overload one side**: Concentrate 3 robots on one flank, draw defenders, then switch play to the weak side.
- **Third-man runs**: Robot A passes to Robot B, Robot C makes a run behind the defense, Robot B passes to Robot C. Requires 3-robot coordination.
- **Shot timing**: Don't always shoot immediately when aligned. Sometimes dribble one more step closer for a higher-percentage shot.
- **Dynamic aim selection**: Instead of scoring 5 fixed positions, compute the exact aim point that maximizes distance from goalie's predicted position at time of arrival.
- **Chip shots**: When goalie is advanced, chip over them (if hardware supports it). The decision logic for when to chip vs ground shot.

---

## 7. Voronoi Planner Integration

### Current state
- **Complete Voronoi planner exists** (`voronoi_planner/`) with Dijkstra pathfinding, obstacle inflation, field boundary handling
- **Completely unused** by any part of the main AI
- Test/demo code only

### Improvements
- **Passing lane validation**: Use Voronoi edges to find the safest passing corridors. A pass is safe if it travels near a Voronoi edge (maximum distance from all obstacles).
- **Territory control**: Voronoi cells show which robot "owns" which area of the field. Use for role assignment — the robot whose Voronoi cell contains the ball is the natural winner.
- **Gap detection**: Large Voronoi cells in the opponent defense indicate exploitable gaps. Direct attacks there.
- **Positioning optimization**: Support robots should position at Voronoi vertices (equidistant from multiple threats) for maximum coverage.
- **Incremental updates**: Don't recompute full Voronoi each frame. Only update when robots move more than a threshold distance. Cache between frames.
- **Weighted Voronoi**: Weight sites by robot speed/threat level so faster robots get larger cells.

---

## 8. Formation System Integration

### Current state
- **Full formation system exists** (`Formation/`) with position definitions, dynamic adjustment, strategic positioning
- **Completely disconnected** from team.py — the main AI uses hardcoded offset arrays instead
- Import paths are broken (absolute instead of relative)

### Improvements
- **Replace hardcoded offsets**: Team.py's `_SUP_OFFSETS` and `_COUNTER_OFFSETS` arrays should be generated from the formation system based on current game state.
- **Formation switching**: 3-2-1 when attacking, 4-1-1 when defending, 2-3-1 in transition. Switch dynamically based on possession state.
- **Ball-relative formations**: Formation shifts as a unit based on ball position. Already partially designed in strategic_position.py.
- **Fix imports**: Change absolute imports to relative so the package works.
- **Add 6v6 formation definitions**: Current formations are designed for 11v11. Need proper 6v6 variants.
- **Role-formation binding**: Each formation position has an associated role (attacker/midfielder/defender). Role assignment reads from formation.

---

## 9. State Estimation & Filtering

### Current state
- Raw vision data used directly — no filtering
- Ball velocity from first/last history point (noisy)
- No robot velocity estimation from vision
- No latency compensation
- No sensor fusion

### Improvements
- **Kalman filter for ball**: Track (x, y, vx, vy) with process noise. Smooths jitter, predicts through occlusion, gives velocity for free.
- **Extended Kalman filter for robots**: Track (x, y, theta, vx, vy, omega). Know not just where robots are but how fast they're moving and turning.
- **Latency compensation**: Vision data is 10-30ms old by the time we act. Predict current state forward by the latency amount.
- **Multi-camera fusion**: If multiple cameras see the same ball, fuse measurements weighted by confidence.
- **Occlusion handling**: When ball disappears behind a robot, continue predicting with the Kalman filter rather than losing track.
- **Velocity-weighted ball history**: Recent samples should matter more than old ones. Exponential weighting on the velocity estimation.

---

## 10. Decision Architecture

### Current state
- Hard-coded if/else chains with fixed thresholds
- Shoot if score > 0.16, pass if score > 0.05, otherwise dribble
- No lookahead, no simulation, no learning

### Improvements
- **Utility-based AI**: Instead of threshold checks, compute a utility score for every possible action (shoot, pass to each teammate, dribble in each direction). Execute the highest-utility action. Much more flexible than if/else.
- **Monte Carlo Tree Search (MCTS)**: Simulate 100-1000 possible futures from current state. Each simulation: pick action → simulate physics → opponent responds → evaluate. Picks the action that leads to the best average outcome. Used by AlphaGo, works great for multi-agent games.
- **Minimax with alpha-beta pruning**: For critical 1v1 situations (striker vs goalie, 1v1 dribble). Look 2-3 moves ahead and pick the move that maximizes our minimum outcome.
- **Hierarchical task network**: High-level plan ("score a goal") decomposes into sub-plans ("get behind ball" → "dribble forward" → "shoot"). More robust than flat state machines.
- **Reinforcement learning**: Train a neural network policy on thousands of simulated games. Start with imitation learning from the current rule-based system, then improve with self-play. Long-term but highest ceiling.

---

## 11. Coordination & Communication

### Current state
- Implicit coordination: team.py assigns roles and targets centrally
- No robot-to-robot communication of intent
- No shared plan or negotiation

### Improvements
- **Intent broadcasting**: Each robot publishes its intended target position and action. Others can avoid conflicts and position complementarily.
- **Auction-based role assignment**: Instead of centralized winner selection, robots "bid" for roles based on their ability to execute. The best-positioned striker bids highest for the attacker role.
- **Cooperative path planning**: When two robots need to go through the same narrow space, negotiate who goes first instead of both trying to avoid each other.
- **Pass request protocol**: A robot in a good position can "call for the ball" by signaling readiness. The ball carrier sees the request and evaluates the pass.
- **Defensive assignments**: Share marking assignments explicitly. "I've got #3, you take #5." Prevents double-marking and leaving opponents free.

---

## 12. Testing & Validation

### Current state
- ~7 test files, all for network/protocol parsing
- **Zero tests for any AI behavior**
- No simulation-based testing
- No regression suite

### Improvements
- **Unit tests for every AI function**: `_pick_aim()`, `_best_intercept()`, `compute_arc_nav()`, `_shot_score()`, `_pass_score()`, all scoring functions.
- **Scenario tests**: Set up specific game situations (ball at position X, robots at positions Y) and verify the correct decision is made. "When striker has ball at penalty spot with clear shot, it should shoot."
- **Regression tests**: Every bug fix gets a test that reproduces the bug. Prevents re-introduction.
- **Simulation harness**: Run full games in headless gRSim with automated scoring. Measure goals scored/conceded per game, possession %, pass completion %. Compare before/after changes.
- **A/B testing framework**: Run two versions of the AI against each other for N games. Statistically determine which is better.
- **Continuous integration**: Every code change triggers the test suite. Block merges that fail tests.
- **Performance benchmarks**: Measure decision loop latency. Alert if it exceeds frame interval (16ms).

---

## 13. Code Quality & Architecture

### Current state
- Ball prediction duplicated in 4 files
- Formation and Voronoi systems built but unused
- Legacy Movement.py partially overlaps path_planner.py
- time_to_intercept.py has broken imports
- Mixed coding styles

### Improvements
- **Single ball physics module**: One `ball_predict(pos, vel, dt)` function used everywhere.
- **Deprecate Movement.py**: Everything useful is in path_planner.py. Remove the dead code.
- **Fix or remove time_to_intercept.py**: Broken imports, not used anywhere.
- **Integrate or archive Voronoi planner**: Either wire it into team.py or move it to an `experimental/` folder so it's not confusing.
- **Fix Formation imports**: Relative imports so the package actually works.
- **Type hints everywhere**: All function signatures should have type annotations. Catches bugs early.
- **Structured logging**: Replace `print()` statements with proper leveled logging. Enable per-module log levels.
- **Configuration validation**: Constants.py should validate ranges (speeds > 0, distances > 0, gains in [0, 10], etc.).

---

## 14. Performance & Real-time

### Current state
- Decision loop runs at ~60Hz (16ms per tick)
- Voronoi planner (if enabled) would be expensive per-frame
- No spatial indexing for nearest-robot queries
- No caching of expensive computations

### Improvements
- **Spatial hashing**: For nearest-robot and obstacle queries, use a grid-based spatial hash. O(1) lookup instead of O(n) scan.
- **Lazy Voronoi**: Only recompute Voronoi when robot positions change by more than a threshold. Cache between frames.
- **Decision budget**: If the decision loop takes too long, skip non-critical computations (support positioning, formation adjustment) and only compute the critical path (ball winner, goalie).
- **Vectorized ball prediction**: Use numpy for batch predictions instead of Python loops. 10-100x speedup on prediction-heavy operations.
- **Profile-guided optimization**: Profile the decision loop to find the actual bottleneck. Optimize that, not guesses.

---

## 15. Game Intelligence

### Improvements that don't fit elsewhere
- **Clock management**: When winning with little time left, possess the ball in safe areas rather than attacking. When losing, increase risk tolerance.
- **Momentum tracking**: If we've scored 2 goals in 2 minutes, our AI is working — don't change strategy. If we've conceded 2, switch to a more defensive formation.
- **Fatigue simulation**: Real robots have battery limits. If a robot has been sprinting for 30s straight, prefer giving it a positioning role so it can "rest" (lower speed commands preserve battery).
- **Substitution strategy**: If a robot malfunctions or is removed, immediately redistribute roles. Don't leave a formation gap.
- **Adaptable aggression**: Tunable parameter from 0 (ultra-defensive) to 1 (ultra-attacking) that shifts all thresholds. Expose in UI for real-time adjustment during matches.
- **Post-goal behavior**: After scoring, immediately sprint back to formation for the kickoff instead of celebrating (wasting time).
- **Timeout strategy**: Use timeouts (if allowed) to break opponent momentum and adjust parameters.

---

## Priority Matrix

| Improvement | Impact | Effort | Priority |
|---|---|---|---|
| Consolidate ball physics into one module | Medium | Low (2h) | **Do first** |
| Kalman filter for ball state | High | Medium (8h) | **Do first** |
| Striker passing ability | High | Medium (6h) | **Do first** |
| Set piece plays (kickoff + free kick) | High | Medium (12h) | **Do first** |
| Opponent trajectory prediction | High | Medium (8h) | **High** |
| Integrate formation system | Medium | Medium (6h) | **High** |
| Unit test suite for AI | High | High (20h) | **High** |
| Defensive line + compactness | High | Medium (10h) | **High** |
| Voronoi integration for passing lanes | Medium | Medium (8h) | **Medium** |
| Utility-based action selection | High | High (16h) | **Medium** |
| One-touch passing | Medium | Low (4h) | **Medium** |
| Goalie sweeper mode | Medium | Low (4h) | **Medium** |
| Counter-press after losing ball | Medium | Low (4h) | **Medium** |
| Latency compensation | Medium | Medium (6h) | **Medium** |
| MCTS for multi-step planning | Very High | Very High (40h) | **Long-term** |
| Reinforcement learning | Very High | Very High (80h+) | **Long-term** |
| Opponent behavioral fingerprinting | High | High (24h) | **Long-term** |
| Graph neural networks for coordination | Very High | Very High (60h+) | **Long-term** |
