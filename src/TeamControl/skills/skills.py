"""Skills registry — imports every skill and exposes BEHAVIOURS / BEHAVIOURS_BY_ID.

To add a new skill: create its .py file in this folder, import it below,
and add a Behaviour entry to the BEHAVIOURS tuple.
"""
from TeamControl.skills._shared import Behaviour, reset_robot_state

from TeamControl.skills.stop                    import stop, compliance as _stop_compliance
from TeamControl.skills.face_ball               import face_ball
from TeamControl.skills.face_target             import face_target
from TeamControl.skills.move_to_ball            import move_to_ball
from TeamControl.skills.move_to_point           import move_to_point
from TeamControl.skills.intercept_ball          import intercept_ball
from TeamControl.skills.dribble_to_point        import dribble_to_point
from TeamControl.skills.kick_at_goal            import kick_at_goal
from TeamControl.skills.kick_at_point           import kick_at_point
from TeamControl.skills.hold_goal_line          import hold_goal_line
from TeamControl.skills.penalty_attacker_stance import penalty_attacker_stance
from TeamControl.skills.kickoff_stance          import kickoff_stance
from TeamControl.skills.move_then_attack        import move_then_attack
from TeamControl.skills.goalie_intercept        import goalie_intercept
from TeamControl.skills.defender_block          import defender_block

BEHAVIOURS: tuple[Behaviour, ...] = (
    # ── Primitives ────────────────────────────────────────────────────────────
    Behaviour("stop",             "Stop (§5.4)",           "Hold position, nudge away if < 0.55 m from ball, cap speed at 1.4 m/s — full STOPPED compliance.", False, stop, _stop_compliance),
    Behaviour("face_ball",        "Face Ball",             "Rotate in place to face the ball.",                                                           False, face_ball),
    Behaviour("face_target",      "Face Target",           "Rotate in place to face a chosen point.",                                                     True,  face_target),
    Behaviour("move_to_ball",     "Move To Ball",          "Face ball then drive toward it, stopping 15 cm short.",                                       False, move_to_ball),
    Behaviour("move_to_point",    "Move To Point",         "Drive to a chosen field position.",                                                           True,  move_to_point),
    Behaviour("intercept_ball",   "Intercept Ball",        "Move to where ball will be in ~0.8 s (velocity prediction), not its current position.",       False, intercept_ball),
    Behaviour("dribble_to_point", "Dribble To Point",      "Carry (dribble) ball to a chosen point — ball placement (§9).",                              True,  dribble_to_point),
    # ── Kick / pass ───────────────────────────────────────────────────────────
    Behaviour("kick_at_goal",     "Kick At Goal",          "Get behind ball → align heading → strike toward the opponent goal.",                          False, kick_at_goal),
    Behaviour("kick_at_point",    "Pass (Kick To Point)",  "Get behind ball → align heading → strike toward a chosen point.",                            True,  kick_at_point),
    # ── Rule-based (SSL rulebook) ─────────────────────────────────────────────
    Behaviour("hold_goal_line",            "Hold Goal Line",               "Keeper holds on own goal line tracking ball y — §8.2 PENALTY_DEFEND.",        False, hold_goal_line),
    Behaviour("penalty_attacker_stance",   "Penalty Attacker Stance",      "Move to the penalty spot facing opponent goal — §8.2.3 PENALTY_SHOOT.",       False, penalty_attacker_stance),
    Behaviour("kickoff_stance",            "Kickoff Stance",               "Move to centre-circle edge, own half, facing ball — §5.3.2 PREPARE_KICKOFF.", False, kickoff_stance),
    # ── Strategy roles ────────────────────────────────────────────────────────
    Behaviour("move_then_attack",  "Attack (Pass / Shoot)", "Move to ball → pass to centre if own half, shoot at goal if enemy half.",                    False, move_then_attack),
    Behaviour("goalie_intercept",  "Goalie — Intercept",    "Sprint to predicted ball crossing on goal line; speed scales with shot speed.",              False, goalie_intercept),
    Behaviour("defender_block",    "Defender — Block",      "Cover goal mouth segment the goalie is not protecting.",                                     False, defender_block),
)

BEHAVIOURS_BY_ID: dict[str, Behaviour] = {b.id: b for b in BEHAVIOURS}
