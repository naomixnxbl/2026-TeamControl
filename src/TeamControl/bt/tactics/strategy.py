"""Team-strategy layer — context-aware team posture for the whole team.

This layer sits ABOVE the role_swap weights and behavior_tree thresholds in
``bt_tuning.yaml``. It exposes a small set of intuitive posture dials
(aggression / formation shape / role stickiness / shooting eagerness / passing
caution / defensive line height) and — crucially — lets those dials change
*dynamically with the live game context*:

    * where the ball is (our third / middle / attacking third),
    * who has the ball (own / opponent / loose),
    * the scoreline (leading / level / trailing, by what margin).

That is the thing static yaml weights cannot do: the weights are fixed
constants, whereas a real team presses higher when the ball is deep in the
opponent half, sits back to protect a late lead, and throws numbers forward
when chasing the game. Each tick the Coordinator builds a :class:`GameContext`
from the Snapshot, fires whichever ``rules`` match, composes their dial deltas
on top of the always-on ``base`` posture, and rewrites the low-level configs
for that tick only.

Two safety guarantees keep the default behaviour identical to before:

1. ``strategy.enabled`` defaults to ``False``. While false, every
   ``apply_strategy_*`` transform returns its input unchanged and the
   Coordinator never recomputes anything — the layer is a complete no-op.
2. Every dial defaults to its neutral value (scales ``1.0``, offsets ``0``) and
   ``rules`` defaults to empty. So even after enabling the layer, nothing
   changes until a dial is moved or a rule is added.

Composition: when several rules fire, their *scale* dials multiply together and
their *offset* dials add together, all on top of the base posture. Because the
Coordinator always transforms from pristine base copies, scales never compound
across ticks.

The per-dial transforms are written generically against dataclass *field names*
(via :func:`dataclasses.replace`) so this module stays decoupled from the
concrete role/tree config classes and free of import cycles.
"""
from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover - depends on libyaml availability
    from yaml import Loader

BT_TUNING_FILENAME = "bt_tuning.yaml"
LEGACY_HEURISTIC_WEIGHT_FILENAME = "heuristic_weight.yaml"

T = TypeVar("T")

# Score-weight fields treated as "pursuit / pressure" terms. ``press_scale``
# multiplies these on top of the per-role weight scale so a single dial can
# make the team hunt the ball harder (or sit off) without re-tuning each role.
_ATTACKER_PRESS_FIELDS: tuple[str, ...] = (
    "ball_close",
    "approach_quality",
    "opponent_has_ball_pressure",
    "loose_ball_pressure",
)
_DEFENDER_PRESS_FIELDS: tuple[str, ...] = (
    "ball_close",
    "opponent_has_ball",
)

# Role-stability fields scaled by ``role_stickiness_scale``. Higher values make
# robots keep their current role longer (less role churn).
_DEFENDER_STABILITY_STICKY_FIELDS: tuple[str, ...] = (
    "stay_bias",
    "cooldown_bias",
    "min_hold_seconds",
    "release_margin",
    "allow_attacker_release_margin",
)

# Defender positioning fractions are clamped to this range after scaling so a
# large ``line_height_scale`` can never push a defender past the ball holder
# (fraction 1.0) or behind its own goal (fraction < 0).
_DEFENDER_FRACTION_MIN: float = 0.0
_DEFENDER_FRACTION_MAX: float = 0.95

# Defaults used when deriving the live game context.
DEFAULT_POSSESSION_RADIUS: float = 0.5    # m — closer than this counts as control
DEFAULT_FIELD_HALF_LENGTH: float = 4.5    # m — Div B goal line at x = ±4.5
DEFAULT_THIRD_FRACTION: float = 1.0 / 3.0  # |progress| past this = a "third"

_BALL_ZONES = ("defensive", "middle", "attacking")
_POSSESSION_STATES = ("own", "opponent", "loose")
_SCORELINES = ("leading", "level", "trailing")


# ---------------------------------------------------------------------------
# Posture dials (all neutral at default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleBiasStrategy:
    """How strongly each role competes during dynamic role assignment.

    Only meaningful when ``heuristic_role_swap`` is enabled (it is in
    ``sim_6v6.yaml``). Each scale multiplies that role's overall suitability
    score, so raising one role's scale makes more robots gravitate to it.
    """

    attacker_weight_scale: float = 1.0
    defender_weight_scale: float = 1.0
    supporter_weight_scale: float = 1.0
    # Extra multiplier applied to the ball-pursuit terms of attacker AND
    # defender scores. >1 = press the ball harder, <1 = sit off and contain.
    press_scale: float = 1.0


@dataclass(frozen=True)
class FormationStrategy:
    """Additive changes to how many robots fill each role.

    Only meaningful when ``heuristic_role_swap`` is enabled. These are added to
    the ``role_swap.role_targets`` counts after they are loaded.
    """

    extra_attackers: int = 0
    extra_defenders: int = 0   # added to both min and max defenders
    extra_supporters: int = 0


@dataclass(frozen=True)
class StabilityStrategy:
    """Global role-churn damping.

    ``role_stickiness_scale`` multiplies every role-swap hysteresis term
    (current/cooldown biases, minimum swap interval, defender hold/stay biases).
    >1 = robots commit to a role longer; <1 = roles react faster but flicker
    more. Only meaningful with ``heuristic_role_swap`` enabled.
    """

    role_stickiness_scale: float = 1.0


@dataclass(frozen=True)
class AttackingStrategy:
    """Shooting / passing posture for the attacker and supporter trees."""

    # Multiplies the max distance from goal at which a clear shot is allowed.
    # >1 = takes longer-range shots; <1 = carries/passes closer first.
    shoot_distance_scale: float = 1.0
    # Multiplies the attacker's allowed heading error before shooting.
    # >1 = shoots with looser aim (more shots); <1 = waits for a cleaner line.
    shot_alignment_tolerance_scale: float = 1.0
    # Multiplies the continuous-possession ticks required before shooting.
    # <1 = shoots sooner after gaining control; >1 = settles longer first.
    settle_time_scale: float = 1.0
    # Multiplies the "is a receiver marked / lane blocked" thresholds.
    # >1 = more cautious passing (needs more space / clearer lanes);
    # <1 = riskier passes into tighter windows.
    pass_caution_scale: float = 1.0


@dataclass(frozen=True)
class DefendingStrategy:
    """Defensive shape for the defender positioning tree."""

    # Multiplies how far up the goal->ball line the defender parks.
    # >1 = a higher, more proactive defensive line (steps toward the ball);
    # <1 = a deeper line that parks closer to our own goal.
    line_height_scale: float = 1.0


@dataclass(frozen=True)
class PostureDials:
    """A complete set of posture dials. Used for the base posture, for each
    rule's adjustment, and for the composed per-tick result."""

    role_bias: RoleBiasStrategy = field(default_factory=RoleBiasStrategy)
    formation: FormationStrategy = field(default_factory=FormationStrategy)
    stability: StabilityStrategy = field(default_factory=StabilityStrategy)
    attacking: AttackingStrategy = field(default_factory=AttackingStrategy)
    defending: DefendingStrategy = field(default_factory=DefendingStrategy)

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "PostureDials":
        if not isinstance(raw, MappingABC):
            return cls()
        return cls(
            role_bias=_dataclass_from_section(RoleBiasStrategy, raw.get("role_bias")),
            formation=_dataclass_from_section(FormationStrategy, raw.get("formation")),
            stability=_dataclass_from_section(StabilityStrategy, raw.get("stability")),
            attacking=_dataclass_from_section(AttackingStrategy, raw.get("attacking")),
            defending=_dataclass_from_section(DefendingStrategy, raw.get("defending")),
        )


# ---------------------------------------------------------------------------
# Live game context + conditional rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameContext:
    """Snapshot-derived game situation the strategy reacts to."""

    ball_zone: str          # one of _BALL_ZONES, relative to our attack direction
    possession: str         # one of _POSSESSION_STATES
    scoreline: str          # one of _SCORELINES
    ball_in_our_half: bool
    goal_margin: int        # own_score - enemy_score (positive = we lead)


@dataclass(frozen=True)
class RuleConditions:
    """Predicate over a :class:`GameContext`. ``None`` means "don't care".

    A rule fires only when ALL of its non-None conditions match.
    """

    ball_zone: str | None = None
    possession: str | None = None
    scoreline: str | None = None
    ball_in_our_half: bool | None = None
    # Fires only when the absolute goal margin is at least this many goals
    # (combine with ``scoreline`` to make it directional, e.g. leading by 2+).
    goal_margin_at_least: int | None = None

    def matches(self, ctx: GameContext) -> bool:
        if self.ball_zone is not None and self.ball_zone != ctx.ball_zone:
            return False
        if self.possession is not None and self.possession != ctx.possession:
            return False
        if self.scoreline is not None and self.scoreline != ctx.scoreline:
            return False
        if (
            self.ball_in_our_half is not None
            and self.ball_in_our_half != ctx.ball_in_our_half
        ):
            return False
        if (
            self.goal_margin_at_least is not None
            and abs(ctx.goal_margin) < self.goal_margin_at_least
        ):
            return False
        return True

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "RuleConditions":
        if not isinstance(raw, MappingABC):
            return cls()
        values: dict[str, object] = {}
        for key in ("ball_zone", "possession", "scoreline"):
            if key in raw and raw[key] is not None:
                values[key] = str(raw[key]).strip().lower()
        if "ball_in_our_half" in raw and raw["ball_in_our_half"] is not None:
            values["ball_in_our_half"] = _coerce_config_value(
                raw["ball_in_our_half"], False
            )
        if "goal_margin_at_least" in raw and raw["goal_margin_at_least"] is not None:
            values["goal_margin_at_least"] = int(raw["goal_margin_at_least"])
        return cls(**values)


@dataclass(frozen=True)
class StrategyRule:
    """A context-triggered posture adjustment."""

    name: str = "unnamed"
    when: RuleConditions = field(default_factory=RuleConditions)
    apply: PostureDials = field(default_factory=PostureDials)

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "StrategyRule":
        if not isinstance(raw, MappingABC):
            return cls()
        return cls(
            name=str(raw.get("name", "unnamed")),
            when=RuleConditions.from_mapping(raw.get("when")),
            apply=PostureDials.from_mapping(raw.get("apply")),
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Top-level team strategy. Disabled, neutral, and rule-free by default.

    The five posture dials at the top level form the always-on *base* posture
    (used while ``enabled`` is true). ``rules`` adjust that base depending on
    the live :class:`GameContext`.
    """

    enabled: bool = False
    role_bias: RoleBiasStrategy = field(default_factory=RoleBiasStrategy)
    formation: FormationStrategy = field(default_factory=FormationStrategy)
    stability: StabilityStrategy = field(default_factory=StabilityStrategy)
    attacking: AttackingStrategy = field(default_factory=AttackingStrategy)
    defending: DefendingStrategy = field(default_factory=DefendingStrategy)
    rules: tuple[StrategyRule, ...] = ()

    @property
    def base_dials(self) -> PostureDials:
        """The always-on base posture as a :class:`PostureDials`."""
        return PostureDials(
            role_bias=self.role_bias,
            formation=self.formation,
            stability=self.stability,
            attacking=self.attacking,
            defending=self.defending,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "StrategyConfig":
        """Build a config from an already-parsed ``strategy:`` mapping."""
        if not isinstance(raw, MappingABC):
            return cls()
        defaults = cls()
        rules_raw = raw.get("rules")
        rules = (
            tuple(StrategyRule.from_mapping(item) for item in rules_raw)
            if isinstance(rules_raw, list)
            else ()
        )
        return cls(
            enabled=_coerce_config_value(
                raw.get("enabled", defaults.enabled), defaults.enabled
            ),
            role_bias=_dataclass_from_section(RoleBiasStrategy, raw.get("role_bias")),
            formation=_dataclass_from_section(FormationStrategy, raw.get("formation")),
            stability=_dataclass_from_section(StabilityStrategy, raw.get("stability")),
            attacking=_dataclass_from_section(AttackingStrategy, raw.get("attacking")),
            defending=_dataclass_from_section(DefendingStrategy, raw.get("defending")),
            rules=rules,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_strategy_config(
    config_filename: str | Path = BT_TUNING_FILENAME,
) -> StrategyConfig:
    """Load the ``strategy:`` section from yaml, preserving defaults.

    A missing file or missing ``strategy:`` section yields the all-default
    (disabled, neutral, rule-free) config, so the team behaves exactly as
    before.
    """
    path = _resolve_utils_config_path(config_filename)
    if not path.exists():
        return StrategyConfig()

    with open(path, "r") as f:
        raw = yaml.load(f, Loader) or {}
    if not isinstance(raw, MappingABC):
        return StrategyConfig()

    return StrategyConfig.from_mapping(raw.get("strategy"))


def is_strategy_active(strategy: StrategyConfig | None) -> bool:
    """Return True only when a non-None strategy has its master switch on."""
    return bool(strategy is not None and strategy.enabled)


# ---------------------------------------------------------------------------
# Context evaluation + per-tick resolution
# ---------------------------------------------------------------------------


def evaluate_game_context(
    snapshot: Any,
    attack_sign: float,
    *,
    possession_radius: float = DEFAULT_POSSESSION_RADIUS,
    field_half_length: float = DEFAULT_FIELD_HALF_LENGTH,
    third_fraction: float = DEFAULT_THIRD_FRACTION,
) -> GameContext:
    """Derive the live :class:`GameContext` from a Snapshot.

    ``attack_sign`` is the Coordinator's ``_attack_sign`` (+1 when we attack
    toward +x, -1 toward -x), so "attacking third" always means *near the
    opponent goal* regardless of which half we defend.
    """
    bx, _by = snapshot.ball_position
    progress = (bx * attack_sign) / max(field_half_length, 1e-6)
    if progress > third_fraction:
        ball_zone = "attacking"
    elif progress < -third_fraction:
        ball_zone = "defensive"
    else:
        ball_zone = "middle"

    ball_in_our_half = (bx * attack_sign) < 0.0
    possession = _estimate_possession(snapshot, possession_radius)

    own_score, enemy_score = snapshot.referee_state.score
    margin = int(own_score) - int(enemy_score)
    scoreline = "leading" if margin > 0 else "trailing" if margin < 0 else "level"

    return GameContext(
        ball_zone=ball_zone,
        possession=possession,
        scoreline=scoreline,
        ball_in_our_half=ball_in_our_half,
        goal_margin=margin,
    )


def resolve_effective_strategy(
    strategy: StrategyConfig | None,
    context: GameContext,
) -> StrategyConfig | None:
    """Compose the base posture with every rule that matches *context*.

    Returns a :class:`StrategyConfig` whose top-level dials are the composed
    result (and whose ``rules`` are cleared, since they have been applied). When
    the strategy is inactive the input is returned unchanged so downstream
    ``apply_strategy_*`` calls stay no-ops.
    """
    if not is_strategy_active(strategy):
        return strategy

    assert strategy is not None
    dials = strategy.base_dials
    for rule in strategy.rules:
        if rule.when.matches(context):
            dials = _compose_posture(dials, rule.apply)

    return replace(
        strategy,
        role_bias=dials.role_bias,
        formation=dials.formation,
        stability=dials.stability,
        attacking=dials.attacking,
        defending=dials.defending,
        rules=(),
    )


# ---------------------------------------------------------------------------
# Transforms — each is identity when the strategy is inactive or neutral
# ---------------------------------------------------------------------------


def apply_strategy_to_role_weights(weights: T, strategy: StrategyConfig | None) -> T:
    """Return role heuristic weights adjusted by the team strategy.

    ``weights`` is a ``RoleHeuristicWeights`` instance (passed structurally to
    avoid an import cycle). When the strategy is inactive the input is returned
    unchanged.
    """
    if not is_strategy_active(strategy):
        return weights

    assert strategy is not None  # for type-checkers; guaranteed by is_active
    role_bias = strategy.role_bias

    attacker = _scale_float_fields(weights.attacker, role_bias.attacker_weight_scale)
    attacker = _scale_float_fields(
        attacker, role_bias.press_scale, only=_ATTACKER_PRESS_FIELDS
    )
    defender = _scale_float_fields(weights.defender, role_bias.defender_weight_scale)
    defender = _scale_float_fields(
        defender, role_bias.press_scale, only=_DEFENDER_PRESS_FIELDS
    )
    supporter = _scale_float_fields(weights.supporter, role_bias.supporter_weight_scale)

    stickiness = strategy.stability.role_stickiness_scale
    stability = _scale_float_fields(weights.stability, stickiness)
    defender_stability = _scale_float_fields(
        weights.defender_stability,
        stickiness,
        only=_DEFENDER_STABILITY_STICKY_FIELDS,
    )

    formation = strategy.formation
    role_targets = weights.role_targets
    role_targets = _offset_int_field(role_targets, "attackers", formation.extra_attackers)
    role_targets = _offset_int_field(
        role_targets, "min_defenders", formation.extra_defenders
    )
    role_targets = _offset_int_field(
        role_targets, "max_defenders", formation.extra_defenders
    )
    role_targets = _offset_int_field(
        role_targets, "min_supporters", formation.extra_supporters
    )

    return replace(
        weights,
        attacker=attacker,
        defender=defender,
        supporter=supporter,
        stability=stability,
        defender_stability=defender_stability,
        role_targets=role_targets,
    )


def apply_strategy_to_attacker_config(config: T, strategy: StrategyConfig | None) -> T:
    """Return an ``AttackerBehaviorConfig`` adjusted by the team strategy."""
    if not is_strategy_active(strategy):
        return config
    assert strategy is not None
    attacking = strategy.attacking

    config = _scale_float_fields(
        config, attacking.shoot_distance_scale, only=("shoot_dist_threshold",)
    )
    config = _scale_float_fields(
        config, attacking.shot_alignment_tolerance_scale, only=("shot_heading_tol",)
    )
    config = _scale_float_fields(
        config,
        attacking.pass_caution_scale,
        only=("pass_marked_distance_frac", "pass_lane_clearance_frac"),
    )
    if attacking.settle_time_scale != 1.0 and hasattr(config, "shot_settle_ticks"):
        new_ticks = max(1, round(config.shot_settle_ticks * attacking.settle_time_scale))
        config = replace(config, shot_settle_ticks=new_ticks)
    return config


def apply_strategy_to_supporter_config(config: T, strategy: StrategyConfig | None) -> T:
    """Return a ``SupporterBehaviorConfig`` adjusted by the team strategy."""
    if not is_strategy_active(strategy):
        return config
    assert strategy is not None
    attacking = strategy.attacking

    config = _scale_float_fields(
        config, attacking.shoot_distance_scale, only=("shoot_dist_threshold",)
    )
    config = _scale_float_fields(
        config, attacking.pass_caution_scale, only=("marked_threshold",)
    )
    return config


def apply_strategy_to_defender_positioning(
    config: T, strategy: StrategyConfig | None
) -> T:
    """Return a ``DefenderPositioningConfig`` adjusted by the team strategy."""
    if not is_strategy_active(strategy):
        return config
    assert strategy is not None
    scale = strategy.defending.line_height_scale
    if scale == 1.0:
        return config

    updates: dict[str, float] = {}
    for name in ("shot_block_fraction_from_goal", "pass_block_fraction_from_carrier"):
        if hasattr(config, name):
            updates[name] = _clamp(
                getattr(config, name) * scale,
                _DEFENDER_FRACTION_MIN,
                _DEFENDER_FRACTION_MAX,
            )
    return replace(config, **updates) if updates else config


# ---------------------------------------------------------------------------
# Generic dataclass helpers
# ---------------------------------------------------------------------------


def _scale_float_fields(
    dc: T,
    factor: float,
    only: Iterable[str] | None = None,
) -> T:
    """Return a copy of *dc* with float fields multiplied by *factor*.

    ``bool`` and ``int`` fields are never scaled (bools are a subclass of int
    and counts must stay integral). When *only* is given, only those named
    fields are scaled. A *factor* of exactly 1.0 returns the input unchanged.
    """
    if factor == 1.0:
        return dc
    allowed = set(only) if only is not None else None
    updates: dict[str, float] = {}
    for f in fields(dc):
        if allowed is not None and f.name not in allowed:
            continue
        value = getattr(dc, f.name)
        if isinstance(value, bool) or not isinstance(value, float):
            continue
        updates[f.name] = value * factor
    return replace(dc, **updates) if updates else dc


def _offset_int_field(dc: T, name: str, delta: int, minimum: int = 0) -> T:
    """Return a copy of *dc* with integer field *name* shifted by *delta*."""
    if delta == 0 or not hasattr(dc, name):
        return dc
    value = int(getattr(dc, name))
    return replace(dc, **{name: max(minimum, value + int(delta))})


def _compose_dials(a: T, b: T) -> T:
    """Compose two dial sub-dataclasses of the same type.

    Float dials (scales, neutral 1.0) multiply; int dials (offsets, neutral 0)
    add; bool dials OR together. Because *b*'s unset fields are neutral, this
    leaves *a* unchanged for any dial *b* did not move.
    """
    updates: dict[str, object] = {}
    for f in fields(a):
        av = getattr(a, f.name)
        bv = getattr(b, f.name)
        if isinstance(av, bool):
            updates[f.name] = bool(av or bv)
        elif isinstance(av, float):
            updates[f.name] = av * bv
        elif isinstance(av, int):
            updates[f.name] = av + bv
        else:
            updates[f.name] = bv
    return replace(a, **updates)


def _compose_posture(base: PostureDials, delta: PostureDials) -> PostureDials:
    """Compose a posture delta on top of a base posture (per sub-dataclass)."""
    return PostureDials(
        role_bias=_compose_dials(base.role_bias, delta.role_bias),
        formation=_compose_dials(base.formation, delta.formation),
        stability=_compose_dials(base.stability, delta.stability),
        attacking=_compose_dials(base.attacking, delta.attacking),
        defending=_compose_dials(base.defending, delta.defending),
    )


def _estimate_possession(snapshot: Any, radius: float) -> str:
    """Classify ball possession as own / opponent / loose by nearest robot."""
    nearest_own = _nearest_distance(snapshot.ball_position, snapshot.own_robots)
    nearest_opp = _nearest_distance(snapshot.ball_position, snapshot.enemy_robots)
    if math.isinf(nearest_own) and math.isinf(nearest_opp):
        return "loose"
    if min(nearest_own, nearest_opp) > radius:
        return "loose"
    if nearest_own == nearest_opp:
        return "loose"
    return "own" if nearest_own < nearest_opp else "opponent"


def _nearest_distance(point: tuple[float, float], robots: Iterable[Any]) -> float:
    best = math.inf
    for robot in robots:
        rx, ry = robot.position
        best = min(best, math.hypot(rx - point[0], ry - point[1]))
    return best


def _dataclass_from_section(cls: type, section: object) -> Any:
    """Build *cls* from a yaml sub-mapping, keeping defaults for missing keys."""
    defaults = cls()
    if not isinstance(section, MappingABC):
        return defaults
    values: dict[str, object] = {}
    for item in fields(cls):
        if item.name not in section:
            continue
        values[item.name] = _coerce_config_value(
            section[item.name], getattr(defaults, item.name)
        )
    return cls(**values)


def _coerce_config_value(value: object, default_value: object) -> object:
    if isinstance(default_value, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    return value


def _resolve_utils_config_path(config_filename: str | Path) -> Path:
    path = Path(config_filename)
    if path.is_absolute():
        return path

    utils_dir = Path(__file__).resolve().parents[2] / "utils"
    path = utils_dir / path
    if path.exists() or path.name != BT_TUNING_FILENAME:
        return path

    legacy_path = utils_dir / LEGACY_HEURISTIC_WEIGHT_FILENAME
    return legacy_path if legacy_path.exists() else path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
