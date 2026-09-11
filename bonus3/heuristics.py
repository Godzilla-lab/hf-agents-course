"""
Battle maths and the battle-state text. No LLM, no network: pure functions on a Battle.

Why does this file exist? Because an LLM is bad at arithmetic and worse at
remembering the 18x18 type chart. poke-env knows the chart exactly, so we do
the multiplication here and hand the model the results ("vs opp x4.0"). The
model then only has to make the judgement call, which is what it is good at.
"""

from poke_env.battle import Battle, Move, Pokemon
from poke_env.data import GenData
from poke_env.player import BattleOrder, Player


TYPE_CHART = GenData.from_gen(9).type_chart  # the 18x18 table, loaded once


# ---------------------------------------------------------------- maths

def effectiveness(move: Move, defender: Pokemon, battle: Battle) -> float:
    """How hard this move hits that Pokemon, from the type chart: 0, 0.25, 0.5, 1, 2 or 4."""
    if move.base_power == 0 or defender is None:
        return 1.0
    types = [t for t in defender.types if t is not None]
    if not types:
        return 1.0
    return move.type.damage_multiplier(types[0], types[1] if len(types) > 1 else None,
                                       type_chart=TYPE_CHART)


def incoming_multiplier(attacker: Pokemon, defender: Pokemon, battle: Battle) -> float:
    """Worst case: the biggest multiplier any of the attacker's OWN types gets on the defender."""
    if attacker is None or defender is None:
        return 1.0
    d_types = [t for t in defender.types if t is not None]
    if not d_types:
        return 1.0
    worst = 0.0
    for atk_type in attacker.types:
        if atk_type is None:
            continue
        worst = max(worst, atk_type.damage_multiplier(
            d_types[0], d_types[1] if len(d_types) > 1 else None, type_chart=TYPE_CHART))
    return worst or 1.0


def is_stab(move: Move, attacker: Pokemon) -> bool:
    return attacker is not None and move.type in attacker.types


def score_move(move: Move, attacker: Pokemon, defender: Pokemon, battle: Battle) -> float:
    """Rough expected damage: power x effectiveness x STAB x accuracy."""
    acc = move.accuracy if isinstance(move.accuracy, (int, float)) and move.accuracy not in (True, 1) else 1.0
    if acc > 1:
        acc = acc / 100
    return move.base_power * effectiveness(move, defender, battle) * (1.5 if is_stab(move, attacker) else 1.0) * acc


def pick_heuristic_order(player: Player, battle: Battle) -> BattleOrder:
    """
    The fallback brain. Used whenever the LLM fails or picks something illegal.
    Best damaging move if we have one; otherwise the switch that resists the
    opponent best; otherwise whatever the server allows (Struggle).
    """
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if battle.available_moves:
        best = max(battle.available_moves, key=lambda m: score_move(m, me, opp, battle))
        return player.create_order(best)
    if battle.available_switches:
        best = min(battle.available_switches, key=lambda p: incoming_multiplier(opp, p, battle))
        return player.create_order(best)
    return player.choose_default_move()


# ---------------------------------------------------------------- text

def _pct(p: Pokemon) -> str:
    return f"{p.current_hp_fraction * 100:.0f}%"


def _types(p: Pokemon) -> str:
    return "/".join(t.name.title() for t in p.types if t is not None)


def _status(p: Pokemon) -> str:
    return p.status.name if p.status else "none"


def _boosts(p: Pokemon) -> str:
    b = {k: v for k, v in p.boosts.items() if v}
    return str(b) if b else "none"


def _mult(x: float) -> str:
    return f"x{x:g}"


def format_state(battle: Battle, history: list) -> str:
    """Everything the model needs, computed, in about 1.5k tokens."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    lines = []

    lines.append(f"Turn {battle.turn} | Weather: {list(battle.weather) or 'none'} | Field: {list(battle.fields) or 'none'}")
    lines.append(f"My side: {dict(battle.side_conditions) or 'none'} | Opponent side: {dict(battle.opponent_side_conditions) or 'none'}")

    if me:
        lines.append(f"\nMy active: {me.species} ({_types(me)}) HP {_pct(me)} status={_status(me)} "
                     f"boosts={_boosts(me)} ability={me.ability or 'unknown'} item={me.item or 'unknown'}"
                     + (f" | can terastallize to {me.tera_type.name.title()}" if battle.can_tera and me.tera_type else ""))
    if opp:
        known = ", ".join(m.id for m in opp.moves.values()) or "none revealed yet"
        lines.append(f"Opponent active: {opp.species} ({_types(opp)}) HP {_pct(opp)} status={_status(opp)} "
                     f"boosts={_boosts(opp)} ability={opp.ability or 'unknown'} item={opp.item or 'unknown'}")
        lines.append(f"  opponent's known moves: {known}")
        if me:
            lines.append(f"  opponent's types hit my active for {_mult(incoming_multiplier(opp, me, battle))}")

    lines.append("\nAvailable moves (use the exact id):")
    if battle.available_moves:
        for m in battle.available_moves:
            eff = effectiveness(m, opp, battle) if opp else 1.0
            lines.append(f"- {m.id} | {m.type.name.title()} | power {m.base_power} | {m.category.name.lower()} | "
                         f"acc {m.accuracy} | pp {m.current_pp}/{m.max_pp} | priority {m.priority} | "
                         f"vs opponent {_mult(eff)}{' | STAB' if is_stab(m, me) else ''} | "
                         f"expected ~{score_move(m, me, opp, battle):.0f}")
    else:
        lines.append("- none (you must switch)")

    lines.append("\nAvailable switches (use the exact species name):")
    if battle.available_switches:
        for p in battle.available_switches:
            lines.append(f"- {p.species} ({_types(p)}) HP {_pct(p)} status={_status(p)} | "
                         f"opponent's types would hit it for {_mult(incoming_multiplier(opp, p, battle)) if opp else 'x1'}")
    else:
        lines.append("- none")

    mine_alive = sum(1 for p in battle.team.values() if not p.fainted)
    opp_seen = len(battle.opponent_team)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
    lines.append(f"\nTeam status: I have {mine_alive}/6 alive. Opponent has revealed {opp_seen}/6, {opp_fainted} fainted.")

    if history:
        lines.append("\nMy last turns:")
        for h in history[-3:]:
            lines.append(f"- turn {h['turn']}: {h['action']} ({h['reason']})")

    return "\n".join(lines)
