"""
The battle agents.

LLMAgentBase is the course's bridge class: it turns a poke-env Battle into text,
asks a brain for ONE tool call (choose_move or choose_switch), and turns that
call back into a legal order. Subclasses only implement _get_llm_decision.

Two brains live here:
  FirstMoveAgent  a fake brain with no API calls (picks the first move), for testing the wiring
  GrokAgent       the real one, xAI Grok through the OpenAI-compatible API with tool calling
"""

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from poke_env.battle import Battle, Move, Pokemon
from poke_env.player import BattleOrder, Player

import config
import heuristics


def normalize_name(name: str) -> str:
    """'Iron Valiant' -> 'ironvaliant', 'Thunder-Bolt' -> 'thunderbolt'."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# The two buttons the model is allowed to press. Same shape as the course's schema.
STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_name": {
                    "type": "string",
                    "description": "The exact id of the move to use (e.g. 'thunderbolt', 'swordsdance'). Must be one of the available moves.",
                },
                "terastallize": {
                    "type": "boolean",
                    "description": "Also terastallize this turn. Only when the state says you can, and only when it clearly helps.",
                },
            },
            "required": ["move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokemon from the bench to switch into.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_name": {
                    "type": "string",
                    "description": "The exact species name of the Pokemon to switch to (e.g. 'Pikachu'). Must be one of the available switches.",
                },
            },
            "required": ["pokemon_name"],
        },
    },
}


class LLMAgentBase(Player):
    """Bridge between a Battle and a brain. Subclasses implement _get_llm_decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.history: Dict[str, list] = defaultdict(list)  # battle_tag -> turns
        self.fallbacks = 0
        self.decisions = 0
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.log_prefix = time.strftime("%Y%m%d-%H%M%S")  # keeps runs apart even if the server reuses battle numbers

    # ------------------------------------------------ text <-> objects

    def _format_battle_state(self, battle: Battle) -> str:
        return heuristics.format_state(battle, self.history[battle.battle_tag])

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        wanted = normalize_name(move_name)
        for move in battle.available_moves:
            if move.id == wanted or normalize_name(move.entry.get("name", "")) == wanted:
                return move
        return None

    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        wanted = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            if normalize_name(pkmn.species) == wanted or normalize_name(pkmn.base_species) == wanted:
                return pkmn
        return None

    # ------------------------------------------------ the turn loop

    async def choose_move(self, battle: Battle) -> BattleOrder:
        # Whatever goes wrong in here, the server must get a legal order back,
        # otherwise the battle silently hangs forever.
        try:
            return await self._choose_move_inner(battle)
        except Exception as e:
            self.fallbacks += 1
            self.decisions += 1
            self._log(battle, "", "heuristic (crash)", f"fallback: {type(e).__name__}: {str(e)[:120]}", 0.0)
            try:
                return heuristics.pick_heuristic_order(self, battle)
            except Exception:
                return self.choose_random_move(battle)

    async def _choose_move_inner(self, battle: Battle) -> BattleOrder:
        state = self._format_battle_state(battle)
        t0 = time.time()
        try:
            result = await asyncio.wait_for(self._get_llm_decision(state), timeout=45)
        except Exception as e:  # the brain must never crash the websocket task
            result = {"error": f"{type(e).__name__}: {e}"}
        self.decisions += 1

        order, action, reason = None, "", ""
        decision = result.get("decision")
        if decision:
            name, args = decision.get("name"), decision.get("arguments", {}) or {}
            if name == "choose_move":
                move = self._find_move_by_name(battle, str(args.get("move_name", "")))
                if move:
                    tera = bool(args.get("terastallize")) and battle.can_tera
                    order = self.create_order(move, terastallize=tera)
                    action, reason = f"move {move.id}{' +tera' if tera else ''}", "llm"
                else:
                    reason = f"fallback: unknown move {args.get('move_name')!r}"
            elif name == "choose_switch":
                pkmn = self._find_pokemon_by_name(battle, str(args.get("pokemon_name", "")))
                if pkmn:
                    order = self.create_order(pkmn)
                    action, reason = f"switch {pkmn.species}", "llm"
                else:
                    reason = f"fallback: unknown switch {args.get('pokemon_name')!r}"
            else:
                reason = f"fallback: unknown tool {name!r}"
        else:
            reason = f"fallback: {result.get('error', 'no decision')}"

        if order is None:
            self.fallbacks += 1
            order = heuristics.pick_heuristic_order(self, battle)
            action = f"heuristic {order.message}"

        self.history[battle.battle_tag].append({"turn": battle.turn, "action": action, "reason": reason})
        self._log(battle, state, action, reason, time.time() - t0)
        return order

    def _log(self, battle: Battle, state: str, action: str, reason: str, seconds: float) -> None:
        path = os.path.join(config.LOG_DIR, f"{self.log_prefix}-{battle.battle_tag}.txt")
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        with open(path, "a") as f:
            f.write(f"turn {battle.turn:>3} | {me.species if me else '?'} {me.current_hp_fraction*100 if me else 0:.0f}% "
                    f"vs {opp.species if opp else '?'} {opp.current_hp_fraction*100 if opp else 0:.0f}% | "
                    f"{action} | {reason} | {seconds:.1f}s\n")
            if battle.turn <= 1:
                f.write("--- state at turn 1 ---\n" + state + "\n--- end ---\n")

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError


class FirstMoveAgent(LLMAgentBase):
    """Fake brain: always presses the first move button. Zero API calls."""

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        m = re.search(r"^- ([a-z0-9]+) \|", battle_state, flags=re.M)
        if not m:
            return {"error": "no move in state"}
        return {"decision": {"name": "choose_move", "arguments": {"move_name": m.group(1)}}}


SYSTEM_PROMPT = """You are a skilled Pokemon battle AI playing gen9randombattle. Each turn you receive
the full battle state, with damage multipliers already computed, and you MUST call exactly
one tool: choose_move or choose_switch.

Rules of thumb, in priority order:
1. If a move can knock the opponent out this turn (their HP is low, or the move is super
   effective with high expected damage), use it. Prefer priority moves to finish a faster foe.
2. If the opponent's types hit your active Pokemon for x4, or you have no move that does at
   least x1 damage, switch to the bench Pokemon that takes the least damage.
3. Use setup moves (swordsdance, nastyplot, calmmind, dragondance, etc.) only when the
   opponent cannot hurt you much this turn.
4. Set entry hazards (stealthrock, spikes) once, early, if the opponent is not threatening.
5. Otherwise use the move with the highest expected damage, preferring STAB and accuracy.
6. Do not switch on consecutive turns unless forced. Never pick a move with 0 PP.
7. Terastallize only when it turns a losing matchup into a winning one or secures a KO.
Use the exact move id or species name shown in the state."""


class GrokAgent(LLMAgentBase):
    """The real brain: xAI Grok via the OpenAI-compatible chat API with tool calling."""

    def __init__(self, model: str = config.DEFAULT_MODEL, temperature: float = 0.1,
                 max_tokens: int = 200, timeout: float = 20.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key=config.require_xai_key(), base_url=config.XAI_BASE_URL,
                                  timeout=timeout, max_retries=0)
        self.tools = [{"type": "function", "function": f} for f in STANDARD_TOOL_SCHEMA.values()]
        self.tokens_in = 0
        self.tokens_out = 0

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current battle state:\n{battle_state}\n\nChoose the best action by calling exactly one tool."},
        ]
        last_error = "unknown"
        for attempt in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools, tool_choice="required",
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(1)
                continue
            if resp.usage:
                self.tokens_in += resp.usage.prompt_tokens or 0
                self.tokens_out += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            if msg.tool_calls:
                call = msg.tool_calls[0]
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    return {"error": f"bad tool arguments: {call.function.arguments!r}"}
                return {"decision": {"name": call.function.name, "arguments": args}}
            # No tool call: maybe the model wrote JSON in plain text. Try to salvage it.
            text = msg.content or ""
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if "move_name" in data:
                        return {"decision": {"name": "choose_move", "arguments": data}}
                    if "pokemon_name" in data:
                        return {"decision": {"name": "choose_switch", "arguments": data}}
                    if "name" in data:
                        return {"decision": data}
                except json.JSONDecodeError:
                    pass
            return {"error": f"no tool call, model said: {text[:120]!r}"}
        return {"error": last_error}

    def cost_line(self) -> str:
        pin, pout = config.PRICE_PER_M.get(self.model, (2.0, 6.0))
        usd = self.tokens_in / 1e6 * pin + self.tokens_out / 1e6 * pout
        return f"{self.model}: {self.tokens_in} in / {self.tokens_out} out tokens, about ${usd:.2f}"
