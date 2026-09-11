"""
Run battles on the LOCAL Showdown server (start it first with: bash bonus3/server.sh).

    ./.venv/bin/python bonus3/battle_local.py --agent random --vs random          plumbing test, no LLM
    ./.venv/bin/python bonus3/battle_local.py --agent stub --vs random            fake brain, no LLM
    ./.venv/bin/python bonus3/battle_local.py --agent grok --vs random            real brain, 1 battle
    ./.venv/bin/python bonus3/battle_local.py --agent grok --vs heuristic --n 5   win rate vs a strong bot
    ./.venv/bin/python bonus3/battle_local.py --play                              you challenge the bot from a browser

Opponents: random (RandomPlayer), maxpower (MaxBasePowerPlayer), heuristic (SimpleHeuristicsPlayer).
Every battle writes a turn-by-turn log to bonus3/logs/<battle_tag>.txt.
"""

import argparse
import asyncio
import os
import time

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

import config
from agents import FirstMoveAgent, GrokAgent

OPPONENTS = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}


def make_agent(kind: str, name: str, model: str):
    common = dict(account_configuration=AccountConfiguration(name, None),
                  server_configuration=LocalhostServerConfiguration,
                  battle_format=config.BATTLE_FORMAT, max_concurrent_battles=1)
    if kind == "random":
        return RandomPlayer(**common)
    if kind == "stub":
        return FirstMoveAgent(**common)
    return GrokAgent(model=model, **common)


async def main(args) -> None:
    me = make_agent(args.agent, args.name, args.model)

    if args.play:
        print(f"Bot '{args.name}' is online on the local server and waiting for ONE challenge.")
        print(f"1. Open {config.LOCAL_WEB_CLIENT}")
        print("2. Choose a name (top right), then Home > Find a user > search '" + args.name + "' > Challenge")
        print(f"3. Pick the format {config.BATTLE_FORMAT} and send it. Watch the terminal and the browser.")
        await me.accept_challenges(None, 1)
    else:
        opp = OPPONENTS[args.vs](account_configuration=AccountConfiguration(f"{args.vs.title()}Bot", None),
                                 server_configuration=LocalhostServerConfiguration,
                                 battle_format=config.BATTLE_FORMAT, max_concurrent_battles=1)
        t0 = time.time()
        for i in range(args.n):  # one challenge at a time avoids poke-env's challenge race
            await me.battle_against(opp, n_battles=1)
            print(f"battle {i + 1}/{args.n} done, running score {me.n_won_battles}/{me.n_finished_battles}")
        print(f"\n{args.agent} vs {args.vs}: won {me.n_won_battles}/{me.n_finished_battles} "
              f"(win rate {me.win_rate:.0%}) in {time.time() - t0:.0f}s")

    if hasattr(me, "decisions") and me.decisions:
        print(f"decisions: {me.decisions}, heuristic fallbacks: {me.fallbacks} ({me.fallbacks / me.decisions:.0%})")
    if hasattr(me, "cost_line"):
        print(me.cost_line())
    print(f"logs: {config.LOG_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=["random", "stub", "grok"], default="grok")
    p.add_argument("--vs", choices=list(OPPONENTS), default="random")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--name", default="PokeGrok")
    p.add_argument("--play", action="store_true", help="wait for a human challenge from the browser")
    asyncio.run(main(p.parse_args()))
