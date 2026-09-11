"""
Put the bot on the course's PUBLIC Showdown server and battle a human there.

    ./.venv/bin/python bonus3/challenge_online.py --human <your web-client name> --yes-i-want-to-go-online

Steps for the human:
  1. Open the course web client (URL printed below), choose a name (top right).
  2. Run this script with that name. The bot logs in and challenges you.
  3. Accept the challenge in the browser and play.

The bot only ever talks to the ONE name you give it, so strangers cannot make it
spend API money. It never starts the battle timer on its own.
"""

import argparse
import asyncio
import random
import string

from poke_env import AccountConfiguration

import config
from agents import GrokAgent


async def main(args) -> None:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    name = f"{args.name}{suffix}"
    me = GrokAgent(model=args.model,
                   account_configuration=AccountConfiguration(name, None),
                   server_configuration=config.COURSE_SERVER,
                   battle_format=config.BATTLE_FORMAT, max_concurrent_battles=1,
                   start_timer_on_battle_start=False)
    print(f"Bot '{name}' connecting to {config.COURSE_SERVER.websocket_url}")
    print(f"Web client: {config.COURSE_WEB_CLIENT}")
    if args.accept:
        print(f"Waiting for a challenge from '{args.human}'...")
        await me.accept_challenges(args.human, 1)
    else:
        print(f"Sending one {config.BATTLE_FORMAT} challenge to '{args.human}'. Accept it in the browser.")
        await me.send_challenges(args.human, 1)
    print(f"\nresult: won {me.n_won_battles}/{me.n_finished_battles}")
    print(f"decisions: {me.decisions}, fallbacks: {me.fallbacks}")
    print(me.cost_line())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", required=True, help="the name you chose in the web client")
    p.add_argument("--name", default="PokeGrok", help="bot name prefix (4 random chars are appended)")
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--accept", action="store_true", help="wait for YOUR challenge instead of sending one")
    p.add_argument("--yes-i-want-to-go-online", action="store_true")
    args = p.parse_args()
    if not args.yes_i_want_to_go_online:
        raise SystemExit("This connects to a public server. Re-run with --yes-i-want-to-go-online.")
    asyncio.run(main(args))
