# Bonus Unit 3: Pokemon battle agent

An LLM plays gen9randombattle on Pokemon Showdown through poke-env. Grok (xAI) is the
brain; the type-chart maths is done in Python and handed to it.

| file | what it is |
| --- | --- |
| `server.sh` | starts a private Pokemon Showdown server on localhost:8000 (clones and installs on first run) |
| `config.py` | reads keys from `../unit4/.env`, model choice, server addresses |
| `heuristics.py` | type effectiveness, expected damage, the fallback move picker, and the battle-state text |
| `agents.py` | `LLMAgentBase` (the course's bridge class, ported to poke-env 0.16), `FirstMoveAgent` (fake brain), `GrokAgent` |
| `battle_local.py` | bot vs bot on the local server, or `--play` to fight it yourself from a browser |
| `challenge_online.py` | the same bot on the course's public server (asks for an explicit flag) |
| `logs/` | one file per battle, one line per turn: matchup, action, why, seconds |

## Run

    cd /Users/godzilla/CLI/HuggingFace/agents-course
    bash bonus3/server.sh                                            # terminal 1, leave it running
    ./.venv/bin/python bonus3/battle_local.py --agent random --vs random   # plumbing, no LLM
    ./.venv/bin/python bonus3/battle_local.py --agent stub --vs random     # fake brain, no LLM
    ./.venv/bin/python bonus3/battle_local.py --agent grok --vs random     # real brain
    ./.venv/bin/python bonus3/battle_local.py --agent grok --vs heuristic --n 5
    ./.venv/bin/python bonus3/battle_local.py --play                       # then open the URL it prints

## Cost

grok-4.3 (default): about $0.15 per battle. grok-4.6 (`--model grok-4.6`): about $0.30.
A battle is 30 to 60 decisions of roughly 1.5k tokens in and 50 out.
