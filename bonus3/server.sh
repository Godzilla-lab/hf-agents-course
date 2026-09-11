#!/usr/bin/env bash
# Start a private Pokemon Showdown server on http://localhost:8000
# First run clones the simulator and installs it (about 2 minutes). Later runs just start it.
# --no-security turns off login and rate limits, which is what lets our bots join with any name.
set -euo pipefail
cd "$(dirname "$0")"
[ -d pokemon-showdown ] || git clone --depth 1 https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
[ -d node_modules ] || npm install
[ -f config/config.js ] || cp config/config-example.js config/config.js
exec node pokemon-showdown start --no-security
