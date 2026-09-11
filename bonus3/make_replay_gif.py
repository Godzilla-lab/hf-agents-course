"""
Turn a battle log into an animated GIF replay.

    ./.venv/bin/python bonus3/make_replay_gif.py bonus3/logs/<battle>.txt media/battle.gif --won

Each turn becomes a few frames: HP bars slide to their new values, the Pokemon
that got hit shakes, and the bot's decision plus its reason scroll in a ticker.
Sprites are the official Pokemon Showdown ones, fetched once into a cache.
"""

import argparse
import io
import math
import os
import re
import sys

import requests
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "logs", "sprites")
SPRITE_URL = "https://play.pokemonshowdown.com/sprites/{kind}/{species}.png"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"

W, H = 900, 506
BG_TOP, BG_BOT = (24, 28, 48), (54, 40, 80)
GREEN, YELLOW, RED = (76, 201, 96), (240, 196, 60), (226, 70, 70)
WHITE, GREY, ACCENT = (245, 245, 250), (150, 150, 170), (120, 190, 255)

TURN_RE = re.compile(r"^turn\s+(\d+) \| (\S+) (\d+)% vs (\S+) (\d+)% \| (.+?) \| (.+?) \| ([\d.]+)s$")


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def sprite(species: str, kind: str) -> Image.Image:
    """Fetch a sprite once; fall back to a grey circle if the species name is odd."""
    os.makedirs(CACHE, exist_ok=True)
    candidates = [species, re.sub(r"(galar|alola|hisui|paldea)$", r"-\1", species), species.replace("mega", "-mega")]
    for name in candidates:
        path = os.path.join(CACHE, f"{kind}-{name}.png")
        if not os.path.exists(path):
            r = requests.get(SPRITE_URL.format(kind=kind, species=name), timeout=20)
            if r.status_code != 200:
                continue
            with open(path, "wb") as f:
                f.write(r.content)
        img = Image.open(path).convert("RGBA")
        return img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img = Image.new("RGBA", (192, 192), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((40, 40, 152, 152), fill=(120, 120, 140, 255))
    return img


def parse_log(path: str) -> list[dict]:
    turns = []
    with open(path) as f:
        for line in f:
            m = TURN_RE.match(line.strip())
            if m:
                turns.append({"turn": int(m[1]), "me": m[2], "me_hp": int(m[3]), "opp": m[4], "opp_hp": int(m[5]),
                              "action": m[6], "reason": m[7], "secs": float(m[8])})
    return turns


def hp_color(pct):
    return GREEN if pct > 50 else YELLOW if pct > 20 else RED


def draw_frame(t, me_hp, opp_hp, shake_me, shake_opp, ticker_text, ticker_x, fonts, sprites, title):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):  # vertical gradient background
        k = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * k) for a, b in zip(BG_TOP, BG_BOT)))
    d.ellipse((470, 150, 860, 250), fill=(60, 110, 70))    # opponent platform
    d.ellipse((30, 330, 430, 440), fill=(60, 110, 70))    # my platform

    big, med, mono = fonts
    d.text((20, 12), title, font=mono, fill=GREY)
    d.text((W - 130, H - 130), f"Turn {t['turn']}", font=big, fill=WHITE)

    opp_img = sprites["opp"]
    ox, oy = 600 + shake_opp, 60
    img.paste(opp_img, (ox, oy), opp_img)
    me_img = sprites["me"]
    mx, my = 120 + shake_me, 220
    img.paste(me_img, (mx, my), me_img)

    def hp_box(x, y, name, pct, label):
        d.rounded_rectangle((x, y, x + 300, y + 62), radius=10, fill=(16, 18, 30))
        d.text((x + 12, y + 8), f"{name}", font=med, fill=WHITE)
        d.text((x + 230, y + 8), label, font=mono, fill=GREY)
        d.rounded_rectangle((x + 12, y + 40, x + 288, y + 52), radius=6, fill=(50, 50, 60))
        if pct > 0:
            d.rounded_rectangle((x + 12, y + 40, x + 12 + int(276 * pct / 100), y + 52), radius=6, fill=hp_color(pct))
        d.text((x + 250, y + 26), f"{int(pct)}%", font=mono, fill=WHITE)

    hp_box(60, 70, t["opp"], opp_hp, "opponent")
    hp_box(540, 300, t["me"], me_hp, "PokeGrok")

    d.rectangle((0, H - 62, W, H), fill=(12, 12, 22))
    d.text((ticker_x, H - 46), ticker_text, font=mono, fill=ACCENT)
    return img


def render(turns, out, won, title, hold=3, tween=6):
    fonts = (load_font(FONT, 30), load_font(FONT, 22), load_font(MONO, 17))
    frames, durations = [], []
    prev_me, prev_opp = 100, 100
    for t in turns:
        sprites = {"me": sprite(t["me"], "gen5-back"), "opp": sprite(t["opp"], "gen5")}
        ticker = f"turn {t['turn']}: {t['action']}   [{t['reason']}, {t['secs']:.1f}s]"
        me_hit = t["me_hp"] < prev_me
        opp_hit = t["opp_hp"] < prev_opp
        for i in range(tween + hold):
            k = min(1.0, i / tween)
            ease = 1 - (1 - k) ** 3
            me_hp = prev_me + (t["me_hp"] - prev_me) * ease
            opp_hp = prev_opp + (t["opp_hp"] - prev_opp) * ease
            shake_me = int(8 * math.sin(i * 2.5)) if me_hit and i < tween else 0
            shake_opp = int(8 * math.sin(i * 2.5)) if opp_hit and i < tween else 0
            frames.append(draw_frame(t, me_hp, opp_hp, shake_me, shake_opp, ticker, 20, fonts, sprites, title))
            durations.append(70)
        prev_me, prev_opp = t["me_hp"], t["opp_hp"]
        if t["me"] != turns[min(len(turns) - 1, turns.index(t) + 1)]["me"]:
            prev_me = 100  # a switch brings in a fresh Pokemon at whatever HP the next line says
    # result card
    last = frames[-1].copy()
    d = ImageDraw.Draw(last)
    d.rectangle((0, 0, W, H), fill=(0, 0, 0))
    last = Image.blend(frames[-1], last, 0.6)
    d = ImageDraw.Draw(last)
    msg = {"won": "PokeGrok wins", "lost": "PokeGrok loses"}.get(won, "End of battle")
    d.text((W // 2 - 140, H // 2 - 40), msg, font=fonts[0], fill=GREEN if won == "won" else RED if won == "lost" else WHITE)
    d.text((W // 2 - 200, H // 2 + 10), f"{len(turns)} decisions by Grok, drawn from the turn log", font=fonts[2], fill=WHITE)
    frames.extend([last] * 25)
    durations.extend([70] * 25)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("out")
    p.add_argument("--won", default="", help="won, lost, or empty for no claim")
    p.add_argument("--title", default="Grok vs SimpleHeuristicsPlayer, gen9randombattle, local Showdown server")
    a = p.parse_args()
    turns = parse_log(a.log)
    if not turns:
        sys.exit("no turn lines found")
    render(turns, a.out, a.won, a.title)
