"""
Animate the GAIA scoreboard: the 20 answers appear one by one, the score climbs to 100%.

    ./.venv/bin/python unit4/make_score_gif.py media/gaia.gif
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "/System/Library/Fonts/Menlo.ttc"
W, H = 900, 640
BG, FG, DIM, GREEN, ACCENT = (14, 16, 26), (230, 232, 240), (120, 125, 145), (76, 201, 96), (120, 190, 255)

KINDS = {1: "wikipedia", 2: "youtube video", 3: "reasoning", 4: "chess image + Stockfish", 5: "wikipedia",
         6: "reasoning", 7: "youtube video", 8: "web", 9: "reasoning", 10: "audio", 11: "web", 12: "run .py",
         13: "web", 14: "audio", 15: "web, multi-hop", 16: "web", 17: "wikipedia", 18: "web", 19: "excel", 20: "web"}


def main(out):
    font = ImageFont.truetype(MONO, 16)
    big = ImageFont.truetype(MONO, 40)
    cache = json.load(open(os.path.join(HERE, "answers.json")))
    rows = sorted(cache.values(), key=lambda r: r["index"])
    frames, durations = [], []

    def draw(n_shown, cursor, splash=0.0):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 16), "$ python unit4/run_local.py --submit", font=font, fill=ACCENT)
        d.text((24, 40), "agent_code : https://huggingface.co/spaces/godzilla1215/gaia-agent", font=font, fill=DIM)
        for i, r in enumerate(rows[:n_shown]):
            y = 76 + i * 24
            ans = r["answer"] if len(r["answer"]) <= 44 else r["answer"][:41] + "..."
            d.text((24, y), f"Q{r['index']:<3}", font=font, fill=DIM)
            d.text((70, y), f"{KINDS.get(r['index'], ''):<24}", font=font, fill=DIM)
            d.text((330, y), ans, font=font, fill=FG)
            d.text((830, y), "ok", font=font, fill=GREEN)
        if cursor and n_shown < len(rows):
            d.text((24, 76 + n_shown * 24), "_", font=font, fill=FG)
        score = int(100 * n_shown / len(rows))
        d.text((24, H - 60), f"score {score:>3}%   correct {n_shown:>2}/20", font=font, fill=GREEN if n_shown == 20 else FG)
        if splash > 0:
            overlay = Image.new("RGB", (W, H), (0, 0, 0))
            img = Image.blend(img, overlay, 0.7 * splash)
            d = ImageDraw.Draw(img)
            if splash >= 1.0:
                d.rounded_rectangle((110, H // 2 - 80, W - 110, H // 2 + 70), radius=16, fill=(18, 40, 26), outline=GREEN, width=2)
                d.text((W // 2 - 240, H // 2 - 55), "20/20 correct. 100%.", font=big, fill=GREEN)
                d.text((W // 2 - 275, H // 2 + 15), "GAIA Level 1, exact match, verified by the course scorer", font=font, fill=FG)
        return img

    frames.append(draw(0, True)); durations.append(600)
    for n in range(1, len(rows) + 1):
        frames.append(draw(n, n % 2 == 0)); durations.append(260)
    for k in range(1, 9):
        frames.append(draw(20, False, splash=k / 8)); durations.append(80)
    frames.extend([draw(20, False, splash=1.0)] * 30); durations.extend([100] * 30)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
