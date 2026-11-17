# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch", "numpy", "pillow", "Hershey-Fonts"]
# ///
"""Trains the tiny model that draws on dexterstorey.com and writes model.bin.

The model drives a pen on a tick. Every tick it reads one number per input stream (the key
pressed, a click, where the click is, where the pen is, its own last move) and writes one
number per output stream (move x, move y, pen down, size, busy).

It learns by copying `Teacher`, a pen that replays drawings made by people: squiggles, zigzags,
coils, waves and rough shapes from Google's Quick, Draw! set (CC BY 4.0). The teacher also
writes what is typed and goes where it is clicked. `Page` is the world they both live in, and
agent.js repeats its rules in the browser.

Training has two parts. The base copies the calm teacher. The fine-tune copies a wild one,
with a shaky hand. model.bin holds the base and the difference, so the page can slide between.

Run: uv run train.py          fetches the drawings once, trains, and writes model.bin
     uv run train.py teacher  only draws what the two teachers do, to teacher-*.png
"""

import json
import math
import os
import random
import sys
import urllib.parse
import urllib.request
from collections import deque
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from HersheyFonts import HersheyFonts
from PIL import Image, ImageDraw

REACH = 24  # the farthest one tick can move on each axis, in units
SCALES = [1, 2, 3, 5, 8, 13]  # pixels per unit
TRAVEL = 5  # the scale used to cross the page with the pen up
KEYS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-\n"  # a key's number is its place here plus one
NEWLINE = len(KEYS)
NOWHERE = 61  # the click stream when there is no click to go to

# What the model reads each tick, and what it writes.
KEY, CLICK, TX, TY, CX, CY, DX, DY, PEN, SCALE = range(10)
INPUTS = [len(KEYS) + 1, 2, 62, 62, 16, 16, 2 * REACH + 1, 2 * REACH + 1, 2, len(SCALES)]
OUTPUTS = [2 * REACH + 1, 2 * REACH + 1, 2, len(SCALES), 2]  # dx, dy, pen, scale, busy


def hop(dx, dy, pen):
    """One move as ticks. A move longer than REACH becomes several."""
    parts = max(1, math.ceil(max(abs(dx), abs(dy)) / REACH))
    return [(dx * (i + 1) // parts - dx * i // parts, dy * (i + 1) // parts - dy * i // parts, pen) for i in range(parts)]


def glyphs():
    """Every key as pen moves, from the public-domain Hershey single-stroke plotter font."""
    font = HersheyFonts()
    font.load_default_font("futural")
    strokes = lambda text: list(font.strokes_for_text(text))
    table = {}
    for number, key in enumerate(KEYS[:-1], 1):
        x, y, moves = 0, 0, []
        for stroke in strokes(key):
            for i, (px, py) in enumerate(stroke):
                moves += hop(px - x, py - y, int(i > 0))
                x, y = px, py
        advance = strokes(key + "l")[-1][0][0] - strokes("l")[0][0][0]
        table[number] = moves + hop(advance - x, -y, 0)
    return table


GLYPHS = glyphs()


SUBJECTS = "squiggle zigzag hurricane tornado ocean river lightning rain line circle triangle square hexagon octagon star sun snowflake stitches diamond"


def fetch():
    """Downloads the first few thousand drawings of each subject and saves them all as pen
    moves in data/moves.npz: every move in one long list, and where each drawing starts."""
    moves, starts = [], [0]
    for subject in SUBJECTS.split():
        address = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{urllib.parse.quote(subject)}.ndjson"
        part = urllib.request.urlopen(urllib.request.Request(address, headers={"Range": "bytes=0-3000000"})).read()
        for line in part.decode(errors="ignore").split("\n")[:-1]:  # the last line is cut short
            sketch = json.loads(line)
            x, y = (round(point[0] / 5) for point in sketch["drawing"][0])
            drawing = []
            for xs, ys in sketch["drawing"]:
                for i, (px, py) in enumerate(zip(xs, ys)):
                    px, py = round(px / 5), round(py / 5)
                    if (px, py) != (x, y):
                        drawing += hop(px - x, py - y, int(i > 0))
                        x, y = px, py
            if sketch["recognized"] and 3 <= len(drawing) <= 120:
                moves += drawing
                starts.append(len(moves))
        print(subject, len(starts) - 1, "drawings so far", flush=True)
    os.makedirs("data", exist_ok=True)
    np.savez("data/moves.npz", moves=np.array(moves, dtype=np.int8), starts=np.array(starts, dtype=np.int32))


DRAWINGS = np.load("data/moves.npz") if os.path.exists("data/moves.npz") else None
MOVES, STARTS = (DRAWINGS["moves"].tolist(), DRAWINGS["starts"].tolist()) if DRAWINGS else ([], [0])


def offset(distance):
    """How far away the click is on one axis, as a stream value. Exact when near, rough when far."""
    if abs(distance) <= REACH:
        return distance + REACH
    return 2 * REACH + 1 + sum(abs(distance) >= edge for edge in (40, 80, 160, 320, 640)) + (6 if distance < 0 else 0)


def toward(tx, ty):
    """The move that closes in on a click, worked out only from what the model can see."""

    def near(value):
        if value <= 2 * REACH:
            return value - REACH
        rough = (25, 40, 80, 160, 320, 640)[(value - 2 * REACH - 1) % 6]
        return rough if value < 2 * REACH + 7 else -rough

    x, y = near(tx), near(ty)
    scale = max((i for i, size in enumerate(SCALES) if size <= max(abs(x), abs(y))), default=0)
    clamp = lambda distance: max(-REACH, min(REACH, int(distance / SCALES[scale])))
    return clamp(x), clamp(y), scale


class Page:
    """The world: a page, a pen, a pending click and a queue of typed keys."""

    def __init__(self, wide, tall, x, y):
        self.wide, self.tall, self.x, self.y = wide, tall, x, y
        self.target = None
        self.keys = deque()
        self.last = (0, 0, 0, 0, 0)

    def frame(self, click=None):
        """What the model reads this tick."""
        if click:
            self.target = click
        dx, dy, pen, scale, busy = self.last
        key = self.keys.popleft() if self.keys and not busy and not self.target else 0
        tx, ty = (offset(self.target[0] - self.x), offset(self.target[1] - self.y)) if self.target else (NOWHERE, NOWHERE)
        cell = lambda at, size: min(15, at * 16 // size)
        return (key, int(bool(click)), tx, ty, cell(self.x, self.wide), cell(self.y, self.tall), dx + REACH, dy + REACH, pen, scale)

    def move(self, action):
        """Applies what the model wrote this tick and returns the line it drew, if any."""
        dx, dy, pen, scale = action[:4]
        before = (self.x, self.y)
        self.x = max(0, min(self.wide - 1, self.x + dx * SCALES[scale]))
        self.y = max(0, min(self.tall - 1, self.y + dy * SCALES[scale]))
        if self.target == (self.x, self.y):
            self.target = None
        self.last = action
        return (before, (self.x, self.y)) if pen else None


class Clicked(Exception):
    pass


class Typed(Exception):
    pass


class Teacher:
    """The pen the model copies. `wild` runs from 0, calm, to 1."""

    def __init__(self, seed, wild):
        self.random = random.Random(seed)
        self.wild = wild
        self.scale = 0

    def pick(self, options, weights=None):
        return self.random.choices(options, weights)[0]

    def chance(self, odds):
        return self.random.random() < odds

    def resize(self):
        calm, wild = [5, 5, 3, 2, 1, 0], [1, 1, 2, 3, 4, 3]
        self.scale = self.pick(range(len(SCALES)), [a + (b - a) * self.wild for a, b in zip(calm, wild)])

    def run(self):
        """One action per tick, forever. Send it the frame, get back the action."""
        self.at = yield
        doing = self.write() if self.at[KEY] else self.idle()
        while True:
            try:
                yield from doing
                doing = self.idle()
            except Clicked:
                doing = self.go()
            except Typed:
                doing = self.write()

    def do(self, dx=0, dy=0, pen=0, scale=None, busy=0, turn=False):
        """One tick. A click always cuts in. A typed key cuts in too, unless this tick was
        a turn to type."""
        if pen and self.chance(self.wild * 0.3):
            shake = self.pick([1, 1, 2])
            dx = max(-REACH, min(REACH, dx + self.random.randint(-shake, shake)))
            dy = max(-REACH, min(REACH, dy + self.random.randint(-shake, shake)))
        self.at = yield (dx, dy, pen, self.scale if scale is None else scale, busy)
        if self.at[CLICK]:
            raise Clicked
        if self.at[KEY] and not turn:
            raise Typed

    def line(self, dx, dy, pen):
        for part in hop(dx, dy, pen):
            yield from self.do(*part)

    def edge(self, dx, dy):
        """True when a move would head off the page."""
        return (dx > 0 and self.at[CX] >= 15) or (dx < 0 and self.at[CX] <= 0) or (dy > 0 and self.at[CY] >= 15) or (dy < 0 and self.at[CY] <= 0)

    def idle(self):
        """Draws, or moves on. Near the edge of the page it always moves on."""
        near_edge = self.at[CX] in (0, 15) or self.at[CY] in (0, 15)
        return self.travel() if near_edge or self.chance(0.3) else self.doodle()

    def go(self):
        """Lifts the pen and closes in on the click, one look at a time. Then draws something there."""
        while self.at[TX] != NOWHERE:
            dx, dy, scale = toward(self.at[TX], self.at[TY])
            yield from self.do(dx, dy, 0, scale)
        self.resize()
        yield from self.doodle()

    def write(self):
        """Writes the key in this frame, then keeps going for as long as keys keep coming."""
        while self.at[KEY]:
            if self.at[KEY] == NEWLINE:
                yield from self.wrap()
            else:
                for dx, dy, pen in GLYPHS[self.at[KEY]]:
                    yield from self.do(dx, dy, pen, busy=1)
                if self.at[CX] >= 14:
                    yield from self.wrap()
            yield from self.do(turn=True)
        yield from self.rest()

    def wrap(self):
        """Back to the left edge and down a line. From the bottom, back to the top."""
        while self.at[CX] > 1:
            yield from self.do(-REACH, 0, 0, busy=1)
        yield from self.do(0, 16, 0, busy=1)
        yield from self.do(0, 16, 0, busy=1)
        if self.at[CY] >= 15:
            while self.at[CY] > 1:
                yield from self.do(0, -REACH, 0, busy=1)

    def doodle(self):
        """Replays one drawing a person made, at the size the pen is set to."""
        drawing = self.random.randrange(len(STARTS) - 1)
        for dx, dy, pen in MOVES[STARTS[drawing] : STARTS[drawing + 1]]:
            yield from self.do(dx, dy, pen)

    def travel(self):
        """Crosses the page with the pen up, more often toward the middle than away."""
        far = lambda cell: self.pick([12, 24]) * self.pick([1, -1], [3, 1] if cell < 8 else [1, 3])
        dx, dy = far(self.at[CX]), self.pick([0, far(self.at[CY])])
        inked = int(self.chance(self.wild * 0.1))
        for _ in range(self.random.randint(1, 4)):
            if self.edge(dx, dy):
                break
            yield from self.do(dx, dy, inked, TRAVEL)
        self.resize()

    def rest(self):
        """Waits a moment after typing, in case another key is coming."""
        while not self.chance(1 / 150 + self.wild):
            yield from self.do()


def episode(job):
    """One page with a teacher on it and a pretend visitor who clicks and types now and then."""
    seed, wild, ticks, size, keep = job
    visitor = random.Random(seed)
    wide, tall = size or (visitor.randint(360, 2400), visitor.randint(500, 1500))
    page = Page(wide, tall, visitor.randrange(wide), visitor.randrange(tall))
    teacher = Teacher(seed, wild).run()
    next(teacher)
    frames, actions, lines, typing = [], [], [], []
    for tick in range(ticks):
        click = (visitor.randrange(page.wide), visitor.randrange(page.tall)) if tick and visitor.random() < 1 / 900 else None
        if visitor.random() < (1 / 1200 if tick else 0.4):
            text = "Dexter Storey" if visitor.random() < 0.3 else "".join(visitor.choice(KEYS[:27] * 4 + KEYS) for _ in range(visitor.randint(1, 10)))
            typing, due = [], tick
            for key in text:
                typing.append((due, KEYS.index(key) + 1))
                due += visitor.randint(3, 60)
        while typing and typing[0][0] <= tick:
            page.keys.append(typing.pop(0)[1])
        frame = page.frame(click)
        action = teacher.send(frame)
        line = page.move(action)
        if line and keep:
            lines.append(line)
        frames.append(frame)
        actions.append((action[0] + REACH, action[1] + REACH, *action[2:]))
    return np.array(frames, dtype=np.uint8), np.array(actions, dtype=np.uint8), (page.wide, page.tall, lines)


def draw(pages, path):
    """Saves a few drawn pages side by side so a person can look at them."""
    sheet = Image.new("L", (sum(wide for wide, _, _ in pages) + 20 * len(pages), max(tall for _, tall, _ in pages)), 200)
    left = 0
    for wide, tall, lines in pages:
        page = Image.new("L", (wide * 2, tall * 2), 255)
        pen = ImageDraw.Draw(page)
        for (ax, ay), (bx, by) in lines:
            pen.line((ax * 2, ay * 2, bx * 2, by * 2), fill=0, width=2)
        sheet.paste(page.resize((wide, tall), Image.LANCZOS), (left, 0))
        left += wide + 20
    sheet.save(path)


if __name__ == "__main__" and not DRAWINGS:
    fetch()
    sys.exit("Fetched the drawings. Run it again to use them.")

if __name__ == "__main__" and sys.argv[1:] == ["teacher"]:
    for wild, ticks in ((0, 6000), (0, 20000), (1, 6000), (1, 20000)):
        draw([episode((seed, wild, ticks, size, True))[2] for seed, size in enumerate([(1280, 800), (390, 800)])], f"teacher-{wild}-{ticks}.png")
    sys.exit()


class Model(nn.Module):
    """One small table per input stream, a recurrent core that carries the state from tick to
    tick, and one small head per output stream."""

    def __init__(self):
        super().__init__()
        self.read = nn.ModuleList(nn.Embedding(size, 64) for size in INPUTS)
        self.core = nn.GRU(64, 256, batch_first=True)
        self.write = nn.ModuleList(nn.Linear(256, size) for size in OUTPUTS)
        self.given = nn.Embedding(OUTPUTS[0], 256)

    def forward(self, frames, state=None):
        """The state after reading these frames."""
        return self.core(sum(table(frames[..., i]) for i, table in enumerate(self.read)), state)

    def scores(self, x, dx):
        """Scores for every output stream. The dy stream also sees the dx that was chosen, so
        the two halves of a move agree with each other."""
        return [head(x + self.given(dx) if i == 1 else x) for i, head in enumerate(self.write)]


def lessons(wild, count, seed):
    """Fresh episodes from the teacher: what it read and what it did, tick by tick."""
    with Pool() as pool:
        played = pool.map(episode, [(seed + i, wild, 2048, None, False) for i in range(count)], chunksize=16)
    return (torch.from_numpy(np.stack([p[i] for p in played])).long() for i in (0, 1))


def study(model, wild, rounds, rate):
    """Copies the teacher. The state carries across each 2048-tick episode, 256 ticks at a time."""
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=rate, total_steps=rounds * 64 * 8, pct_start=0.05)
    for round_ in range(rounds):
        frames, actions = lessons(wild, 64 * 48, 1_000_000 * round_ + int(wild * 500_000))
        wrong = torch.zeros(len(OUTPUTS))
        for batch in range(64):
            rows, state = slice(batch * 48, batch * 48 + 48), None
            for start in range(0, 2048, 256):
                x, y = frames[rows, start : start + 256], actions[rows, start : start + 256]
                hidden, state = model(x, state)
                state = state.detach()
                guesses = model.scores(hidden, y[..., 0])
                loss = sum(F.cross_entropy(guess.flatten(0, 1), y[..., i].flatten()) for i, guess in enumerate(guesses))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                schedule.step()
            wrong += torch.tensor([(guess.argmax(2) != y[..., i]).float().mean().item() for i, guess in enumerate(guesses)]) / 64
        print(f"wild {wild} round {round_} loss {loss.item():.4f} wrong per stream (dx dy pen scale busy) {[round(w, 4) for w in wrong.tolist()]}", flush=True)


@torch.no_grad()
def play(model, ticks, heat, size, seed):
    """Lets the model draw on its own page. A scripted visitor types and clicks, to see it react."""
    torch.manual_seed(seed)
    page, state, lines = Page(*size, size[0] // 2, size[1] // 2), None, []
    choose = lambda scores: torch.multinomial(F.softmax(scores[0, 0] / heat, 0), 1)
    for tick in range(ticks):
        if tick == ticks // 4:
            page.keys.extend(KEYS.index(key) + 1 for key in "hello")
        hidden, state = model(torch.tensor([[page.frame((size[0] // 4, size[1] // 4) if tick == ticks // 2 else None)]]), state)
        dx = choose(model.write[0](hidden))
        action = [choose(scores).item() for scores in model.scores(hidden, dx.view(1, 1))]
        line = page.move((dx.item() - REACH, action[1] - REACH, *action[2:]))
        if line:
            lines.append(line)
    return (*size, lines)


def quantize(model, base=None):
    """Rounds every tensor to 8 bits in place and returns the bytes: scales, then weights.
    With `base`, it stores only the difference from the base."""
    scales, weights = [], []
    with torch.no_grad():
        for tensor, start in zip(model.parameters(), base.parameters() if base else model.parameters()):
            delta = tensor - start if base else tensor
            scale = max(delta.abs().max().item(), 1e-9) / 127
            rounded = (delta / scale).round().clamp(-127, 127)
            tensor.copy_(rounded * scale + (start if base else 0))
            scales.append(scale)
            weights.append(rounded.numpy().astype(np.int8).tobytes())
    return np.array(scales, dtype=np.float32).tobytes() + b"".join(weights)


if __name__ == "__main__":
    torch.manual_seed(0)
    calm, wild = Model(), Model()
    if os.path.exists("calm.pt"):
        calm.load_state_dict(torch.load("calm.pt"))
    else:
        study(calm, 0, 12, 2e-3)
        torch.save(calm.state_dict(), "calm.pt")
    calm_bytes = quantize(calm)

    wild.load_state_dict(calm.state_dict())
    if os.path.exists("wild.pt"):
        wild.load_state_dict(torch.load("wild.pt"))
    else:
        study(wild, 1, 2, 3e-4)
        torch.save(wild.state_dict(), "wild.pt")
    wild_bytes = quantize(wild, calm)
    open("model.bin", "wb").write(calm_bytes + wild_bytes)
    print(sum(p.numel() for p in calm.parameters()), "parameters,", len(calm_bytes + wild_bytes), "bytes")

    # Look at what it draws as the weights slide from calm (0) to wild (1) and past it.
    for slide in (0, 0.1, 0.2, 0.3, 0.5, 1):
        mixed = Model()
        mixed.load_state_dict({name: value + slide * (wild.state_dict()[name] - value) for name, value in calm.state_dict().items()})
        draw([play(mixed, 6000, 0.9, size, 0) for size in [(1280, 800), (390, 800)]], f"model-{slide}.png")
