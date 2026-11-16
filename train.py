# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch", "numpy", "pillow", "Hershey-Fonts"]
# ///
"""Trains the tiny model that draws on dexterstorey.com and writes model.bin.

The model drives a plotter pen on a tick. Every tick it reads one number per input stream
(the key pressed, a click, where the click is, where the pen is, its own last move) and writes
one number per output stream (move x, move y, pen down, size, busy).

It learns by copying `Teacher`, a hand-written pen that draws lines, rings, waves, grids and
other plain figures, writes what is typed and goes where it is clicked. `Page` is the world
they both live in, and agent.js repeats its rules in the browser.

Training has two parts. The base copies the calm teacher. The fine-tune copies a wild one.
model.bin holds the base and the difference, so the page can slide from calm to wild.

Run: uv run train.py          trains and writes model.bin
     uv run train.py teacher  only draws what the two teachers do, to teacher-*.png
"""

import math
import os
import random
import sys
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
        """Picks what to draw next. Each row is an act, how often when calm, how often when wild."""
        acts = [
            (self.travel, 6, 4), (self.rest, 3, 0.3), (self.rule, 3, 2), (self.circle, 3, 3), (self.wave, 3, 3),
            (self.polygon, 2.5, 2), (self.rays, 2, 3), (self.grid, 2, 1), (self.meander, 2, 3), (self.hatch, 2, 3),
            (self.spiral, 1.5, 2), (self.stairs, 1, 1), (self.scribble, 0, 2),
        ]  # fmt: skip
        return self.pick([act for act, _, _ in acts], [calm + (wild - calm) * self.wild for _, calm, wild in acts])()

    def go(self):
        """Lifts the pen and closes in on the click, one look at a time. Then draws something there."""
        while self.at[TX] != NOWHERE:
            dx, dy, scale = toward(self.at[TX], self.at[TY])
            yield from self.do(dx, dy, 0, scale)
        self.resize()
        yield from self.pick([self.circle, self.rays, self.polygon, self.spiral, self.rest], [3, 3, 2, 2, 1])()

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

    # Every figure comes in a few fixed sizes and stops either on a count the model can see
    # or on a coin toss, so the model can learn each one exactly instead of guessing.

    def rule(self):
        """Straight lines side by side, solid or dashed."""
        ax, ay = self.pick([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1)], [6, 3, 2, 2, 1, 1])
        dashed = self.chance(0.25)
        step = 6 if dashed else self.pick([12, 24])
        gap, length = self.pick([2, 3, 5, 8, 13]), self.pick([4, 8, 16, 32])
        while True:
            for i in range(length):
                if self.edge(ax, ay):
                    break
                yield from self.do(ax * step, ay * step, int(not dashed or i % 2 == 0))
            if self.chance(0.45):
                return
            yield from self.do(gap * (ax == 0), gap * (ax != 0), 0)
            ax, ay = -ax, -ay

    def circle(self):
        """Rings around one centre, whole or half."""
        radius, grow, half = self.pick([4, 7, 12, 20, 32]), self.pick([2, 4]), self.chance(0.2)
        angle, turn = 0, self.pick([1, -1])
        while True:
            steps = max(8, round(radius * 1.2)) // (2 if half else 1)
            x, y = round(radius * math.cos(angle)), round(radius * math.sin(angle))
            for _ in range(steps):
                angle += turn * math.pi / steps * (1 if half else 2)
                nx, ny = round(radius * math.cos(angle)), round(radius * math.sin(angle))
                yield from self.do(nx - x, ny - y, 1)
                x, y = nx, ny
            if self.chance(0.55):
                return
            yield from self.do(round(grow * math.cos(angle)), round(grow * math.sin(angle)), 0)
            radius += grow
            turn = -turn if half else turn

    def spiral(self):
        radius, grow, angle, turn = 2, self.pick([3, 5]) / math.tau, 0, self.pick([1, -1])
        x, y = radius, 0
        while radius < 45 and not (radius > 6 and self.chance(1 / 40)):
            swing = min(0.6, 6 / radius)
            angle += turn * swing
            radius += grow * swing
            nx, ny = round(radius * math.cos(angle)), round(radius * math.sin(angle))
            yield from self.do(nx - x, ny - y, 1)
            x, y = nx, ny

    def polygon(self):
        """Triangles, squares and hexagons, nested and sometimes twisting as they grow."""
        sides = self.pick([3, 4, 6], [2, 5, 2])
        radius, grow, twist = self.pick([6, 12, 20]), self.pick([3, 5]), self.pick([0, 0, math.pi / 12])
        angle = math.pi / sides
        corner = lambda k: (round(radius * math.cos(angle + math.tau * k / sides)), round(radius * math.sin(angle + math.tau * k / sides)))
        x, y = corner(0)
        while True:
            for k in range(1, sides + 1):
                nx, ny = corner(k)
                yield from self.line(nx - x, ny - y, 1)
                x, y = nx, ny
            if self.chance(0.4):
                return
            radius, angle = radius + grow, angle + twist
            nx, ny = corner(0)
            yield from self.line(nx - x, ny - y, 0)
            x, y = nx, ny

    def rays(self):
        """Lines out from one point, all the same or long and short in turn."""
        count, length, uneven = self.pick([8, 12, 16, 24]), self.pick([6, 12, 24, 40]), self.chance(0.3)
        for k in range(count):
            reach = length / 2 if uneven and k % 2 else length
            dx, dy = round(reach * math.cos(math.tau * k / count)), round(reach * math.sin(math.tau * k / count))
            yield from self.line(dx, dy, 1)
            yield from self.line(-dx, -dy, 0)

    def wave(self):
        """Rows of the same wave, each row in step with the last or a quarter turn behind."""
        tall, period = self.pick([2, 4, 8]), self.pick([8, 12, 16])
        shift, step, gap = self.pick([0, period // 4]), self.pick([3, 4, 6]), self.pick([4, 6, 8])
        length = period * self.pick([1, 2, 3, 4])
        height = lambda i, row: round(tall * math.sin(math.tau * (i + shift * row) / period))
        i, way, row = 0, 1, 0
        while True:
            for _ in range(length):
                if self.edge(way, 0):
                    break
                yield from self.do(way * step, height(i + way, row) - height(i, row), 1)
                i += way
            if self.chance(0.2):
                return
            yield from self.line(0, gap + height(i, row + 1) - height(i, row), 0)
            way, row = -way, row + 1

    def grid(self):
        cell, columns, rows = self.pick([4, 6, 8, 12]), self.pick([2, 3, 4, 6]), self.pick([2, 3, 4, 6])
        way, up = 1, -1
        for row in range(rows + 1):
            if row:
                yield from self.line(0, cell, 0)
            yield from self.line(way * columns * cell, 0, 1)
            way = -way
        for column in range(columns + 1):
            if column:
                yield from self.line(way * cell, 0, 0)
            yield from self.line(0, up * rows * cell, 1)
            up = -up

    def meander(self):
        """One long line that wanders and never turns sharply."""
        heading, step, curl = self.random.random() * math.tau, self.random.randint(3, 8), self.pick([0.05, 0.1, 0.2])
        bend = 0
        while not self.chance(1 / 70):
            bend = max(-0.5, min(0.5, bend + self.random.gauss(0, curl)))
            heading += bend
            dx, dy = round(step * math.cos(heading)), round(step * math.sin(heading))
            if self.edge(dx, dy):
                heading += math.pi
            yield from self.do(dx, dy, 1)

    def stairs(self):
        across, down = self.pick([3, 6, 10]) * self.pick([1, -1]), self.pick([3, 6, 10]) * self.pick([1, -1])
        while not self.chance(1 / 12) and not self.edge(across, down):
            yield from self.do(across, 0, 1)
            yield from self.do(0, down, 1)

    def hatch(self):
        """Short strokes packed side by side, to shade a patch."""
        ax, ay = self.pick([(6, 6), (6, -6), (0, 8), (8, 0), (8, 3)])
        gap, length = self.pick([1, 2, 3]), self.pick([1, 2, 3, 4])
        while True:
            for _ in range(length):
                yield from self.do(ax, ay, 1)
            if self.chance(0.07):
                return
            yield from self.do(gap * (ay != 0), gap * (ay == 0), 0)
            ax, ay = -ax, -ay

    def scribble(self):
        """A loose tangle: the pen keeps moving and its heading jumps about. Only the wild teacher does this."""
        heading, step = self.random.random() * math.tau, self.random.randint(3, 9)
        while not self.chance(1 / 50):
            heading += self.random.gauss(0, 0.6)
            dx, dy = round(step * math.cos(heading)), round(step * math.sin(heading))
            if self.edge(dx, dy):
                heading += math.pi
            yield from self.do(dx, dy, 1)

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
        while not self.chance(1 / 120 + self.wild):
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

    def forward(self, frames, state=None):
        x, state = self.core(sum(table(frames[..., i]) for i, table in enumerate(self.read)), state)
        return [head(x) for head in self.write], state


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
                guesses, state = model(x, state)
                state = state.detach()
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
    page, state, lines = Page(*size, 16, 24), None, []
    page.keys.extend(KEYS.index(key) + 1 for key in "Dexter Storey")
    for tick in range(ticks):
        if tick == ticks // 4:
            page.keys.extend(KEYS.index(key) + 1 for key in "hello")
        guesses, state = model(torch.tensor([[page.frame((size[0] // 2, size[1] // 2) if tick == ticks // 2 else None)]]), state)
        line = page.move([torch.multinomial(F.softmax(guess[0, 0] / heat, 0), 1).item() - (REACH if i < 2 else 0) for i, guess in enumerate(guesses)])
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
        draw([play(mixed, 6000, 0.7, size, 0) for size in [(1280, 800), (390, 800)]], f"model-{slide}.png")
