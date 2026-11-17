// The page the model draws on. Every tick the model reads one frame (the key you pressed,
// your click, where the pen is, its own last move) and writes one small move of the pen.
// Left alone it draws. Type and it writes. Click and it goes there. It never stops.
//
// Four inks draw at once. The grey one leads. Cyan, magenta and yellow each take their own
// sample of the same move and are pulled back to the lead. Where the model is sure, they hide
// under the grey line. Where it is not, they show beside it as color. The inks are thin and
// pale, like pencil, so the page darkens slowly where lines pile up.
//
// The heat rises and falls in a slow wave, and every key and click pushes it up for a while.
// Heat slides the weights from the calm model toward the wild one, loosens the sampling and
// speeds up the ticks. Nothing is ever erased.
// Open with #60 on the end to start 60 seconds in, at the top of the wave.

import { load } from './model.js'

const REACH = 24 // the farthest one tick can move on each axis, in units
const SCALES = [1, 2, 3, 5, 8, 13] // pixels per unit
const KEYS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-\n"
const NOWHERE = 61
const WAVE = 120 // seconds from one calm moment to the next

const canvas = document.createElement('canvas')
Object.assign(canvas.style, { position: 'fixed', inset: 0, width: '100vw', height: '100vh' })
const paper = canvas.getContext('2d')

let wide, tall // the page, in pixels
let target = null // the click the pen is on its way to
let clicked = 0 // 1 for the one tick after a click
let nudge = 0 // heat added by keys and clicks, which wears off
let last = { dx: 0, dy: 0, pen: 0, scale: 0, busy: 0 } // what the model wrote on the previous tick
const keys = [] // typed keys waiting for their turn
let typed = 0 // when the last key was pressed
const inks = ['#888', '#aff', '#faf', '#ffa'].map(color => ({ color, x: Math.round(innerWidth / 2), y: Math.round(innerHeight / 2), path: new Path2D() }))

function fit() {
	wide = innerWidth
	tall = innerHeight
	canvas.width = wide * devicePixelRatio
	canvas.height = tall * devicePixelRatio
	paper.scale(devicePixelRatio, devicePixelRatio)
	paper.fillStyle = '#fff'
	paper.fillRect(0, 0, wide, tall)
	paper.lineCap = paper.lineJoin = 'round'
	paper.lineWidth = 0.75
	paper.globalCompositeOperation = 'multiply'
}

// The rules of the page. train.py has the same ones in `Page`.
const cell = (at, size) => Math.min(15, Math.floor((at * 16) / size))
const hold = (at, size) => Math.max(0, Math.min(size - 1, at))
const offset = distance =>
	Math.abs(distance) <= REACH ? distance + REACH : 49 + [40, 80, 160, 320, 640].filter(edge => Math.abs(distance) >= edge).length + (distance < 0 ? 6 : 0)

// Picks from the scores. Low heat picks the likely one, high heat picks almost anything.
function choose(scores, heat) {
	const best = Math.max(...scores)
	const odds = Array.from(scores, score => Math.exp((score - best) / heat))
	let roll = Math.random() * odds.reduce((a, b) => a + b)
	return Math.max(0, odds.findIndex(chance => (roll -= chance) < 0))
}

// One tick. Returns false when the pen neither moved nor drew, so the page can skip the pause.
function tick(model, heat) {
	const [lead] = inks
	const key = last.busy || target ? 0 : (keys.shift() ?? 0)
	const score = model.tick([
		key,
		clicked,
		target ? offset(target.x - lead.x) : NOWHERE,
		target ? offset(target.y - lead.y) : NOWHERE,
		cell(lead.x, wide),
		cell(lead.y, tall),
		last.dx + REACH,
		last.dy + REACH,
		last.pen,
		last.scale
	])
	clicked = 0

	// A move is dx, then dy given that dx. Every ink takes its own. They share the rest.
	const across = score(0)
	const pen = choose(score(2), heat)
	const scale = choose(score(3), heat)
	const busy = choose(score(4), heat)

	for (const ink of inks) {
		const dx = choose(across, heat)
		const dy = choose(score(1, dx), heat)
		const from = [ink.x, ink.y]
		ink.x = hold(ink.x + (dx - REACH) * SCALES[scale], wide)
		ink.y = hold(ink.y + (dy - REACH) * SCALES[scale], tall)
		if (ink === lead) last = { dx: dx - REACH, dy: dy - REACH, pen, scale, busy }
		else {
			ink.x += (lead.x - ink.x) * 0.85
			ink.y += (lead.y - ink.y) * 0.85
		}
		if (pen) {
			ink.path.moveTo(...from)
			ink.path.lineTo(ink.x, ink.y)
		}
	}
	if (target && target.x === lead.x && target.y === lead.y) target = null
	return last.dx !== 0 || last.dy !== 0 || last.pen === 1
}

addEventListener('pointerdown', event => {
	target = { x: hold(Math.round(event.clientX), wide), y: hold(Math.round(event.clientY), tall) }
	clicked = 1
	nudge += 0.05
})

addEventListener('keydown', event => {
	const key = KEYS.indexOf(event.key === 'Enter' ? '\n' : event.key)
	if (key < 0 || event.metaKey || event.ctrlKey || event.altKey) return
	event.preventDefault()
	keys.push(key + 1)
	typed = performance.now()
	nudge += 0.01
})

addEventListener('resize', fit)

if (matchMedia('(prefers-reduced-motion: reduce)').matches) document.body.append('Dexter Storey')
else {
	const model = await load()
	const started = performance.now() / 1000 - (Number(location.hash.slice(1)) || 0)
	let slid = -1 // the heat the weights were last set for
	document.body.append(canvas)
	fit()

	requestAnimationFrame(function frame(now) {
		const seconds = now / 1000 - started
		const heat = Math.min(1, 0.33 - 0.28 * Math.cos((seconds / WAVE) * 2 * Math.PI) + nudge)
		nudge *= 0.998
		if (Math.abs(heat - slid) > 0.01) model.slide(0.7 * (slid = heat) ** 3)

		// A bigger page gets more ticks, so a phone and a wide screen fill at the same pace.
		// A tick where the pen rests does not count, so it never sits still, unless someone
		// is typing: then it waits for the next key the way it learned to.
		let ticks = ((4 + heat * 4) * Math.sqrt(wide * tall)) / 1000
		while (ticks > 0 && performance.now() - now < 8) if (tick(model, 0.85 + heat * 0.15) || now - typed < 1500) ticks--
		for (const ink of inks) {
			paper.strokeStyle = ink.color
			paper.stroke(ink.path)
			ink.path = new Path2D()
		}
		requestAnimationFrame(frame)
	})
}
