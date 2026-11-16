// The page the model draws on. Every tick the model reads one frame (the key you pressed,
// your click, where the pen is, its own last move) and writes one small move of the pen.
// Left alone it draws. Type and it writes. Click and it goes there. It never stops.
//
// Three inks draw at once: magenta leads, cyan and yellow each take their own sample of the
// same move and are pulled back to the lead. Where the model is sure, the three land on one
// line and print black. Where it is not, they split into color.
//
// The heat rises and falls in a slow wave, and every key and click pushes it up for a while.
// Heat slides the weights from the calm model toward the wild one, loosens the sampling and
// speeds up the ticks. Old ink fades to white in about two minutes, so there is always room.
// Open with #120 on the end to start 120 seconds in, at the top of the wave.

import { load } from './model.js'

const REACH = 24 // the farthest one tick can move on each axis, in units
const SCALES = [1, 2, 3, 5, 8, 13] // pixels per unit
const KEYS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-\n"
const NOWHERE = 61
const WAVE = 240 // seconds from one calm moment to the next
const NAME = Array.from('Dexter Storey', key => KEYS.indexOf(key) + 1)

const canvas = document.createElement('canvas')
Object.assign(canvas.style, { position: 'fixed', inset: 0, width: '100vw', height: '100vh' })
const paper = canvas.getContext('2d')

let wide, tall // the page, in pixels
let target = null // the click the pen is on its way to
let clicked = 0 // 1 for the one tick after a click
let nudge = 0 // heat added by keys and clicks, which wears off
let last = { dx: 0, dy: 0, pen: 0, scale: 0, busy: 0 } // what the model wrote on the previous tick
const keys = [...NAME] // typed keys waiting for their turn. The name is the first thing it writes.
const inks = ['#f0f', '#0ff', '#ff0'].map(color => ({ color, x: 16, y: 24, path: new Path2D() }))

function fit() {
	wide = innerWidth
	tall = innerHeight
	canvas.width = wide * devicePixelRatio
	canvas.height = tall * devicePixelRatio
	paper.scale(devicePixelRatio, devicePixelRatio)
	paper.fillStyle = '#fff'
	paper.fillRect(0, 0, wide, tall)
	paper.lineCap = paper.lineJoin = 'round'
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

function tick(model, heat) {
	const [lead] = inks
	const key = last.busy || target ? 0 : (keys.shift() ?? 0)
	const scores = model.tick([
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

	const [dx, dy, pen, scale, busy] = scores.map(stream => choose(stream, heat))
	last = { dx: dx - REACH, dy: dy - REACH, pen, scale, busy }

	for (const ink of inks) {
		const mine = ink === lead ? [dx, dy, pen] : scores.slice(0, 3).map(stream => choose(stream, heat))
		const from = [ink.x, ink.y]
		ink.x = hold(ink.x + (mine[0] - REACH) * SCALES[scale], wide)
		ink.y = hold(ink.y + (mine[1] - REACH) * SCALES[scale], tall)
		if (ink !== lead) {
			ink.x += (lead.x - ink.x) * 0.7
			ink.y += (lead.y - ink.y) * 0.7
		}
		if (mine[2]) {
			ink.path.moveTo(...from)
			ink.path.lineTo(ink.x, ink.y)
		}
	}
	if (target && target.x === lead.x && target.y === lead.y) target = null
}

function go(x, y) {
	target = { x: hold(Math.round(x), wide), y: hold(Math.round(y), tall) }
	clicked = 1
}

addEventListener('pointerdown', event => {
	go(event.clientX, event.clientY)
	nudge += 0.05
})

addEventListener('keydown', event => {
	const key = KEYS.indexOf(event.key === 'Enter' ? '\n' : event.key)
	if (key < 0 || event.metaKey || event.ctrlKey || event.altKey) return
	event.preventDefault()
	keys.push(key + 1)
	nudge += 0.01
})

addEventListener('resize', fit)

if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
	const model = await load()
	const started = performance.now() / 1000 - (Number(location.hash.slice(1)) || 0)
	let slid = -1 // the heat the weights were last set for
	let faded = 0 // when the ink last faded a step
	let signed = 0 // how many waves the name has been written for
	document.body.replaceChildren(canvas) // the model writes the name from here on
	fit()

	requestAnimationFrame(function frame(now) {
		const seconds = now / 1000 - started
		const heat = Math.min(1, 0.33 - 0.28 * Math.cos((seconds / WAVE) * 2 * Math.PI) + nudge)
		nudge *= 0.998
		if (Math.abs(heat - slid) > 0.01) model.slide(0.7 * (slid = heat) ** 3)

		// Each time the wave comes back to calm, it goes to the corner and signs again.
		if (seconds / WAVE > signed + 1) {
			signed++
			go(16, 24)
			keys.push(...NAME)
		}

		paper.globalCompositeOperation = 'multiply'
		// A bigger page gets more ticks, so a phone and a wide screen fill at the same pace.
		for (let ticks = ((4 + heat * 5) * Math.sqrt(wide * tall)) / 1000; ticks > 0 && performance.now() - now < 8; ticks--) tick(model, 0.6 + heat * 0.3)
		for (const ink of inks) {
			paper.strokeStyle = ink.color
			paper.stroke(ink.path)
			ink.path = new Path2D()
		}

		// Every half second all the ink gets one shade lighter. Black is gone in two minutes.
		if (now - faded > 450) {
			faded = now
			paper.globalCompositeOperation = 'lighter'
			paper.fillStyle = '#010101'
			paper.fillRect(0, 0, wide, tall)
		}
		requestAnimationFrame(frame)
	})
}
