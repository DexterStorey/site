// The page. A model on the server is an artist, this site is its sketchbook, and this file
// is its hand. The model writes brush commands, one per line (api/draw.js lists them). The
// brush here paints each one, stroke by stroke, on a sheet of paper. When a drawing is
// finished the model says what it would change and starts the next on a new sheet underneath.
// It never stops.
//
// Click the paper and type to write on it. The model reads what you wrote and answers.
//
// seeds.txt holds drawings the model made ahead of time. One of them starts at once on every
// visit, so the brush is moving before the model has answered, and they fill in whenever the
// model is away.

const INKS = { black: '#141414', grey: '#8f8f8f' }
const WASHES = { cyan: [0.3, 0.82, 0.96], magenta: [1, 0.36, 0.62], yellow: [1, 0.86, 0.2] }
const LINKS = /^https:\/\/(www\.)?(rubriclabs\.com|github\.com|x\.com|instagram\.com|youtube\.com)(\/|$)/
const NIB = 0.6 // the angle the brush is held at, in radians
const still = matchMedia('(prefers-reduced-motion: reduce)').matches

// Every sheet is the size of the window. The model works in units, on a paper of the same
// shape and always a million square units, so a phone and a wide screen get the same detail.
const wide = innerWidth
const tall = innerHeight
const live = [wide, tall].map(size => Math.round(size / Math.sqrt((wide * tall) / 1e6)))

const story = { drafts: [], title: null, lines: [], paper: null } // what is on the paper: finished drawings as [title, note], and this one's size and lines
const queue = [] // lines the model has written that the brush has not reached
let line = null // the line the brush is on
let strokes = [] // the strokes of that line still to paint
let sheet = null // the paper being drawn on: { section, ink, scale, left, top }
let listening = null // the open request to the model
let wait = Infinity // no new request before this time
let cursor = null // where the visitor is writing: { x, y, span }
let pause = null // the timer that sends what the visitor wrote
let seeds = [] // drawings made ahead of time, each a list of lines, in the order they will be shown

const cut = text => (text.includes(' ') ? [text.slice(0, text.indexOf(' ')), text.slice(text.indexOf(' ') + 1)] : [text, ''])

// Adds one line to a story. Used for the paper as it is, and to work out what it will be
// once the brush has caught up with the model.
function fold(story, line) {
	const [word, rest] = cut(line)
	if (word === 'paper') story.paper = rest.split(' ').map(Number)
	else if (word === 'draft') story.title = rest
	else if (word === 'next') {
		if (story.title) story.drafts.push([story.title, rest])
		Object.assign(story, { title: null, lines: [], paper: null })
	} else story.lines.push(line)
	return story
}

// A smooth curve through the points, as a point every unit and a half.
function curve(numbers, closed) {
	const points = []
	for (let i = 0; i + 1 < numbers.length; i += 2) points.push([numbers[i], numbers[i + 1]])
	const at = i => (closed ? points[(i + points.length) % points.length] : points[Math.max(0, Math.min(points.length - 1, i))])
	const out = []
	for (let i = 0; i < points.length - (closed ? 0 : 1); i++) {
		const [a, b, c, d] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
		const steps = Math.ceil(Math.hypot(c[0] - b[0], c[1] - b[1]) / 1.5)
		for (let step = 0; step < steps; step++) {
			const t = step / steps
			out.push(
				[0, 1].map(
					k => 0.5 * (2 * b[k] + (c[k] - a[k]) * t + (2 * a[k] - 5 * b[k] + 4 * c[k] - d[k]) * t * t + (3 * b[k] - a[k] - 3 * c[k] + d[k]) * t * t * t)
				)
			)
		}
	}
	out.push(at(closed ? 0 : points.length - 1))
	return out
}

// An ink line. The brush is held at one angle, so the line is thick going one way and thin
// going the other, and it lands and lifts with a taper.
function brush(ink, weight, numbers) {
	const closed = numbers.length > 5 && numbers[0] === numbers.at(-2) && numbers[1] === numbers.at(-1)
	const path = curve(closed ? numbers.slice(0, -2) : numbers, closed)
	const ends = Math.min(40, path.length / 3)
	let width = 0
	const widths = path.map(([x, y], i) => {
		const [nx, ny] = path[Math.min(path.length - 1, i + 4)]
		const lean = Math.abs(Math.sin(Math.atan2(ny - y, nx - x) + NIB))
		const taper = Math.min(1, (0.15 + Math.min(i, path.length - 1 - i)) / ends)
		width += (weight * (0.4 + 0.8 * lean) - width) * (i ? 0.08 : 1)
		return Math.max(0.3, width * (0.2 + 0.8 * taper * (2 - taper)))
	})
	const { ink: paper } = sheet
	return {
		length: path.length * 1.5,
		to(from, upto) {
			paper.globalCompositeOperation = 'darken'
			paper.strokeStyle = INKS[ink] ?? INKS.black
			for (let i = Math.floor(from / 1.5); i < Math.min(path.length - 1, Math.ceil(upto / 1.5)); i++) {
				paper.lineWidth = widths[i]
				paper.beginPath()
				paper.moveTo(...path[i])
				paper.lineTo(...path[i + 1])
				paper.stroke()
			}
		}
	}
}

// A flat shape of colour. It soaks in over twelve thin coats that multiply up to the colour.
function wash(colour, numbers) {
	const coat = `rgb(${(WASHES[colour] ?? WASHES.yellow).map(part => Math.round(255 * part ** (1 / 12)))})`
	const shape = new Path2D()
	for (const point of curve(numbers, true)) shape.lineTo(...point)
	const { ink: paper } = sheet
	return {
		length: 360,
		to(from, upto) {
			paper.globalCompositeOperation = 'multiply'
			paper.fillStyle = coat
			for (let i = Math.floor(from / 30); i < Math.floor(upto / 30); i++) paper.fill(shape)
		}
	}
}

// Words on the sheet, typed on a letter at a time. x and y are in units, size is a capital's height.
function words(text, x, y, size, name = 'span') {
	const { section, scale, left, top } = sheet
	const element = section.appendChild(document.createElement(name))
	const height = Math.max(11, size * scale * 1.4)
	const across = Math.max(8, Math.min(left + x * scale, wide - 16 - text.length * height * 0.61)) // kept on the paper
	Object.assign(element.style, { left: `${across}px`, top: `${top + y * scale - height}px`, fontSize: `${height}px` })
	return { element, length: text.length * 9, to: (from, upto) => (element.textContent = text.slice(0, Math.ceil(upto / 9))) }
}

// Turns one line from the model into the strokes the brush will make for it.
function begin(line) {
	const [word, rest] = cut(line)
	if (word === 'draft') {
		const following = innerHeight + scrollY > document.documentElement.scrollHeight - 80
		const [across, down] = story.paper ?? live
		const scale = Math.min(wide / across, tall / down)
		const section = document.body.appendChild(document.createElement('section'))
		const canvas = section.appendChild(document.createElement('canvas'))
		Object.assign(section.style, { width: `${wide}px`, height: `${tall}px` })
		canvas.width = wide * devicePixelRatio
		canvas.height = tall * devicePixelRatio
		sheet = { section, ink: canvas.getContext('2d'), scale, left: (wide - across * scale) / 2, top: (tall - down * scale) / 2 }
		sheet.ink.setTransform(scale * devicePixelRatio, 0, 0, scale * devicePixelRatio, sheet.left * devicePixelRatio, sheet.top * devicePixelRatio)
		sheet.ink.lineCap = 'round'
		cursor = null
		if (following) section.scrollIntoView({ behavior: still ? 'auto' : 'smooth' })
		if (document.querySelectorAll('section').length > 40) document.querySelector('section').remove()
		const caption = section.appendChild(document.createElement('header'))
		return [{ length: rest.length * 9, to: (from, upto) => (caption.textContent = `${story.drafts.length + 1}. ${rest}`.slice(0, 3 + Math.ceil(upto / 9))) }]
	}
	if (!sheet) return []
	if (word === 'idea') {
		const thought = sheet.section.querySelector('header').appendChild(document.createElement('p'))
		return [{ length: rest.length * 9, to: (from, upto) => (thought.textContent = rest.slice(0, Math.ceil(upto / 9))) }]
	}
	if (word === 'next') {
		const note = sheet.section.appendChild(document.createElement('footer'))
		return [{ length: rest.length * 9, to: (from, upto) => (note.textContent = rest.slice(0, Math.ceil(upto / 9))) }]
	}
	const parts = rest.split(' ')
	if (word === 'wash') {
		const numbers = parts.slice(1).map(Number)
		return numbers.length > 5 && numbers.every(isFinite) ? [wash(parts[0], numbers)] : []
	}
	if (word === 'line') {
		const numbers = parts.slice(2).map(Number).filter(isFinite)
		return numbers.length > 3 ? [brush(parts[0], Math.max(1, Math.min(8, +parts[1] || 2)), numbers)] : []
	}
	const [x, y, size] = parts.map(Number)
	if (!(isFinite(x) && isFinite(y) && size > 0)) return []
	if (word === 'write') return [words(parts.slice(3).join(' '), x, y, size)]
	if (word === 'link' && LINKS.test(parts[3])) {
		const link = words(parts.slice(4).join(' '), x, y, size, 'a')
		link.element.href = parts[3]
		link.element.target = '_blank'
		return [link]
	}
	return []
}

// Moves the brush on by a distance, through as many strokes and lines as that covers.
function draw(distance) {
	while (distance > 0) {
		const stroke = strokes[0]
		if (!stroke) {
			if (line) fold(story, line)
			line = queue.shift()
			if (!line) return
			strokes = begin(line).map(stroke => ({ ...stroke, drawn: 0 }))
			continue
		}
		const step = Math.min(distance, stroke.length - stroke.drawn)
		stroke.to(stroke.drawn, (stroke.drawn += step))
		distance -= step
		if (stroke.drawn >= stroke.length) strokes.shift()
	}
}

// Asks the model to carry on from what the paper will hold once the brush has caught up.
async function ask(visitor) {
	listening?.abort()
	const control = (listening = new AbortController())
	const { drafts, title, lines, paper } = [line, ...(visitor ? [] : queue)].filter(Boolean).reduce(fold, structuredClone(story))
	const [across, down] = paper ?? live
	let heard = 0
	try {
		const response = await fetch('/api/draw', {
			method: 'POST',
			body: JSON.stringify({ wide: across, tall: down, count: drafts.length, drafts: drafts.slice(-16), title, lines, visitor }),
			signal: control.signal
		})
		if (!response.ok) throw new Error(response.status)
		const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
		for (let text = ''; ; ) {
			const { done, value = '\n' } = await reader.read()
			const lines = (text + value).split('\n')
			text = lines.pop()
			const whole = lines.map(line => line.trim()).filter(Boolean)
			if (visitor && !heard && whole.length) queue.length = 0 // the answer replaces what was planned
			queue.push(...whole)
			heard += whole.length
			if (done) break
		}
	} catch {}
	if (listening !== control) return
	listening = null
	if (!heard) wait = performance.now() + 30000 // the model is away
}

// Draws one from before, after closing the sheet the model left unfinished.
function rest() {
	if (!seeds.length) return
	if ([line].filter(Boolean).reduce(fold, structuredClone(story)).title) queue.push('next')
	queue.push(...seeds[0])
	seeds.push(seeds.shift())
}

// The visitor's own writing goes straight onto the paper, and a moment after they stop
// the model is told. Its answer replaces what it had planned but not yet drawn.
function send() {
	clearTimeout(pause)
	pause = null
	const { scale, left, top } = sheet
	story.lines.push(`visitor ${Math.round((cursor.x - left) / scale)} ${Math.round((cursor.y - top) / scale)} ${cursor.span.textContent}`.trim())
	cursor = null
	ask(true)
}

function point(x, y) {
	if (pause) send()
	const span = sheet.section.appendChild(document.createElement('span'))
	span.className = 'visitor'
	Object.assign(span.style, { left: `${x}px`, top: `${y - 18}px` })
	cursor = { x, y, span }
	pause = setTimeout(send, 1500)
}

addEventListener('pointerdown', event => {
	if (!sheet || event.target.closest('section') !== sheet.section || event.target.closest('a')) return
	const box = sheet.section.getBoundingClientRect()
	point(event.clientX - box.left, event.clientY - box.top)
})

addEventListener('keydown', event => {
	if (!sheet || !line || event.metaKey || event.ctrlKey || event.altKey) return
	if (event.key === 'Enter' && pause) return send()
	if (!/^[ -~]$/.test(event.key)) return
	event.preventDefault()
	if (!cursor) point(24, tall - 64)
	cursor.span.textContent += event.key
	clearTimeout(pause)
	pause = setTimeout(send, 1500)
})

// One of the earlier drawings starts at once: those made for the paper nearest the window's shape.
fetch('seeds.txt')
	.then(response => response.text())
	.then(text => {
		const all = text.trim().split('\n\n').map(seed => seed.split('\n'))
		const off = ([paper]) => Math.abs(Math.log(((wide / tall) * paper.split(' ')[2]) / paper.split(' ')[1]))
		const nearest = Math.min(...all.map(off))
		seeds = all.filter(seed => off(seed) < nearest + 0.2).sort(() => Math.random() - 0.5)
		rest()
	})
	.catch(() => {})
	.finally(() => (wait = 0))

// The brush goes faster the further the model is ahead of it, so it neither stalls nor falls behind.
let before = performance.now()
requestAnimationFrame(function frame(now) {
	const seconds = Math.min(0.05, (now - before) / 1000)
	before = now
	draw(still ? Infinity : seconds * Math.min(1600, 600 + 40 * queue.length))
	if (!listening && queue.length < 12) {
		if (now > wait) ask()
		else if (!queue.length && !strokes.length && !still) rest()
	}
	requestAnimationFrame(frame)
})
