// The tiny model. A recurrent network that reads one frame a tick and scores every choice
// for each output stream. train.py made model.bin and explains the streams.
// These numbers match train.py.
const INPUTS = [71, 2, 62, 62, 16, 16, 49, 49, 2, 6]
const OUTPUTS = [49, 49, 2, 6, 2]
const WIDTH = 64
const STATE = 256

// The size of every tensor, in the order train.py wrote them:
// a table per input stream, the recurrent core, then a head per output stream.
const sizes = [...INPUTS.map(size => size * WIDTH), 3 * STATE * WIDTH, 3 * STATE * STATE, 3 * STATE, 3 * STATE, ...OUTPUTS.flatMap(size => [size * STATE, size])]

// One set of tensors: a scale per tensor, then every weight as a single byte.
function unpack(buffer, offset) {
	const scales = new Float32Array(buffer.slice(offset, offset + sizes.length * 4))
	offset += sizes.length * 4
	return sizes.map((size, index) => {
		const tensor = Float32Array.from(new Int8Array(buffer, offset, size), byte => byte * scales[index])
		offset += size
		return tensor
	})
}

// matrix · x + bias, for a matrix stored row by row.
function linear(matrix, bias, x, rows) {
	const columns = x.length
	const out = new Float32Array(rows)
	for (let row = 0; row < rows; row++) {
		let sum = bias[row]
		for (let column = 0, at = row * columns; column < columns; column++) sum += matrix[at + column] * x[column]
		out[row] = sum
	}
	return out
}

const squash = value => 1 / (1 + Math.exp(-value))

export async function load() {
	const buffer = await (await fetch('model.bin')).arrayBuffer()
	const calm = unpack(buffer, 0)
	const drift = unpack(buffer, buffer.byteLength / 2) // the wild fine-tune, as a difference from calm
	const weights = calm.map(tensor => tensor.slice())
	const state = new Float32Array(STATE)

	return {
		// Moves the weights from calm (0) toward wild (1), or past it.
		slide(amount) {
			weights.forEach((tensor, index) => {
				for (let i = 0; i < tensor.length; i++) tensor[i] = calm[index][i] + amount * drift[index][i]
			})
		},

		forget() {
			state.fill(0)
		},

		// Reads one frame, updates the state, and returns the scores for each output stream.
		tick(frame) {
			const x = new Float32Array(WIDTH)
			frame.forEach((value, stream) => {
				for (let i = 0; i < WIDTH; i++) x[i] += weights[stream][value * WIDTH + i]
			})
			const [fromInput, fromState, inputBias, stateBias] = weights.slice(INPUTS.length)
			const a = linear(fromInput, inputBias, x, 3 * STATE)
			const b = linear(fromState, stateBias, state, 3 * STATE)
			for (let i = 0; i < STATE; i++) {
				const reset = squash(a[i] + b[i])
				const keep = squash(a[STATE + i] + b[STATE + i])
				const fresh = Math.tanh(a[2 * STATE + i] + reset * b[2 * STATE + i])
				state[i] = (1 - keep) * fresh + keep * state[i]
			}
			const heads = weights.slice(INPUTS.length + 4)
			return OUTPUTS.map((size, stream) => linear(heads[stream * 2], heads[stream * 2 + 1], state, size))
		}
	}
}
