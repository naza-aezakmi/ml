// Reuse the production decoder and terrainStats: no second implementation of metrics.
// node scripts/presentation_extract.cjs ../api ../reports/presentation-cache
const fs = require('node:fs')
const path = require('node:path')
const { gunzipSync } = require('node:zlib')
const { createHash } = require('node:crypto')
const api = path.resolve(process.argv[2] || '../api')
const out = path.resolve(process.argv[3] || '../reports/presentation-cache')
fs.mkdirSync(out, { recursive: true })
const directory = path.join(api, 'terrain-atlas')
const read = name => fs.readFileSync(path.join(directory, name), 'utf8')
const manifest = JSON.parse(read('relief/index.json'))
const screen = gunzipSync(fs.readFileSync(path.join(directory, 'relief/field-screen-v1.gz')))
if (screen.length !== manifest.worlds.earth.count) throw new Error('Invalid screen')
const codec = read('relief-codec.js')
const worker = read('relief-worker.js').replace('importScripts("relief-codec.js");', '')
const { decodeReliefProfile, terrainStats } = new Function('fetch', 'self',
  codec + '\n' + worker + '\nreturn { decodeReliefProfile, terrainStats };'
)(async () => ({ ok: true, json: async () => ({}) }), {})
const metadata = { samples: manifest.samples, step: manifest.step, scale: manifest.scale,
  features: ['range_m', 'plane_slope_deg', 'roughness_m'], scope: 'All stored production windows; Earth filtered by the production water mask',
  worker_sha256: createHash('sha256').update(read('relief-worker.js')).digest('hex'), bodies: {} }
for (const body of ['earth', 'moon', 'mars']) {
  const info = manifest.worlds[body]
  const count = body === 'earth' ? screen.reduce((sum, flag) => sum + Boolean(flag & 1), 0) : info.count
  const output = new Float64Array(count * 3)
  const values = new Int16Array(manifest.samples ** 2)
  let n = 0
  let stored = 0
  const digest = createHash('sha256')
  for (const part of info.tiles) {
    if (!part.file) continue
    const compressed = fs.readFileSync(path.join(directory, 'relief', body, part.file))
    digest.update(compressed)
    const plain = gunzipSync(compressed)
    const buffer = plain.buffer.slice(plain.byteOffset, plain.byteOffset + plain.byteLength)
    const view = new DataView(decodeReliefProfile(buffer, part.count, manifest.samples, body === 'earth'))
    const stride = values.length * 2 + (body === 'earth' ? 4 : 0)
    if (view.byteLength !== part.count * stride) throw new Error('Invalid decoded tile')
    for (let cell = 0; cell < part.count; cell++) {
      stored++
      let offset = cell * stride
      if (body === 'earth') {
        if (!(screen[view.getUint32(offset, true)] & 1)) continue
        offset += 4
      }
      for (let i = 0; i < values.length; i++) values[i] = view.getInt16(offset + i * 2, true)
      const stats = terrainStats(values, manifest.scale)
      if (![stats.range, stats.slope, stats.roughness].every(Number.isFinite)) throw new Error('Invalid descriptor')
      output[n * 3] = stats.range
      output[n * 3 + 1] = stats.slope
      output[n * 3 + 2] = stats.roughness
      n++
    }
  }
  if (n !== count) throw new Error(`Window count mismatch for ${body}: ${n} != ${count}`)
  fs.writeFileSync(path.join(out, body + '.f64'), Buffer.from(output.buffer))
  metadata.bodies[body] = { windows: n, stored_windows: stored, label: info.label, source_url: info.url,
    tile_stream_sha256: digest.digest('hex'), dtype: 'little-endian float64', shape: [n, 3] }
  console.log(JSON.stringify({ body, windows: n, stored, output_bytes: output.byteLength }))
}
fs.writeFileSync(path.join(out, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n')
