'use strict'
// node test_encoder.js — pure-function tests, no Minecraft needed.
const assert = require('node:assert/strict')
const E = require('./encoder.js')

function frame (over = {}) {
  return {
    x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
    simYawDeg: 0, simPitchDeg: 0, hp: 20, maxHp: 20,
    onGround: true, atkCd: 0, hurt: 0, absorption: 0,
    shield: false, eating: false, hunger: 20, selected: 0,
    counts: {}, pearlCd: 0, gappleCd: 0, bowCd: 0,
    eatLeft: 0, speedT: 0, strengthT: 0, damage: 0, combo: 0,
    lastAttempt: -1e9, lastHit: -1e9, lastAttemptHit: 0,
    history: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...over
  }
}
const arena = { size: 32, lo: 1, hi: 31 }

// 1. shape + finiteness
{
  const o = E.buildObs({ me: frame(), foe: frame(), tick: 100, proj: null, nproj: 0, arena, pillars: [] })
  assert.equal(o.length, 73)
  assert.ok(o.every(Number.isFinite))
}

// 2. spot checks against hand-computed schema values
{
  const me = frame({ x: 8, z: 16, hp: 15 })
  const foe = frame({ x: 24, z: 16, hp: 10 })
  const o = E.buildObs({ me, foe, tick: 50, proj: null, nproj: 0, arena, pillars: [] })
  assert.equal(o[0], 8 / 32) // self x / size
  assert.equal(o[9], 15 / 20) // self hp
  assert.equal(o[13], 16 / 32) // foe dx / size
  assert.equal(o[22], 10 / 20) // foe hp
  assert.equal(o[26], 16 / 32) // dist / size
  assert.equal(o[31], (8 - 1) / 32) // wall x-lo
  assert.equal(o[39], 0) // bearing: foe straight ahead (simYaw 0 = +X)
  assert.equal(o[40], 1)
  assert.equal(o[44], 5) // min(tick-attempt,100)/20, attempt at -1e9
  assert.equal(o[63], 1) // hunger 20/20
}

// 3. yaw frame: MC yaw -90 (facing +X) -> sim yaw 0
assert.equal(E.simYawDeg(-90), 0)
assert.equal(E.wrapYaw(190), -170)

// 4. hotbar counts follow defs.py:76 order
{
  const c = E.hotbarCounts({ iron_sword: 1, shield: 1, bread: 5, golden_apple: 1, ender_pearl: 4, bow: 1, arrow: 16 })
  assert.deepEqual(c, [1, 1, 5, 1, 4, 0, 0, 0, 1])
}

// 5. decoder: valid + illegal
{
  const c = E.decodeAction([1, 1, 0, 1, 0, 0, 0, 1, 1])
  assert.deepEqual([c.moveZ, c.sprint, c.attack, c.slot], [1, true, true, 0])
  // sprint gated by forward motion (engine.py); attack beats block/use
  assert.equal(E.decodeAction([0, 1, 0, 0, 0, 0, 0, 1, 1]).sprint, false)
  const b = E.decodeAction([1, 0, 0, 1, 1, 1, 0, 1, 1])
  assert.equal(b.block, false)
  assert.equal(b.use, false)
  assert.equal(E.decodeAction([1, 0, 0, 0, 0, 0, 0, 1, 1]).yawDelta, 0)
  assert.equal(E.decodeAction([1, 0, 0, 0, 0, 0, 0, 0, 2]).yawDelta, -15)
  assert.equal(E.decodeAction([1, 0, 0, 0, 0, 0, 0, 1, 2]).pitchDelta, 10)
  assert.equal(E.decodeAction([5, 0, 0, 0, 0, 0, 0, 1, 1]), null)
  assert.equal(E.decodeAction([0, 0, 0, 0, 0, 0, 9, 1, 1]), null)
  assert.equal(E.decodeAction([1, 0]), null)
}

console.log('encoder tests: ALL PASS')
