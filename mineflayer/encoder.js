'use strict'
// EXACT 73-dim observation v4 encoder + 9-slot action decoder, pure JS.
// Single source of truth for semantics: rl/environment/obs_schema.py and
// rl/environment/action_map.py. Index table: SELF 0-12, ENEMY 13-25,
// COMBAT 26-30, WORLD 31-38, V2 39-43, V3 44-46, V4 47-72.
//
// Coordinate frame (must match the sim): simYaw = mcYaw + 90 (deg),
// simPitch = -mcPitch (deg). Deltas transfer 1:1.
const OBS_DIM = 73
const NVEC = [5, 2, 2, 2, 2, 2, 9, 3, 3]
const YAW_STEP = 15.0 // action_map.YAW_STEP
const PITCH_STEP = 10.0 // action_map.PITCH_STEP
// HOTBAR order = sandbox/items/defs.py:76
const HOTBAR = ['sword', 'shield', 'bread', 'golden_apple', 'ender_pearl',
  'cobweb', 'block', 'strength_potion', 'bow']

function wrapYaw (deg) {
  while (deg > 180) deg -= 360
  while (deg <= -180) deg += 360
  return deg
}
function simYawDeg (mcYawDeg) { return wrapYaw(mcYawDeg + 90) } // engine.look_dir
function clip (x) { return Math.max(-5, Math.min(5, x)) } // obs_schema._clip

// counts: plain {itemName: n} using MINECRAFT item names.
function hotbarCounts (counts) {
  const has = (...names) => names.reduce((s, n) => s + (counts[n] || 0), 0)
  return [
    has('stone_sword', 'iron_sword', 'diamond_sword'), // sword
    has('shield'), // shield
    has('bread'), // bread
    has('golden_apple'), // golden_apple
    has('ender_pearl'), // ender_pearl
    has('cobweb'), // cobweb
    has('dirt', 'cobblestone'), // block
    has('splash_potion'), // strength_potion
    has('bow') > 0 ? 1 : 0 // bow: owned flag (schema counts the tool, not arrows)
  ]
}

// ctx: { me, foe, tick, proj: [dx,dy,dz]|null, nproj, arena: {size, lo, hi}, pillars: [[x0,z0,x1,z1]] }
// me/foe: { x,y,z, vx,vy,vz (blocks/sec), simYawDeg, simPitchDeg, hp, maxHp,
//   onGround, atkCd, hurt, absorption, shield, eating, hunger, selected,
//   counts{}, pearlCd, gappleCd, bowCd, eatLeft, speedT, strengthT,
//   damage, combo, lastAttempt, lastHit, history: [[x,y,z]x4, oldest first] }
function buildObs (ctx) {
  const { me, foe, tick, arena } = ctx
  const s = arena.size
  const o = new Array(OBS_DIM).fill(0)
  const sy = me.simYawDeg * Math.PI / 180
  // SELF 0-12
  o[0] = clip(me.x / s); o[1] = clip(me.y / 10); o[2] = clip(me.z / s)
  o[3] = clip(me.vx / 10); o[4] = clip(me.vy / 10); o[5] = clip(me.vz / 10)
  o[6] = Math.sin(sy); o[7] = Math.cos(sy)
  o[8] = clip(me.simPitchDeg / 90)
  o[9] = clip(me.hp / me.maxHp)
  o[10] = me.onGround ? 1 : 0
  o[11] = clip(me.atkCd / 12); o[12] = clip(me.hurt / 10)
  // ENEMY 13-25
  const dx = foe.x - me.x, dy = foe.y - me.y, dz = foe.z - me.z
  o[13] = clip(dx / s); o[14] = clip(dy / 10); o[15] = clip(dz / s)
  o[16] = clip((foe.vx - me.vx) / 10)
  o[17] = clip((foe.vy - me.vy) / 10)
  o[18] = clip((foe.vz - me.vz) / 10)
  const dyaw = (foe.simYawDeg - me.simYawDeg) * Math.PI / 180
  o[19] = Math.sin(dyaw); o[20] = Math.cos(dyaw)
  o[21] = clip(foe.simPitchDeg / 90)
  o[22] = clip(foe.hp / foe.maxHp)
  o[23] = foe.onGround ? 1 : 0
  o[24] = clip(foe.atkCd / 12); o[25] = clip(foe.hurt / 10)
  // COMBAT 26-30
  o[26] = clip(Math.sqrt(dx * dx + dy * dy + dz * dz) / s)
  o[27] = clip(me.damage / 20); o[28] = clip(foe.damage / 20)
  o[29] = clip(me.combo / 5)
  o[30] = clip(Math.min(tick - me.lastHit, 100) / 20)
  // WORLD 31-38
  o[31] = clip((me.x - arena.lo) / s); o[32] = clip((arena.hi - me.x) / s)
  o[33] = clip((me.z - arena.lo) / s); o[34] = clip((arena.hi - me.z) / s)
  const cell = sensor(me.x, me.z, arena, ctx.pillars || [])
  o[35] = cell[0]; o[36] = cell[1]; o[37] = cell[2]; o[38] = cell[3]
  // V2 39-43
  const bearing = Math.atan2(dz, dx) - me.simYawDeg * Math.PI / 180
  o[39] = Math.sin(bearing); o[40] = Math.cos(bearing)
  o[41] = clip(Math.sqrt(foe.vx * foe.vx + foe.vz * foe.vz) / 10)
  const old = foe.history[0]
  o[42] = clip((foe.x - old[0]) / s); o[43] = clip((foe.z - old[2]) / s)
  // V3 44-46
  o[44] = clip(Math.min(tick - me.lastAttempt, 100) / 20)
  o[45] = clip(Math.min(tick - foe.lastAttempt, 100) / 20)
  o[46] = me.lastAttemptHit ? 1 : 0
  // V4 47-72
  o[47] = clip(me.selected / 8)
  hotbarCounts(me.counts).forEach((c, i) => { o[48 + i] = clip(c / 16) })
  o[57] = clip(me.pearlCd / 20)
  o[58] = clip(me.gappleCd / 100)
  o[59] = clip(me.bowCd / 15)
  o[60] = clip(me.absorption / 8)
  o[61] = me.shield ? 1 : 0
  o[62] = clip(me.eatLeft / 32)
  o[63] = clip(me.hunger / 20)
  o[64] = clip(foe.absorption / 8)
  o[65] = foe.shield ? 1 : 0
  o[66] = foe.eating ? 1 : 0
  o[67] = clip(ctx.nproj / 4)
  if (ctx.proj) {
    o[68] = clip(ctx.proj[0] / 20)
    o[69] = clip(ctx.proj[1] / 10)
    o[70] = clip(ctx.proj[2] / 20)
  }
  o[71] = clip(me.speedT / 1200)
  o[72] = clip(me.strengthT / 1200)
  return o
}

function sensor (x, z, arena, pillars) {
  const out = [0, 0, 0, 0]
  const dirs = [[2, 0], [-2, 0], [0, 2], [0, -2]]
  for (let i = 0; i < 4; i++) {
    for (const t of [0.75, 1.5, 2.0]) {
      const sx = x + dirs[i][0] * (t / 2), sz = z + dirs[i][1] * (t / 2)
      if (sx >= arena.lo && sx <= arena.hi && sz >= arena.lo && sz <= arena.hi &&
          pillars.some(p => p[0] <= sx && sx <= p[2] && p[1] <= sz && sz <= p[3])) {
        out[i] = 1
        break
      }
    }
  }
  return out
}

// Returns controller or null when illegal (caller holds course).
// action_map.to_controller: move 0 stop/1 fwd/2 back/3 left/4 right;
// sprint only applies with moveZ>0 (engine.py); attack beats block (sim.py);
// yaw/pitch deltas applied ONCE per decision (frame_skip=4).
function decodeAction (a) {
  if (!Array.isArray(a) || a.length !== NVEC.length) return null
  for (let i = 0; i < NVEC.length; i++) {
    if (!Number.isInteger(a[i]) || a[i] < 0 || a[i] >= NVEC[i]) return null
  }
  const c = { moveX: 0, moveZ: 0, yawDelta: (a[7] - 1) * YAW_STEP, pitchDelta: (a[8] - 1) * PITCH_STEP }
  if (a[0] === 1) c.moveZ = 1
  else if (a[0] === 2) c.moveZ = -1
  else if (a[0] === 3) c.moveX = -1
  else if (a[0] === 4) c.moveX = 1
  c.sprint = a[1] === 1 && c.moveZ > 0
  c.jump = a[2] === 1
  c.attack = a[3] === 1
  c.block = a[4] === 1 && !c.attack
  c.use = a[5] === 1 && !c.attack
  c.slot = a[6]
  return c
}

function describe (a) {
  const names = ['stop', 'fwd', 'back', 'left', 'right']
  const yaw = ['L', '-', 'R'][a[7]]
  return names[a[0]] + (a[1] ? 'S' : '') + (a[2] ? 'J' : '') + (a[3] ? 'X' : '') +
    (a[4] ? 'B' : '') + (a[5] ? 'U' + a[6] : '') + yaw
}

module.exports = {
  OBS_DIM, NVEC, HOTBAR, wrapYaw, simYawDeg, clip,
  hotbarCounts, buildObs, sensor, decodeAction, describe
}
