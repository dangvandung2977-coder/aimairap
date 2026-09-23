'use strict'
// 2-bot PvP: two mineflayer bots fight each other, each driven by the trained
// PPO checkpoint through the SAME localhost protocol as the NeoForge mod
// (deployment/protocol.py, version 1). The Python side needs no changes.
//
//   npm install
//   node bot2v2.js [--host 127.0.0.1] [--port 25565] [--version 1.21.1]
//                   [--names BotA,BotB] [--bridge 127.0.0.1:25575]
//                   [--arena-size 64] [--dump-dir ./dumps]
//
// Each bot opens its OWN bridge connection per decision and builds obs from
// ITS OWN perspective (foe = the other bot). Decision every 4 physics ticks
// (= training frame_skip). Bridge down -> neutral controls, retry next time.
const net = require('node:net')
const fs = require('node:fs')
const mineflayer = require('mineflayer')
const E = require('./encoder.js')

const DECISION_EVERY = 4 // training frame_skip
const ATTEMPT_WINDOW = 6 // ticks to attribute a hit to a swing

function args (name, def) {
  const i = process.argv.indexOf('--' + name)
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def
}
const HOST = args('host', '127.0.0.1')
const PORT = parseInt(args('port', '25565'), 10)
const VERSION = args('version', '1.21.1')
const NAMES = args('names', 'PvPBot_A,PvPBot_B').split(',')
const [BHOST, BPORT] = args('bridge', '127.0.0.1:25575').split(':')
const ARENA_SIZE = parseFloat(args('arena-size', '64'))
const DUMP_DIR = args('dump-dir', null)
if (DUMP_DIR) fs.mkdirSync(DUMP_DIR, { recursive: true })

function queryBridge (tick, obs) {
  return new Promise((resolve) => {
    const s = net.connect(parseInt(BPORT, 10), BHOST)
    let buf = ''
    const done = (v) => { try { s.destroy() } catch {} resolve(v) }
    const to = setTimeout(() => done(null), 800)
    s.on('connect', () => s.write(JSON.stringify(
      { version: 1, tick, observation: obs, metadata: {} }) + '\n'))
    s.on('data', (d) => {
      buf += d
      if (!buf.endsWith('\n')) return
      clearTimeout(to)
      try {
        const m = JSON.parse(buf)
        done(m && Array.isArray(m.action) ? m : null)
      } catch { done(null) }
    })
    s.on('error', () => { clearTimeout(to); done(null) })
  })
}

// MC item names present in the kit (see README). Counts scan ALL slots so
// the offhand shield is counted as owned (mineflayer hides it from items()).
function countItems (bot) {
  const c = {}
  for (const s of bot.inventory.slots) {
    if (s) c[s.name] = (c[s.name] || 0) + s.count
  }
  return c
}
function findSlot (bot, names) {
  for (let i = 0; i < 9; i++) {
    const it = bot.inventory.slots[bot.inventory.hotbarStart + i]
    if (it && names.includes(it.name)) return i
  }
  return -1
}

class Fighter {
  constructor (bot, foeName, arena, shared) {
    this.bot = bot
    this.foeName = foeName
    this.arena = arena
    this.shared = shared // other Fighter (same process: exact foe block/eat/effect state)
    this.tick = 0
    this.inflight = false
    this.lastAction = null
    this.ctrl = { moveX: 0, moveZ: 0 }
    // tracked sim-counterparts (sim.py per-player state)
    this.damage = 0; this.foeDamage = 0; this.combo = 0; this.comboTimer = 0
    this.lastAttempt = -1e9; this.lastAttemptHit = 0; this.lastHit = -1e9
    this.atkCd = 0; this.hurt = 0
    this.pearlCd = 0; this.bowCd = 0
    this.eating = 0; this.drawing = false; this.blocking = false
    this.speedT = 0; this.strengthT = 0
    this.prevHp = 20; this.prevFoeHp = 20
    this.pendingHit = false
    this.foePrev = null
    this.foeVx = 0; this.foeVy = 0; this.foeVz = 0
    this.foeHist = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    this.foeAtkCd = 0; this.foeHurt = 0
    this.foeLastAttempt = -1e9; this.foeLastAttemptHit = 0; this.foePendingHit = false
    this.foeSwingTick = -1e9
    this.px = 0; this.py = 0; this.pz = 0
    this.vx = 0; this.vy = 0; this.vz = 0
    bot.on('physicsTick', () => this.onTick())
    bot.on('entityHurt', (e) => {
      if (e === bot.entity) {
        this.hurt = 10
        if (this.eating > 0) { // sim interrupts eating on damage (use.tick_eating)
          try { bot.deactivateItem() } catch {}
          this.eating = 0
        }
      } else if (this.foeEnt() && e === this.foeEnt()) {
        this.foeHurt = 10
      }
    })
    bot.on('entitySwingArm', (e) => {
      if (this.foeEnt() && e === this.foeEnt()) {
        this.foeLastAttempt = this.tick
        this.foeLastAttemptHit = 0
        this.foeSwingTick = this.tick
        this.foePendingHit = true
        this.foeAtkCd = 12
      }
    })
    bot.on('death', () => this.onDeath())
  }

  foeEnt () { return this.bot.players[this.foeName]?.entity || null }

  log (...a) { console.log(`[${this.bot.username} t=${this.tick}]`, ...a) }

  onDeath () {
    this.log(`died (hp tracking reset, re-equip on spawn)`)
    this.bot.clearControlStates()
    try { this.bot.deactivateItem() } catch {}
    this.damage = 0; this.combo = 0; this.comboTimer = 0
    this.eating = 0; this.drawing = false; this.blocking = false
    this.lastAction = null
  }

  async equipKit () {
    const inv = this.bot.inventory
    const sword = inv.items().find(i => /sword/.test(i.name))
    if (sword) { try { await this.bot.equip(sword, 'hand') } catch {} }
    const shield = inv.items().find(i => i.name === 'shield')
    if (shield) { try { await this.bot.equip(shield, 'off-hand') } catch {} }
    if (!sword) this.log('WARN no sword in inventory (see README kit)')
    if (!shield) this.log('WARN no shield in inventory (see README kit)')
  }

  mcDeg (rad) { return rad * 180 / Math.PI }

  snapshotSelf () {
    const b = this.bot, e = b.entity
    const counts = countItems(b)
    return {
      x: e.position.x, y: e.position.y, z: e.position.z,
      vx: this.vx, vy: this.vy, vz: this.vz,
      simYawDeg: E.simYawDeg(this.mcDeg(e.yaw)),
      simPitchDeg: -this.mcDeg(e.pitch),
      hp: Math.max(0, b.health), maxHp: 20,
      onGround: !!e.onGround, atkCd: this.atkCd, hurt: this.hurt,
      absorption: 0, // not exposed by mineflayer; documented limitation
      shield: this.blocking, eating: this.eating > 0,
      hunger: b.food, selected: b.quickBarSlot,
      counts, pearlCd: this.pearlCd, gappleCd: 0, bowCd: this.bowCd,
      eatLeft: this.eating, speedT: 0, strengthT: this.strengthT,
      damage: this.damage, combo: this.combo,
      lastAttempt: this.lastAttempt, lastAttemptHit: this.lastAttemptHit,
      lastHit: this.lastHit
    }
  }

  snapshotFoe (fe) {
    const s = this.shared // exact block/eat/effect state (same-process 2v2 only)
    return {
      x: fe.position.x, y: fe.position.y, z: fe.position.z,
      vx: this.foeVx, vy: this.foeVy, vz: this.foeVz,
      simYawDeg: E.simYawDeg(this.mcDeg(fe.yaw)),
      simPitchDeg: -this.mcDeg(fe.pitch),
      hp: Math.max(0, fe.health !== undefined ? fe.health : 20), maxHp: 20,
      onGround: !!fe.onGround, atkCd: this.foeAtkCd, hurt: this.foeHurt,
      absorption: 0, shield: s ? s.blocking : false, eating: s ? s.eating > 0 : false,
      hunger: 20, selected: 0, counts: {},
      pearlCd: 0, gappleCd: 0, bowCd: 0, eatLeft: 0,
      speedT: 0, strengthT: s ? s.strengthT : 0,
      damage: this.foeDamage, combo: 0,
      lastAttempt: this.foeLastAttempt, lastAttemptHit: this.foeLastAttemptHit,
      lastHit: -1e9, history: this.foeHist
    }
  }

  async onTick () {
    const b = this.bot
    if (!b.entity) return
    this.tick++
    const e = b.entity
    // velocities from deltas (blocks/sec), uniform, no unit risk
    this.vx = (e.position.x - this.px) * 20
    this.vy = (e.position.y - this.py) * 20
    this.vz = (e.position.z - this.pz) * 20
    this.px = e.position.x; this.py = e.position.y; this.pz = e.position.z
    for (const k of ['atkCd', 'hurt', 'pearlCd', 'bowCd', 'foeAtkCd', 'foeHurt', 'speedT', 'strengthT']) {
      if (this[k] > 0) this[k]--
    }
    if (this.eating > 0 && --this.eating === 0) { /* food consumed server-side */ }
    if (this.comboTimer > 0 && --this.comboTimer === 0) this.combo = 0
    const fe = this.foeEnt()
    if (fe) {
      // foe velocity + 4-entry history (sim prev_pos)
      if (this.foePrev) {
        this.foeVx = (fe.position.x - this.foePrev.x) * 20
        this.foeVy = (fe.position.y - this.foePrev.y) * 20
        this.foeVz = (fe.position.z - this.foePrev.z) * 20
      } else { this.foeVx = this.foeVy = this.foeVz = 0 }
      this.foePrev = fe.position.clone()
      this.foeHist.push([fe.position.x, fe.position.y, fe.position.z])
      if (this.foeHist.length > 4) this.foeHist.shift()
      // health-delta damage attribution (exact observed damage, all sources)
      const fhp = Math.max(0, fe.health !== undefined ? fe.health : 20)
      if (fhp < this.prevFoeHp) {
        const d = this.prevFoeHp - fhp
        this.damage += d
        this.combo = this.comboTimer > 0 ? this.combo + 1 : 1
        this.comboTimer = 60
        this.lastHit = this.tick
        if (this.pendingHit) { this.lastAttemptHit = 1; this.pendingHit = false }
      }
      this.prevFoeHp = fhp
      const mhp = Math.max(0, b.health)
      if (mhp < this.prevHp) {
        this.foeDamage = (this.foeDamage || 0) + (this.prevHp - mhp)
        if (this.shared) { this.shared.combo = 0; this.shared.comboTimer = 0 }
        if (this.foePendingHit && this.tick - this.foeSwingTick <= ATTEMPT_WINDOW) {
          this.foeLastAttemptHit = 1; this.foePendingHit = false
        }
      }
      this.prevHp = mhp
    }
    if (this.pendingHit && this.tick - this.lastAttempt > ATTEMPT_WINDOW) {
      this.pendingHit = false // whiff confirmed
    }
    if (this.tick % DECISION_EVERY === 0 && fe && !this.inflight) {
      this.inflight = true
      try { await this.decide(fe) } finally { this.inflight = false }
    }
    if (this.tick % 40 === 0 && fe) {
      this.log(`hp=${b.health} foe=${Math.round(fe.health || 0)} ` +
        `d=${b.entity.position.distanceTo(fe.position).toFixed(1)} ` +
        `act=${this.lastAction ? E.describe(this.lastAction) : 'none'}`)
    }
  }

  async decide (fe) {
    const b = this.bot
    const fx = fe.position.x - b.entity.position.x
    const fz = fe.position.z - b.entity.position.z
    const dist = Math.hypot(fx, fz)
    let proj = null; let nproj = 0
    for (const id of Object.keys(b.entities)) {
      const p = b.entities[id]
      if (!p || !p.isValid || (p.name !== 'arrow' && p.name !== 'ender_pearl')) continue
      nproj++
      const dx = p.position.x - b.entity.position.x
      const dy = p.position.y - b.entity.position.y
      const dz = p.position.z - b.entity.position.z
      if (!proj || dx * dx + dy * dy + dz * dz < proj.d2) proj = { d2: dx * dx + dy * dy + dz * dz, v: [dx, dy, dz] }
    }
    const obs = E.buildObs({
      me: this.snapshotSelf(), foe: this.snapshotFoe(fe), tick: this.tick,
      proj: proj ? proj.v : null, nproj,
      arena: this.arena, pillars: []
    })
    if (DUMP_DIR) {
      fs.writeFileSync(`${DUMP_DIR}/obs_${b.username}_${this.tick}.json`,
        JSON.stringify({ tick: this.tick, observation: obs }))
    }
    const t0 = Date.now()
    const resp = await queryBridge(this.tick, obs)
    if (!resp) { this.log('bridge unavailable, holding course'); return } // safe fallback
    const ctrl = E.decodeAction(resp.action)
    if (!ctrl) return // illegal: hold course, never crash
    this.lastAction = resp.action
    if (resp.inference_ms !== undefined && this.tick % 200 === 0) {
      this.log(`infer=${resp.inference_ms}ms rtt=${Date.now() - t0}ms dist=${dist.toFixed(1)}`)
    }
    this.apply(ctrl, fe)
  }

  apply (ctrl, fe) {
    const b = this.bot
    // aim ONCE per decision (trained dynamics, frame_skip=4)
    const yaw = b.entity.yaw + ctrl.yawDelta * Math.PI / 180
    const pitch = b.entity.pitch - ctrl.pitchDelta * Math.PI / 180 // sim pitch+ = up
    try { b.look(yaw, pitch) } catch {}
    // hotbar select (slot 1 = shield lives in offhand; keep current)
    if (ctrl.slot !== 1) { try { b.setQuickBarSlot(ctrl.slot) } catch {} }
    // movement (MC forward is camera-relative, like sim yaw-space wish)
    b.setControlState('forward', ctrl.moveZ > 0)
    b.setControlState('back', ctrl.moveZ < 0)
    b.setControlState('left', ctrl.moveX < 0)
    b.setControlState('right', ctrl.moveX > 0)
    b.setControlState('jump', ctrl.jump)
    b.setControlState('sprint', ctrl.sprint)
    // attack takes precedence over block (sim.py)
    if (ctrl.attack) {
      if (this.blocking) { try { b.deactivateItem() } catch {}; this.blocking = false }
      try { b.attack(fe) } catch {}
      this.lastAttempt = this.tick
      this.lastAttemptHit = 0
      this.pendingHit = true
      this.atkCd = 12
    } else if (ctrl.block) {
      if (!this.blocking) {
        try { b.activateItem(true) } catch {} // off-hand shield
        this.blocking = true
      }
    } else if (this.blocking) {
      try { b.deactivateItem() } catch {}
      this.blocking = false
    }
    // item use through the real item pipeline
    if (ctrl.use) this.onUse(fe)
    else if (this.drawing) {
      try { b.deactivateItem() } catch {} // release bow shot
      this.drawing = false
      this.bowCd = 15
    }
  }

  onUse (fe) {
    const b = this.bot
    const held = b.heldItem
    if (!held) return
    if (held.name === 'ender_pearl') {
      if (this.pearlCd <= 0) {
        try { b.activateItem(false) } catch {}
        this.pearlCd = 20
      }
    } else if (held.name === 'bow') {
      const arrows = countItems(b).arrow || 0
      if (arrows > 0 && this.bowCd <= 0 && !this.drawing) {
        try { b.activateItem(false) } catch {}
        this.drawing = true
      }
    } else if (['bread', 'golden_apple'].includes(held.name)) {
      if (this.eating <= 0) {
        try { b.activateItem(false) } catch {}
        this.eating = 32 // sim use_time_ticks; interrupted on damage
      }
    }
    // cobweb/block/strength_potion: selected only (later phase, like the mod)
  }
}

function launch (username, foeName, arena) {
  const bot = mineflayer.createBot({
    host: HOST, port: PORT, username, version: VERSION
  })
  let fighter = null
  bot.once('spawn', async () => {
    console.log(`[${username}] spawned at`, bot.entity.position.toString())
    if (!arena.center) {
      arena.center = { x: Math.round(bot.entity.position.x), z: Math.round(bot.entity.position.z) }
      arena.lo = arena.center.x - ARENA_SIZE / 2 + 1
      arena.hi = arena.center.x + ARENA_SIZE / 2 - 1
      arena.size = ARENA_SIZE
      console.log(`[${username}] arena center`, arena.center, `size ${ARENA_SIZE}`)
    }
    fighter = new Fighter(bot, foeName, arena, null)
    fighters[username] = fighter
    // link shared foe state (same-process 2v2 observability shortcut)
    for (const n of Object.keys(fighters)) {
      for (const m of Object.keys(fighters)) {
        if (n !== m) fighters[n].shared = fighters[m]
      }
    }
    await fighter.equipKit()
    fighter.px = bot.entity.position.x
    fighter.py = bot.entity.position.y
    fighter.pz = bot.entity.position.z
    fighter.prevHp = bot.health
    const fe = bot.players[foeName]?.entity
    fighter.prevFoeHp = fe && fe.health !== undefined ? fe.health : 20
  })
  bot.on('kicked', (r) => { console.log(`[${username}] kicked:`, r); process.exit(1) })
  bot.on('end', (r) => { console.log(`[${username}] connection ended:`, r); process.exit(1) })
  bot.on('error', (e) => console.log(`[${username}] error:`, e.message))
  return bot
}

const fighters = {}
const arena = { size: ARENA_SIZE, lo: -ARENA_SIZE / 2 + 1, hi: ARENA_SIZE / 2 - 1, center: null }
process.on('unhandledRejection', (e) => console.log('[warn] unhandled rejection:', e && e.message))
process.on('SIGINT', () => { console.log('quitting'); process.exit(0) })
console.log(`connecting ${NAMES[0]} vs ${NAMES[1]} to ${HOST}:${PORT} (${VERSION}), bridge ${BHOST}:${BPORT}`)
launch(NAMES[0], NAMES[1], arena)
setTimeout(() => launch(NAMES[1], NAMES[0], arena), 3000)
