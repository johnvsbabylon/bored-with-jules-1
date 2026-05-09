#!/usr/bin/env python3
"""
brain.py — Neurons only. Does a brain form?

Seed neurons grow fractal processes (the same entangled Julia iteration
from the Gateway Fractal), form synapses, and develop Hebbian plasticity.

Question: given only FitzHugh-Nagumo dynamics + fractal growth +
Hebbian learning, does a brain-like network self-organize?

Signs of a brain forming:
  - Traveling waves of activation
  - Synchronized oscillations (brain rhythms)
  - Stable firing circuits (the same neurons firing in sequence)
  - Functional clusters (regions that fire together)
  - Strong hubs (neurons with many connections)
"""

import numpy as np
import pygame
import sys
import os
from scipy.ndimage import gaussian_filter
from datetime import datetime

SIZE   = 896
SCALE  = 1
HUD_W  = 220
FPS    = 60
DT     = 1.0

NEURON = 1
EMPTY  = 0

# FitzHugh-Nagumo (Hodgkin-Huxley reduced)
FHN_A   = 0.7
FHN_B   = 0.8
FHN_TAU = 12.5

# Synaptic weights (Hebbian plasticity)
SYN_INIT   = 0.08   # weight at birth of new connection
SYN_MAX    = 1.0
SYN_LTP    = 0.018  # long-term potentiation: fire together → stronger
SYN_LTD    = 0.004  # long-term depression: passive decay
SYN_THRESH = 0.25   # threshold for "established" connection

# Energy (neurons sustained by uniform nutrient supply — like a brain's blood supply)
NUTRIENT   = 0.65
E_GAIN     = 0.016
E_DECAY    = 0.0025
E_FIRE     = 0.008   # metabolic cost of firing
DIV_THRESH = 0.72
DEATH_THR  = -0.18

# Noise and spontaneous activity
NOISE_AMP   = 0.04
SPONT_RATE  = 0.0008  # random spike probability per neuron per step

# Fractal growth
COUPLING      = 0.15   # entanglement α (same as Gateway Fractal)
GROW_INTERVAL = 6      # steps between growth attempts
GROW_N        = 120    # neurons attempting to grow per interval


class Brain:
    def __init__(self):
        S = SIZE
        self.t = 0

        self.cell_type  = np.zeros((S, S), dtype=np.int8)
        self.energy     = np.zeros((S, S), dtype=np.float32)

        # FitzHugh-Nagumo state
        self.v          = np.zeros((S, S), dtype=np.float32)  # membrane potential
        self.w          = np.zeros((S, S), dtype=np.float32)  # recovery variable

        # Firing
        self.firing     = np.zeros((S, S), dtype=bool)
        self.fire_count = np.zeros((S, S), dtype=np.int32)    # total fire events

        # Synaptic weights
        # syn_h[i,j]: connection strength between (i,j) and (i,j+1)
        # syn_v[i,j]: connection strength between (i,j) and (i+1,j)
        self.syn_h = np.zeros((S, S), dtype=np.float32)
        self.syn_v = np.zeros((S, S), dtype=np.float32)

        # Stats history
        self.history = {k: [] for k in
                        ('neuron','firing','synchrony','connectivity','mean_weight')}

        # Synchrony detector: track firing fraction over last 100 steps
        self.fire_buffer = []

    def seed(self, n=24):
        rng = np.random.default_rng(42)
        xs  = rng.integers(30, SIZE-30, n)
        ys  = rng.integers(30, SIZE-30, n)
        for x, y in zip(xs, ys):
            self.cell_type[x, y] = NEURON
            self.energy[x, y]    = 0.45
            self.v[x, y]         = rng.uniform(-0.5, 0.5)
            self.w[x, y]         = rng.uniform(-0.2, 0.2)

    def _lap(self, Z):
        return (np.roll(Z,1,0) + np.roll(Z,-1,0) +
                np.roll(Z,1,1) + np.roll(Z,-1,1) - 4*Z)

    def step(self):
        self.t += 1
        nm = self.cell_type == NEURON
        if not nm.any():
            return

        # ── Synaptic input ─────────────────────────────────────────────────
        # Each neuron receives the weighted membrane potential of its neighbors.
        # This propagates excitation through the network via established weights.
        I_syn = (np.roll(self.v, -1, 1) * self.syn_h                   +  # right
                 np.roll(self.v,  1, 1) * np.roll(self.syn_h, 1, 1)    +  # left
                 np.roll(self.v, -1, 0) * self.syn_v                   +  # down
                 np.roll(self.v,  1, 0) * np.roll(self.syn_v, 1, 0))      # up

        # Noise
        I_noise = np.random.normal(0, NOISE_AMP, (SIZE, SIZE)).astype(np.float32)

        # Spontaneous spikes
        spont = (np.random.random((SIZE, SIZE)) < SPONT_RATE) & nm
        self.v[spont] += 1.2

        I_total = 0.28 + I_syn * 0.45 + I_noise

        # ── FitzHugh-Nagumo ────────────────────────────────────────────────
        dv = self.v - (self.v**3) / 3.0 - self.w + I_total
        dw = (self.v + FHN_A - FHN_B * self.w) / FHN_TAU

        self.v += (DT / 10.0) * dv * nm
        self.w += (DT / 10.0) * dw * nm

        # ── Firing ─────────────────────────────────────────────────────────
        self.firing      = nm & (self.v > 1.5)
        self.fire_count += self.firing

        # ── Hebbian plasticity ─────────────────────────────────────────────
        # Neurons that fire together wire together (LTP).
        # All connections slowly decay (LTD) unless reinforced.
        fire_r = np.roll(self.firing, -1, 1)
        fire_d = np.roll(self.firing, -1, 0)
        nm_r   = np.roll(nm, -1, 1)
        nm_d   = np.roll(nm, -1, 0)

        # LTP: both cells firing simultaneously
        self.syn_h += SYN_LTP * (self.firing & fire_r & nm & nm_r)
        self.syn_v += SYN_LTP * (self.firing & fire_d & nm & nm_d)

        # LTD: passive decay of all existing connections
        self.syn_h -= SYN_LTD * (self.syn_h > 0)
        self.syn_v -= SYN_LTD * (self.syn_v > 0)

        np.clip(self.syn_h, 0, SYN_MAX, out=self.syn_h)
        np.clip(self.syn_v, 0, SYN_MAX, out=self.syn_v)

        # ── Energy ─────────────────────────────────────────────────────────
        self.energy += E_GAIN * NUTRIENT * nm
        self.energy -= E_DECAY * nm
        self.energy -= E_FIRE  * self.firing

        # ── Division ───────────────────────────────────────────────────────
        self._divide()

        # ── Death ──────────────────────────────────────────────────────────
        self._die()

        # ── Fractal growth ─────────────────────────────────────────────────
        if self.t % GROW_INTERVAL == 0:
            self._grow()

        # ── Stats ──────────────────────────────────────────────────────────
        if self.t % 15 == 0:
            self._record()

    def _divide(self):
        cands = np.argwhere((self.energy > DIV_THRESH) & (self.cell_type == NEURON))
        if not len(cands):
            return
        np.random.shuffle(cands)
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for x, y in cands[:50]:
            order = dirs.copy()
            np.random.shuffle(order)
            for dx, dy in order:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                    continue
                if self.cell_type[nx, ny] != EMPTY:
                    continue
                self.cell_type[nx, ny] = NEURON
                self.energy[nx, ny]    = self.energy[x, y] * 0.5
                self.energy[x, y]     *= 0.5
                self.v[nx, ny]         = self.v[x, y] * 0.25
                # Parent-child synapse
                if   dx==0 and dy== 1: self.syn_h[x,  y]  = SYN_INIT
                elif dx==0 and dy==-1: self.syn_h[nx, ny] = SYN_INIT
                elif dx== 1 and dy==0: self.syn_v[x,  y]  = SYN_INIT
                elif dx==-1 and dy==0: self.syn_v[nx, ny] = SYN_INIT
                break

    def _die(self):
        dead = (self.cell_type == NEURON) & (self.energy < DEATH_THR)
        if not dead.any():
            return
        self.cell_type[dead]  = EMPTY
        self.energy[dead]     = 0.0
        self.v[dead]          = 0.0
        self.w[dead]          = 0.0
        self.fire_count[dead] = 0

    def _grow(self):
        # Entangled Julia iteration as growth rule.
        # c is determined by local neuron density gradient:
        #   isolated neurons → grow toward other neurons (social gradient)
        #   dense neurons    → fractal branching (explore and connect)
        npos = np.argwhere(self.cell_type == NEURON)
        if not len(npos):
            return

        density = gaussian_filter(
            (self.cell_type == NEURON).astype(np.float32), sigma=6)

        np.random.shuffle(npos)
        for x, y in npos[:GROW_N]:
            if self.energy[x, y] < 0.25:
                continue

            # Map to complex plane
            zr = (x / SIZE - 0.5) * 4.0
            zi = (y / SIZE - 0.5) * 4.0
            wr = -zi
            wi =  zr

            local_d = float(density[x, y])

            if local_d < 0.04:
                # Isolated — follow density gradient toward other neurons
                gx = (density[min(x+2,SIZE-1), y] - density[max(x-2,0), y]) * 0.5
                gy = (density[x, min(y+2,SIZE-1)] - density[x, max(y-2,0)]) * 0.5
                cr = 0.5 * gx
                ci = 0.5 * gy
            else:
                # Near others — fractal branching, guided by membrane potential
                cr = 0.30 * (local_d - 0.08)
                ci = 0.20 * float(self.v[x, y])

            # Entangled Julia iteration (Gateway Fractal equation)
            zr_n = zr*zr - zi*zi + cr
            zi_n = 2.0*zr*zi + ci
            wr_n = wr*wr - wi*wi + cr
            wi_n = 2.0*wr*wi + ci

            # Coupling α
            zr_c = zr_n + COUPLING * (wr_n - zr_n)
            zi_c = zi_n + COUPLING * (wi_n - zi_n)

            mag = np.sqrt(zr_c**2 + zi_c**2)
            if mag < 0.1 or mag > 2.5:
                continue

            angle = np.arctan2(zi_c, zr_c)
            dx    = int(np.round(np.cos(angle)))
            dy    = int(np.round(np.sin(angle)))
            if dx == 0 and dy == 0:
                continue

            nx, ny = x+dx, y+dy
            if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                continue

            if self.cell_type[nx, ny] == EMPTY:
                self.cell_type[nx, ny] = NEURON
                self.energy[nx, ny]    = self.energy[x, y] * 0.55
                self.energy[x, y]     *= 0.65
                self.v[nx, ny]         = self.v[x, y] * 0.15
                if   dx==0 and dy== 1: self.syn_h[x,  y]  = SYN_INIT
                elif dx==0 and dy==-1: self.syn_h[nx, ny] = SYN_INIT
                elif dx== 1 and dy==0: self.syn_v[x,  y]  = SYN_INIT
                elif dx==-1 and dy==0: self.syn_v[nx, ny] = SYN_INIT

    def _record(self):
        nm = self.cell_type == NEURON
        n  = int(nm.sum())
        if n == 0:
            return
        h  = self.history
        fr = int(self.firing.sum())
        h['neuron'].append(n)
        h['firing'].append(fr)
        h['synchrony'].append(float(fr) / max(n, 1))
        c_h = int((self.syn_h > SYN_THRESH).sum())
        c_v = int((self.syn_v > SYN_THRESH).sum())
        h['connectivity'].append(c_h + c_v)
        syn_active = self.syn_h[self.syn_h > 0]
        h['mean_weight'].append(float(syn_active.mean()) if len(syn_active) else 0.0)

        self.fire_buffer.append(float(fr) / max(n, 1))
        if len(self.fire_buffer) > 60:
            self.fire_buffer.pop(0)

    def synchrony_score(self):
        if len(self.fire_buffer) < 10:
            return 0.0
        arr = np.array(self.fire_buffer)
        return float(arr.std())

    def counts(self):
        nm = self.cell_type == NEURON
        n  = int(nm.sum())
        fr = int(self.firing.sum())
        return {
            'neuron':    n,
            'firing':    fr,
            'frac':      float(fr) / max(n, 1),
            'conn':      int((self.syn_h > SYN_THRESH).sum() +
                             (self.syn_v > SYN_THRESH).sum()),
            'mean_w':    float(self.syn_h[self.syn_h > 0].mean())
                         if (self.syn_h > 0).any() else 0.0,
            'synchrony': self.synchrony_score(),
        }


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, brain, save_frames=False):
        self.brain       = brain
        self.save_frames = save_frames
        self.frame_dir   = None
        self.frame_n     = 0

        if save_frames:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.frame_dir = os.path.join(os.path.dirname(__file__), f'brain_frames_{ts}')
            os.makedirs(self.frame_dir, exist_ok=True)

        self.win_w  = SIZE * SCALE + HUD_W
        self.win_h  = SIZE * SCALE
        self.view   = 'cells'   # 'cells' 'weights' 'activity' 'potential'
        self.paused = False
        self.speed  = 1

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Brain Genesis")
        self.font_sm = pygame.font.SysFont('monospace', 11)
        self.font_md = pygame.font.SysFont('monospace', 13, bold=True)
        self.clock   = pygame.time.Clock()
        self.graph_w = HUD_W - 16
        self.graph_h = 70

    def render(self):
        b = self.brain
        S = SIZE

        pixels = np.zeros((S, S, 3), dtype=np.uint8)
        nm     = b.cell_type == NEURON

        if self.view == 'cells':
            # Membrane potential → brightness (cyan)
            # w_rec (recovery/refractory) → red tint
            v_norm  = np.clip((b.v + 2.0) / 4.0, 0, 1)   # [-2,2] → [0,1]
            w_norm  = np.clip(b.w / 2.0, 0, 1)
            pixels[:,:,0] = np.clip((w_norm * 80)  * nm, 0, 255).astype(np.uint8)
            pixels[:,:,1] = np.clip((v_norm * 200) * nm, 0, 255).astype(np.uint8)
            pixels[:,:,2] = np.clip((v_norm * 255 + 30) * nm, 0, 255).astype(np.uint8)
            # Highly active neurons glow brighter (fire_count)
            hub = np.clip(b.fire_count / max(b.fire_count.max(), 1), 0, 1).astype(np.float32)
            pixels[:,:,1] = np.clip(pixels[:,:,1].astype(np.float32) + hub * nm * 80, 0, 255).astype(np.uint8)

        elif self.view == 'weights':
            # Synaptic weight strength per cell
            total_w  = (b.syn_h + np.roll(b.syn_h, 1, 1) +
                        b.syn_v + np.roll(b.syn_v, 1, 0))
            w_norm   = np.clip(total_w / 2.0, 0, 1)
            pixels[:,:,0] = np.clip(w_norm * nm * 60,  0, 255).astype(np.uint8)
            pixels[:,:,1] = np.clip(w_norm * nm * 255, 0, 255).astype(np.uint8)
            pixels[:,:,2] = np.clip(w_norm * nm * 180, 0, 255).astype(np.uint8)

        elif self.view == 'activity':
            # Total fire count (heatmap) — reveals which neurons are "hubs"
            fc_norm  = np.clip(b.fire_count / max(b.fire_count.max(), 1), 0, 1)
            pixels[:,:,0] = np.clip(fc_norm * nm * 255, 0, 255).astype(np.uint8)
            pixels[:,:,1] = np.clip(fc_norm * nm * 120, 0, 255).astype(np.uint8)
            pixels[:,:,2] = np.clip(fc_norm * nm * 30,  0, 255).astype(np.uint8)

        elif self.view == 'potential':
            # Raw membrane potential — blue=hyperpolarized, red=depolarized
            v = b.v
            pos = np.clip( v / 2.0, 0, 1) * nm
            neg = np.clip(-v / 2.0, 0, 1) * nm
            pixels[:,:,0] = (pos * 255).astype(np.uint8)
            pixels[:,:,2] = (neg * 255).astype(np.uint8)
            pixels[:,:,1] = (np.abs(v) / 2.0 * nm * 60).astype(np.uint8).clip(0,255)

        # Firing neurons always flash white regardless of view
        pixels[b.firing, :] = 255

        surf   = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (SIZE*SCALE, SIZE*SCALE))
        self.screen.blit(scaled, (0, 0))

        # ── HUD ───────────────────────────────────────────────────────────
        hx = SIZE * SCALE
        pygame.draw.rect(self.screen, (4, 4, 10), (hx, 0, HUD_W, self.win_h))

        c  = b.counts()
        y  = 10

        def txt(s, col=(155,155,185), big=False):
            nonlocal y
            f = self.font_md if big else self.font_sm
            su = f.render(s, True, col)
            self.screen.blit(su, (hx+8, y))
            y += su.get_height() + 2

        def bar(label, val, maxv, col):
            nonlocal y
            txt(label)
            bw, bh = self.graph_w, 9
            pygame.draw.rect(self.screen, (20,20,32), (hx+8, y, bw, bh))
            fw = int(bw * min(val/max(maxv,1), 1.0))
            if fw > 0:
                pygame.draw.rect(self.screen, col, (hx+8, y, fw, bh))
            y += bh + 5

        txt("BRAIN GENESIS", (80,220,255), big=True)
        txt(f"t = {b.t:,}  |  ×{self.speed}", (90,90,120))
        y += 3

        txt("─ NETWORK ─", (45,45,70))
        bar(f"neurons  {c['neuron']:6d}", c['neuron'],    SIZE*SIZE//2, (55,185,255))
        bar(f"firing   {c['firing']:6d}", c['firing'],    max(c['neuron']//5,1), (200,200,255))
        bar(f"fr frac  {c['frac']:.3f}", c['frac'],       0.3,            (150,150,220))

        y += 3
        txt("─ PLASTICITY ─", (45,45,70))
        bar(f"connects {c['conn']:6d}", c['conn'],         5000,           (80,220,180))
        bar(f"mean w   {c['mean_w']:.3f}", c['mean_w'],    SYN_MAX,        (60,180,140))

        y += 3
        txt("─ SYNCHRONY ─", (45,45,70))
        bar(f"score  {c['synchrony']:.4f}", c['synchrony'], 0.15, (200,180,80))

        sync_s = c['synchrony']
        if sync_s < 0.01:
            txt("  chaotic / independent", (120,80,80))
        elif sync_s < 0.04:
            txt("  weak oscillation", (160,140,60))
        elif sync_s < 0.08:
            txt("  rhythmic bursting", (200,180,80))
        else:
            txt("  SYNCHRONIZED", (255,220,100))

        y += 3
        txt("─ VIEW ─", (45,45,70))
        views = [('X','cells'),('W','weights'),('A','activity'),('P','potential')]
        for k, v in views:
            col = (80,220,255) if self.view == v else (70,70,90)
            txt(f"  [{k}] {v}", col)

        if self.paused:
            txt("[ PAUSED ]", (255,210,60))

        # Synchrony oscillation graph
        y += 4
        txt("─ FIRING RHYTHM ─", (45,45,70))
        self._draw_sync_graph(hx+8, y)
        y += self.graph_h + 6

        # Population graph
        txt("─ GROWTH ─", (45,45,70))
        self._draw_pop_graph(hx+8, y)
        y += self.graph_h + 6

        txt("─ KEYS ─", (35,35,55))
        for k, v in [("SPC","pause"),("↑↓","speed"),("Q","quit")]:
            txt(f"  {k:<5} {v}", (65,65,85))

        pygame.display.flip()

        if self.save_frames and not self.paused and self.frame_n % 4 == 0:
            path = os.path.join(self.frame_dir, f'frame_{self.frame_n:06d}.png')
            pygame.image.save(self.screen, path)
        self.frame_n += 1

    def _draw_sync_graph(self, gx, gy):
        pygame.draw.rect(self.screen, (8,8,16), (gx, gy, self.graph_w, self.graph_h))
        data = self.brain.fire_buffer
        if len(data) < 2:
            return
        n   = min(len(data), self.graph_w)
        pts = []
        for i, v in enumerate(data[-n:]):
            px = gx + int(i * self.graph_w / n)
            py = gy + self.graph_h - int(min(v/0.3, 1.0) * (self.graph_h-2)) - 1
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(self.screen, (200,180,80), False, pts, 1)

    def _draw_pop_graph(self, gx, gy):
        pygame.draw.rect(self.screen, (8,8,16), (gx, gy, self.graph_w, self.graph_h))
        h = self.brain.history
        for key, col, maxv in [
            ('neuron',       (55,185,255), SIZE*SIZE//4),
            ('connectivity', (80,220,180), 5000),
        ]:
            data = h[key]
            if len(data) < 2:
                continue
            n   = min(len(data), self.graph_w)
            pts = []
            for i, v in enumerate(data[-n:]):
                px = gx + int(i * self.graph_w / n)
                py = gy + self.graph_h - int(min(v/maxv,1.0)*(self.graph_h-2)) - 1
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, pts, 1)

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:     return False
                if e.key == pygame.K_SPACE: self.paused = not self.paused
                if e.key == pygame.K_UP:    self.speed = min(self.speed+1, 20)
                if e.key == pygame.K_DOWN:  self.speed = max(self.speed-1, 1)
                if e.key == pygame.K_x:     self.view = 'cells'
                if e.key == pygame.K_w:     self.view = 'weights'
                if e.key == pygame.K_a:     self.view = 'activity'
                if e.key == pygame.K_p:     self.view = 'potential'
        return True

    def tick(self):
        self.clock.tick(FPS)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    save  = '--save' in sys.argv
    brain = Brain()
    brain.seed(n=96)
    renderer = Renderer(brain, save_frames=save)

    print("Brain Genesis")
    print("="*50)
    print(f"Seed: 24 neurons scattered across {SIZE}×{SIZE} grid")
    print()
    print("Growth rule: entangled Julia iteration (Gateway Fractal equation)")
    print("Plasticity:  Hebbian LTP/LTD  (fire together → wire together)")
    print("Dynamics:    FitzHugh-Nagumo  (Hodgkin-Huxley reduced)")
    print()
    print("Watch for:")
    print("  Fractal dendritic branching as neurons grow toward each other")
    print("  White flashes traveling through the network (action potentials)")
    print("  Synchrony score rising  (brain rhythms forming)")
    print("  Activity view [A]: hotspot hubs appearing")
    print("  Weights view  [W]: strong circuit paths lighting up")
    print()
    print("Controls: SPACE pause | ↑↓ speed | X/W/A/P views | Q quit")
    print()

    running = True
    while running:
        running = renderer.handle_events()
        if not renderer.paused:
            for _ in range(renderer.speed):
                brain.step()
        renderer.render()
        renderer.tick()

    pygame.quit()

    c = brain.counts()
    print(f"\nSimulation ended at t={brain.t}")
    print(f"Neurons: {c['neuron']} | Connections: {c['conn']} | Synchrony: {c['synchrony']:.4f}")
    h = brain.history
    if h['synchrony']:
        peak = max(h['synchrony'])
        print(f"Peak synchrony: {peak:.4f}")
        if peak > 0.08:
            print("→ Synchronized oscillations emerged. A brain rhythm formed.")
        elif peak > 0.03:
            print("→ Weak rhythmic bursting. Partial network coordination.")
        else:
            print("→ Remained chaotic. No stable circuits.")


if __name__ == '__main__':
    main()

