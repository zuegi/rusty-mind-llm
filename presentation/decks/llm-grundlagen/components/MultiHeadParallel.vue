<script lang="ts">
let mhUid = 0
</script>

<script setup lang="ts">
const uid = `mh-${++mhUid}`
const arrowDarkId = `${uid}-arrow-dark`
const arrowOrangeId = `${uid}-arrow-orange`
const arrowDarkUrl = `url(#${arrowDarkId})`
const arrowOrangeUrl = `url(#${arrowOrangeId})`

// 4 staggered 3D cards viewed obliquely from front-left (back-right -> front-left)
// Drawn back-to-front so front cards naturally overlap rear cards.
const heads = [
  { id: 'Head H', sub: '… bis num_heads', x: 184, y: 110, fill: '#fff7ed', side: '#fed7aa', top: '#ffedd5', stroke: '#fb923c', text: '#9a3412', dashed: true },
  { id: 'Head 3', sub: 'Blickwinkel 3', x: 138, y: 142, fill: '#ffedd5', side: '#fdba74', top: '#fff7ed', stroke: '#f97316', text: '#9a3412', dashed: false },
  { id: 'Head 2', sub: 'Blickwinkel 2', x: 92, y: 174, fill: '#ffedd5', side: '#fdba74', top: '#fff7ed', stroke: '#ea580c', text: '#7c2d12', dashed: false },
  { id: 'Head 1', sub: 'Blickwinkel 1', x: 46, y: 206, fill: '#fed7aa', side: '#fb923c', top: '#ffedd5', stroke: '#c2410c', text: '#7c2d12', dashed: false },
]

const cardW = 166
const cardH = 106
const depthX = 14
const depthY = -8
</script>

<template>
  <div class="mh-wrapper">
    <svg
      viewBox="0 0 420 468"
      width="420"
      height="468"
      class="mh-svg"
      role="img"
      aria-label="Multi-Head Attention: parallele Köpfe in Schrägansicht"
    >
      <defs>
        <marker
          :id="arrowDarkId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
        </marker>
        <marker
          :id="arrowOrangeId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#ea580c" />
        </marker>
      </defs>

      <!-- TOP: Concat + Linear Output -->
      <rect
        x="48"
        y="12"
        width="324"
        height="52"
        rx="9"
        fill="#e0e7ff"
        stroke="#4f46e5"
        stroke-width="1.8"
      />
      <text x="210" y="33" text-anchor="middle" class="t-title" fill="#312e81">
        Zusammenführen (Concat + Linear)
      </text>
      <text x="210" y="51" text-anchor="middle" class="t-sub" fill="#4338ca">
        H × head_dim = wieder volle Länge (z. B. 12 × 64 = 768)
      </text>

      <!-- Merge arrows from heads up to Concat -->
      <line x1="128" y1="198" x2="128" y2="68" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="3 2" :marker-end="arrowOrangeUrl" />
      <line x1="175" y1="166" x2="175" y2="68" stroke="#ea580c" stroke-width="1.5" :marker-end="arrowOrangeUrl" />
      <line x1="221" y1="134" x2="221" y2="68" stroke="#ea580c" stroke-width="1.5" :marker-end="arrowOrangeUrl" />
      <line x1="268" y1="102" x2="268" y2="68" stroke="#fb923c" stroke-width="1.4" stroke-dasharray="3 2" :marker-end="arrowOrangeUrl" />

      <!-- Split arrows from Input up to each Head (drawn behind cards) -->
      <path d="M 140 398 L 140 374 L 128 374 L 128 317" fill="none" stroke="#ea580c" stroke-width="1.6" :marker-end="arrowOrangeUrl" />
      <path d="M 192 398 L 192 368 L 176 368 L 176 285" fill="none" stroke="#ea580c" stroke-width="1.5" :marker-end="arrowOrangeUrl" />
      <path d="M 244 398 L 244 362 L 222 362 L 222 253" fill="none" stroke="#ea580c" stroke-width="1.5" :marker-end="arrowOrangeUrl" />
      <path d="M 296 398 L 296 356 L 268 356 L 268 221" fill="none" stroke="#fb923c" stroke-width="1.4" stroke-dasharray="3 2" :marker-end="arrowOrangeUrl" />

      <!-- 3D Oblique Parallel Head Cards (back-to-front) -->
      <g v-for="(h, idx) in heads" :key="h.id" :class="{ 'front-glow': idx === heads.length - 1 }">
        <!-- Top 3D bevel -->
        <polygon
          :points="`${h.x},${h.y} ${h.x + depthX},${h.y + depthY} ${h.x + cardW + depthX},${h.y + depthY} ${h.x + cardW},${h.y}`"
          :fill="h.top"
          :stroke="h.stroke"
          stroke-width="1.3"
          :stroke-dasharray="h.dashed ? '4 3' : undefined"
        />
        <!-- Right 3D bevel -->
        <polygon
          :points="`${h.x + cardW},${h.y} ${h.x + cardW + depthX},${h.y + depthY} ${h.x + cardW + depthX},${h.y + cardH + depthY} ${h.x + cardW},${h.y + cardH}`"
          :fill="h.side"
          :stroke="h.stroke"
          stroke-width="1.3"
          :stroke-dasharray="h.dashed ? '4 3' : undefined"
        />
        <!-- Front face -->
        <rect
          :x="h.x"
          :y="h.y"
          :width="cardW"
          :height="cardH"
          rx="4"
          :fill="h.fill"
          :stroke="h.stroke"
          :stroke-width="idx === heads.length - 1 ? 2 : 1.5"
          :stroke-dasharray="h.dashed ? '4 3' : undefined"
        />

        <!-- Card header -->
        <text :x="h.x + 12" :y="h.y + 16" class="t-head" :fill="h.text">
          {{ h.id }}
        </text>
        <text :x="h.x + cardW - 10" :y="h.y + 15" text-anchor="end" class="t-chip" :fill="h.text">
          head_dim
        </text>

        <!-- Details visible on front card -->
        <g v-if="idx === heads.length - 1">
          <line :x1="h.x + 12" :y1="h.y + 24" :x2="h.x + cardW - 12" :y2="h.y + 24" stroke="#ea580c" stroke-opacity="0.35" stroke-width="1" />
          <text :x="h.x + 12" :y="h.y + 42" class="t-small" fill="#7c2d12">Eigene Gewichte</text>
          <text :x="h.x + 12" :y="h.y + 57" class="t-tiny" fill="#9a3412">Maskierte Token-Bezüge</text>

          <!-- Mini attention weight bars (illustrative visual cue) -->
          <rect :x="h.x + 12" :y="h.y + 69" width="142" height="24" rx="4" fill="#fff7ed" stroke="#fdba74" stroke-width="1" />
          <rect :x="h.x + 18" :y="h.y + 75" width="28" height="5" rx="2" fill="#fed7aa" />
          <rect :x="h.x + 18" :y="h.y + 83" width="66" height="5" rx="2" fill="#ea580c" />
          <text :x="h.x + 148" :y="h.y + 87" text-anchor="end" class="t-micro" fill="#9a3412">Token-Matrix</text>
        </g>
      </g>

      <!-- Oblique Parallelism Callout (bottom-right of stack) -->
      <g>
        <line x1="234" y1="336" x2="386" y2="236" stroke="#b7410e" stroke-width="1.8" stroke-dasharray="4 3" :marker-start="arrowOrangeUrl" :marker-end="arrowOrangeUrl" />
        <rect x="266" y="294" width="142" height="44" rx="7" fill="#fff7ed" stroke="#b7410e" stroke-width="1.4" />
        <text x="337" y="313" text-anchor="middle" class="t-badge" fill="#9a3412">GLEICHZEITIG</text>
        <text x="337" y="328" text-anchor="middle" class="t-tiny" fill="#7c2d12">alle Heads parallel (GPU)</text>
      </g>

      <!-- Split label -->
      <rect x="148" y="366" width="124" height="20" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />
      <text x="210" y="380" text-anchor="middle" class="t-micro" fill="#334155">Aufteilen in H Teile</text>

      <!-- BOTTOM: Input Vector (segmented) -->
      <rect
        x="48"
        y="398"
        width="324"
        height="58"
        rx="9"
        fill="#fce7f3"
        stroke="#db2777"
        stroke-width="1.8"
      />
      <text x="210" y="418" text-anchor="middle" class="t-title" fill="#831843">
        Eingabe-Vektor pro Token (embedding_dim)
      </text>
      <!-- 4 visual segments showing the split -->
      <g transform="translate(64, 427)">
        <rect x="0" y="0" width="68" height="19" rx="4" fill="#fed7aa" stroke="#c2410c" stroke-width="1.1" />
        <text x="34" y="13" text-anchor="middle" class="t-micro" fill="#7c2d12">Teil 1</text>

        <rect x="74" y="0" width="68" height="19" rx="4" fill="#ffedd5" stroke="#ea580c" stroke-width="1.1" />
        <text x="108" y="13" text-anchor="middle" class="t-micro" fill="#7c2d12">Teil 2</text>

        <rect x="148" y="0" width="68" height="19" rx="4" fill="#ffedd5" stroke="#f97316" stroke-width="1.1" />
        <text x="182" y="13" text-anchor="middle" class="t-micro" fill="#9a3412">Teil 3</text>

        <rect x="222" y="0" width="68" height="19" rx="4" fill="#fff7ed" stroke="#fb923c" stroke-width="1.1" stroke-dasharray="3 2" />
        <text x="256" y="13" text-anchor="middle" class="t-micro" fill="#9a3412">… Teil H</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.mh-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
}

.mh-svg {
  width: 100%;
  max-width: 365px;
  height: auto;
  display: block;
}

.front-glow {
  filter: drop-shadow(0 3px 4px rgba(154, 52, 18, 0.16));
}

.t-title {
  font-size: 12.5px;
  font-weight: 700;
}

.t-head {
  font-size: 13px;
  font-weight: 700;
}

.t-sub {
  font-size: 10.5px;
  font-weight: 500;
}

.t-small {
  font-size: 11px;
  font-weight: 600;
}

.t-tiny {
  font-size: 9.8px;
  font-weight: 400;
}

.t-chip {
  font-size: 9.5px;
  font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.t-badge {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.t-micro {
  font-size: 9px;
  font-weight: 600;
}
</style>
