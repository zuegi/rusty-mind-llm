<script setup lang="ts">
const trainLoss = [
  11.186, 10.2675, 8.7421, 8.0989, 7.7052, 7.4433, 7.2367, 7.0602, 6.8942,
  6.7553, 6.6564, 6.5431, 6.415, 6.3214, 6.237, 6.1563, 6.0976, 5.9851,
  5.937, 5.8693, 5.7849,
]

const valLoss = [
  11.1926, 10.4021, 8.893, 8.236, 7.7753, 7.5009, 7.3582, 7.2215, 7.0724,
  7.0062, 6.9228, 6.8938, 6.8049, 6.7589, 6.7653, 6.7223, 6.6642, 6.671,
  6.7338, 6.7123, 6.7019,
]

const padLeft = 42
const padRight = 14
const padTop = 18
const padBottom = 34
const width = 420
const height = 220

const minY = 5.5
const maxY = 11.5
const maxEpoch = 20

const xOf = (epoch: number) =>
  padLeft + (epoch / maxEpoch) * (width - padLeft - padRight)

const yOf = (val: number) =>
  padTop + ((maxY - val) / (maxY - minY)) * (height - padTop - padBottom)

const toPolyline = (values: number[]) =>
  values.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(' ')

const trainPoints = toPolyline(trainLoss)
const valPoints = toPolyline(valLoss)

const bestEpoch = 16
const bestX = xOf(bestEpoch)
const bestY = yOf(valLoss[bestEpoch])
const yTicks = [6, 8, 10, 11]
const xTicks = [0, 5, 10, 16, 20]
</script>

<template>
  <svg viewBox="0 0 420 220" class="loss-chart-svg" role="img" aria-label="Trainings- und Prüfungsfehler über 20 Durchläufe">
    <!-- Overfitting zone after epoch 16 -->
    <rect
      :x="bestX"
      :y="padTop"
      :width="xOf(20) - bestX"
      :height="height - padTop - padBottom"
      fill="#fef2f2"
    />
    <text
      :x="bestX + (xOf(20) - bestX) / 2"
      :y="padTop + 13"
      text-anchor="middle"
      class="lc-warn"
    >
      schlechter
    </text>

    <!-- Horizontal grid lines -->
    <g v-for="t in yTicks" :key="t">
      <line
        :x1="padLeft"
        :y1="yOf(t)"
        :x2="width - padRight"
        :y2="yOf(t)"
        stroke="#e2e8f0"
        stroke-width="1"
      />
      <text
        :x="padLeft - 6"
        :y="yOf(t) + 3.5"
        text-anchor="end"
        class="lc-tick"
      >
        {{ t }}
      </text>
    </g>

    <!-- Axes -->
    <line :x1="padLeft" :y1="padTop" :x2="padLeft" :y2="height - padBottom" stroke="#94a3b8" stroke-width="1.2" />
    <line :x1="padLeft" :y1="height - padBottom" :x2="width - padRight" :y2="height - padBottom" stroke="#94a3b8" stroke-width="1.2" />

    <!-- X ticks -->
    <g v-for="e in xTicks" :key="e">
      <line :x1="xOf(e)" :y1="height - padBottom" :x2="xOf(e)" :y2="height - padBottom + 4" stroke="#94a3b8" stroke-width="1.2" />
      <text
        :x="xOf(e)"
        :y="height - padBottom + 15"
        text-anchor="middle"
        :class="e === bestEpoch ? 'lc-tick-best' : 'lc-tick'"
      >
        {{ e }}
      </text>
    </g>
    <text :x="padLeft + (width - padLeft - padRight) / 2" :y="height - 4" text-anchor="middle" class="lc-axis">
      Durchlauf (Epoche)
    </text>
    <text x="12" :y="padTop + (height - padTop - padBottom) / 2" text-anchor="middle" class="lc-axis" :transform="`rotate(-90 12 ${padTop + (height - padTop - padBottom) / 2})`">
      Fehler (Loss)
    </text>

    <!-- Best epoch vertical line -->
    <line
      :x1="bestX"
      :y1="padTop"
      :x2="bestX"
      :y2="height - padBottom"
      stroke="#0d9488"
      stroke-width="1.3"
      stroke-dasharray="4 3"
    />

    <!-- Curves -->
    <polyline :points="trainPoints" fill="none" stroke="#2563eb" stroke-width="2.3" stroke-linejoin="round" />
    <polyline :points="valPoints" fill="none" stroke="#dc2626" stroke-width="2.3" stroke-linejoin="round" />

    <!-- Best point marker -->
    <circle :cx="bestX" :cy="bestY" r="4.5" fill="#0d9488" stroke="#ffffff" stroke-width="1.5" />
    <text :x="bestX - 8" :y="bestY - 12" text-anchor="end" class="lc-best">
      Bester Stand: Epoche 16 (6.66)
    </text>

    <!-- Legend -->
    <g transform="translate(170, 26)">
      <rect x="0" y="0" width="122" height="38" rx="6" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" />
      <line x1="8" y1="13" x2="24" y2="13" stroke="#2563eb" stroke-width="2.3" />
      <text x="30" y="16.5" class="lc-legend">Übungsdaten</text>
      <line x1="8" y1="28" x2="24" y2="28" stroke="#dc2626" stroke-width="2.3" />
      <text x="30" y="31.5" class="lc-legend">Prüfungsdaten</text>
    </g>
  </svg>
</template>

<style scoped>
.loss-chart-svg {
  width: 100%;
  height: auto;
  font-family: Arial, sans-serif;
}

.lc-tick {
  font-size: 9.5px;
  fill: #64748b;
}

.lc-tick-best {
  font-size: 9.5px;
  font-weight: 700;
  fill: #0f766e;
}

.lc-axis {
  font-size: 10px;
  fill: #475569;
}

.lc-warn {
  font-size: 9.5px;
  font-weight: 700;
  fill: #b91c1c;
}

.lc-best {
  font-size: 10px;
  font-weight: 700;
  fill: #0f766e;
}

.lc-legend {
  font-size: 10px;
  fill: #1e293b;
}
</style>
