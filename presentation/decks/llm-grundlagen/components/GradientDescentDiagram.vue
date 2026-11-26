<script lang="ts">
let gdUid = 0
</script>

<script setup lang="ts">
const arrowId = `gd-arrow-${++gdUid}`
const arrowUrl = `url(#${arrowId})`
</script>

<template>
  <svg
    class="gd-svg"
    viewBox="0 0 420 168"
    role="img"
    aria-label="Anschauung Gradientenabstieg: vom Zufallsstart in kleinen Schritten Richtung kleinerem Fehler"
  >
    <defs>
      <marker :id="arrowId" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
      </marker>
    </defs>

    <!-- Axes -->
    <line x1="46" y1="16" x2="46" y2="140" stroke="#94a3b8" stroke-width="1.2" />
    <line x1="46" y1="140" x2="404" y2="140" stroke="#94a3b8" stroke-width="1.2" />
    <text x="14" y="80" class="gd-axis" transform="rotate(-90 14 80)">Fehler (Loss)</text>
    <text x="225" y="159" class="gd-axis">Gewicht (Stellschraube)</text>

    <!-- Loss curve (smooth valley) -->
    <path
      d="M 68 28 Q 225 224 382 28"
      fill="none"
      stroke="#cbd5e1"
      stroke-width="2.5"
      stroke-linecap="round"
    />

    <!-- Tangent line at start showing slope (gradient) -->
    <line x1="76" y1="30" x2="148" y2="96" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="4 3" />
    <text x="152" y="64" class="gd-hint-amber">Gradient = Steigung</text>
    <text x="152" y="76" class="gd-hint-amber">(zeigt bergab)</text>

    <!-- Descent steps -->
    <path d="M 98 56 Q 124 72 136 86" fill="none" stroke="#dc2626" stroke-width="2" :marker-end="arrowUrl" />
    <path d="M 142 91 Q 168 106 184 114" fill="none" stroke="#dc2626" stroke-width="2" :marker-end="arrowUrl" />
    <path d="M 190 116 Q 208 122 221 124" fill="none" stroke="#dc2626" stroke-width="2" :marker-end="arrowUrl" />

    <!-- Points -->
    <circle cx="96" cy="54" r="5" fill="#dc2626" />
    <circle cx="139" cy="89" r="3.5" fill="#f87171" />
    <circle cx="187" cy="115" r="3.5" fill="#f87171" />
    <circle cx="225" cy="126" r="5" fill="#059669" />

    <!-- Labels -->
    <text x="104" y="42" class="gd-label-red">Zufallsstart</text>
    <text x="64" y="118" class="gd-hint-red">Schritt = Lernrate</text>
    <text x="236" y="129" class="gd-label-green">kleinerer Fehler</text>

    <!-- Overshoot warning hint in clean space inside valley -->
    <text x="285" y="48" class="gd-note">Zu große Lernrate →</text>
    <text x="285" y="60" class="gd-note">springt übers Tal</text>
  </svg>
</template>

<style scoped>
.gd-svg {
  width: 100%;
  max-width: 430px;
  height: auto;
  display: block;
  background: #fafafa;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.gd-axis {
  font-size: 10px;
  fill: #475569;
  font-weight: 600;
  text-anchor: middle;
}

.gd-label-red {
  font-size: 10px;
  fill: #b91c1c;
  font-weight: 700;
}

.gd-hint-red {
  font-size: 9.5px;
  fill: #dc2626;
  font-weight: 600;
}

.gd-hint-amber {
  font-size: 9.5px;
  fill: #b45309;
  font-weight: 600;
}

.gd-label-green {
  font-size: 10px;
  fill: #047857;
  font-weight: 700;
}

.gd-note {
  font-size: 9px;
  fill: #64748b;
  text-anchor: middle;
}
</style>
