<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    step: number
  }>(),
  {
    step: 9,
  },
)

const isOverview = () => props.step >= 9

const show = (minStep: number) => props.step >= minStep

const isActive = (steps: number[]) => isOverview() || steps.includes(props.step)

const opacityFor = (steps: number[]) => (isActive(steps) ? 1 : 0.62)

const strokeWidthFor = (steps: number[]) =>
  !isOverview() && steps.includes(props.step) ? 2.4 : 1.3
</script>

<template>
  <div class="llm-arch-wrapper">
    <svg
      viewBox="0 0 434 510"
      class="llm-arch-svg"
      role="img"
      :aria-label="`LLM-Architektur Schritt ${step}`"
    >
      <defs>
        <marker
          id="arrow-dark"
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
          id="arrow-green"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
        </marker>
        <marker
          id="arrow-red"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
        </marker>
        <filter id="active-glow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0f172a" flood-opacity="0.16" />
        </filter>
      </defs>

      <!-- STEP 1: BLACKBOX -->
      <g v-if="step === 1">
        <rect x="75" y="425" width="284" height="48" rx="10" fill="#f1f5f9" stroke="#64748b" stroke-width="1.8" />
        <text x="217" y="446" text-anchor="middle" class="t-title" fill="#0f172a">Eingabetext</text>
        <text x="217" y="463" text-anchor="middle" class="t-sub" fill="#475569">„Der Hund …“</text>

        <line x1="217" y1="425" x2="217" y2="342" stroke="#475569" stroke-width="2" marker-end="url(#arrow-dark)" />

        <rect
          x="75"
          y="165"
          width="284"
          height="170"
          rx="14"
          fill="#f8fafc"
          stroke="#334155"
          stroke-width="2.2"
          stroke-dasharray="7 5"
          filter="url(#active-glow)"
        />
        <text x="217" y="238" text-anchor="middle" class="t-big" fill="#0f172a">Sprachmodell</text>
        <text x="217" y="264" text-anchor="middle" class="t-sub" fill="#64748b">(Blackbox — wird Schritt für Schritt geöffnet)</text>

        <line x1="217" y1="165" x2="217" y2="94" stroke="#475569" stroke-width="2" marker-end="url(#arrow-dark)" />

        <rect x="75" y="38" width="284" height="48" rx="10" fill="#d1fae5" stroke="#059669" stroke-width="1.8" />
        <text x="217" y="59" text-anchor="middle" class="t-title" fill="#065f46">Fortsetzung</text>
        <text x="217" y="76" text-anchor="middle" class="t-sub" fill="#047857">„… rennt“ (ein Token nach dem anderen)</text>
      </g>

      <!-- STEPS 2..9: GROWING VASWANI-STYLE DECODER-ONLY ARCHITECTURE -->
      <g v-else>
        <!-- Dashed GptModel Core Boundary -->
        <rect
          x="24"
          y="78"
          width="228"
          height="362"
          rx="12"
          fill="#ffffff"
          stroke="#475569"
          stroke-width="1.5"
          stroke-dasharray="6 4"
        />
        <text x="244" y="92" text-anchor="end" class="t-badge" fill="#475569">GptModel (Kern)</text>

        <!-- STEP 2: Tokenizer (outside) -->
        <g
          :opacity="opacityFor([2])"
          :filter="isActive([2]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <rect
            x="24"
            y="456"
            width="228"
            height="42"
            rx="8"
            fill="#f1f5f9"
            stroke="#64748b"
            :stroke-width="strokeWidthFor([2])"
          />
          <text x="138" y="473" text-anchor="middle" class="t-title" fill="#0f172a">Tokenizer (außerhalb)</text>
          <text x="138" y="489" text-anchor="middle" class="t-sub" fill="#475569">Text → Token-IDs</text>
          <line x1="138" y1="456" x2="138" y2="429" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow-dark)" />
        </g>

        <!-- STEP 2: Token Embedding + Positional Embedding -->
        <g
          :opacity="opacityFor([2])"
          :filter="isActive([2]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <rect
            x="44"
            y="393"
            width="188"
            height="32"
            rx="7"
            fill="#fce7f3"
            stroke="#db2777"
            :stroke-width="strokeWidthFor([2])"
          />
          <text x="138" y="413" text-anchor="middle" class="t-title" fill="#831843">Token Embedding</text>

          <line x1="138" y1="393" x2="138" y2="375" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow-dark)" />

          <!-- Positional Addition Circle -->
          <circle cx="138" cy="363" r="10" fill="#f8fafc" stroke="#475569" stroke-width="1.5" />
          <text x="138" y="367" text-anchor="middle" class="t-plus" fill="#0f172a">+</text>

          <!-- Positional Embedding Box -->
          <rect
            x="162"
            y="350"
            width="82"
            height="26"
            rx="6"
            fill="#f3f4f6"
            stroke="#6b7280"
            :stroke-width="strokeWidthFor([2])"
          />
          <text x="203" y="361" text-anchor="middle" class="t-tiny" fill="#1f2937">Positional</text>
          <text x="203" y="372" text-anchor="middle" class="t-tiny" fill="#1f2937">Embedding</text>
          <line x1="162" y1="363" x2="151" y2="363" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />
        </g>

        <!-- STEP 3: N× Transformer-Block (Pre-LayerNorm) -->
        <g
          v-if="show(3)"
          :opacity="opacityFor([3])"
          :filter="isActive([3]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line x1="138" y1="353" x2="138" y2="336" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow-dark)" />

          <!-- N× Frame -->
          <rect
            x="36"
            y="142"
            width="204"
            height="198"
            rx="10"
            fill="#f8fafc"
            stroke="#94a3b8"
            :stroke-width="strokeWidthFor([3])"
          />
          <text x="41" y="156" class="t-nx" fill="#334155">N×</text>

          <!-- LayerNorm 1 (Pre-LN) -->
          <rect x="56" y="308" width="152" height="22" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.3" />
          <text x="132" y="323" text-anchor="middle" class="t-small" fill="#713f12">LayerNorm</text>
          <line x1="132" y1="308" x2="132" y2="299" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />

          <!-- Masked Multi-Head Attention -->
          <rect x="56" y="266" width="152" height="30" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.4" />
          <text x="132" y="279" text-anchor="middle" class="t-small" fill="#7c2d12">Masked Multi-Head</text>
          <text x="132" y="291" text-anchor="middle" class="t-tiny" fill="#9a3412">Attention (+ Dropout)</text>
          <line x1="132" y1="266" x2="132" y2="258" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />

          <!-- Residual Add 1 -->
          <circle cx="132" cy="248" r="8" fill="#ffffff" stroke="#475569" stroke-width="1.4" />
          <text x="132" y="252" text-anchor="middle" class="t-plus-sm" fill="#0f172a">+</text>
          <!-- Residual Bypass 1 -->
          <path
            d="M 138 336 L 224 336 L 224 248 L 143 248"
            fill="none"
            stroke="#64748b"
            stroke-width="1.3"
            marker-end="url(#arrow-dark)"
          />

          <line x1="132" y1="240" x2="132" y2="232" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />

          <!-- LayerNorm 2 (Pre-LN) -->
          <rect x="56" y="207" width="152" height="22" rx="6" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.3" />
          <text x="132" y="222" text-anchor="middle" class="t-small" fill="#713f12">LayerNorm</text>
          <line x1="132" y1="207" x2="132" y2="198" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />

          <!-- Feed-Forward -->
          <rect x="56" y="170" width="152" height="25" rx="6" fill="#e0f2fe" stroke="#0284c7" stroke-width="1.4" />
          <text x="132" y="186" text-anchor="middle" class="t-small" fill="#0c4a6e">Feed-Forward (+ Dropout)</text>
          <line x1="132" y1="170" x2="132" y2="162" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow-dark)" />

          <!-- Residual Add 2 -->
          <circle cx="132" cy="153" r="8" fill="#ffffff" stroke="#475569" stroke-width="1.4" />
          <text x="132" y="157" text-anchor="middle" class="t-plus-sm" fill="#0f172a">+</text>
          <!-- Residual Bypass 2 -->
          <path
            d="M 132 236 L 224 236 L 224 153 L 143 153"
            fill="none"
            stroke="#64748b"
            stroke-width="1.3"
            marker-end="url(#arrow-dark)"
          />
        </g>

        <!-- STEP 4: Linear (output_projection) -> Logits -->
        <g
          v-if="show(4)"
          :opacity="opacityFor([4])"
          :filter="isActive([4]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line x1="132" y1="145" x2="132" y2="131" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow-dark)" />
          <rect
            x="44"
            y="98"
            width="188"
            height="30"
            rx="7"
            fill="#e0e7ff"
            stroke="#4f46e5"
            :stroke-width="strokeWidthFor([4])"
          />
          <text x="138" y="112" text-anchor="middle" class="t-small" fill="#312e81">Linear (output_projection)</text>
          <text x="138" y="123" text-anchor="middle" class="t-tiny" fill="#4338ca">→ Logits [V]</text>
        </g>

        <!-- STEP 4 & 7: Inferenz (outside, top) + Autoregressive Loop -->
        <g
          v-if="show(4)"
          :opacity="opacityFor([4, 7])"
          :filter="isActive([4, 7]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line x1="138" y1="98" x2="138" y2="67" stroke="#059669" stroke-width="1.7" marker-end="url(#arrow-green)" />
          <rect
            x="24"
            y="14"
            width="228"
            height="49"
            rx="8"
            fill="#d1fae5"
            stroke="#059669"
            :stroke-width="strokeWidthFor([4, 7])"
          />
          <text x="138" y="33" text-anchor="middle" class="t-title" fill="#065f46">Inferenz (außerhalb)</text>
          <text x="138" y="48" text-anchor="middle" class="t-sub" fill="#047857">Softmax · Greedy / Sampling</text>
          <text x="138" y="59" text-anchor="middle" class="t-tiny" fill="#065f46">Token anhängen → erneut berechnen</text>

          <!-- Autoregressive feedback loop along far left -->
          <path
            d="M 24 38 L 9 38 L 9 477 L 20 477"
            fill="none"
            stroke="#059669"
            stroke-width="1.6"
            stroke-dasharray="4 3"
            marker-end="url(#arrow-green)"
          />
        </g>

        <!-- STEP 5: Training (outside, right top) -->
        <g
          v-if="show(5)"
          :opacity="opacityFor([5])"
          :filter="isActive([5]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line x1="232" y1="113" x2="270" y2="113" stroke="#dc2626" stroke-width="1.6" marker-end="url(#arrow-red)" />
          <rect
            x="274"
            y="84"
            width="150"
            height="108"
            rx="8"
            fill="#fee2e2"
            stroke="#dc2626"
            :stroke-width="strokeWidthFor([5])"
          />
          <text x="349" y="102" text-anchor="middle" class="t-title" fill="#991b1b">Training</text>
          <text x="349" y="119" text-anchor="middle" class="t-sub" fill="#7f1d1d">Zieltoken + Logits</text>
          <text x="349" y="135" text-anchor="middle" class="t-sub" fill="#7f1d1d">↓ Cross-Entropy Loss</text>
          <text x="349" y="151" text-anchor="middle" class="t-sub" fill="#7f1d1d">↓ Gradienten</text>
          <text x="349" y="167" text-anchor="middle" class="t-sub" fill="#7f1d1d">↓ Optimizer</text>
          <text x="349" y="183" text-anchor="middle" class="t-tiny-bold" fill="#991b1b">Gewichte ändern</text>

          <!-- Weight update arrow back into GptModel -->
          <line x1="274" y1="176" x2="256" y2="176" stroke="#dc2626" stroke-width="1.7" marker-end="url(#arrow-red)" />
        </g>

        <!-- STEP 6: Validierung (outside, right middle) -->
        <g
          v-if="show(6)"
          :opacity="opacityFor([6])"
          :filter="isActive([6]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line
            x1="252"
            y1="256"
            x2="270"
            y2="256"
            stroke="#0891b2"
            stroke-width="1.5"
            stroke-dasharray="4 3"
            marker-end="url(#arrow-dark)"
          />
          <rect
            x="274"
            y="214"
            width="150"
            height="84"
            rx="8"
            fill="#cffafe"
            stroke="#0891b2"
            :stroke-width="strokeWidthFor([6])"
          />
          <text x="349" y="233" text-anchor="middle" class="t-title" fill="#155e75">Validierung</text>
          <text x="349" y="250" text-anchor="middle" class="t-sub" fill="#164e63">Getrennte Daten</text>
          <text x="349" y="266" text-anchor="middle" class="t-sub" fill="#164e63">Loss messen</text>
          <text x="349" y="284" text-anchor="middle" class="t-tiny-bold" fill="#155e75">Feste Gewichte · kein Update</text>
        </g>

        <!-- STEP 8: Checkpoints (outside, right bottom) -->
        <g
          v-if="show(8)"
          :opacity="opacityFor([8])"
          :filter="isActive([8]) && !isOverview() ? 'url(#active-glow)' : undefined"
        >
          <line
            x1="252"
            y1="376"
            x2="270"
            y2="376"
            stroke="#7c3aed"
            stroke-width="1.5"
            marker-end="url(#arrow-dark)"
            marker-start="url(#arrow-dark)"
          />
          <rect
            x="274"
            y="320"
            width="150"
            height="112"
            rx="8"
            fill="#f3e8ff"
            stroke="#7c3aed"
            :stroke-width="strokeWidthFor([8])"
          />
          <text x="349" y="339" text-anchor="middle" class="t-title" fill="#5b21b6">Checkpoints</text>
          <text x="349" y="359" text-anchor="middle" class="t-sub-bold" fill="#4c1d95">last/</text>
          <text x="349" y="373" text-anchor="middle" class="t-tiny" fill="#5b21b6">Modell + Optimizer (Resume)</text>
          <text x="349" y="396" text-anchor="middle" class="t-sub-bold" fill="#4c1d95">best/</text>
          <text x="349" y="410" text-anchor="middle" class="t-tiny" fill="#5b21b6">Bester Val.-Loss (Anwenden)</text>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.llm-arch-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
}

.llm-arch-svg {
  width: 100%;
  max-height: 415px;
  height: auto;
  font-family: Arial, sans-serif;
}

.t-big {
  font-size: 18px;
  font-weight: 700;
}

.t-title {
  font-size: 12.5px;
  font-weight: 700;
}

.t-small {
  font-size: 11px;
  font-weight: 600;
}

.t-sub {
  font-size: 10.5px;
  font-weight: 400;
}

.t-sub-bold {
  font-size: 11px;
  font-weight: 700;
}

.t-tiny {
  font-size: 9.5px;
  font-weight: 400;
}

.t-tiny-bold {
  font-size: 9.5px;
  font-weight: 700;
}

.t-badge {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.t-nx {
  font-size: 12px;
  font-weight: 700;
}

.t-plus {
  font-size: 15px;
  font-weight: 700;
}

.t-plus-sm {
  font-size: 13px;
  font-weight: 700;
}
</style>
