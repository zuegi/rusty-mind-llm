<script lang="ts">
let archUid = 0
</script>

<script setup lang="ts">
const uid = `arch-${++archUid}`
const arrowDarkId = `${uid}-arrow-dark`
const arrowGreenId = `${uid}-arrow-green`
const arrowRedId = `${uid}-arrow-red`
const arrowDarkUrl = `url(#${arrowDarkId})`
const arrowGreenUrl = `url(#${arrowGreenId})`
const arrowRedUrl = `url(#${arrowRedId})`

const props = withDefaults(
  defineProps<{
    step: number
    hideInference?: boolean
    highlightModel?: boolean
  }>(),
  {
    step: 13,
    hideInference: false,
    highlightModel: false,
  },
)

const isOverview = () => props.step >= 13

const isCompact = () => props.step >= 2 && props.step <= 7

const svgViewBox = () => (isCompact() ? '0 0 268 536' : '0 0 434 536')
const svgWidth = () => (isCompact() ? 268 : 434)

const show = (minStep: number) => props.step >= minStep

const isModelPart = (steps: number[]) =>
  steps.some((s) => s >= 3 && s <= 6) && !steps.includes(2) && !steps.includes(7)

const isActive = (steps: number[]) =>
  isOverview() ||
  steps.includes(props.step) ||
  (props.highlightModel && isModelPart(steps))

const isOutputActive = () =>
  isOverview() ||
  props.step === 12 ||
  props.highlightModel ||
  (props.step === 6 && props.hideInference)

const opacityFor = (steps: number[]) => (isActive(steps) ? 1 : 0.48)

const strokeWidthFor = (steps: number[]) =>
  !isOverview() && !props.highlightModel && steps.includes(props.step) ? 2.4 : 1.3

const activeClassFor = (steps: number[]) =>
  !props.highlightModel && isActive(steps) && !isOverview() ? 'active-glow' : undefined
</script>

<template>
  <div class="llm-arch-wrapper">
    <svg
      :viewBox="svgViewBox()"
      :width="svgWidth()"
      height="536"
      :class="['llm-arch-svg', { 'llm-arch-svg--compact': isCompact() }]"
      role="img"
      :aria-label="`LLM-Architektur Schritt ${step}`"
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
          :id="arrowGreenId"
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
          :id="arrowRedId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
        </marker>
      </defs>

      <!-- STEP 1: BLACKBOX -->
      <g v-if="step === 1">
        <rect x="75" y="438" width="284" height="48" rx="10" fill="#f1f5f9" stroke="#64748b" stroke-width="1.8" />
        <text x="217" y="459" text-anchor="middle" class="t-title" fill="#0f172a">Eingabetext</text>
        <text x="217" y="476" text-anchor="middle" class="t-sub" fill="#475569">„Der Hund …“</text>

        <line x1="217" y1="438" x2="217" y2="355" stroke="#475569" stroke-width="2" :marker-end="arrowDarkUrl" />

        <rect
          x="75"
          y="178"
          width="284"
          height="170"
          rx="14"
          fill="#f8fafc"
          stroke="#334155"
          stroke-width="2.2"
          stroke-dasharray="7 5"
          class="active-glow"
        />
        <text x="217" y="251" text-anchor="middle" class="t-big" fill="#0f172a">Sprachmodell</text>
        <text x="217" y="277" text-anchor="middle" class="t-sub" fill="#64748b">(Blackbox — wird Schritt für Schritt geöffnet)</text>

        <line x1="217" y1="178" x2="217" y2="107" stroke="#475569" stroke-width="2" :marker-end="arrowDarkUrl" />

        <rect x="75" y="51" width="284" height="48" rx="10" fill="#d1fae5" stroke="#059669" stroke-width="1.8" />
        <text x="217" y="72" text-anchor="middle" class="t-title" fill="#065f46">Fortsetzung</text>
        <text x="217" y="89" text-anchor="middle" class="t-sub" fill="#047857">„… rennt“ (ein Token nach dem anderen)</text>
      </g>

      <!-- STEPS 2..13: GROWING VASWANI-STYLE DECODER-ONLY ARCHITECTURE -->
      <g v-else>
        <!-- Dashed GptModel Core Boundary -->
        <rect
          x="24"
          y="76"
          width="228"
          height="390"
          rx="12"
          :fill="step === 9 ? '#fff5f5' : props.highlightModel ? '#f0fdf4' : '#ffffff'"
          :stroke="step === 9 ? '#dc2626' : props.highlightModel ? '#059669' : '#475569'"
          :stroke-width="step === 9 || props.highlightModel ? 2.2 : 1.5"
          stroke-dasharray="6 4"
        />
        <text
          x="244"
          y="91"
          text-anchor="end"
          class="t-badge"
          :fill="step === 9 ? '#b91c1c' : props.highlightModel ? '#065f46' : '#475569'"
        >
          {{
            step === 9
              ? 'GptModel (Gewichte ändern)'
              : props.highlightModel
                ? 'GptModel (Kern) — komplett'
                : 'GptModel (Kern)'
          }}
        </text>

        <!-- STEP 2: Tokenizer (outside) -->
        <g
          :opacity="opacityFor([2, 12])"
          :class="activeClassFor([2, 12])"
        >
          <rect
            x="24"
            y="484"
            width="228"
            height="42"
            rx="8"
            fill="#f1f5f9"
            stroke="#64748b"
            :stroke-width="strokeWidthFor([2, 12])"
          />
          <text x="138" y="501" text-anchor="middle" class="t-title" fill="#0f172a">Tokenizer (außerhalb)</text>
          <text x="138" y="517" text-anchor="middle" class="t-sub" fill="#475569">Text → Nummern (IDs)</text>
          <line x1="138" y1="484" x2="138" y2="456" stroke="#475569" stroke-width="1.6" :marker-end="arrowDarkUrl" />
        </g>

        <!-- STEP 3: GptEmbedding (Token Embedding + Positional Embedding) -->
        <g
          v-if="show(3)"
          :opacity="opacityFor([3, 12])"
          :class="activeClassFor([3])"
        >
          <rect
            x="36"
            y="372"
            width="204"
            height="86"
            rx="10"
            fill="#fdf2f8"
            fill-opacity="0.45"
            stroke="#94a3b8"
            :stroke-width="strokeWidthFor([3])"
          />
          <text x="44" y="387" text-anchor="start" class="t-badge" fill="#334155">GptEmbedding</text>

          <rect
            x="44"
            y="418"
            width="188"
            height="34"
            rx="7"
            fill="#fce7f3"
            stroke="#db2777"
            :stroke-width="strokeWidthFor([3])"
          />
          <text x="138" y="433" text-anchor="middle" class="t-small" fill="#831843">Token-Embedding</text>
          <text x="138" y="446" text-anchor="middle" class="t-tiny" fill="#9d174d">ID → Vektor</text>

          <line x1="138" y1="418" x2="138" y2="402" stroke="#475569" stroke-width="1.6" :marker-end="arrowDarkUrl" />

          <!-- Positional Addition Circle -->
          <circle cx="138" cy="390" r="10" fill="#f8fafc" stroke="#475569" stroke-width="1.5" />
          <text x="138" y="394" text-anchor="middle" class="t-plus" fill="#0f172a">+</text>

          <!-- Positional Embedding Box -->
          <rect
            x="156"
            y="377"
            width="76"
            height="26"
            rx="6"
            fill="#f3f4f6"
            stroke="#6b7280"
            :stroke-width="strokeWidthFor([3])"
          />
          <text x="194" y="388" text-anchor="middle" class="t-tiny" fill="#1f2937">Positions-</text>
          <text x="194" y="399" text-anchor="middle" class="t-tiny" fill="#1f2937">Embedding</text>
          <line x1="156" y1="390" x2="150" y2="390" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />
        </g>

        <!-- STEP 4 & 5: N× Transformer-Block (Attention on step 4, full block on step 5) -->
        <g v-if="show(4)">
          <!-- N× Frame (visible from step 5) -->
          <g
            v-if="show(5)"
            :opacity="opacityFor([5, 12])"
            :class="activeClassFor([5])"
          >
            <rect
              x="36"
              y="148"
              width="204"
              height="204"
              rx="10"
              fill="#f8fafc"
              stroke="#94a3b8"
              :stroke-width="strokeWidthFor([5])"
            />
            <text x="240" y="144" text-anchor="end" class="t-badge" style="font-size: 8.5px" fill="#334155">N× Transformer-Block</text>
          </g>

          <!-- Entry arrow into block -->
          <line
            x1="138"
            y1="380"
            x2="138"
            :y2="show(5) ? 344 : 311"
            stroke="#475569"
            stroke-width="1.6"
            :marker-end="arrowDarkUrl"
            :opacity="opacityFor([4, 5, 12])"
          />

          <!-- LayerNorm 1 (Pre-LN) — step 5 -->
          <g
            v-if="show(5)"
            :opacity="opacityFor([5, 12])"
            :class="activeClassFor([5])"
          >
            <rect x="56" y="319" width="152" height="22" rx="6" fill="#fef9c3" stroke="#ca8a04" :stroke-width="strokeWidthFor([5])" />
            <text x="132" y="334" text-anchor="middle" class="t-small" fill="#713f12">LayerNorm</text>
            <line x1="132" y1="319" x2="132" y2="310" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />
          </g>

          <!-- Masked Multi-Head Attention — step 4 -->
          <g
            :opacity="opacityFor([4, 12])"
            :class="activeClassFor([4])"
          >
            <rect
              x="56"
              y="277"
              width="152"
              height="30"
              rx="6"
              fill="#ffedd5"
              stroke="#ea580c"
              :stroke-width="strokeWidthFor([4])"
            />
            <text x="132" y="290" text-anchor="middle" class="t-small" fill="#7c2d12">Masked Multi-Head</text>
            <text x="132" y="302" text-anchor="middle" class="t-tiny" fill="#9a3412">Attention (Token-Bezüge)</text>
          </g>

          <!-- Rest of Transformer-Block: Residual 1, LayerNorm 2, Feed-Forward, Residual 2 — step 5 -->
          <g
            v-if="show(5)"
            :opacity="opacityFor([5, 12])"
            :class="activeClassFor([5])"
          >
            <line x1="132" y1="277" x2="132" y2="269" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />

            <!-- Residual Add 1 -->
            <circle cx="132" cy="259" r="8" fill="#ffffff" stroke="#475569" stroke-width="1.4" />
            <text x="132" y="263" text-anchor="middle" class="t-plus-sm" fill="#0f172a">+</text>
            <!-- Residual Bypass 1 -->
            <path
              d="M 138 347 L 224 347 L 224 259 L 143 259"
              fill="none"
              stroke="#64748b"
              stroke-width="1.3"
              :marker-end="arrowDarkUrl"
            />

            <line x1="132" y1="251" x2="132" y2="243" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />

            <!-- LayerNorm 2 (Pre-LN) -->
            <rect x="56" y="218" width="152" height="22" rx="6" fill="#fef9c3" stroke="#ca8a04" :stroke-width="strokeWidthFor([5])" />
            <text x="132" y="233" text-anchor="middle" class="t-small" fill="#713f12">LayerNorm</text>
            <line x1="132" y1="218" x2="132" y2="209" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />

            <!-- Feed-Forward -->
            <rect x="56" y="179" width="152" height="27" rx="6" fill="#e0f2fe" stroke="#0284c7" :stroke-width="strokeWidthFor([5])" />
            <text x="132" y="191" text-anchor="middle" class="t-small" fill="#0c4a6e">Feed-Forward</text>
            <text x="132" y="202" text-anchor="middle" class="t-tiny" fill="#0369a1">pro Token verarbeiten</text>
            <line x1="132" y1="179" x2="132" y2="172" stroke="#475569" stroke-width="1.4" :marker-end="arrowDarkUrl" />

            <!-- Residual Add 2 -->
            <circle cx="132" cy="162" r="8" fill="#ffffff" stroke="#475569" stroke-width="1.4" />
            <text x="132" y="166" text-anchor="middle" class="t-plus-sm" fill="#0f172a">+</text>
            <!-- Residual Bypass 2 -->
            <path
              d="M 132 246 L 224 246 L 224 162 L 143 162"
              fill="none"
              stroke="#64748b"
              stroke-width="1.3"
              :marker-end="arrowDarkUrl"
            />
          </g>
        </g>

        <!-- STEP 6: Ausgabeschicht (Linear) -> Punkte (Logits) -->
        <g
          v-if="show(6)"
          :opacity="isOutputActive() ? 1 : 0.48"
          :class="!props.highlightModel && isOutputActive() && !isOverview() ? 'active-glow' : undefined"
        >
          <line x1="132" y1="154" x2="132" y2="132" stroke="#475569" stroke-width="1.6" :marker-end="arrowDarkUrl" />
          <rect
            x="44"
            y="98"
            width="188"
            height="30"
            rx="7"
            fill="#e0e7ff"
            stroke="#4f46e5"
            :stroke-width="!isOverview() && !props.highlightModel && props.step === 6 && props.hideInference ? 2.4 : 1.3"
          />
          <text x="138" y="111" text-anchor="middle" class="t-small" fill="#312e81">Ausgabeschicht (Linear)</text>
          <text x="138" y="123" text-anchor="middle" class="t-tiny" fill="#4338ca">→ Punkte pro Token (Logits)</text>
        </g>

        <!-- STEP 6, 7 & 12: Inferenz (outside, top) -->
        <g
          v-if="show(6) && !props.hideInference"
          :opacity="opacityFor([6, 7, 12])"
          :class="activeClassFor([6, 7, 12])"
        >
          <line x1="138" y1="98" x2="138" y2="63" stroke="#059669" stroke-width="1.7" :marker-end="arrowGreenUrl" />
          <rect
            x="24"
            y="10"
            width="228"
            height="49"
            rx="8"
            fill="#d1fae5"
            stroke="#059669"
            :stroke-width="strokeWidthFor([6, 7, 12])"
          />
          <text x="138" y="29" text-anchor="middle" class="t-small" fill="#065f46">Decoding / Token-Wahl (außerhalb)</text>
          <text x="138" y="43" text-anchor="middle" class="t-tiny" fill="#047857">Softmax (100 %) · Greedy / Sampling</text>
          <text v-if="show(7)" x="138" y="54" text-anchor="middle" class="t-tiny" fill="#065f46">Token anhängen → neu berechnen</text>
        </g>

        <!-- STEP 7 & 12: Autoregressive feedback loop along far left -->
        <g
          v-if="show(7)"
          :opacity="opacityFor([7, 12])"
          :class="activeClassFor([7, 12])"
        >
          <path
            d="M 24 34 L 9 34 L 9 505 L 20 505"
            fill="none"
            stroke="#059669"
            :stroke-width="strokeWidthFor([7, 12])"
            stroke-dasharray="4 3"
            :marker-end="arrowGreenUrl"
          />
        </g>

        <!-- STEP 8 & 9: Training (outside, right top) -->
        <g
          v-if="show(8)"
          :opacity="opacityFor([8, 9])"
          :class="activeClassFor([8, 9])"
        >
          <line x1="232" y1="113" x2="270" y2="113" stroke="#dc2626" stroke-width="1.6" :marker-end="arrowRedUrl" />
          <rect
            x="274"
            y="84"
            width="150"
            height="108"
            rx="8"
            fill="#fee2e2"
            stroke="#dc2626"
            :stroke-width="strokeWidthFor([8, 9])"
          />
          <text x="349" y="102" text-anchor="middle" class="t-title" fill="#991b1b">Training</text>
          <text x="349" y="119" text-anchor="middle" class="t-sub" fill="#7f1d1d">Ziel-Token + Punkte</text>
          <text x="349" y="135" text-anchor="middle" class="t-sub" fill="#7f1d1d">↓ Fehlerzahl (Loss)</text>
          <text
            x="349"
            y="151"
            text-anchor="middle"
            :class="step === 9 ? 't-sub-bold' : 't-sub'"
            :fill="step === 9 ? '#991b1b' : '#7f1d1d'"
          >↓ Backpropagation</text>
          <text
            x="349"
            y="167"
            text-anchor="middle"
            :class="step === 9 ? 't-sub-bold' : 't-sub'"
            :fill="step === 9 ? '#991b1b' : '#7f1d1d'"
          >↓ Optimizer</text>
          <text x="349" y="183" text-anchor="middle" class="t-tiny-bold" fill="#991b1b">← alle Gewichte anpassen</text>

          <!-- Weight update arrow back into GptModel -->
          <line
            x1="274"
            y1="176"
            x2="256"
            y2="176"
            stroke="#dc2626"
            :stroke-width="step === 9 ? 2.4 : 1.7"
            :marker-end="arrowRedUrl"
          />

          <!-- Backpropagation downward flow through all layers inside GptModel (step 9) -->
          <g v-if="step === 9">
            <line
              x1="245"
              y1="113"
              x2="245"
              y2="435"
              stroke="#dc2626"
              stroke-width="2"
              stroke-dasharray="4 2.5"
              :marker-end="arrowRedUrl"
            />
            <line x1="245" y1="113" x2="236" y2="113" stroke="#dc2626" stroke-width="1.6" :marker-end="arrowRedUrl" />
            <line x1="245" y1="266" x2="239" y2="266" stroke="#dc2626" stroke-width="1.6" :marker-end="arrowRedUrl" />
            <line x1="245" y1="435" x2="236" y2="435" stroke="#dc2626" stroke-width="1.6" :marker-end="arrowRedUrl" />
            <rect x="274" y="200" width="150" height="32" rx="6" fill="#fef2f2" stroke="#dc2626" stroke-width="1.2" />
            <text x="349" y="213" text-anchor="middle" class="t-tiny-bold" fill="#991b1b">↓ Backpropagation</text>
            <text x="349" y="225" text-anchor="middle" class="t-tiny" fill="#7f1d1d">Fehler durch alle Schichten</text>
          </g>
        </g>

        <!-- STEP 10: Validierung (outside, right middle) -->
        <g
          v-if="show(10)"
          :opacity="opacityFor([10])"
          :class="activeClassFor([10])"
        >
          <line
            x1="252"
            y1="266"
            x2="270"
            y2="266"
            stroke="#0891b2"
            stroke-width="1.5"
            stroke-dasharray="4 3"
            :marker-end="arrowDarkUrl"
          />
          <rect
            x="274"
            y="224"
            width="150"
            height="84"
            rx="8"
            fill="#cffafe"
            stroke="#0891b2"
            :stroke-width="strokeWidthFor([10])"
          />
          <text x="349" y="243" text-anchor="middle" class="t-title" fill="#155e75">Validierung</text>
          <text x="349" y="260" text-anchor="middle" class="t-sub" fill="#164e63">Prüfungstexte</text>
          <text x="349" y="276" text-anchor="middle" class="t-sub" fill="#164e63">Fehler messen</text>
          <text x="349" y="294" text-anchor="middle" class="t-tiny-bold" fill="#155e75">Gewichte fest · kein Update</text>
        </g>

        <!-- STEP 11: Checkpoints (outside, right bottom) -->
        <g
          v-if="show(11)"
          :opacity="opacityFor([11, 12])"
          :class="activeClassFor([11])"
        >
          <line
            x1="252"
            y1="396"
            x2="270"
            y2="396"
            :stroke="step === 12 ? '#059669' : '#7c3aed'"
            :stroke-width="step === 12 ? 2 : 1.5"
            :marker-end="step === 12 ? undefined : arrowDarkUrl"
            :marker-start="step === 12 ? arrowGreenUrl : arrowDarkUrl"
          />
          <rect
            x="274"
            y="340"
            width="150"
            height="112"
            rx="8"
            fill="#f3e8ff"
            stroke="#7c3aed"
            :stroke-width="strokeWidthFor([11])"
          />
          <text x="349" y="359" text-anchor="middle" class="t-title" fill="#5b21b6">Checkpoints</text>
          <text x="349" y="379" text-anchor="middle" class="t-sub-bold" fill="#4c1d95">last/</text>
          <text x="349" y="393" text-anchor="middle" class="t-tiny" fill="#5b21b6">Weitertrainieren (Resume)</text>
          <text
            x="349"
            y="416"
            text-anchor="middle"
            class="t-sub-bold"
            :fill="step === 12 ? '#047857' : '#4c1d95'"
          >best/</text>
          <text
            x="349"
            y="430"
            text-anchor="middle"
            class="t-tiny"
            :fill="step === 12 ? '#047857' : '#5b21b6'"
          >Anwenden (bester Prüffehler)</text>
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
  max-width: 324px;
  max-height: 396px;
  height: auto;
  display: block;
}

.llm-arch-svg--compact {
  max-width: 198px;
  max-height: 396px;
  font-family: Arial, sans-serif;
}

.active-glow {
  filter: drop-shadow(0 2px 2.5px rgba(15, 23, 42, 0.16));
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

.t-plus {
  font-size: 15px;
  font-weight: 700;
}

.t-plus-sm {
  font-size: 13px;
  font-weight: 700;
}
</style>
