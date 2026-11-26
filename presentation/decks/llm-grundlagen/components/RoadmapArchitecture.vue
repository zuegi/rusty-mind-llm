<script lang="ts">
let roadmapUid = 0
</script>

<script setup lang="ts">
const uid = `roadmap-${++roadmapUid}`
const arrowDarkId = `${uid}-arrow-dark`
const arrowGreenId = `${uid}-arrow-green`
const arrowBlueId = `${uid}-arrow-blue`
const arrowPurpleId = `${uid}-arrow-purple`
const arrowDarkUrl = `url(#${arrowDarkId})`
const arrowGreenUrl = `url(#${arrowGreenId})`
const arrowBlueUrl = `url(#${arrowBlueId})`
const arrowPurpleUrl = `url(#${arrowPurpleId})`
</script>

<template>
  <div class="roadmap-arch-wrapper">
    <svg
      viewBox="0 0 434 510"
      width="434"
      height="510"
      class="roadmap-arch-svg"
      role="img"
      aria-label="Ausbau-Architektur für Kapitel 19 bis 28"
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
          :id="arrowBlueId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7" />
        </marker>
        <marker
          :id="arrowPurpleId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed" />
        </marker>
      </defs>

      <!-- Legend -->
      <g>
        <rect x="64" y="10" width="14" height="10" rx="2.5" fill="#f8fafc" stroke="#334155" stroke-width="1.8" />
        <text x="84" y="19" class="t-legend" fill="#334155">Umgesetzt (Kap. 1–18)</text>

        <rect x="230" y="10" width="14" height="10" rx="2.5" fill="#ffffff" stroke="#64748b" stroke-width="1.5" stroke-dasharray="3 2" />
        <text x="250" y="19" class="t-legend" fill="#475569">Geplant (Kap. 19–28)</text>
      </g>

      <!-- LAYER 1: Betrieb, Sicherheit & Evaluation (Kap. 27-28) + Eigener Chat (Kap. 23-24) -->
      <g>
        <rect
          x="14"
          y="36"
          width="406"
          height="118"
          rx="12"
          fill="#fefce8"
          stroke="#ca8a04"
          stroke-width="1.5"
          stroke-dasharray="6 4"
        />
        <text x="26" y="54" class="t-badge" fill="#854d0e">Evaluation · Sicherheit · Betrieb</text>
        <text x="408" y="54" text-anchor="end" class="t-kap" fill="#a16207">Kap. 27–28</text>

        <!-- Inner Chat Box -->
        <rect
          x="28"
          y="64"
          width="378"
          height="78"
          rx="9"
          fill="#d1fae5"
          stroke="#059669"
          stroke-width="1.6"
          stroke-dasharray="5 3"
        />
        <text x="42" y="83" class="t-title" fill="#065f46">Eigener Chat</text>
        <text x="394" y="83" text-anchor="end" class="t-kap" fill="#047857">Kap. 23–24</text>

        <!-- 3 Interface Chips -->
        <rect x="42" y="95" width="106" height="32" rx="6" fill="#ffffff" stroke="#10b981" stroke-width="1.2" />
        <text x="95" y="115" text-anchor="middle" class="t-small" fill="#065f46">Terminal-Chat</text>

        <rect x="164" y="95" width="106" height="32" rx="6" fill="#ffffff" stroke="#10b981" stroke-width="1.2" />
        <text x="217" y="115" text-anchor="middle" class="t-small" fill="#065f46">HTTP-API</text>

        <rect x="286" y="95" width="106" height="32" rx="6" fill="#ffffff" stroke="#10b981" stroke-width="1.2" />
        <text x="339" y="115" text-anchor="middle" class="t-small" fill="#065f46">Web-UI</text>
      </g>

      <!-- Arrows up to Eigener Chat -->
      <line x1="121" y1="184" x2="121" y2="145" stroke="#059669" stroke-width="1.8" :marker-end="arrowGreenUrl" />
      <line x1="335" y1="184" x2="335" y2="145" stroke="#0284c7" stroke-width="1.6" stroke-dasharray="4 3" :marker-end="arrowBlueUrl" />

      <!-- LAYER 2 LEFT: Existing Core (Kap. 1-18, SOLID) -->
      <g>
        <rect
          x="14"
          y="184"
          width="214"
          height="168"
          rx="12"
          fill="#f8fafc"
          stroke="#334155"
          stroke-width="2.2"
        />
        <text x="26" y="204" class="t-title" fill="#0f172a">rusty-mind-llm Kern</text>
        <rect x="160" y="191" width="58" height="17" rx="8.5" fill="#0f172a" />
        <text x="189" y="203" text-anchor="middle" class="t-pill" fill="#ffffff">Kap. 1–18</text>

        <!-- Sub-blocks inside core -->
        <rect x="26" y="218" width="190" height="44" rx="7" fill="#ffffff" stroke="#64748b" stroke-width="1.3" />
        <text x="121" y="236" text-anchor="middle" class="t-small" fill="#0f172a">GptModel + Tokenizer</text>
        <text x="121" y="252" text-anchor="middle" class="t-tiny" fill="#475569">Embeddings · N× Transformer · Logits</text>

        <rect x="26" y="272" width="90" height="32" rx="6" fill="#fee2e2" stroke="#dc2626" stroke-width="1.2" />
        <text x="71" y="292" text-anchor="middle" class="t-tiny-bold" fill="#991b1b">Training</text>

        <rect x="126" y="272" width="90" height="32" rx="6" fill="#cffafe" stroke="#0891b2" stroke-width="1.2" />
        <text x="171" y="292" text-anchor="middle" class="t-tiny-bold" fill="#155e75">Validierung</text>

        <rect x="26" y="312" width="190" height="28" rx="6" fill="#f3e8ff" stroke="#7c3aed" stroke-width="1.2" />
        <text x="121" y="330" text-anchor="middle" class="t-tiny-bold" fill="#5b21b6">Checkpoints (last/ &amp; best/) · Inferenz</text>
      </g>

      <!-- LAYER 2 RIGHT: RAG (Kap. 25) + Fine-Tuning (Kap. 26) -->
      <g>
        <!-- RAG -->
        <rect
          x="250"
          y="184"
          width="170"
          height="74"
          rx="10"
          fill="#e0f2fe"
          stroke="#0284c7"
          stroke-width="1.6"
          stroke-dasharray="5 3"
        />
        <text x="262" y="203" class="t-title" fill="#0c4a6e">RAG</text>
        <text x="408" y="203" text-anchor="end" class="t-kap" fill="#0369a1">Kap. 25</text>
        <text x="262" y="222" class="t-sub" fill="#075985">Eigenes Wissen</text>
        <text x="262" y="238" class="t-tiny" fill="#0369a1">Dokumente suchen &amp; als</text>
        <text x="262" y="250" class="t-tiny" fill="#0369a1">Kontext mitgeben</text>

        <!-- Fine-/Instruction-Tuning -->
        <rect
          x="250"
          y="278"
          width="170"
          height="74"
          rx="10"
          fill="#ffedd5"
          stroke="#ea580c"
          stroke-width="1.6"
          stroke-dasharray="5 3"
        />
        <text x="262" y="297" class="t-title" fill="#7c2d12">Feinschliff</text>
        <text x="408" y="297" text-anchor="end" class="t-kap" fill="#c2410c">Kap. 26</text>
        <text x="262" y="316" class="t-sub" fill="#9a3412">Fine-/Instruction-Tuning</text>
        <text x="262" y="332" class="t-tiny" fill="#9a3412">Frage-Antwort-Stil &amp;</text>
        <text x="262" y="344" class="t-tiny" fill="#9a3412">Aufgaben gezielt trainieren</text>

        <!-- Arrow Fine-Tuning -> Core -->
        <line x1="250" y1="315" x2="231" y2="315" stroke="#ea580c" stroke-width="1.6" stroke-dasharray="4 3" :marker-end="arrowDarkUrl" />
      </g>

      <!-- LAYER 3 BOTTOM: Daten & Tempo (Kap. 19-20) + GPT-2 Import (Kap. 21-22) -->
      <g>
        <!-- Daten & Tempo -->
        <rect
          x="14"
          y="396"
          width="192"
          height="94"
          rx="10"
          fill="#e0e7ff"
          stroke="#4f46e5"
          stroke-width="1.6"
          stroke-dasharray="5 3"
        />
        <text x="26" y="416" class="t-title" fill="#312e81">Daten &amp; Tempo</text>
        <text x="194" y="416" text-anchor="end" class="t-kap" fill="#4338ca">Kap. 19–20</text>
        <text x="26" y="436" class="t-sub" fill="#3730a3">• Saubere Texte</text>
        <text x="26" y="453" class="t-sub" fill="#3730a3">• Unabhängiges Testset</text>
        <text x="26" y="470" class="t-sub" fill="#3730a3">• Laufzeit &amp; Speicher</text>

        <!-- Arrow Daten & Tempo -> Core -->
        <line x1="110" y1="396" x2="110" y2="355" stroke="#475569" stroke-width="1.6" stroke-dasharray="4 3" :marker-end="arrowDarkUrl" />

        <!-- Fertiges Wissen laden (GPT-2) -->
        <rect
          x="228"
          y="396"
          width="192"
          height="94"
          rx="10"
          fill="#f3e8ff"
          stroke="#7c3aed"
          stroke-width="1.6"
          stroke-dasharray="5 3"
        />
        <text x="240" y="416" class="t-title" fill="#5b21b6">Fertiges Wissen</text>
        <text x="408" y="416" text-anchor="end" class="t-kap" fill="#6d28d9">Kap. 21–22</text>
        <text x="240" y="436" class="t-sub" fill="#4c1d95">• GPT-2-Gewichte laden</text>
        <text x="240" y="453" class="t-sub" fill="#4c1d95">• Architektur-Abgleich</text>
        <text x="240" y="470" class="t-sub" fill="#4c1d95">• Start mit Sprachwissen</text>

        <!-- Arrow GPT-2 -> Core -->
        <path
          d="M 324 396 L 324 374 L 184 374 L 184 355"
          fill="none"
          stroke="#7c3aed"
          stroke-width="1.6"
          stroke-dasharray="4 3"
          :marker-end="arrowPurpleUrl"
        />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.roadmap-arch-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
}

.roadmap-arch-svg {
  width: 100%;
  max-width: 353px;
  height: auto;
  display: block;
}

.t-title {
  font-size: 12px;
  font-weight: 700;
}

.t-small {
  font-size: 11px;
  font-weight: 600;
}

.t-sub {
  font-size: 10.5px;
  font-weight: 500;
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
  font-size: 10.5px;
  font-weight: 700;
}

.t-kap {
  font-size: 9.5px;
  font-weight: 700;
}

.t-pill {
  font-size: 9px;
  font-weight: 700;
}

.t-legend {
  font-size: 10px;
  font-weight: 600;
}
</style>
