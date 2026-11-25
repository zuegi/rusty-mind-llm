<script setup lang="ts">
import { computed } from 'vue'

type Phase = 'inference' | 'training' | 'validation' | 'overview' | 'summary'

const props = defineProps<{
  phase: Phase
  label?: string
}>()

const config: Record<Phase, { text: string; dot: string; box: string }> = {
  inference: {
    text: 'Laufendes Modell · Inferenz',
    dot: '#10b981',
    box: 'background:#ecfdf5;color:#065f46;border-color:#a7f3d0;',
  },
  training: {
    text: 'Training · Modell lernt',
    dot: '#ef4444',
    box: 'background:#fef2f2;color:#991b1b;border-color:#fecaca;',
  },
  validation: {
    text: 'Validierung · Prüfung',
    dot: '#0891b2',
    box: 'background:#ecfeff;color:#155e75;border-color:#a5f3fc;',
  },
  overview: {
    text: 'Überblick · Zusammenspiel',
    dot: '#64748b',
    box: 'background:#f8fafc;color:#334155;border-color:#cbd5e1;',
  },
  summary: {
    text: 'Fazit & Ausblick',
    dot: '#4f46e5',
    box: 'background:#eef2ff;color:#3730a3;border-color:#c7d2fe;',
  },
}

const active = computed(() => config[props.phase])
</script>

<template>
  <div class="phase-badge" :style="active.box">
    <span class="phase-dot" :style="{ backgroundColor: active.dot }" />
    <span>{{ label ?? active.text }}</span>
  </div>
</template>

<style scoped>
.phase-badge {
  position: absolute;
  top: 22px;
  right: 32px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  border: 1px solid;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  line-height: 1.2;
}

.phase-dot {
  width: 7px;
  height: 7px;
  border-radius: 9999px;
  flex-shrink: 0;
}
</style>
