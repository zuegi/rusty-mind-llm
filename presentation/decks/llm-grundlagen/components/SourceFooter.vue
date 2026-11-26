<script setup lang="ts">
defineProps<{
  chapter: string
  code?: string
}>()
</script>

<template>
  <div class="source-footer">
    <span>Buch: {{ chapter }}</span>
    <template v-if="code">
      <span class="sep">·</span>
      <code>{{ code }}</code>
    </template>
  </div>
</template>

<style scoped>
.source-footer {
  position: absolute;
  bottom: 0.55rem;
  left: 3.5rem;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 10.5px;
  color: #94a3b8;
  line-height: 1;
  z-index: 5;
  pointer-events: none;
}

.source-footer code {
  font-size: 10px;
  color: #64748b;
  background: transparent;
  padding: 0;
}

.sep {
  opacity: 0.6;
}
</style>
