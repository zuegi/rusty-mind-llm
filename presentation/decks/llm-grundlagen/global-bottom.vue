<template>
  <img
    src="/avatar.png"
    alt="René Weishaupt"
    class="slide-avatar"
  />
</template>

<style scoped>
.slide-avatar {
  position: fixed;
  right: 1rem;
  bottom: 0.75rem;
  width: 44px;
  height: 44px;
  border-radius: 9999px;
  object-fit: cover;
  object-position: center 20%;
  border: 2px solid #ffffff;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.18);
  z-index: 20;
  pointer-events: none;
}
</style>
