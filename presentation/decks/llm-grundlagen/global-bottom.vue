<template>
  <div class="rust-stripe" aria-hidden="true" />
  <img
    src="/avatar.png"
    alt="René Weishaupt"
    class="slide-avatar"
  />
</template>

<style scoped>
.rust-stripe {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 6px;
  background: linear-gradient(180deg, #b7410e 0%, #9a3412 55%, #7c2d12 100%);
  z-index: 20;
  pointer-events: none;
}

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
