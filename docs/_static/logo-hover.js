document.addEventListener("DOMContentLoaded", () => {
  const logo = document.querySelector(".navbar .logo");
  const logoImg = logo?.querySelector(".logo-img");
  if (!logo || !logoImg) {
    return;
  }

  let hoverTimer = null;

  const clearHoverTimer = () => {
    if (hoverTimer) {
      clearTimeout(hoverTimer);
      hoverTimer = null;
    }
  };

  logo.addEventListener("mouseenter", () => {
    clearHoverTimer();
    hoverTimer = setTimeout(() => {
      logoImg.classList.add("logo-img-zoomed");
    }, 1500);
  });

  logo.addEventListener("mouseleave", () => {
    clearHoverTimer();
    logoImg.classList.remove("logo-img-zoomed");
  });
});
