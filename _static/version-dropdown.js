document.addEventListener("DOMContentLoaded", () => {
  const versionMenus = document.querySelectorAll("details.rst-versions");
  if (!versionMenus.length) {
    return;
  }

  document.addEventListener("click", (event) => {
    versionMenus.forEach((menu) => {
      if (menu.open && !menu.contains(event.target)) {
        menu.open = false;
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      versionMenus.forEach((menu) => {
        menu.open = false;
      });
    }
  });
});
