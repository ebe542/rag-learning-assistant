const refreshIntervalMilliseconds = 2000;

const refreshPackageStatus = async () => {
  const region = document.querySelector("#package-status-region");
  if (!region || region.dataset.refresh !== "true") {
    return;
  }

  try {
    const response = await fetch("/library/packages/status", {
      headers: { "X-Requested-With": "package-status" },
    });
    if (response.ok) {
      region.outerHTML = await response.text();
    }
  } finally {
    const updatedRegion = document.querySelector("#package-status-region");
    if (updatedRegion?.dataset.refresh === "true") {
      window.setTimeout(refreshPackageStatus, refreshIntervalMilliseconds);
    }
  }
};

window.setTimeout(refreshPackageStatus, refreshIntervalMilliseconds);
