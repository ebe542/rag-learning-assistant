const form = document.querySelector("[data-study-form]");

if (form) {
  const button = form.querySelector("[data-submit-button]");
  const status = form.querySelector("[data-submission-status]");

  const resetSubmissionState = () => {
    form.removeAttribute("aria-busy");
    button.disabled = false;
    button.textContent = "Evaluate answer";
    status.hidden = true;
  };

  form.addEventListener("submit", () => {
    form.setAttribute("aria-busy", "true");
    button.disabled = true;
    button.textContent = "Evaluating…";
    status.hidden = false;
  });

  window.addEventListener("pageshow", resetSubmissionState);
}
