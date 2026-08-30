const uploadForm = document.querySelector(".package-create-form");
const uploadError = document.querySelector("[data-upload-error]");

if (uploadForm && uploadError) {
  for (const field of uploadForm.querySelectorAll('input[name="name"], input[name="pdf"]')) {
    const clearUploadError = () => uploadError.remove();
    field.addEventListener("input", clearUploadError, { once: true });
    field.addEventListener("change", clearUploadError, { once: true });
  }
}
