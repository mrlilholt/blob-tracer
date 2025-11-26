// frontend/main.js

// Change this to your deployed backend URL:
const BACKEND_URL = "https://blob-tracer-backend.onrender.com";

const form = document.getElementById("blobForm");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("resultSection");
const outputVideo = document.getElementById("outputVideo");
const downloadLink = document.getElementById("downloadLink");
const submitBtn = document.getElementById("submitBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("videoFile");
  const maxBoxesInput = document.getElementById("maxBoxes");
  const minAreaInput = document.getElementById("minArea");

  if (!fileInput.files.length) {
    statusEl.textContent = "Please choose a video file.";
    return;
  }

  const file = fileInput.files[0];
  const maxBoxes = maxBoxesInput.value || "";
  const minArea = minAreaInput.value || 50;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_boxes", maxBoxes);
  formData.append("min_area", minArea);

  statusEl.textContent = "Uploading and processing video...";
  submitBtn.disabled = true;
  resultSection.classList.add("hidden");

  try {
    const response = await fetch(`${BACKEND_URL}/process-video`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    // Show video
    outputVideo.src = url;
    outputVideo.load();

    // Set download link
    downloadLink.href = url;

    resultSection.classList.remove("hidden");
    statusEl.textContent = "Done!";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error processing video. Check console for details.";
  } finally {
    submitBtn.disabled = false;
  }
});
