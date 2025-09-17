document.getElementById("sendBtn").addEventListener("click", async () => {
  const fIn = document.getElementById("fileInput");
  const file = fIn.files[0];
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  result.innerText = "";
  if (!file) {
    status.innerText = "Select a video file first.";
    return;
  }
  status.innerText = "Uploading...";
  const form = new FormData();
  form.append("video", file);
  try {
    const resp = await fetch("/predict", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) {
      status.innerText = "Error: " + data.error;
    } else {
      status.innerText = "Done";
      result.innerText = "Hypothesis: " + data.hypothesis;
    }
  } catch (err) {
    status.innerText = "Upload failed: " + err;
  }
});
