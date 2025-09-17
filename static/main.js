document.getElementById("sendBtn").addEventListener("click", async () => {
  const fIn = document.getElementById("fileInput");
  const aIn = document.getElementById("alignInput");
  const file = fIn.files[0];
  const alignFile = aIn.files[0];
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
  if (alignFile) form.append("align", alignFile);
  try {
    const resp = await fetch("/predict", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) {
      status.innerText = "Error: " + data.error;
    } else {
      status.innerText = "Done";
      let out = "Hypothesis: " + (data.hypothesis || "<empty>") + "\n";
      if (data.ground_truth !== undefined) {
        out += "Ground truth: " + (data.ground_truth || "<empty>") + "\n";
      }
      if (data.wer !== undefined) {
        out += "WER: " + data.wer;
      }
      result.innerText = out;
    }
  } catch (err) {
    status.innerText = "Upload failed: " + err;
  }
});
