let popup = null;
let query = "";

document.addEventListener("mouseup", (e) => {
  if (popup) return;

  const selection = window.getSelection();
  query = selection.toString().trim();

  if (!query) return;

  const rect = selection.getRangeAt(0).getBoundingClientRect();
  createPopup(rect);
});

document.addEventListener("mousedown", (e) => {
  if (popup && !popup.contains(e.target)) {
    destroyPopup();
  }
});

function createPopup(rect) {
  const container = document.createElement("div");
  container.style.position = "absolute";
  container.style.top = (window.scrollY + rect.bottom + 8) + "px";
  container.style.left = (window.scrollX + rect.left) + "px";
  container.style.zIndex = 999999;
  container.style.background = "#ffffff";
  container.style.border = "1px solid #cccccc";
  container.style.borderRadius = "6px";
  container.style.padding = "8px";
  container.style.minWidth = "120px";
  container.style.boxShadow = "0 4px 12px #00000022";
  container.style.fontFamily = "Arial, sans-serif";

  const button = document.createElement("button");
  button.textContent = "WikiCheck";
  button.style.width = "100%";

  const result = document.createElement("div");
  result.style.display = "none";
  result.style.marginTop = "4px";
  result.style.fontSize  = "16px";
  result.style.lineHeight = "1.4";
  result.style.maxWidth = "50vw";

  button.onclick = async () => {
    button.disabled = true;
    button.textContent = "Checking...";

    try {
      const res = await fetch("http://127.0.0.1:8000/check", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({"method": "smbert", "query": query})
      });
      const data = await res.json();
      let topResult = data[0];

      const answer = document.createElement("span");
      answer.textContent = topResult.paragraph + " ";
      
      const link = document.createElement("a");
      link.textContent = "[" + topResult.title + "]";
      link.href = topResult.url;
      link.target = "_blank";
      link.style.fontWeight = "bold";

      result.appendChild(answer);
      result.appendChild(link);
    } catch (err) {
      result.textContent = "Error calling server: " + err;
    }

    button.style.display = "none";
    result.style.display = "block";
  };

  container.appendChild(button);
  container.appendChild(result);

  popup = container;
  document.body.appendChild(popup);
}

function destroyPopup() {
  if (popup) {
    popup.remove();
    popup = null;
  }
}
