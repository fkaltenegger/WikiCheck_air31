const radios = document.querySelectorAll("input[name='model']");

chrome.storage.sync.get(["model"], (result) => {
  const model = result.model || "sbert";
  document.querySelector("input[value='" + model + "']").checked = true;
});

radios.forEach((radio) => {
  radio.addEventListener("change", () => {
    chrome.storage.sync.set({ model: radio.value });
  });
});
