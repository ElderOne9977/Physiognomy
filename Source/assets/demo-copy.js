// Event delegation survives report updates. Read text only; never insert model markup.
element.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-copy]");
  if (!button || !element.contains(button)) return;
  const feedback = element.querySelector(".copy-feedback");
  const cards = button.dataset.copy === "section"
    ? [button.closest(".reading-card")]
    : [...element.querySelectorAll(".reading-card")];
  const text = cards.map((card) => {
    const body = card.querySelector(".report-text");
    return body.dataset.available === "true"
      ? `${card.querySelector(".report-title").textContent}:\n${body.textContent}`
      : `${card.querySelector(".report-title").textContent}:\n`;
  }).join("\n\n");
  try {
    await navigator.clipboard.writeText(text);
    if (feedback) feedback.textContent = "Đã sao chép vào clipboard.";
  } catch (_) {
    if (feedback) feedback.textContent = "Trình duyệt chưa cho phép clipboard. Bạn có thể chọn văn bản để sao chép.";
  }
});
