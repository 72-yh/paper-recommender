const form = document.querySelector("#recommend-form");
const statusNode = document.querySelector("#status");
const resultsNode = document.querySelector("#results");
const template = document.querySelector("#result-template");

function setStatus(message) {
  statusNode.textContent = message;
}

async function parseJsonOrNull(response) {
  try {
    return await response.json();
  } catch (error) {
    return null;
  }
}

function normalizeErrorDetail(body, fallbackMessage) {
  const detail = body && body.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const firstMessage = detail.find((item) => item && typeof item.msg === "string")?.msg;
    return firstMessage || "Invalid request";
  }

  return fallbackMessage;
}

function renderResults(results) {
  resultsNode.replaceChildren();

  if (results.length === 0) {
    setStatus("No results");
    return;
  }

  setStatus("");
  for (const result of results) {
    const item = template.content.cloneNode(true);
    const publishedDate = result.published_date || "Unknown date";
    const score = Number(result.similarity_score);

    item.querySelector(".paper-id").textContent = result.arxiv_id;
    item.querySelector(".score").textContent = Number.isFinite(score) ? score.toFixed(3) : "";
    item.querySelector(".meta").textContent = `${result.primary_category} | ${publishedDate}`;
    item.querySelector(".arxiv-link").href = result.url;
    resultsNode.appendChild(item);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Searching...");
  resultsNode.replaceChildren();

  const formData = new FormData(form);
  const payload = {
    url: formData.get("url"),
    category: formData.get("category") || null,
    date_from: formData.get("date_from") || null,
    date_to: formData.get("date_to") || null,
    top_k: Number(formData.get("top_k") || 10),
  };

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await parseJsonOrNull(response);

    if (!response.ok) {
      setStatus(normalizeErrorDetail(body, response.status === 422 ? "Invalid request" : "Request failed"));
      return;
    }

    renderResults((body && body.results) || []);
  } catch (error) {
    setStatus("Request failed");
  }
});
