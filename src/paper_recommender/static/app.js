const form = document.querySelector("#recommend-form");
const indexStatusNode = document.querySelector("#index-status");
const statusNode = document.querySelector("#status");
const resultsNode = document.querySelector("#results");
const template = document.querySelector("#result-template");

function setStatus(message) {
  statusNode.textContent = message;
}

async function parseJsonOrNull(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function normalizeErrorDetail(detail) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const itemWithMessage = detail.find((item) => item && typeof item.msg === "string");
    if (itemWithMessage) {
      return itemWithMessage.msg;
    }
  }

  return "Invalid request";
}

function errorStatusText(body) {
  if (body && body.detail !== undefined) {
    return normalizeErrorDetail(body.detail);
  }

  return "Request failed";
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function formatIndexStatus(status) {
  const papers = formatCount(status.indexed_papers);
  const kind = status.index_kind || "unknown";
  const datestamp = status.last_oai_datestamp || "unknown";
  return `${papers} papers | ${kind} | OAI through ${datestamp}`;
}

async function loadIndexStatus() {
  try {
    const response = await fetch("/api/status");
    const body = await parseJsonOrNull(response);
    if (!response.ok || !body) {
      return;
    }
    indexStatusNode.textContent = formatIndexStatus(body);
  } catch {
    indexStatusNode.textContent = "Index status unavailable";
  }
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
      setStatus(errorStatusText(body));
      return;
    }

    renderResults((body && body.results) || []);
  } catch (error) {
    setStatus("Request failed");
  }
});

loadIndexStatus();
