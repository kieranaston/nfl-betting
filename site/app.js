const DATA = "data";
const PICK_WINDOW_MS = 48 * 60 * 60 * 1000;

const CHART_MUTED = "#6b7280";
const CHART_GRID = "#e5e7eb";
const CHART_ACCENT = "#2563eb";
const CHART_SECONDARY = "#0f766e";

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.error(`Failed to load ${path}`, err);
    return null;
  }
}

function fmtNum(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function fmtPct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtGap(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  return (n > 0 ? "+" : "") + n.toFixed(1);
}

function withinNext48Hours(pick) {
  if (!pick?.commence_time) return false;
  const kickoff = Date.parse(pick.commence_time);
  if (Number.isNaN(kickoff)) return false;
  const now = Date.now();
  return kickoff >= now && kickoff <= now + PICK_WINDOW_MS;
}

async function init() {
  const summary = await loadJSON(`${DATA}/summary.json`);
  if (summary) {
    renderSummary(summary);
    renderCharts(summary);
  }

  const picks = await findLatestPicks();
  if (picks) renderPicks(picks);
  else {
    document.getElementById("picks-meta").textContent = "No picks file found";
    document.getElementById("picks-body").innerHTML =
      `<tr><td colspan="10" class="empty">No picks yet</td></tr>`;
  }
}

async function findLatestPicks() {
  const latest = await loadJSON(`${DATA}/latest_picks.json`);
  if (latest) return latest;

  for (const week of [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) {
    for (const season of [2026, 2025, 2024]) {
      const path = `${DATA}/week_${season}_${String(week).padStart(2, "0")}_picks.json`;
      const data = await loadJSON(path);
      if (data) return data;
    }
  }
  return null;
}

function renderSummary(s) {
  document.getElementById("mae").textContent = fmtNum(s.mae);
  document.getElementById("bias").textContent = fmtNum(s.bias);
  document.getElementById("brier").textContent = fmtNum(s.brier);
  document.getElementById("log-loss").textContent = fmtNum(s.log_loss);
}

function bookCell(label, href, title) {
  const text = label || "—";
  const titleAttr = title ? ` title="${title}"` : "";
  if (!href) return `<td class="book-cell"${titleAttr}>${text}</td>`;
  return `<td class="book-cell"${titleAttr}><a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a></td>`;
}

function renderPicks(data) {
  const tbody = document.getElementById("picks-body");
  const allPicks = data.picks || [];
  const picks = allPicks.filter(withinNext48Hours);
  const meta = document.getElementById("picks-meta");

  const when = data.generated_at
    ? new Date(data.generated_at).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : null;
  meta.textContent = [
    data.season != null && data.week != null ? `${data.season} · Week ${data.week}` : null,
    `${picks.length} pick${picks.length === 1 ? "" : "s"} in next 48h`,
    allPicks.length !== picks.length ? `${allPicks.length} week total` : null,
    data.min_side_prob != null ? `min model ${(data.min_side_prob * 100).toFixed(0)}%` : null,
    when ? `updated ${when}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  if (!picks.length) {
    const emptyMsg = allPicks.length
      ? "No picks for games within the next 48 hours"
      : "No picks this week (none cleared the model-confidence threshold)";
    tbody.innerHTML = `<tr><td colspan="10" class="empty">${emptyMsg}</td></tr>`;
    return;
  }

  tbody.innerHTML = picks
    .map((p) => {
      const modelProb = p.model_prob;
      const gap = p.mu_gap ?? (p.model_mu != null && p.line != null ? p.model_mu - p.line : null);
      const price =
        p.price == null ? "—" : p.price > 0 ? `+${p.price}` : `${p.price}`;
      const team = p.team ? `<span class="player-label-team">${p.team}</span>` : "";
      const lineBook = p.line_book_title || p.line_book || "—";
      const priceBook = p.price_book_title || p.price_book || "—";
      const lineHint =
        p.num_books != null
          ? `Median line across ${p.num_books} books; shown book posts this line`
          : "";
      return `
    <tr>
      <td>
        <span class="player-label">
          <span class="player-label-name">${p.player}</span>
          ${team}
        </span>
      </td>
      <td>${p.position}</td>
      <td class="num">${p.line}</td>
      ${bookCell(lineBook, p.line_link, lineHint)}
      <td class="pick-${p.pick}">${p.pick.toUpperCase()}</td>
      <td class="num ${modelProb != null && modelProb >= 0.55 ? "ev-pos" : ""}">${fmtPct(modelProb)}</td>
      <td class="num">${fmtGap(gap)}</td>
      <td class="num">${p.model_mu}</td>
      <td class="num">${price}</td>
      ${bookCell(priceBook, p.price_link)}
      <td>${p.matchup || ""}</td>
    </tr>`;
    })
    .join("");
}

function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: CHART_MUTED, maxRotation: 45, font: { size: 11 } },
        grid: { color: CHART_GRID },
        border: { color: CHART_GRID },
      },
      y: {
        ticks: { color: CHART_MUTED, font: { size: 11 } },
        grid: { color: CHART_GRID },
        border: { color: CHART_GRID },
      },
    },
  };
}

function renderLineChart(canvasId, emptyId, series, valueKey, color, yLabel) {
  const canvas = document.getElementById(canvasId);
  const empty = document.getElementById(emptyId);
  if (!series.length) {
    canvas.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  canvas.hidden = false;
  const defaults = chartDefaults();
  new Chart(canvas, {
    type: "line",
    data: {
      labels: series.map((d) => d.week),
      datasets: [
        {
          data: series.map((d) => d[valueKey]),
          borderColor: color,
          backgroundColor: color === CHART_ACCENT ? "rgba(37, 99, 235, 0.08)" : "rgba(15, 118, 110, 0.08)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          pointBackgroundColor: color,
        },
      ],
    },
    options: {
      ...defaults,
      scales: {
        ...defaults.scales,
        y: {
          ...defaults.scales.y,
          title: { display: true, text: yLabel, color: CHART_MUTED, font: { size: 11 } },
        },
      },
    },
  });
}

function renderCharts(summary) {
  renderLineChart(
    "mae-chart",
    "mae-empty",
    summary.cumulative_mae || [],
    "mae",
    CHART_SECONDARY,
    "MAE"
  );
  renderLineChart(
    "brier-chart",
    "brier-empty",
    summary.cumulative_brier || [],
    "brier",
    CHART_ACCENT,
    "Brier"
  );
}

init();
