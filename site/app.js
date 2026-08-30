const DATA = "data";

const CHART_MUTED = "#6b7280";
const CHART_GRID = "#e5e7eb";
const CHART_ACCENT = "#2563eb";
const CHART_SUCCESS = "#059669";
const CHART_DANGER = "#dc2626";

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

function fmtPct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtEV(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

async function init() {
  const summary = await loadJSON(`${DATA}/summary.json`);
  if (summary) renderSummary(summary);

  const picks = await findLatestPicks();
  if (picks) renderPicks(picks);
  else {
    document.getElementById("picks-meta").textContent = "No picks file found";
    document.getElementById("picks-body").innerHTML =
      `<tr><td colspan="10" class="empty">No picks yet</td></tr>`;
  }

  if (summary) renderCharts(summary);
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
  document.getElementById("record").textContent =
    `${s.total_wins ?? 0}-${s.total_losses ?? 0}`;
  document.getElementById("win-pct").textContent = fmtPct(s.win_pct);

  const briers = (s.cumulative_brier || []).map((d) => d.brier).filter(Boolean);
  const avgBrier = briers.length
    ? (briers.reduce((a, b) => a + b, 0) / briers.length).toFixed(3)
    : "—";
  document.getElementById("avg-brier").textContent = avgBrier;

  const clvs = (s.cumulative_clv || []).map((d) => d.clv).filter((v) => v != null);
  const avgClv = clvs.length
    ? (clvs.reduce((a, b) => a + b, 0) / clvs.length).toFixed(2)
    : "—";
  document.getElementById("avg-clv").textContent = avgClv;
}

function renderPicks(data) {
  const tbody = document.getElementById("picks-body");
  const picks = data.picks || [];
  const meta = document.getElementById("picks-meta");

  const when = data.generated_at
    ? new Date(data.generated_at).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : null;
  meta.textContent = [
    data.season != null && data.week != null ? `${data.season} · Week ${data.week}` : null,
    `${picks.length} pick${picks.length === 1 ? "" : "s"}`,
    data.min_ev != null ? `min EV ${(data.min_ev * 100).toFixed(0)}%` : null,
    when ? `updated ${when}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  if (!picks.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty">No picks this week (no edge above threshold)</td></tr>`;
    return;
  }

  tbody.innerHTML = picks
    .map((p) => {
      const ev = p.ev ?? p.edge;
      const price = p.price > 0 ? `+${p.price}` : `${p.price}`;
      const team = p.team ? `<span class="player-label-team">${p.team}</span>` : "";
      const lineBook = p.line_book_title || p.line_book || "—";
      const priceBook = p.price_book_title || p.price_book || "—";
      const lineTitle =
        p.num_books != null
          ? `title="Median line across ${p.num_books} books; shown book posts this line"`
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
      <td class="book-cell" ${lineTitle}>${lineBook}</td>
      <td class="pick-${p.pick}">${p.pick.toUpperCase()}</td>
      <td class="num ${ev != null && ev >= 0 ? "ev-pos" : ""}">${fmtEV(ev)}</td>
      <td class="num">${p.model_mu}</td>
      <td class="num">${price}</td>
      <td class="book-cell">${priceBook}</td>
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

function renderCharts(summary) {
  const brierData = summary.cumulative_brier || [];
  const clvData = summary.cumulative_clv || [];
  const defaults = chartDefaults();

  if (!brierData.length) {
    document.getElementById("brier-chart").hidden = true;
    document.getElementById("brier-empty").hidden = false;
  } else {
    new Chart(document.getElementById("brier-chart"), {
      type: "line",
      data: {
        labels: brierData.map((d) => d.week),
        datasets: [
          {
            data: brierData.map((d) => d.brier),
            borderColor: CHART_ACCENT,
            backgroundColor: "rgba(37, 99, 235, 0.08)",
            fill: true,
            tension: 0.3,
            pointRadius: 3,
            pointBackgroundColor: CHART_ACCENT,
          },
        ],
      },
      options: {
        ...defaults,
        scales: {
          ...defaults.scales,
          y: {
            ...defaults.scales.y,
            title: { display: true, text: "Brier", color: CHART_MUTED, font: { size: 11 } },
          },
        },
      },
    });
  }

  if (!clvData.length) {
    document.getElementById("clv-chart").hidden = true;
    document.getElementById("clv-empty").hidden = false;
  } else {
    new Chart(document.getElementById("clv-chart"), {
      type: "bar",
      data: {
        labels: clvData.map((d) => d.week),
        datasets: [
          {
            data: clvData.map((d) => d.clv),
            backgroundColor: clvData.map((d) => (d.clv >= 0 ? CHART_SUCCESS : CHART_DANGER)),
          },
        ],
      },
      options: {
        ...defaults,
        scales: {
          ...defaults.scales,
          y: {
            ...defaults.scales.y,
            title: { display: true, text: "CLV (line)", color: CHART_MUTED, font: { size: 11 } },
          },
        },
      },
    });
  }
}

init();
