const DATA = "data";

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) return null;
  return res.json();
}

function fmtPct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtEdge(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

async function init() {
  const summary = await loadJSON(`${DATA}/summary.json`);
  if (summary) renderSummary(summary);

  const picksFiles = await findLatestPicks();
  if (picksFiles) renderPicks(picksFiles);

  if (summary) renderCharts(summary);
}

async function findLatestPicks() {
  // Try common week patterns — sync script copies recent files
  for (const week of [5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]) {
    for (const season of [2025, 2024]) {
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

  if (!picks.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">No picks this week (no edge above threshold)</td></tr>`;
    return;
  }

  tbody.innerHTML = picks
    .map(
      (p) => `
    <tr>
      <td>${p.player}</td>
      <td>${p.position}</td>
      <td>${p.line}</td>
      <td class="pick-${p.pick}">${p.pick.toUpperCase()}</td>
      <td>${fmtEdge(p.edge)}</td>
      <td>${p.model_mu}</td>
      <td>${p.price > 0 ? "+" + p.price : p.price}</td>
      <td>${p.matchup || ""}</td>
    </tr>`
    )
    .join("");
}

function renderCharts(summary) {
  const brierData = summary.cumulative_brier || [];
  const clvData = summary.cumulative_clv || [];

  const chartDefaults = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#8899aa", maxRotation: 45 }, grid: { color: "#2d3a4d" } },
      y: { ticks: { color: "#8899aa" }, grid: { color: "#2d3a4d" } },
    },
  };

  if (brierData.length) {
    new Chart(document.getElementById("brier-chart"), {
      type: "line",
      data: {
        labels: brierData.map((d) => d.week),
        datasets: [
          {
            data: brierData.map((d) => d.brier),
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59,130,246,0.1)",
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: {
        ...chartDefaults,
        scales: {
          ...chartDefaults.scales,
          y: { ...chartDefaults.scales.y, title: { display: true, text: "Brier", color: "#8899aa" } },
        },
      },
    });
  }

  if (clvData.length) {
    new Chart(document.getElementById("clv-chart"), {
      type: "bar",
      data: {
        labels: clvData.map((d) => d.week),
        datasets: [
          {
            data: clvData.map((d) => d.clv),
            backgroundColor: clvData.map((d) => (d.clv >= 0 ? "#22c55e" : "#ef4444")),
          },
        ],
      },
      options: {
        ...chartDefaults,
        scales: {
          ...chartDefaults.scales,
          y: { ...chartDefaults.scales.y, title: { display: true, text: "CLV (line)", color: "#8899aa" } },
        },
      },
    });
  }
}

init();
