/* md_interactions GUI: pick atoms in 3D, define interactions, run, see figures. */

const state = {
  viewer: null,
  indices: [],        // PDB serial (1-based) -> universe index
  picked: [],         // [{serial, index, resname, resid, name}]
  interactions: [],   // [{name, kind, atoms, threshold}]
  labels: [],
};

const $ = (id) => document.getElementById(id);
const KIND_BY_COUNT = { 2: "distance", 3: "angle", 4: "dihedral" };

// --------------------------------------------------------------------------
// system + structure
// --------------------------------------------------------------------------
async function loadSystem() {
  const info = await (await fetch("/api/system")).json();
  const ligands = info.ligands.map((l) => `${l.resname}${l.resid}`).join(", ") || "none";
  $("system-info").textContent =
    `${info.n_atoms.toLocaleString()} atoms &middot; ${info.n_protein_residues} protein residues &middot; ` +
    `${info.n_frames} frames &middot; ligands: ${ligands}`;
  $("stride").value = info.stride;
}

async function loadStructure() {
  const waters = $("show-waters").checked ? 1 : 0;
  $("pick-status").textContent = "loading structure...";
  const data = await (await fetch(`/api/structure?waters=${waters}`)).json();
  state.indices = data.indices;

  const element = $("viewer");
  element.innerHTML = "";
  state.viewer = $3Dmol.createViewer(element, { backgroundColor: "#0e1116" });
  // keepH: 3Dmol drops hydrogens by default and you need to click on them
  // (a transferred proton is exactly the kind of atom worth measuring)
  state.viewer.addModel(data.pdb, "pdb", { keepH: true });
  applyStyle();
  state.viewer.zoomTo();
  state.viewer.render();

  state.viewer.setClickable({}, true, (atom) => pickAtom(atom));
  $("pick-status").textContent = "click atoms in the structure";
  redrawPicks();
}

function applyStyle() {
  const viewer = state.viewer;
  viewer.setStyle({}, { cartoon: { color: "#8fa3b8", opacity: 0.75 } });
  if ($("show-sidechains").checked) {
    viewer.setStyle({ hetflag: false }, {
      cartoon: { color: "#8fa3b8", opacity: 0.6 },
      stick: { radius: 0.12, colorscheme: "default" },
    });
  }
  viewer.setStyle({ hetflag: true }, { stick: { radius: 0.2, colorscheme: "greenCarbon" } });
}

// --------------------------------------------------------------------------
// atom picking
// --------------------------------------------------------------------------
function pickAtom(atom) {
  // Map through the PDB serial, not 3Dmol's own index: the parser may skip
  // atoms, which would shift every index and silently measure the wrong pair.
  const serial = atom.serial - 1;
  const universeIndex = state.indices[serial];
  if (universeIndex === undefined) return;
  if (state.picked.some((p) => p.serial === serial)) return;
  if (state.picked.length >= 4) {
    $("pick-status").innerHTML = '<span class="error">4 atoms maximum; add or clear</span>';
    return;
  }
  state.picked.push({
    serial,
    index: universeIndex,
    resname: atom.resn,
    resid: atom.resi,
    name: atom.atom,
    x: atom.x, y: atom.y, z: atom.z,
  });
  redrawPicks();
}

function redrawPicks() {
  const container = $("picked");
  container.innerHTML = "";
  state.labels.forEach((l) => state.viewer && state.viewer.removeLabel(l));
  state.labels = [];

  state.picked.forEach((atom, position) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `${atom.resname}${atom.resid}:${atom.name}<span class="drop">✕</span>`;
    chip.querySelector(".drop").onclick = () => {
      state.picked.splice(position, 1);
      redrawPicks();
    };
    container.appendChild(chip);

    if (state.viewer) {
      state.labels.push(state.viewer.addLabel(
        `${atom.resname}${atom.resid}:${atom.name}`,
        { position: { x: atom.x, y: atom.y, z: atom.z },
          backgroundColor: "#0072B2", backgroundOpacity: 0.85,
          fontColor: "white", fontSize: 11 },
      ));
    }
  });
  if (state.viewer) state.viewer.render();

  const kind = KIND_BY_COUNT[state.picked.length];
  $("add-interaction").disabled = !kind;
  $("measure-hint").textContent = kind
    ? `${state.picked.length} atoms selected -> ${translate(kind)}`
    : "Click 2 atoms for a distance, 3 for an angle, 4 for a dihedral.";
}

function translate(kind) {
  return { distance: "distance", angle: "angle", dihedral: "dihedral" }[kind] || kind;
}

// --------------------------------------------------------------------------
// interaction list
// --------------------------------------------------------------------------
function addInteraction(atoms, threshold) {
  const kind = KIND_BY_COUNT[atoms.length];
  if (!kind) return;
  const shortName = atoms.map((a) => `${a.resname}${a.resid}${a.name}`).join("_");
  const prefix = { distance: "d", angle: "a", dihedral: "chi" }[kind];
  state.interactions.push({
    name: `${prefix}_${shortName}`.slice(0, 40),
    kind,
    atoms,
    threshold: threshold ?? (kind === "distance" ? 3.5 : null),
  });
  state.picked = [];
  redrawPicks();
  renderInteractions();
}

function renderInteractions() {
  const body = document.querySelector("#interactions tbody");
  body.innerHTML = "";
  state.interactions.forEach((entry, position) => {
    const row = document.createElement("tr");

    const name = document.createElement("input");
    name.value = entry.name;
    name.onchange = () => { entry.name = name.value; };
    const nameCell = document.createElement("td");
    nameCell.appendChild(name);

    const kindCell = document.createElement("td");
    kindCell.textContent = translate(entry.kind);

    const atomsCell = document.createElement("td");
    atomsCell.textContent = entry.atoms
      .map((a) => `${a.resname}${a.resid}:${a.name}`).join(" – ");

    const thresholdCell = document.createElement("td");
    if (entry.kind === "distance") {
      const threshold = document.createElement("input");
      threshold.type = "number";
      threshold.step = "0.1";
      threshold.style.width = "4.2rem";
      threshold.value = entry.threshold ?? "";
      threshold.onchange = () => {
        entry.threshold = threshold.value ? parseFloat(threshold.value) : null;
      };
      thresholdCell.appendChild(threshold);
    }

    const removeCell = document.createElement("td");
    const remove = document.createElement("button");
    remove.className = "ghost";
    remove.textContent = "✕";
    remove.onclick = () => {
      state.interactions.splice(position, 1);
      renderInteractions();
    };
    removeCell.appendChild(remove);

    row.append(nameCell, kindCell, atomsCell, thresholdCell, removeCell);
    body.appendChild(row);
  });

  $("empty-list").hidden = state.interactions.length > 0;
  $("run").disabled = state.interactions.length === 0;
}

// --------------------------------------------------------------------------
// automatic detection
// --------------------------------------------------------------------------
async function detect(mode) {
  const output = $("detect-output");
  output.innerHTML = '<span class="spinner"></span> analysing the trajectory...';
  try {
    const response = await fetch("/api/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        region: $("region").value.trim(),
        cutoff: parseFloat($("cutoff").value),
      }),
    });
    const data = await response.json();
    if (data.error) {
      output.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }
    output.innerHTML = mode === "contacts"
      ? renderContacts(data.rows)
      : renderChanges(data);
  } catch (error) {
    output.innerHTML = `<p class="error">${error}</p>`;
  }
}

function badge(kind) {
  const cls = kind === "hbond" ? "hbond"
    : kind === "salt bridge" ? "salt"
      : kind === "water" ? "water" : "polar";
  return `<span class="badge ${cls}">${kind}</span>`;
}

function renderContacts(rows) {
  if (!rows.length) return '<p class="muted small">Nothing above the threshold.</p>';
  const body = rows.map((row, position) => `
    <tr class="clickable" data-row="${position}" title="click to add it to the list">
      <td>${row.atom_a}</td><td>${row.atom_b}</td><td>${badge(row.kind)}</td>
      <td>${row["occupancy_%"].toFixed(0)}%</td>
      <td>${row.distance_mean == null ? "" : row.distance_mean.toFixed(2)}</td>
    </tr>`).join("");
  setTimeout(() => bindContactRows(rows), 0);
  return `<div class="detected"><table>
      <thead><tr><th>A</th><th>B</th><th>type</th><th>occup.</th><th>d (A)</th></tr></thead>
      <tbody>${body}</tbody></table>
      <p class="muted small">Click a row to measure it.</p></div>`;
}

function bindContactRows(rows) {
  document.querySelectorAll("#detect-output tr.clickable").forEach((element) => {
    element.onclick = () => {
      const row = rows[parseInt(element.dataset.row, 10)];
      if (String(row.atom_b).includes("(any)")) {
        $("detect-output").insertAdjacentHTML("beforeend",
          '<p class="muted small">Aggregated water is not a fixed atom: use mode ' +
          '"min" from the input file to follow it.</p>');
        return;
      }
      addInteraction([parseLabel(row.atom_a), parseLabel(row.atom_b)], 3.5);
    };
  });
}

function parseLabel(label) {
  const [residue, name] = String(label).split(":");
  const resid = parseInt(residue.replace(/^[A-Za-z]+/, ""), 10);
  const resname = residue.replace(/[0-9]+$/, "");
  return { resname, resid, name };
}

function renderChanges(data) {
  const bonds = data.bonds || [];
  const protons = data.protons || [];
  if (!bonds.length && !protons.length) {
    return '<p class="ok small">No chemical changes: the topology describes this trajectory correctly.</p>';
  }
  let html = '<div class="detected">';
  if (bonds.length) {
    html += "<table><thead><tr><th>event</th><th>A</th><th>B</th><th>state</th></tr></thead><tbody>";
    html += bonds.map((row) => `<tr><td>${row.event}</td><td>${row.atom_a}</td>
      <td>${row.atom_b}</td><td class="muted">${row.state || ""}${
  row.note ? "<br>" + row.note : ""}</td></tr>`).join("");
    html += "</tbody></table>";
  }
  if (protons.length) {
    html += "<p class='small'><b>Transferred protons</b></p><table><tbody>";
    html += protons.map((row) => `<tr><td>${row.hydrogen}</td>
      <td class="muted">${row.hosts}</td></tr>`).join("");
    html += "</tbody></table>";
  }
  html += '<p class="error small">Reactive trajectory: the topology only describes the starting structure.</p>';
  return html + "</div>";
}

// --------------------------------------------------------------------------
// run
// --------------------------------------------------------------------------
async function run() {
  $("run").disabled = true;
  $("run-status").innerHTML = '<span class="spinner"></span> starting...';
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      interactions: state.interactions,
      stride: parseInt($("stride").value, 10),
      time_unit: $("time-unit").value,
    }),
  });
  const data = await response.json();
  if (data.error) {
    $("run-status").innerHTML = `<p class="error">${data.error}</p>`;
    $("run").disabled = false;
    return;
  }
  pollJob();
}

async function pollJob() {
  const job = await (await fetch("/api/job")).json();
  if (job.status === "running") {
    $("run-status").innerHTML = `<span class="spinner"></span> ${job.message}`;
    setTimeout(pollJob, 800);
    return;
  }
  $("run").disabled = false;
  if (job.status === "error") {
    $("run-status").innerHTML = `<p class="error">${job.message}</p>`;
    return;
  }
  if (job.status === "done") {
    $("run-status").innerHTML = '<p class="ok">Analysis finished</p>';
    showResults(job.result);
  }
}

function showResults(result) {
  const container = $("results");
  $("results-card").hidden = false;
  let html = "";
  if (result.summary && result.summary.length) {
    html += "<table><thead><tr><th>observable</th><th>mean &plusmn; sd</th><th>min</th><th>max</th></tr></thead><tbody>";
    html += result.summary.map((row) => `<tr>
      <td>${row.observable}</td>
      <td>${fmt(row.mean)} ± ${fmt(row.std)} ${row.unit || ""}</td>
      <td>${fmt(row.min)}</td><td>${fmt(row.max)}</td></tr>`).join("");
    html += "</tbody></table>";
  }
  result.failures.forEach((failure) => {
    html += `<p class="error small">${failure.analysis}: ${failure.message}</p>`;
  });
  html += result.plots.map((name) =>
    `<img src="/results/plots/${name}?t=${Date.now()}" alt="${name}">`).join("");
  html += `<p class="muted small">Written to <code>${result.directory}</code></p>`;
  container.innerHTML = html;
}

function fmt(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(3);
}

// --------------------------------------------------------------------------
// wiring
// --------------------------------------------------------------------------
$("detect-contacts").onclick = () => detect("contacts");
$("detect-changes").onclick = () => detect("changes");
$("add-interaction").onclick = () => addInteraction([...state.picked]);
$("clear-picks").onclick = () => { state.picked = []; redrawPicks(); };
$("reset-view").onclick = () => { state.viewer.zoomTo(); state.viewer.render(); };
$("show-waters").onchange = loadStructure;
$("show-sidechains").onchange = () => {
  applyStyle();
  state.viewer.render();
};
$("run").onclick = run;

loadSystem().then(loadStructure);
