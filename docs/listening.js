const holder = document.querySelector("#audio-examples");
const examples = (window.tipsterExamples || []).filter(example => example.title && example.source && example.reference && example.output);

function createElement(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

for (const example of examples) {
  const card = createElement("article");
  card.append(createElement("h3", example.title));
  const tracks = createElement("div", "", "tracks");
  for (const [key, label] of [["source", "Source performance"], ["reference", "Timbre reference"], ["output", "TIPSTER output"], ["baseline", "AFTER baseline"]]) {
    if (!example[key]) continue;
    const track = createElement("div");
    track.append(createElement("div", label, "track-label"));
    const audio = createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = example[key];
    audio.setAttribute("aria-label", label + " for " + example.title);
    track.append(audio);
    tracks.append(track);
  }
  card.append(tracks);
  holder.append(card);
}

document.querySelector("#audio-status").hidden = examples.length > 0;
holder.addEventListener("play", event => {
  for (const player of holder.querySelectorAll("audio")) {
    if (player !== event.target) player.pause();
  }
}, true);