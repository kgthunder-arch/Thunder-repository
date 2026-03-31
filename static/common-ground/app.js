const mapNodes = [
  { left: "16%", top: "36%" },
  { left: "24%", top: "58%" },
  { left: "38%", top: "28%" },
  { left: "46%", top: "52%" },
  { left: "58%", top: "34%" },
  { left: "67%", top: "64%" },
  { left: "76%", top: "38%" },
  { left: "84%", top: "48%" }
];

const tickerItems = [
  {
    title: "Climate adaptation workshop",
    detail: "Urban planners in Jakarta, Rotterdam, and Accra just opened a live room."
  },
  {
    title: "New mentor match",
    detail: "A product founder in Nairobi paired with a civic designer in Toronto."
  },
  {
    title: "Forum trend",
    detail: "Local manufacturing in Latin America is the fastest-growing topic today."
  }
];

const panels = {
  discover: {
    title: "Discover communities with purpose, not noise.",
    intro:
      "The discovery layer blends geography, themes, and professional intent. Users can explore region-driven communities, topic clusters, and active rooms from one calm command center.",
    layout: "two-column",
    primary: `
      <article class="panel-card panel-copy">
        <h3>Global exploration feed</h3>
        <p>People browse by mission, region, or momentum. Instead of chasing followers, they can see where useful energy is building and why.</p>
        <ul class="panel-list">
          <li><strong>Signal-first map:</strong> see emerging discussions by place, topic, and urgency.</li>
          <li><strong>Community highlights:</strong> rotating spotlights explain what each group is working on right now.</li>
          <li><strong>Timezone-aware entry points:</strong> live rooms, async threads, and recap bundles are clearly separated.</li>
        </ul>
      </article>
    `,
    secondary: `
      <article class="panel-card">
        <div class="room-header">
          <div>
            <span class="eyebrow">Today across the platform</span>
            <h3>Featured rooms</h3>
          </div>
          <span class="story-pill">Live now</span>
        </div>
        <ul class="room-list">
          <li>
            <strong>Designing public health campaigns across languages</strong>
            <div class="translation-meta">Cape Town, Manila, and Bogota - 214 participants</div>
          </li>
          <li>
            <strong>How local crafts become global businesses</strong>
            <div class="translation-meta">Marrakech, Kyoto, and Mexico City - 96 participants</div>
          </li>
          <li>
            <strong>AI tools for teachers in under-resourced schools</strong>
            <div class="translation-meta">Chennai, Nairobi, and Lima - 148 participants</div>
          </li>
        </ul>
      </article>
    `
  },
  translate: {
    title: "Make every message feel native, respectful, and clear.",
    intro:
      "Translation is built into the conversation itself. The app keeps original phrasing available, adapts idioms when needed, and adds cultural context when misunderstandings are likely.",
    layout: "two-column",
    primary: `
      <article class="translation-card">
        <div class="translation-header">
          <div>
            <span class="eyebrow">Conversation bridge</span>
            <h3>Live translation thread</h3>
          </div>
          <div class="translation-language-toggle" id="languageToggle"></div>
        </div>
        <div class="translation-list" id="translationList"></div>
      </article>
    `,
    secondary: `
      <article class="panel-card">
        <h3>Context tools that reduce friction</h3>
        <ul class="panel-list">
          <li><strong>Etiquette hints:</strong> prompts explain regional communication norms before a meeting starts.</li>
          <li><strong>Tone assist:</strong> rewrites messages to sound clearer, warmer, or more direct without losing meaning.</li>
          <li><strong>Meeting recap:</strong> every room creates a multilingual summary with next steps and decisions.</li>
        </ul>
      </article>
    `
  },
  match: {
    title: "Use AI to connect complementary people, not identical profiles.",
    intro:
      "Matching prioritizes complementary strengths, shared causes, and realistic overlap windows. It is designed to spark useful collaboration between people who would not usually find each other.",
    layout: "single-column",
    primary: `
      <article class="panel-card">
        <div class="match-header">
          <div>
            <span class="eyebrow">Suggested connections</span>
            <h3>High-potential matches for a civic innovation brief</h3>
          </div>
          <span class="story-pill">Updated 2 min ago</span>
        </div>
        <div class="match-grid" id="matchGrid"></div>
      </article>
    `
  },
  collaborate: {
    title: "Move from a conversation to shared execution in one tap.",
    intro:
      "Every promising discussion can become a collaboration room with notes, a live canvas, tasks, and an async handoff. The goal is to preserve momentum when global teams cannot all be online together.",
    layout: "two-column",
    primary: `
      <article class="room-card">
        <div class="room-header">
          <div>
            <span class="eyebrow">Live workspace</span>
            <h3>Cross-border project room</h3>
          </div>
          <span class="story-pill">Sprint 03</span>
        </div>
        <ul class="room-list">
          <li>
            <strong>Shared brief</strong>
            <div class="translation-meta">Problem statement translated into 4 languages with local examples layered underneath.</div>
          </li>
          <li>
            <strong>Whiteboard and notes</strong>
            <div class="translation-meta">Sticky notes are auto-grouped by theme, then summarized for teammates joining later.</div>
          </li>
          <li>
            <strong>Async handoff</strong>
            <div class="translation-meta">The system packages decisions, open questions, and ownership into a recap card.</div>
          </li>
        </ul>
      </article>
    `,
    secondary: `
      <article class="panel-card">
        <h3>Designed for real global teamwork</h3>
        <p class="room-note">Smart scheduling surfaces the best overlap windows, then recommends async rituals when those windows are too short.</p>
        <ul class="panel-list">
          <li><strong>Presence cues:</strong> see who is live, who is asleep, and who has already handed off work.</li>
          <li><strong>Shared milestones:</strong> turn forum threads into project goals without re-entering context.</li>
          <li><strong>Knowledge memory:</strong> AI-generated summaries stop information from disappearing into chat history.</li>
        </ul>
      </article>
    `
  }
};

const translationConversations = {
  english: [
    {
      author: "Amina - Nairobi",
      text: "We should test the workshop in markets first, because that is where trust already exists.",
      meta: "Translated from Swahili with context note: community-first outreach"
    },
    {
      author: "Mateo - Bogota",
      text: "Agreed. If we frame it as a co-design session, more local organizers will show up.",
      meta: "Original in Spanish"
    },
    {
      author: "Yuna - Seoul",
      text: "I can convert the ideas into a simple toolkit and a slide deck for remote facilitators.",
      meta: "Original in Korean"
    }
  ],
  spanish: [
    {
      author: "Amina - Nairobi",
      text: "Debemos probar primero el taller en los mercados, porque ahi ya existe confianza.",
      meta: "Traducido del suajili con nota de contexto: divulgacion liderada por la comunidad"
    },
    {
      author: "Mateo - Bogota",
      text: "De acuerdo. Si lo presentamos como una sesion de codiseno, se sumaran mas organizadores locales.",
      meta: "Mensaje original en espanol"
    },
    {
      author: "Yuna - Seoul",
      text: "Puedo convertir las ideas en una guia sencilla y una presentacion para facilitadores remotos.",
      meta: "Traducido del coreano"
    }
  ],
  hindi: [
    {
      author: "Amina - Nairobi",
      text: "Hame workshop ko pehle bazaaron mein test karna chahiye, kyunki vahan pehle se bharosa bana hua hai.",
      meta: "Swahili se anuvad, sandarbh note: community-first outreach"
    },
    {
      author: "Mateo - Bogota",
      text: "Sahmat. Agar hum ise co-design session ke roop mein rakhen, to zyada local organizers judenge.",
      meta: "Mool sandesh Spanish mein tha"
    },
    {
      author: "Yuna - Seoul",
      text: "Main in ideas ko remote facilitators ke liye ek simple toolkit aur slide deck mein badal sakti hoon.",
      meta: "Korean se anuvad"
    }
  ]
};

const matches = [
  {
    name: "Sara Okafor",
    role: "Community health strategist - Lagos",
    score: "96% fit",
    reason: "Pairs field research strength with your policy prototype work.",
    tags: ["Public health", "Community design", "GMT+1 overlap"]
  },
  {
    name: "Jin Park",
    role: "Civic data designer - Seoul",
    score: "93% fit",
    reason: "Brings systems mapping and workshop synthesis for distributed teams.",
    tags: ["Data storytelling", "Facilitation", "Async-friendly"]
  },
  {
    name: "Lucia Mendes",
    role: "Education entrepreneur - Sao Paulo",
    score: "90% fit",
    reason: "Matches your youth innovation track and offers pilot access in schools.",
    tags: ["Education", "Pilot networks", "Portuguese + English"]
  },
  {
    name: "Samir Qureshi",
    role: "Climate program lead - Dubai",
    score: "88% fit",
    reason: "Adds operations depth and cross-regional partnership experience.",
    tags: ["Climate", "Operations", "MENA partnerships"]
  }
];

const forums = [
  {
    category: "innovation",
    title: "How do local makers scale globally without losing identity?",
    description: "Founders, artisans, and brand strategists compare models from India, Morocco, and Mexico.",
    activity: "184 replies",
    region: "Asia + Africa + LATAM"
  },
  {
    category: "culture",
    title: "What does respectful collaboration look like across very different work styles?",
    description: "A moderated conversation on trust, directness, hierarchy, and communication rituals.",
    activity: "92 replies",
    region: "Global"
  },
  {
    category: "climate",
    title: "Which city-led climate solutions translate well between regions?",
    description: "Urban practitioners share which interventions adapt cleanly and which need local redesign.",
    activity: "141 replies",
    region: "Coastal cities"
  },
  {
    category: "education",
    title: "How are teachers using AI when bandwidth is limited?",
    description: "Educators trade practical classroom workflows for low-connectivity environments.",
    activity: "117 replies",
    region: "South Asia + East Africa"
  },
  {
    category: "innovation",
    title: "What kind of matchmaking actually helps early-stage collaboration?",
    description: "Product builders and researchers debate the best signals for idea compatibility.",
    activity: "76 replies",
    region: "Remote-first teams"
  },
  {
    category: "culture",
    title: "Which local traditions should global platforms celebrate more intentionally?",
    description: "A joyful thread collecting rituals, festivals, and community habits that shape belonging.",
    activity: "203 replies",
    region: "Every region"
  }
];

const signalMap = document.getElementById("signalMap");
const ticker = document.getElementById("ticker");
const panelStage = document.getElementById("panelStage");
const forumGrid = document.getElementById("forumGrid");

function renderMap() {
  mapNodes.forEach((node, index) => {
    const dot = document.createElement("span");
    dot.className = "map-node";
    dot.style.left = node.left;
    dot.style.top = node.top;
    dot.style.animationDelay = `${index * 0.25}s`;
    signalMap.appendChild(dot);
  });
}

function renderTicker() {
  ticker.innerHTML = "";
  tickerItems.forEach((item) => {
    const block = document.createElement("article");
    block.className = "ticker-item";
    block.innerHTML = `<strong>${item.title}</strong><span>${item.detail}</span>`;
    ticker.appendChild(block);
  });
}

function renderTranslation(language) {
  const toggle = document.getElementById("languageToggle");
  const list = document.getElementById("translationList");
  const languages = [
    { id: "english", label: "English" },
    { id: "spanish", label: "Spanish" },
    { id: "hindi", label: "Hindi" }
  ];

  toggle.innerHTML = "";
  languages.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `language-button${item.id === language ? " is-active" : ""}`;
    button.textContent = item.label;
    button.addEventListener("click", () => renderTranslation(item.id));
    toggle.appendChild(button);
  });

  list.innerHTML = "";
  translationConversations[language].forEach((entry) => {
    const bubble = document.createElement("article");
    bubble.className = "translation-bubble";
    bubble.innerHTML = `
      <strong>${entry.author}</strong>
      <span>${entry.text}</span>
      <small class="translation-meta">${entry.meta}</small>
    `;
    list.appendChild(bubble);
  });
}

function renderMatches() {
  const matchGrid = document.getElementById("matchGrid");
  matchGrid.innerHTML = "";
  matches.forEach((match) => {
    const card = document.createElement("article");
    card.className = "match-card";
    card.innerHTML = `
      <span class="score">${match.score}</span>
      <div>
        <strong>${match.name}</strong>
        <div class="match-meta">${match.role}</div>
      </div>
      <p class="match-meta">${match.reason}</p>
      <div class="tag-row">${match.tags.map((tag) => `<span>${tag}</span>`).join("")}</div>
    `;
    matchGrid.appendChild(card);
  });
}

function renderPanel(key) {
  const panel = panels[key];
  panelStage.innerHTML = `
    <div class="panel-layout ${panel.layout}">
      <div>${panel.primary}</div>
      ${panel.secondary ? `<div>${panel.secondary}</div>` : ""}
    </div>
  `;

  const introCard = document.createElement("article");
  introCard.className = "panel-card panel-copy";
  introCard.innerHTML = `
    <h3>${panel.title}</h3>
    <p>${panel.intro}</p>
  `;

  const wrapper = panelStage.querySelector(".panel-layout");
  wrapper.prepend(introCard);

  if (key === "translate") {
    renderTranslation("english");
  }

  if (key === "match") {
    renderMatches();
  }
}

function setActiveTab(key) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    const isCurrent = button.dataset.panel === key;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-selected", String(isCurrent));
  });
  renderPanel(key);
}

function renderForums(filter = "all") {
  forumGrid.innerHTML = "";
  const visible = forums.filter((forum) => filter === "all" || forum.category === filter);

  visible.forEach((forum) => {
    const card = document.createElement("article");
    card.className = "forum-card";
    card.innerHTML = `
      <div class="forum-header">
        <span class="story-pill">${forum.category}</span>
        <span class="translation-meta">${forum.region}</span>
      </div>
      <strong>${forum.title}</strong>
      <p>${forum.description}</p>
      <footer>
        <span>${forum.activity}</span>
        <span>Open discussion</span>
      </footer>
    `;
    forumGrid.appendChild(card);
  });
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.panel));
});

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("is-active"));
    button.classList.add("is-active");
    renderForums(button.dataset.filter);
  });
});

renderMap();
renderTicker();
setActiveTab("discover");
renderForums();
