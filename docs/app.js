/* 강화학습 100제 — 문항 로드 및 화면 구성 */

const STORE_PREFIX = 'dlfs4-p100:ch';

const TYPE_LABEL = {
  ox: 'O/X',
  blank: '빈칸',
  short: '단답',
  explain: '설명',
};

/* ---------- 진행률 저장 ---------- */

function loadDone(chapterId) {
  try {
    const raw = localStorage.getItem(STORE_PREFIX + chapterId);
    return new Set(raw ? JSON.parse(raw) : []);
  } catch (e) {
    return new Set(); // 사생활 보호 모드 등으로 접근이 막히면 그냥 저장을 포기한다
  }
}

function saveDone(chapterId, doneSet) {
  try {
    localStorage.setItem(STORE_PREFIX + chapterId, JSON.stringify([...doneSet]));
  } catch (e) {
    /* 저장 실패는 무시 — 화면 동작에는 지장이 없다 */
  }
}

/* ---------- 데이터 ---------- */

async function loadData() {
  const res = await fetch('questions.json');
  if (!res.ok) throw new Error('questions.json 을 불러오지 못했습니다 (' + res.status + ')');
  return res.json();
}

function showError(err) {
  const main = document.getElementById('main');
  main.innerHTML =
    '<p class="status">문항을 불러오지 못했습니다.<br>' +
    '<code>file://</code> 로 열면 브라우저가 막습니다. ' +
    '<code>python -m http.server</code> 로 띄워주세요.</p>';
  console.error(err);
}

/* ---------- 랜딩 ---------- */

function renderIndex(data) {
  const main = document.getElementById('main');
  const grid = document.createElement('div');
  grid.className = 'chapter-grid';

  for (const ch of data.chapters) {
    const done = loadDone(ch.id).size;
    const total = ch.questions.length;

    const card = document.createElement('a');
    card.className = 'chapter-card';
    card.href = 'chapter.html#' + ch.id;

    const num = document.createElement('div');
    num.className = 'num';
    num.textContent = ch.id + '장';

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = ch.title;

    const count = document.createElement('div');
    count.className = 'count';
    if (done > 0) {
      const mark = document.createElement('span');
      mark.className = 'done-mark';
      mark.textContent = done + '/' + total;
      count.append(mark, ' 확인');
    } else {
      count.textContent = total + '문항';
    }

    card.append(num, name, count);
    grid.append(card);
  }

  main.replaceChildren(grid);
}

/* ---------- 장 페이지 ---------- */

function currentChapterId(data) {
  const raw = parseInt(location.hash.slice(1), 10);
  return data.chapters.some((c) => c.id === raw) ? raw : data.chapters[0].id;
}

function renderChapter(data) {
  const main = document.getElementById('main');
  const id = currentChapterId(data);
  const ch = data.chapters.find((c) => c.id === id);
  const done = loadDone(id);

  document.title = ch.id + '장 ' + ch.title + ' — 강화학습 100제';
  document.getElementById('chapter-title').textContent = ch.id + '장 ' + ch.title;
  document.getElementById('chapter-sub').textContent = ch.questions.length + '문항';

  /* --- 도구 막대 --- */
  const progress = document.createElement('div');
  progress.className = 'progress';

  const btnAll = document.createElement('button');
  const btnReset = document.createElement('button');
  btnReset.textContent = '진행률 초기화';

  const toolbar = document.createElement('div');
  toolbar.className = 'toolbar';
  toolbar.append(progress, btnAll, btnReset);

  /* --- 문항 --- */
  const list = document.createElement('ol');
  list.className = 'q-list';

  const answers = [];

  ch.questions.forEach((q, i) => {
    const item = document.createElement('li');
    item.className = 'q-item';

    const head = document.createElement('div');
    head.className = 'q-head';

    const num = document.createElement('span');
    num.className = 'q-num';
    num.textContent = i + 1 + '.';

    const badge = document.createElement('span');
    badge.className = 'badge badge-' + q.type;
    badge.textContent = TYPE_LABEL[q.type] || q.type;

    head.append(num, badge);

    if (q.ref) {
      const ref = document.createElement('span');
      ref.className = 'q-ref';
      ref.textContent = q.ref + '절';
      head.append(ref);
    }

    const text = document.createElement('p');
    text.className = 'q-text';
    text.textContent = q.q;

    item.append(head, text);

    if (q.fig) {
      const fig = document.createElement('p');
      fig.className = 'q-fig';
      const img = document.createElement('img');
      img.src = q.fig;
      img.alt = '문항 ' + (i + 1) + ' 그림';
      img.loading = 'lazy';
      fig.append(img);
      item.append(fig);
    }

    const answer = document.createElement('div');
    answer.className = 'answer';
    answer.textContent = q.a;
    answer.hidden = true;
    answers.push(answer);

    const reveal = document.createElement('button');
    reveal.className = 'reveal';
    reveal.textContent = '정답 보기 ▾';
    reveal.setAttribute('aria-expanded', 'false');
    reveal.addEventListener('click', () => {
      answer.hidden = !answer.hidden;
      reveal.textContent = answer.hidden ? '정답 보기 ▾' : '정답 접기 ▴';
      reveal.setAttribute('aria-expanded', String(!answer.hidden));
      syncRevealAll();
    });

    const check = document.createElement('label');
    check.className = 'check';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = done.has(q.id);
    box.addEventListener('change', () => {
      if (box.checked) done.add(q.id);
      else done.delete(q.id);
      item.classList.toggle('is-done', box.checked);
      saveDone(id, done);
      syncProgress();
    });
    check.append(box, document.createTextNode('확인'));

    item.classList.toggle('is-done', box.checked);

    const foot = document.createElement('div');
    foot.className = 'q-foot';
    foot.append(reveal, check);

    item.append(answer, foot);
    list.append(item);
  });

  /* --- 도구 막대 동작 --- */
  function syncProgress() {
    progress.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = done.size;
    progress.append(strong, '/' + ch.questions.length + ' 확인');
  }

  function syncRevealAll() {
    const allOpen = answers.every((a) => !a.hidden);
    btnAll.textContent = allOpen ? '정답 모두 접기' : '정답 모두 펼치기';
  }

  btnAll.addEventListener('click', () => {
    const open = answers.some((a) => a.hidden); // 하나라도 접혀 있으면 전부 펼친다
    answers.forEach((a) => {
      a.hidden = !open;
    });
    list.querySelectorAll('.reveal').forEach((b) => {
      b.textContent = open ? '정답 접기 ▴' : '정답 보기 ▾';
      b.setAttribute('aria-expanded', String(open));
    });
    syncRevealAll();
  });

  btnReset.addEventListener('click', () => {
    done.clear();
    saveDone(id, done);
    list.querySelectorAll('.check input').forEach((b) => {
      b.checked = false;
    });
    list.querySelectorAll('.q-item').forEach((el) => el.classList.remove('is-done'));
    syncProgress();
  });

  syncProgress();
  syncRevealAll();

  /* --- 이전/다음 장 --- */
  const nav = document.createElement('div');
  nav.className = 'chapter-nav';
  const pos = data.chapters.findIndex((c) => c.id === id);

  if (pos > 0) {
    const prev = data.chapters[pos - 1];
    const a = document.createElement('a');
    a.href = 'chapter.html#' + prev.id;
    a.textContent = '← ' + prev.id + '장 ' + prev.title;
    nav.append(a);
  } else {
    nav.append(document.createElement('span'));
  }

  if (pos < data.chapters.length - 1) {
    const next = data.chapters[pos + 1];
    const a = document.createElement('a');
    a.href = 'chapter.html#' + next.id;
    a.textContent = next.id + '장 ' + next.title + ' →';
    nav.append(a);
  }

  main.replaceChildren(toolbar, list, nav);
  window.scrollTo(0, 0);
}

/* ---------- 시작 ---------- */

async function main() {
  const page = document.body.dataset.page;
  try {
    const data = await loadData();
    if (page === 'index') {
      renderIndex(data);
    } else {
      renderChapter(data);
      window.addEventListener('hashchange', () => renderChapter(data));
    }
  } catch (err) {
    showError(err);
  }
}

main();
