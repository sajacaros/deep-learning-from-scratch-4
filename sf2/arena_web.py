"""arena.py와 같지만 화면을 브라우저로 보여준다(WSLg 창이 안 뜰 때).

  uv run python sf2/arena_web.py runs/dqn/qnet_final.pt runs/double/qnet_final.pt ...
  → Windows 브라우저에서 http://localhost:8765

라운드마다 상대와 시작을 바꾸는 방식은 arena.py와 같다.
모델마다 프로세스가 공유 메모리에 프레임을 쓰고(arena.worker), 이 서버는 그것을 JPEG로 바꿔
MJPEG 스트림으로 내보낸다. Play 버튼은 모든 모델에 한 라운드를 동시에 시작시킨다.
"""
import argparse
import io
import json
import multiprocessing as mp
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from arena import H, W, label_of, new_round, state_label, worker

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>SF2 Arena</title>
<style>
  html, body { height: 100%; }
  body { margin: 0; background: #18181c; color: #eee; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; align-items: center; gap: 16px; padding: 8px 16px; flex: none; }
  button { font-size: 20px; padding: 6px 28px; border: 0; border-radius: 8px; color: #fff;
           background: #c44034; cursor: pointer; }
  button:disabled { background: #555; cursor: default; }
  /* 창 크기에 맞춰 네 칸이 한 화면에 다 들어가게 한다 */
  .grid { flex: 1; min-height: 0; display: grid; gap: 8px; padding: 0 16px 12px;
          grid-template-columns: repeat(var(--cols), 1fr); grid-template-rows: repeat(var(--rows), 1fr); }
  .cell { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
  .head { display: flex; justify-content: space-between; padding: 2px; font-size: 14px; flex: none; }
  img { flex: 1; min-height: 0; width: 100%; object-fit: contain; image-rendering: pixelated; display: block; }
  .win { color: #78dc78; } .lose { color: #e66e64; }
</style></head><body>
<header><button id="play" disabled>Play</button><span id="round"></span>
<span style="color:#999">(Space: Play)</span></header>
<div class="grid" id="grid"></div>
<script>
const grid = document.getElementById('grid'), btn = document.getElementById('play');
let built = false;
function build(slots) {
  const cols = slots.length === 1 ? 1 : 2;
  grid.style.setProperty('--cols', cols);
  grid.style.setProperty('--rows', Math.ceil(slots.length / cols));
  grid.innerHTML = slots.map((s, i) => `<div class="cell"><div class="head"><span id="n${i}"></span>
    <span id="r${i}"></span></div><img src="/stream/${i}"></div>`).join('');
  built = true;
}
async function poll() {
  try {
    const st = await (await fetch('/status')).json();
    if (!built) build(st.slots);
    st.slots.forEach((s, i) => {
      document.getElementById('n' + i).textContent = `${s.name} ${s.kind}   W ${s.wins} / L ${s.losses}`;
      const r = document.getElementById('r' + i);
      r.textContent = s.last; r.className = s.last.startsWith('WIN') ? 'win' : 'lose';
    });
    btn.disabled = !st.can_play;
    btn.textContent = st.can_play ? 'Play' : (st.slots.every(s => s.ready) ? 'Playing…' : 'Loading…');
    document.getElementById('round').textContent = `round ${st.rounds}` + (st.opponent ? `  ·  ${st.opponent}` : '');
  } catch (e) {}
  setTimeout(poll, 300);
}
const play = () => { if (!btn.disabled) { btn.disabled = true; fetch('/play', {method: 'POST'}); } };
btn.onclick = play;
document.addEventListener('keydown', e => { if (e.code === 'Space') { e.preventDefault(); play(); } });
poll();
</script></body></html>"""


class Arena:
    def __init__(self, models, epsilon):
        ctx = mp.get_context('spawn')
        self.lock = threading.Lock()
        self.rounds = 0
        self.opponent = ''
        self.rng = np.random.default_rng()
        self.slots = []
        for path in models:
            shm = ctx.RawArray('B', H * W * 3)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(path, shm, child, epsilon), daemon=True)
            proc.start()
            self.slots.append(dict(name=label_of(path), frame=np.frombuffer(shm, dtype=np.uint8).reshape(H, W, 3),
                                   conn=parent, proc=proc, ready=False, playing=False,
                                   wins=0, losses=0, last='', kind=''))
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):  # 각 프로세스가 보내는 준비·결과 소식을 받는다
        while True:
            for s in self.slots:
                while s['conn'].poll():
                    msg = s['conn'].recv()
                    with self.lock:
                        if msg[0] == 'ready':
                            s['ready'], s['kind'] = True, msg[1]
                        else:
                            _, win, reward, my_hp, enemy_hp = msg
                            s['playing'] = False
                            s['wins' if win else 'losses'] += 1
                            s['last'] = f'{"WIN" if win else "LOSE"}  reward {reward:.2f}  hp {my_hp} vs {enemy_hp}'
            time.sleep(0.05)

    def can_play(self):
        return all(s['ready'] and not s['playing'] for s in self.slots)

    def play(self):
        with self.lock:
            if not self.can_play():
                return
            self.rounds += 1
            msg = new_round(self.rng)
            self.opponent = state_label(msg[1])
            for s in self.slots:
                s['playing'], s['last'] = True, ''
                s['conn'].send(msg)

    def status(self):
        with self.lock:
            keys = ('name', 'kind', 'ready', 'playing', 'wins', 'losses', 'last')
            return dict(rounds=self.rounds, opponent=self.opponent, can_play=self.can_play(),
                        slots=[{k: s[k] for k in keys} for s in self.slots])

    def jpeg(self, i):
        buf = io.BytesIO()
        Image.fromarray(self.slots[i]['frame'].copy()).save(buf, format='JPEG', quality=85)
        return buf.getvalue()


def make_handler(arena, fps):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self._send(PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/status':
                self._send(json.dumps(arena.status()).encode(), 'application/json')
            elif self.path.startswith('/stream/'):
                i = int(self.path.rsplit('/', 1)[1])
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    while True:
                        jpg = arena.jpeg(i)
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                         + f'Content-Length: {len(jpg)}\r\n\r\n'.encode() + jpg + b'\r\n')
                        time.sleep(1 / fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == '/play':
                arena.play()
                self._send(b'{}', 'application/json')
            else:
                self.send_error(404)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+')
    p.add_argument('--epsilon', type=float, default=0.0)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--fps', type=int, default=30)  # 스트림 프레임률(게임 속도와는 무관)
    args = p.parse_args()
    assert len(args.models) <= 4, '모델은 4개까지'

    arena = Arena(args.models, args.epsilon)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(arena, args.fps))
    server.daemon_threads = True
    print(f'http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
