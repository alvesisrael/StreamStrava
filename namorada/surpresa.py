"""
💛 Atleta do Ano — Surpresa Dia dos Namorados
Rode com: streamlit run namorada/surpresa.py
"""

import base64
import os
import streamlit as st
import streamlit.components.v1 as components

# ════════════════════════════════════════════════
# ✏️  CONFIGURE AQUI
# ════════════════════════════════════════════════
NOME_DELA     = "Thais"
SEU_NOME      = "Israel"
SPOTIFY_TRACK = "3IbztmgBw5OwtVhKfzhuWp"
# Fotos em namorada/fotos/:
#   capa.jpg   → retrato dela (3:4)
#   foto1.jpg  → vocês dois juntos (wide)
#   foto2.jpg  → ela treinando (quadrado)
#   foto3.jpg  → dia de prova (quadrado)
#   foto4.jpg  → momento especial (quadrado)
#   foto5.jpg  → foto do coração (wide)
# Música: coloque musica.mp3 em namorada/fotos/
# ════════════════════════════════════════════════

st.set_page_config(
    page_title=f"Atleta do Ano · {NOME_DELA}",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"] { background: #0a0a0a; }
  iframe { display: block; }
</style>
""", unsafe_allow_html=True)


def to_b64(path: str, mime_override: str = None) -> str | None:
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = mime_override or {"jpg":"jpeg","jpeg":"jpeg","png":"png","webp":"webp","mp3":"mpeg","mp4":"mp4"}.get(ext, "jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{'audio' if mime == 'mpeg' else 'image'}/{mime};base64,{data}"


BASE = os.path.join(os.path.dirname(__file__), "fotos")

def foto(nome):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        p = os.path.join(BASE, f"{nome}.{ext}")
        u = to_b64(p)
        if u: return u
    return None

def audio_b64():
    p = os.path.join(BASE, "musica.mp3")
    if not os.path.exists(p): return None
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:audio/mpeg;base64,{data}"

f_capa  = foto("capa")
f_foto1 = foto("foto1")
f_foto2 = foto("foto2")
f_foto3 = foto("foto3")
f_foto4 = foto("foto4")
f_foto5 = foto("foto5")
f_audio = audio_b64()

def img_tag(uri, icon, label):
    if uri:
        return f'<img src="{uri}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>'
    return f'<span style="font-size:28px;opacity:.2">{icon}</span><span style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#555">{label}</span>'

def capa_tag(uri):
    if uri:
        return f'<img src="{uri}" alt="Foto dela" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>'
    return '<span style="font-size:48px;opacity:.3">📸</span><span style="font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:#888">Coloque capa.jpg na pasta fotos/</span>'

audio_tag = f'<audio id="bgAudio" src="{f_audio}" loop></audio>' if f_audio else ''
audio_play = 'document.getElementById("bgAudio").play().catch(()=>{{}});' if f_audio else ''

spotify_bar = f"""
<div style="background:#0f0f0f;border-top:1px solid #1a1a1a;border-bottom:1px solid #1a1a1a;padding:12px 24px;">
  <iframe style="border-radius:8px;display:block"
    src="https://open.spotify.com/embed/track/{SPOTIFY_TRACK}?utm_source=generator&theme=0"
    width="100%" height="80" frameBorder="0" allowfullscreen=""
    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
    loading="lazy"></iframe>
</div>
""" if SPOTIFY_TRACK else ""

# ════════════════════════════════════════════════
html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Inter:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --gold:#C9A84C;--gold-light:#F0D080;--gold-dark:#8B6914;
  --black:#0a0a0a;--dark:#111;--darker:#080808;--white:#f5f0e8;--gray:#888;
}}
html{{scroll-behavior:smooth}}
body{{background:var(--black);color:var(--white);font-family:'Inter',sans-serif;overflow-x:hidden}}

/* ── INTRO ── */
#intro{{
  width:100%;height:100vh;min-height:500px;
  background:var(--black);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:20px;cursor:pointer;user-select:none;
  position:relative;overflow:hidden;
}}
#intro .tag{{
  font-size:clamp(11px,2vw,14px);letter-spacing:.45em;text-transform:uppercase;
  color:var(--gold);font-family:'Inter',sans-serif;
}}
#intro h1{{
  font-family:'Playfair Display',serif;font-size:clamp(52px,14vw,108px);
  font-weight:900;line-height:.9;text-align:center;
  color:var(--gold-light);
}}
#intro .year{{
  font-family:'Playfair Display',serif;font-size:clamp(18px,5vw,32px);
  letter-spacing:.55em;color:var(--gold);
}}
#intro .divider{{width:100px;height:1px;background:var(--gold-dark);}}
/* botão pulsante */
#openBtn{{
  margin-top:8px;
  width:clamp(64px,18vw,80px);height:clamp(64px,18vw,80px);
  border-radius:50%;border:2px solid var(--gold);
  background:transparent;color:var(--gold);
  font-size:clamp(26px,7vw,34px);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;animation:ringPulse 2s ease infinite;
  transition:background .2s;
}}
#openBtn:hover{{background:rgba(201,168,76,.15)}}
#intro .hint{{font-size:12px;letter-spacing:.3em;text-transform:uppercase;color:var(--gray)}}
.particles{{position:absolute;inset:0;pointer-events:none;overflow:hidden}}
.p{{position:absolute;width:2px;height:2px;background:var(--gold);border-radius:50%;opacity:0;animation:floatUp linear infinite}}
@keyframes ringPulse{{0%,100%{{box-shadow:0 0 0 0 rgba(201,168,76,.5)}}60%{{box-shadow:0 0 0 16px rgba(201,168,76,0)}}}}
@keyframes floatUp{{0%{{opacity:0;transform:translateY(0)}}10%{{opacity:.8}}90%{{opacity:.3}}100%{{opacity:0;transform:translateY(-100vh)}}}}

/* ── MAIN ── */
#main{{opacity:0;transition:opacity 1s ease}}
#main.on{{opacity:1}}

/* ── HERO ── */
.hero{{
  min-height:100vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;
  padding:60px 24px;
  background:radial-gradient(ellipse 80% 60% at 50% 50%,#1a1200 0%,var(--black) 70%);
}}
.eyebrow{{font-size:11px;letter-spacing:.5em;text-transform:uppercase;color:var(--gold);margin-bottom:28px}}
.trophy{{width:90px;height:110px;filter:drop-shadow(0 0 24px rgba(201,168,76,.5));margin-bottom:32px}}
.hero-title{{
  font-family:'Playfair Display',serif;font-size:clamp(52px,14vw,120px);
  font-weight:900;line-height:.9;color:var(--gold-light);margin-bottom:12px;
}}
.hero-name{{
  font-family:'Playfair Display',serif;font-size:clamp(32px,8vw,72px);
  font-weight:900;font-style:italic;color:var(--white);margin-bottom:12px;
}}
.hero-ed{{font-size:12px;letter-spacing:.4em;text-transform:uppercase;color:var(--gray)}}
.hero-line{{width:180px;height:1px;background:var(--gold-dark);margin:36px auto}}
.scroll-hint{{font-size:12px;letter-spacing:.35em;text-transform:uppercase;color:var(--gray);animation:blink 2s infinite}}
@keyframes blink{{0%,100%{{opacity:.3}}50%{{opacity:.9}}}}

/* ── PHOTO MAIN ── */
.photo-sec{{padding:60px 20px;background:var(--darker);text-align:center}}
.photo-main{{
  width:min(380px,88vw);aspect-ratio:3/4;margin:0 auto 32px;
  border:1px solid var(--gold-dark);background:#1a1200;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;
  position:relative;overflow:hidden;
}}
.photo-main::before,.photo-main::after{{content:'';position:absolute;width:20px;height:20px;border-color:var(--gold);border-style:solid;z-index:2}}
.photo-main::before{{top:10px;left:10px;border-width:2px 0 0 2px}}
.photo-main::after{{bottom:10px;right:10px;border-width:0 2px 2px 0}}
.photo-caption{{font-family:'Playfair Display',serif;font-size:clamp(15px,3vw,18px);font-style:italic;color:var(--gold-light);max-width:500px;margin:0 auto}}

/* ── STATS ── */
.stats{{background:#0c0c0c;border-top:1px solid #1a1a1a;border-bottom:1px solid #1a1a1a;padding:48px 20px}}
.stats-inner{{max-width:860px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:32px;text-align:center}}
.stat-n{{font-family:'Playfair Display',serif;font-size:clamp(32px,7vw,52px);font-weight:900;color:var(--gold);line-height:1}}
.stat-l{{font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--gray);margin-top:6px}}

/* ── CATEGORIES ── */
.sec{{padding:60px 20px;max-width:880px;margin:0 auto}}
.sec-label{{font-size:11px;letter-spacing:.5em;text-transform:uppercase;color:var(--gold);margin-bottom:14px}}
.sec-title{{font-family:'Playfair Display',serif;font-size:clamp(26px,5vw,44px);font-weight:700;margin-bottom:40px;line-height:1.2}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}}
.card{{background:#111;border:1px solid #1e1e1e;border-top:2px solid var(--gold-dark);padding:28px 24px;transition:transform .3s,border-color .3s}}
.card:hover{{transform:translateY(-4px);border-top-color:var(--gold)}}
.medal{{font-size:30px;margin-bottom:14px}}
.cat-name{{font-size:10px;letter-spacing:.4em;text-transform:uppercase;color:var(--gold);margin-bottom:10px}}
.cat-title{{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;margin-bottom:10px}}
.cat-desc{{font-size:13px;line-height:1.7;color:#aaa}}

/* ── GALLERY ── */
.gallery-sec{{padding:60px 20px;background:var(--darker)}}
.gallery-inner{{max-width:880px;margin:0 auto}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:36px}}
.grid .g1{{grid-column:1/3}}
.grid .g4{{grid-column:2/4}}
.frame{{background:#1a1a0a;border:1px solid #2a2a1a;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;overflow:hidden;position:relative}}
.g1,.g4{{aspect-ratio:16/9}}

/* ── DECLARATION ── */
.decl{{padding:80px 24px;max-width:660px;margin:0 auto;text-align:center}}
.quote-mark{{font-family:'Playfair Display',serif;font-size:100px;line-height:.4;color:var(--gold-dark);opacity:.4;margin-bottom:20px}}
.decl-text{{font-family:'Playfair Display',serif;font-size:clamp(17px,3vw,24px);line-height:1.85;font-style:italic;color:var(--white);margin-bottom:40px}}
.decl-text em{{color:var(--gold-light);font-style:normal}}
.decl-sig{{font-family:'Playfair Display',serif;font-size:clamp(20px,4vw,26px);color:var(--gold);font-style:italic}}
.decl-date{{font-size:11px;letter-spacing:.3em;text-transform:uppercase;color:var(--gray);margin-top:10px}}
.decl-div{{width:70px;height:1px;background:var(--gold-dark);margin:28px auto}}

/* ── FOOTER ── */
footer{{padding:48px 20px;text-align:center;border-top:1px solid #1a1a1a}}
.footer-logo{{font-family:'Playfair Display',serif;font-size:13px;letter-spacing:.4em;text-transform:uppercase;color:var(--gold);margin-bottom:12px}}
.footer-txt{{font-size:11px;color:#444;letter-spacing:.2em}}

/* ── REVEAL ── */
.reveal{{opacity:0;transform:translateY(28px);transition:opacity .8s ease,transform .8s ease}}
.reveal.on{{opacity:1;transform:none}}

/* ── MOBILE ── */
@media(max-width:560px){{
  .grid{{grid-template-columns:1fr 1fr}}
  .grid .g1,.grid .g4{{grid-column:1/-1}}
  .cards{{grid-template-columns:1fr}}
  .sec{{padding:40px 16px}}
  .stats-inner{{gap:24px}}
  .decl{{padding:60px 16px}}
}}
</style>
</head>
<body>
{audio_tag}

<!-- INTRO -->
<div id="intro">
  <div class="particles" id="ptcl"></div>
  <p class="tag">12 de Junho · Dia dos Namorados</p>
  <h1>Atleta<br>do Ano</h1>
  <p class="year">2 0 2 6</p>
  <div class="divider"></div>
  <button id="openBtn" onclick="openCeremony()" aria-label="Abrir surpresa">▶</button>
  <p class="hint">toque para abrir</p>
</div>

<!-- MAIN -->
<div id="main">

  <!-- HERO -->
  <section class="hero">
    <p class="eyebrow">A premiação mais especial de todas as edições</p>
    <svg class="trophy" viewBox="0 0 100 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#8B6914"/><stop offset="50%" stop-color="#F0D080"/><stop offset="100%" stop-color="#C9A84C"/>
      </linearGradient></defs>
      <path d="M30 10 H70 L62 60 H38 Z" fill="url(#tg)" opacity=".9"/>
      <path d="M30 18 Q10 20 12 38 Q14 52 30 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="M70 18 Q90 20 88 38 Q86 52 70 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
      <rect x="44" y="60" width="12" height="22" fill="url(#tg)" rx="2"/>
      <rect x="28" y="82" width="44" height="10" fill="url(#tg)" rx="3"/>
      <rect x="22" y="92" width="56" height="8" fill="url(#tg)" rx="3"/>
      <path d="M50 22 L52.4 29.2 L60 29.2 L54 33.8 L56.4 41 L50 36.4 L43.6 41 L46 33.8 L40 29.2 L47.6 29.2 Z" fill="#fff" opacity=".9"/>
    </svg>
    <p class="eyebrow">12 de Junho · Dia dos Namorados 2026</p>
    <h1 class="hero-title">ATLETA<br>DO ANO</h1>
    <p class="hero-name">{NOME_DELA} 🏆</p>
    <p class="hero-ed">Categoria: Coração de Campeã · 12.06.2026</p>
    <div class="hero-line"></div>
    <p class="scroll-hint">↓ Continue descobrindo</p>
  </section>

  <!-- MUSIC BAR -->
  {spotify_bar}

  <!-- FOTO PRINCIPAL -->
  <section class="photo-sec">
    <div class="photo-main reveal">{capa_tag(f_capa)}</div>
    <p class="photo-caption reveal">"A campeã que escolhi amar todos os dias — dentro e fora das pistas."</p>
  </section>

  <!-- STATS -->
  <div class="stats">
    <div class="stats-inner">
      <div class="reveal"><div class="stat-n" data-target="1">0</div><div class="stat-l">namorado sortudo</div></div>
      <div class="reveal"><div class="stat-n" data-target="12">0</div><div class="stat-l">de junho · nosso dia</div></div>
      <div class="reveal"><div class="stat-n" data-target="100">0</div><div class="stat-l">% do meu coração</div></div>
      <div class="reveal"><div class="stat-n">∞</div><div class="stat-l">razões pra te amar</div></div>
    </div>
  </div>

  <!-- CATEGORIES -->
  <section class="sec">
    <p class="sec-label reveal">Dia dos Namorados · Categorias premiadas</p>
    <h2 class="sec-title reveal">Tudo que você é<br>e que me conquistou</h2>
    <div class="cards">
      <div class="card reveal">
        <div class="medal">🥇</div>
        <div class="cat-name">Melhor Performance · 2026</div>
        <div class="cat-title">Força Inabalável</div>
        <div class="cat-desc">Pela capacidade de acordar antes do sol, honrar cada treino e nunca deixar que o cansaço vença a determinação. Você prova todo dia que ouro é feito de repetição — e eu tenho sorte de ver isso de perto.</div>
      </div>
      <div class="card reveal">
        <div class="medal">❤️</div>
        <div class="cat-name">Prêmio do Coração · 12.06</div>
        <div class="cat-title">A Que Me Conquistou</div>
        <div class="cat-desc">Não foi num dia especial, não teve momento exato. Você foi chegando, do jeito que você é — determinada, intensa, bonita — e eu percebi que não queria mais ninguém ao meu lado.</div>
      </div>
      <div class="card reveal">
        <div class="medal">🏅</div>
        <div class="cat-name">Categoria Relacionamento</div>
        <div class="cat-title">Melhor Parceira</div>
        <div class="cat-desc">Por me mostrar o que é comprometimento de verdade. Por trazer leveza mesmo nos dias pesados. Por celebrar minhas vitórias com a mesma energia que celebra as suas.</div>
      </div>
      <div class="card reveal">
        <div class="medal">⭐</div>
        <div class="cat-name">Prêmio Especial · Dia dos Namorados</div>
        <div class="cat-title">Meu Amor</div>
        <div class="cat-desc">Hoje, 12 de junho, quero que você saiba: não precisa de prova, nem de medalha, nem de plateia pra ser especial pra mim. Você já é, todo dia, só por ser quem você é.</div>
      </div>
    </div>
  </section>

  <!-- GALLERY -->
  <section class="gallery-sec">
    <div class="gallery-inner">
      <p class="sec-label reveal">12 de Junho · Arquivo de memórias</p>
      <h2 class="sec-title reveal">Os nossos momentos<br>que ficam pra sempre</h2>
      <div class="grid">
        <div class="frame g1 reveal">{img_tag(f_foto1,"📸","foto1.jpg")}</div>
        <div class="frame reveal">{img_tag(f_foto2,"🏃‍♀️","foto2.jpg")}</div>
        <div class="frame reveal">{img_tag(f_foto3,"🎽","foto3.jpg")}</div>
        <div class="frame reveal">{img_tag(f_foto4,"💛","foto4.jpg")}</div>
        <div class="frame g4 reveal">{img_tag(f_foto5,"🌟","foto5.jpg")}</div>
      </div>
    </div>
  </section>

  <!-- DECLARATION -->
  <section class="decl">
    <p class="quote-mark reveal">"</p>
    <p class="decl-text reveal">
      Hoje é 12 de junho. O dia que o mundo escolheu pra falar de amor.<br/><br/>
      Mas eu não preciso de uma data pra saber o que sinto. Sinto isso quando te vejo acordar cedo pra treinar enquanto eu ainda estou dormindo. Quando você volta cansada e mesmo assim sorri. Quando você fala dos seus objetivos com aquela vontade nos olhos.<br/><br/>
      <em>Você é atleta na pista e no amor — com a mesma entrega, a mesma garra, a mesma presença.</em><br/><br/>
      Nesse dia dos namorados, eu quero que você saiba que torço por você em tudo — nas provas, nos treinos, nos sonhos, na vida. <em>E que não existe prêmio melhor do que poder te chamar de minha.</em>
    </p>
    <div class="decl-div"></div>
    <p class="decl-sig reveal">Com todo o meu amor,<br/>{SEU_NOME} 💛</p>
    <p class="decl-date reveal">12 de Junho de 2026 · Dia dos Namorados</p>
  </section>

  <!-- FOOTER -->
  <footer>
    <p class="footer-logo">Atleta do Ano · 12 de Junho · 2026</p>
    <p class="footer-txt">Edição Especial Dia dos Namorados · Feita com muito amor, só pra você 💛</p>
  </footer>

</div>

<script>
(function(){{
  const c = document.getElementById('ptcl');
  for(let i=0;i<40;i++){{
    const p=document.createElement('div'); p.className='p';
    p.style.left=Math.random()*100+'%';
    p.style.bottom='-4px';
    p.style.animationDuration=(4+Math.random()*7)+'s';
    p.style.animationDelay=(Math.random()*8)+'s';
    p.style.opacity=Math.random()*.7;
    c.appendChild(p);
  }}
}})();

function openCeremony(){{
  // toca a música
  {audio_play}

  // esconde intro
  const intro=document.getElementById('intro');
  intro.style.transition='opacity .8s ease';
  intro.style.opacity='0';
  intro.style.pointerEvents='none';
  setTimeout(()=>{{ intro.style.display='none'; }}, 900);

  // mostra conteúdo
  const main=document.getElementById('main');
  main.classList.add('on');
  setTimeout(()=>{{ main.scrollIntoView({{behavior:'smooth'}}); }}, 200);

  // reveal on scroll
  const io=new IntersectionObserver(entries=>{{
    entries.forEach(e=>{{ if(e.isIntersecting){{ e.target.classList.add('on'); io.unobserve(e.target); }} }});
  }},{{threshold:0.1}});
  document.querySelectorAll('.reveal').forEach(el=>io.observe(el));

  // counters
  const io2=new IntersectionObserver(entries=>{{
    entries.forEach(e=>{{
      if(e.isIntersecting){{
        const el=e.target, t=parseInt(el.dataset.target);
        if(isNaN(t)) return;
        let cur=0; const step=Math.ceil(t/60);
        const timer=setInterval(()=>{{
          cur=Math.min(cur+step,t);
          el.textContent=cur+(t===100?'%':'');
          if(cur>=t) clearInterval(timer);
        }},24);
        io2.unobserve(el);
      }}
    }});
  }},{{threshold:0.5}});
  document.querySelectorAll('[data-target]').forEach(el=>io2.observe(el));
}}
</script>
</body>
</html>"""

components.html(html, height=5400, scrolling=True)
