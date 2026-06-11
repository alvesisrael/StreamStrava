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
# Fotos em namorada/fotos/:
#   capa.jpg   → retrato dela (3:4)
#   disco.jpg  → foto circular do disco (quadrada, vira vinil girando)
#   foto1.jpg  → vocês dois juntos (wide)
#   foto2.jpg  → ela treinando (quadrado)
#   foto3.jpg  → dia de prova (quadrado)
#   foto4.jpg  → momento especial (quadrado)
#   foto5.jpg  → foto do coração (wide)
#   musica.mp3 → música de vocês (toca automático)
# ════════════════════════════════════════════════

st.set_page_config(
    page_title=f"Atleta do Ano · {NOME_DELA}",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove TODO o chrome do Streamlit e faz o iframe preencher a tela
st.markdown("""
<style>
  [data-testid="stHeader"]        { display: none !important; }
  [data-testid="stToolbar"]       { display: none !important; }
  [data-testid="stDecoration"]    { display: none !important; }
  [data-testid="stStatusWidget"]  { display: none !important; }
  #MainMenu, footer               { display: none !important; }
  .block-container                { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"] { padding: 0 !important; }
  [data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
  iframe                          { height: 100vh !important; border: none !important; display: block !important; }
</style>
""", unsafe_allow_html=True)


BASE = os.path.join(os.path.dirname(__file__), "fotos")


def to_b64_img(nome):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        p = os.path.join(BASE, f"{nome}.{ext}")
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "jpeg" if ext in ("jpg","jpeg") else ext
            return f"data:image/{mime};base64,{data}"
    return None


def to_b64_audio():
    p = os.path.join(BASE, "musica.mp3")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:audio/mpeg;base64,{data}"


f_capa  = to_b64_img("capa")
f_disco = to_b64_img("disco")
f_foto1 = to_b64_img("foto1")
f_foto2 = to_b64_img("foto2")
f_foto3 = to_b64_img("foto3")
f_foto4 = to_b64_img("foto4")
f_foto5 = to_b64_img("foto5")
f_audio = to_b64_audio()


def img_tag(uri, icon, label):
    if uri:
        return f'<img src="{uri}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>'
    return f'<span style="font-size:26px;opacity:.2">{icon}</span><span style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#555">{label}</span>'


def capa_tag(uri):
    if uri:
        return f'<img src="{uri}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>'
    return '<span style="font-size:44px;opacity:.3">📸</span><span style="font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:#888">capa.jpg</span>'


# Disco de vinil — centro com foto ou padrão dourado
if f_disco:
    disc_center = f'background-image:url("{f_disco}");background-size:cover;background-position:center'
else:
    disc_center = "background:radial-gradient(circle,#C9A84C 0%,#8B6914 100%)"

audio_src   = f_audio or ""
has_audio   = "true" if f_audio else "false"

# ════════════════════════════════════════════════
html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Inter:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --gold:#C9A84C;--gl:#F0D080;--gd:#8B6914;
  --black:#0a0a0a;--dark:#111;--darker:#080808;--white:#f5f0e8;--gray:#777;
}}
html,body{{height:100%;scroll-behavior:smooth}}
body{{background:var(--black);color:var(--white);font-family:'Inter',sans-serif;overflow-x:hidden}}

/* ══ INTRO ══ */
#intro{{
  height:100vh;min-height:100%;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:18px;position:relative;overflow:hidden;background:var(--black);
}}
.ptcl{{position:absolute;inset:0;pointer-events:none;overflow:hidden}}
.p{{position:absolute;width:2px;height:2px;background:var(--gold);border-radius:50%;opacity:0;animation:floatUp linear infinite}}
.tag{{font-size:clamp(11px,2.5vw,14px);letter-spacing:.45em;text-transform:uppercase;color:var(--gold)}}
#intro h1{{
  font-family:'Playfair Display',serif;font-size:clamp(48px,13vw,100px);
  font-weight:900;line-height:.9;text-align:center;color:var(--gl);
}}
.year{{font-family:'Playfair Display',serif;font-size:clamp(16px,4vw,28px);letter-spacing:.55em;color:var(--gold)}}
.idivider{{width:90px;height:1px;background:var(--gd)}}
.play-btn{{
  margin-top:4px;
  width:clamp(68px,16vw,84px);height:clamp(68px,16vw,84px);border-radius:50%;
  border:2px solid var(--gold);background:transparent;color:var(--gl);
  font-size:clamp(28px,7vw,36px);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;animation:pulse 2s ease infinite;
}}
.play-btn:active{{transform:scale(.93)}}
.hint{{font-size:11px;letter-spacing:.35em;text-transform:uppercase;color:var(--gray);animation:blink 2s infinite}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(201,168,76,.5)}}60%{{box-shadow:0 0 0 18px rgba(201,168,76,0)}}}}
@keyframes blink{{0%,100%{{opacity:.25}}50%{{opacity:.85}}}}
@keyframes floatUp{{0%{{opacity:0;transform:translateY(0)}}10%{{opacity:.9}}90%{{opacity:.3}}100%{{opacity:0;transform:translateY(-100vh)}}}}

/* ══ MAIN ══ */
#main{{opacity:0;transition:opacity 1s ease}}
#main.on{{opacity:1}}

/* ══ HERO ══ */
.hero{{
  min-height:100vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;padding:60px 20px;
  background:radial-gradient(ellipse 80% 60% at 50% 50%,#1a1200,var(--black));
}}
.eyebrow{{font-size:11px;letter-spacing:.5em;text-transform:uppercase;color:var(--gold);margin-bottom:24px}}
.trophy{{width:80px;height:96px;filter:drop-shadow(0 0 20px rgba(201,168,76,.5));margin-bottom:28px}}
.htitle{{font-family:'Playfair Display',serif;font-size:clamp(48px,13vw,110px);font-weight:900;line-height:.88;color:var(--gl);margin-bottom:10px}}
.hname{{font-family:'Playfair Display',serif;font-size:clamp(28px,7vw,64px);font-weight:900;font-style:italic;color:var(--white);margin-bottom:10px}}
.hed{{font-size:11px;letter-spacing:.4em;text-transform:uppercase;color:var(--gray)}}
.hline{{width:160px;height:1px;background:var(--gd);margin:32px auto}}
.scrollhint{{font-size:11px;letter-spacing:.35em;text-transform:uppercase;color:var(--gray);animation:blink 2s infinite}}

/* ══ DISCO PLAYER ══ */
.player-wrap{{
  background:#0d0d0d;border-top:1px solid #1c1c1c;border-bottom:1px solid #1c1c1c;
  padding:32px 20px;display:flex;flex-direction:column;align-items:center;gap:20px;
}}
.vinyl{{
  width:clamp(200px,50vw,260px);height:clamp(200px,50vw,260px);
  border-radius:50%;position:relative;
  background:radial-gradient(circle,
    transparent 0%,transparent 31%,
    #1a1a1a 32%,#252525 34%,#1a1a1a 36%,#252525 38%,#1a1a1a 40%,
    #252525 42%,#1a1a1a 44%,#252525 46%,#1a1a1a 48%,#252525 50%,
    #111 52%,#1a1a1a 100%);
  animation:spin 5s linear infinite;
  animation-play-state:paused;
  box-shadow:0 0 40px rgba(0,0,0,.9),0 0 0 2px #222;
}}
.vinyl.playing{{animation-play-state:running}}
.vinyl-label{{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:65%;height:65%;border-radius:50%;overflow:hidden;
  border:3px solid #444;
  {disc_center};
}}
.vinyl-hole{{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:5%;height:5%;border-radius:50%;background:#0a0a0a;
  border:1px solid #333;z-index:2;
}}
@keyframes spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
.player-ctrl{{display:flex;align-items:center;gap:16px}}
.player-btn{{
  width:48px;height:48px;border-radius:50%;border:1.5px solid var(--gold);
  background:transparent;color:var(--gold);font-size:18px;
  display:flex;align-items:center;justify-content:center;cursor:pointer;
  transition:background .2s;
}}
.player-btn:hover{{background:rgba(201,168,76,.15)}}
.player-info{{text-align:center}}
.player-song{{font-size:14px;font-weight:500;color:var(--white);margin-bottom:3px}}
.player-artist{{font-size:12px;color:var(--gray)}}
.player-status{{font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--gold);margin-top:4px;min-height:14px}}

/* ══ PHOTO ══ */
.photo-sec{{padding:56px 20px;background:var(--darker);text-align:center}}
.photo-main{{
  width:min(360px,86vw);aspect-ratio:3/4;margin:0 auto 28px;
  border:1px solid var(--gd);background:#1a1200;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;
  position:relative;overflow:hidden;
}}
.photo-main::before,.photo-main::after{{content:'';position:absolute;width:18px;height:18px;border-color:var(--gold);border-style:solid;z-index:2}}
.photo-main::before{{top:10px;left:10px;border-width:2px 0 0 2px}}
.photo-main::after{{bottom:10px;right:10px;border-width:0 2px 2px 0}}
.photo-cap{{font-family:'Playfair Display',serif;font-size:clamp(14px,3vw,17px);font-style:italic;color:var(--gl);max-width:480px;margin:0 auto;line-height:1.6}}

/* ══ STATS ══ */
.stats{{background:#0c0c0c;border-top:1px solid #1a1a1a;border-bottom:1px solid #1a1a1a;padding:44px 20px}}
.stats-g{{max-width:840px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:28px;text-align:center}}
.sn{{font-family:'Playfair Display',serif;font-size:clamp(30px,7vw,50px);font-weight:900;color:var(--gold);line-height:1}}
.sl{{font-size:10px;letter-spacing:.3em;text-transform:uppercase;color:var(--gray);margin-top:6px}}

/* ══ CARDS ══ */
.sec{{padding:56px 20px;max-width:860px;margin:0 auto}}
.sl2{{font-size:11px;letter-spacing:.5em;text-transform:uppercase;color:var(--gold);margin-bottom:12px}}
.st{{font-family:'Playfair Display',serif;font-size:clamp(24px,5vw,42px);font-weight:700;margin-bottom:36px;line-height:1.2}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}}
.card{{background:#111;border:1px solid #1e1e1e;border-top:2px solid var(--gd);padding:26px 22px;transition:transform .3s,border-color .3s}}
.card:hover{{transform:translateY(-4px);border-top-color:var(--gold)}}
.medal{{font-size:28px;margin-bottom:12px}}
.cn{{font-size:10px;letter-spacing:.4em;text-transform:uppercase;color:var(--gold);margin-bottom:8px}}
.ct{{font-family:'Playfair Display',serif;font-size:19px;font-weight:700;margin-bottom:8px}}
.cd{{font-size:13px;line-height:1.7;color:#999}}

/* ══ GALLERY ══ */
.gal-sec{{padding:56px 20px;background:var(--darker)}}
.gal-inner{{max-width:860px;margin:0 auto}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:32px}}
.grid .g1{{grid-column:1/3}}.grid .g4{{grid-column:2/4}}
.frame{{background:#1a1a0a;border:1px solid #2a2a1a;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;overflow:hidden;position:relative}}
.g1,.g4{{aspect-ratio:16/9}}

/* ══ DECLARATION ══ */
.decl{{padding:72px 20px;max-width:640px;margin:0 auto;text-align:center}}
.qm{{font-family:'Playfair Display',serif;font-size:90px;line-height:.3;color:var(--gd);opacity:.4;margin-bottom:16px}}
.dt{{font-family:'Playfair Display',serif;font-size:clamp(16px,3vw,22px);line-height:1.9;font-style:italic;color:var(--white);margin-bottom:36px}}
.dt em{{color:var(--gl);font-style:normal}}
.dsig{{font-family:'Playfair Display',serif;font-size:clamp(18px,4vw,24px);color:var(--gold);font-style:italic}}
.ddate{{font-size:11px;letter-spacing:.3em;text-transform:uppercase;color:var(--gray);margin-top:8px}}
.ddiv{{width:60px;height:1px;background:var(--gd);margin:24px auto}}

/* ══ FOOTER ══ */
footer{{padding:44px 20px;text-align:center;border-top:1px solid #1a1a1a}}
.flogo{{font-family:'Playfair Display',serif;font-size:12px;letter-spacing:.4em;text-transform:uppercase;color:var(--gold);margin-bottom:10px}}
.ftxt{{font-size:11px;color:#444;letter-spacing:.15em}}

/* ══ REVEAL ══ */
.reveal{{opacity:0;transform:translateY(24px);transition:opacity .8s ease,transform .8s ease}}
.reveal.on{{opacity:1;transform:none}}

/* ══ MOBILE ══ */
@media(max-width:560px){{
  .grid{{grid-template-columns:1fr 1fr}}
  .grid .g1,.grid .g4{{grid-column:1/-1}}
  .cards{{grid-template-columns:1fr}}
  .sec{{padding:40px 16px}}
  .decl{{padding:56px 16px}}
}}
</style>
</head>
<body>

<audio id="snd" src="{audio_src}" loop preload="auto"></audio>

<!-- ══ INTRO ══ -->
<div id="intro">
  <div class="ptcl" id="ptcl"></div>
  <p class="tag">12 de Junho · Dia dos Namorados</p>
  <h1>Atleta<br>do Ano</h1>
  <p class="year">2 0 2 6</p>
  <div class="idivider"></div>
  <button class="play-btn" onclick="openCeremony()" aria-label="Abrir surpresa">▶</button>
  <p class="hint">toque para abrir</p>
</div>

<!-- ══ MAIN ══ -->
<div id="main">

  <!-- HERO -->
  <section class="hero">
    <p class="eyebrow">A premiação mais especial de todas as edições</p>
    <svg class="trophy" viewBox="0 0 100 120" fill="none">
      <defs><linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#8B6914"/><stop offset="50%" stop-color="#F0D080"/><stop offset="100%" stop-color="#C9A84C"/>
      </linearGradient></defs>
      <path d="M30 10H70L62 60H38Z" fill="url(#tg)" opacity=".9"/>
      <path d="M30 18Q10 20 12 38Q14 52 30 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="M70 18Q90 20 88 38Q86 52 70 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
      <rect x="44" y="60" width="12" height="22" fill="url(#tg)" rx="2"/>
      <rect x="28" y="82" width="44" height="10" fill="url(#tg)" rx="3"/>
      <rect x="22" y="92" width="56" height="8" fill="url(#tg)" rx="3"/>
      <path d="M50 22L52.4 29.2L60 29.2L54 33.8L56.4 41L50 36.4L43.6 41L46 33.8L40 29.2L47.6 29.2Z" fill="#fff" opacity=".9"/>
    </svg>
    <p class="eyebrow">12 de Junho · Dia dos Namorados 2026</p>
    <h1 class="htitle">ATLETA<br>DO ANO</h1>
    <p class="hname">{NOME_DELA} 🏆</p>
    <p class="hed">Categoria: Coração de Campeã · 12.06.2026</p>
    <div class="hline"></div>
    <p class="scrollhint">↓ Continue descobrindo</p>
  </section>

  <!-- DISCO PLAYER -->
  <div class="player-wrap">
    <div class="vinyl" id="vinyl">
      <div class="vinyl-label"></div>
      <div class="vinyl-hole"></div>
    </div>
    <div class="player-info">
      <div class="player-song">Diagnóstico</div>
      <div class="player-artist">Ryu, The Runner, BIN, Thiago Sub</div>
      <div class="player-status" id="playerStatus"></div>
    </div>
    <div class="player-ctrl">
      <button class="player-btn" id="playPauseBtn" onclick="togglePlay()" aria-label="Play/Pause">▶</button>
    </div>
  </div>

  <!-- FOTO PRINCIPAL -->
  <section class="photo-sec">
    <div class="photo-main reveal">{capa_tag(f_capa)}</div>
    <p class="photo-cap reveal">"A campeã que escolhi amar todos os dias."</p>
  </section>

  <!-- STATS -->
  <div class="stats">
    <div class="stats-g">
      <div class="reveal"><div class="sn" data-target="1">0</div><div class="sl">namorado sortudo</div></div>
      <div class="reveal"><div class="sn" data-target="12">0</div><div class="sl">de junho · nosso dia</div></div>
      <div class="reveal"><div class="sn" data-target="100">0</div><div class="sl">% do meu coração</div></div>
      <div class="reveal"><div class="sn">∞</div><div class="sl">razões pra te amar</div></div>
    </div>
  </div>

  <!-- CATEGORIES -->
  <section class="sec">
    <p class="sl2 reveal">Dia dos Namorados · Categorias premiadas</p>
    <h2 class="st reveal">Tudo que você é<br>e que me conquistou</h2>
    <div class="cards">
      <div class="card reveal"><div class="medal">🥇</div><div class="cn">Melhor Performance · 2026</div><div class="ct">Força Inabalável</div><div class="cd">Pela capacidade de acordar antes do sol, honrar cada treino e nunca deixar que o cansaço vença a determinação. Você prova todo dia que ouro é feito de repetição — e eu tenho sorte de ver isso de perto e poder ajudar em alguns dias.</div></div>
      <div class="card reveal"><div class="medal">❤️</div><div class="cn">Prêmio do Coração · 12.06</div><div class="ct">A Que Me Conquistou</div><div class="cd">Não foi num dia especial, não teve momento exato. Me aproximei o quanto antes porque sabia que você era a pessoa certa e eu não queria mais ninguém ao meu lado.</div></div>
      <div class="card reveal"><div class="medal">🏅</div><div class="cn">Categoria Relacionamento</div><div class="ct">Melhor Parceira</div><div class="cd">Por me mostrar o que é comprometimento de verdade. Por trazer leveza mesmo nos dias pesados. Por celebrar minhas vitórias com a mesma energia que celebra as suas.</div></div>
      <div class="card reveal"><div class="medal">⭐</div><div class="cn">Prêmio Especial · Dia dos Namorados</div><div class="ct">Meu Amor</div><div class="cd">Hoje, 12 de junho, quero que você saiba: não precisa de prova, nem de medalha, nem de plateia pra ser especial pra mim. Você já é, todo dia, só por ser quem você é TE AMO nenem.</div></div>
    </div>
  </section>

  <!-- GALLERY -->
  <section class="gal-sec">
    <div class="gal-inner">
      <p class="sl2 reveal">12 de Junho · Arquivo de memórias</p>
      <h2 class="st reveal">Os nossos momentos<br>que ficam pra sempre</h2>
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
    <p class="qm reveal">"</p>
    <p class="dt reveal">
      Hoje é 12 de junho. O dia que o mundo escolheu pra falar de amor.<br/><br/>
      Mas eu não preciso de uma data pra saber o que sinto. Sinto isso quando te vejo acordar cedo pra treinar enquanto eu ainda estou dormindo com a ratinha. Quando você volta cansada e mesmo assim sorri. Quando você fala dos seus objetivos com aquela vontade nos olhos.<br/><br/>
      <em>Você é atleta na pista e no amor — com a mesma entrega, a mesma garra, a mesma presença.</em><br/><br/>
      Nesse dia dos namorados, eu quero que você saiba que estou com você em tudo — nas provas, nos treinos, nos sonhos, na vida. <em>E que não existe prêmio melhor do que poder te chamar de minha.</em>
    </p>
    <div class="ddiv"></div>
    <p class="dsig reveal">Com todo o meu amor,<br/>{SEU_NOME} 💛</p>
    <p class="ddate reveal">12 de Junho de 2026 · Dia dos Namorados</p>
  </section>

  <footer>
    <p class="flogo">Atleta do Ano · 12 de Junho · 2026</p>
    <p class="ftxt">Edição Especial Dia dos Namorados · Feita com muito amor, só pra você 💛</p>
  </footer>

</div><!-- /#main -->

<script>
// ── partículas
(function(){{
  const c=document.getElementById('ptcl');
  for(let i=0;i<40;i++){{
    const p=document.createElement('div'); p.className='p';
    p.style.cssText=`left:${{Math.random()*100}}%;bottom:-4px;animation-duration:${{4+Math.random()*7}}s;animation-delay:${{Math.random()*8}}s;opacity:${{Math.random()*.7}}`;
    c.appendChild(p);
  }}
}})();

// ── áudio
const snd       = document.getElementById('snd');
const vinyl     = document.getElementById('vinyl');
const playBtn   = document.getElementById('playPauseBtn');
const statusEl  = document.getElementById('playerStatus');
const hasAudio  = {has_audio};
let playing     = false;

function setPlaying(v){{
  playing = v;
  vinyl.classList.toggle('playing', v);
  playBtn.textContent = v ? '⏸' : '▶';
  statusEl.textContent = v ? '♪ tocando agora' : '';
}}

function togglePlay(){{
  if(!hasAudio){{ statusEl.textContent='Adicione musica.mp3 na pasta fotos/'; return; }}
  if(playing){{ snd.pause(); setPlaying(false); }}
  else {{ snd.play().then(()=>setPlaying(true)).catch(()=>{{ statusEl.textContent='Erro ao tocar — tente novamente'; }}); }}
}}

// ── abrir cerimônia
function openCeremony(){{
  // toca música automaticamente (usuário acabou de clicar = interação válida)
  if(hasAudio){{
    snd.play().then(()=>setPlaying(true)).catch(()=>{{}});
  }}

  const intro = document.getElementById('intro');
  intro.style.transition='opacity .8s ease';
  intro.style.opacity='0';
  intro.style.pointerEvents='none';
  setTimeout(()=>{{ intro.style.display='none'; }}, 900);

  const main = document.getElementById('main');
  main.classList.add('on');
  setTimeout(()=>{{ window.scrollTo({{top:0,behavior:'smooth'}}); }}, 100);

  // reveal por scroll
  const io = new IntersectionObserver(en=>{{
    en.forEach(e=>{{ if(e.isIntersecting){{ e.target.classList.add('on'); io.unobserve(e.target); }} }});
  }},{{threshold:0.08}});
  document.querySelectorAll('.reveal').forEach(el=>io.observe(el));

  // contadores
  const io2 = new IntersectionObserver(en=>{{
    en.forEach(e=>{{
      if(e.isIntersecting){{
        const el=e.target, t=parseInt(el.dataset.target);
        if(isNaN(t)) return;
        let cur=0; const step=Math.ceil(t/60);
        const tmr=setInterval(()=>{{ cur=Math.min(cur+step,t); el.textContent=cur+(t===100?'%':''); if(cur>=t)clearInterval(tmr); }},24);
        io2.unobserve(el);
      }}
    }});
  }},{{threshold:0.5}});
  document.querySelectorAll('[data-target]').forEach(el=>io2.observe(el));
}}
</script>
</body>
</html>"""

components.html(html, height=800, scrolling=True)
