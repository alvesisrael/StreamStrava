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
NOME_DELA       = "Thais"           # Nome dela
SEU_NOME        = "Israel"        # Seu nome
SPOTIFY_TRACK   = "3IbztmgBw5OwtVhKfzhuWp"  # ID da música no Spotify
# Para pegar o ID: abra a música no Spotify → Compartilhar → Copiar link
# O link é tipo: https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC
# O ID é a parte depois de /track/

# Fotos — coloque os arquivos na pasta namorada/fotos/
# Nomes esperados (jpg ou png):
#   capa.jpg       → foto principal grande dela
#   foto1.jpg      → foto favorita de vocês dois (wide)
#   foto2.jpg      → ela treinando
#   foto3.jpg      → dia de prova
#   foto4.jpg      → momento especial
#   foto5.jpg      → foto do coração (wide)
# ════════════════════════════════════════════════

st.set_page_config(
    page_title=f"Atleta do Ano · {NOME_DELA}",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Esconde menu e rodapé do Streamlit
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  [data-testid="stAppViewContainer"] { background: #0a0a0a; }
</style>
""", unsafe_allow_html=True)


def img_to_b64(path: str) -> str | None:
    """Converte imagem local para base64 URI."""
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


# Carrega fotos
BASE = os.path.join(os.path.dirname(__file__), "fotos")

def foto(nome):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        path = os.path.join(BASE, f"{nome}.{ext}")
        uri = img_to_b64(path)
        if uri:
            return uri
    return None

f_capa  = foto("capa")
f_foto1 = foto("foto1")
f_foto2 = foto("foto2")
f_foto3 = foto("foto3")
f_foto4 = foto("foto4")
f_foto5 = foto("foto5")

def img_tag(uri, placeholder_icon, placeholder_label):
    if uri:
        return f'<img src="{uri}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" />'
    return f'<span class="ph-icon">{placeholder_icon}</span><span class="ph-label">{placeholder_label}</span>'

def capa_tag(uri):
    if uri:
        return f'<img src="{uri}" alt="Foto dela" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" />'
    return '<span class="photo-placeholder-icon">📸</span><span class="photo-placeholder-text">Coloque capa.jpg na pasta fotos/</span>'

# Spotify embed
spotify_html = f"""
<iframe style="border-radius:8px"
  src="https://open.spotify.com/embed/track/{SPOTIFY_TRACK}?utm_source=generator&theme=0"
  width="100%" height="80" frameBorder="0" allowfullscreen=""
  allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
  loading="lazy">
</iframe>
""" if SPOTIFY_TRACK else ""

# ════════════════════════════════════════════════
# HTML COMPLETO
# ════════════════════════════════════════════════
html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --gold: #C9A84C; --gold-light: #F0D080; --gold-dark: #8B6914;
      --black: #0a0a0a; --darker: #080808; --white: #f5f0e8; --gray: #888;
    }}
    html {{ scroll-behavior: smooth; }}
    body {{ background: var(--black); color: var(--white); font-family: 'Inter', sans-serif; overflow-x: hidden; }}

    /* INTRO */
    #intro {{
      position: sticky; top: 0;
      width: 100%; height: 100vh;
      background: var(--black); z-index: 100;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 24px; cursor: pointer;
      transition: opacity 0.8s ease;
    }}
    #intro.hidden {{ opacity: 0; pointer-events: none; margin-top: -100vh; }}
    .intro-logo {{
      font-family: 'Playfair Display', serif; font-size: clamp(14px,2vw,18px);
      letter-spacing: 0.4em; text-transform: uppercase; color: var(--gold);
    }}
    .intro-title {{
      font-family: 'Playfair Display', serif; font-size: clamp(42px,10vw,96px);
      font-weight: 900; line-height: 1; text-align: center;
      background: linear-gradient(135deg, var(--gold-dark), var(--gold-light), var(--gold));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .intro-year {{
      font-family: 'Playfair Display', serif; font-size: clamp(18px,4vw,36px);
      letter-spacing: 0.6em; color: var(--gold-light);
    }}
    .intro-divider {{
      width: 120px; height: 1px;
      background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }}
    .intro-hint {{
      font-size: 12px; letter-spacing: 0.3em; text-transform: uppercase; color: var(--gray);
      animation: pulse 2s ease infinite;
    }}
    .gold-particles {{ position: absolute; inset: 0; pointer-events: none; overflow: hidden; }}
    .particle {{
      position: absolute; width: 2px; height: 2px; background: var(--gold);
      border-radius: 50%; animation: floatUp linear infinite; opacity: 0;
    }}

    /* MAIN */
    #main {{ opacity: 0; transition: opacity 1s ease; }}
    #main.visible {{ opacity: 1; }}

    /* HERO */
    .hero {{
      min-height: 100vh; display: flex; flex-direction: column;
      align-items: center; justify-content: center; text-align: center;
      padding: 60px 24px; position: relative;
      background: radial-gradient(ellipse 80% 60% at 50% 50%, #1a1200 0%, var(--black) 70%);
      overflow: hidden;
    }}
    .hero::before {{
      content: ''; position: absolute; inset: 0;
      background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c9a84c' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }}
    .hero-eyebrow {{
      font-size: 11px; letter-spacing: 0.5em; text-transform: uppercase;
      color: var(--gold); margin-bottom: 32px;
    }}
    .trophy-wrap {{ margin-bottom: 40px; }}
    .trophy-svg {{ width: 100px; height: 120px; filter: drop-shadow(0 0 30px rgba(201,168,76,0.5)); }}
    .hero-title {{
      font-family: 'Playfair Display', serif; font-size: clamp(48px,12vw,120px);
      font-weight: 900; line-height: 0.9;
      background: linear-gradient(160deg, var(--gold-dark) 0%, var(--gold-light) 50%, var(--gold) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}
    .hero-name {{
      font-family: 'Playfair Display', serif; font-size: clamp(36px,8vw,80px);
      font-weight: 900; font-style: italic; color: var(--gold-light);
      margin-bottom: 16px;
    }}
    .hero-edition {{
      font-size: 13px; letter-spacing: 0.4em; text-transform: uppercase; color: var(--gray);
    }}
    .hero-line {{
      width: 200px; height: 1px;
      background: linear-gradient(90deg, transparent, var(--gold), transparent);
      margin: 40px auto;
    }}
    .scroll-hint {{
      font-size: 11px; letter-spacing: 0.4em; text-transform: uppercase; color: var(--gray);
      animation: pulse 2s ease infinite;
    }}

    /* MUSIC */
    .music-bar {{
      background: #0f0f0f; border-top: 1px solid #1a1a1a; border-bottom: 1px solid #1a1a1a;
      padding: 16px 32px; position: sticky; top: 0; z-index: 50;
    }}

    /* PHOTO SECTION */
    .photo-section {{ padding: 80px 24px; background: var(--darker); text-align: center; }}
    .photo-frame-main {{
      width: min(420px, 90vw); aspect-ratio: 3/4; margin: 0 auto 40px;
      border: 1px solid var(--gold-dark); background: #1a1200;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 16px; position: relative; overflow: hidden;
    }}
    .photo-frame-main::before, .photo-frame-main::after {{
      content: ''; position: absolute; width: 24px; height: 24px;
      border-color: var(--gold); border-style: solid; z-index: 2;
    }}
    .photo-frame-main::before {{ top: 12px; left: 12px; border-width: 2px 0 0 2px; }}
    .photo-frame-main::after  {{ bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }}
    .photo-placeholder-icon {{ font-size: 48px; opacity: 0.3; }}
    .photo-placeholder-text {{ font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--gray); }}
    .photo-caption {{ font-family: 'Playfair Display', serif; font-size: 18px; font-style: italic; color: var(--gold-light); }}

    /* STATS */
    .stats-section {{
      background: #0c0c0c; border-top: 1px solid #1a1a1a; border-bottom: 1px solid #1a1a1a;
      padding: 60px 24px;
    }}
    .stats-inner {{
      max-width: 900px; margin: 0 auto;
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 40px; text-align: center;
    }}
    .stat-number {{
      font-family: 'Playfair Display', serif; font-size: clamp(36px,6vw,56px);
      font-weight: 900; color: var(--gold); line-height: 1;
    }}
    .stat-label {{ font-size: 11px; letter-spacing: 0.3em; text-transform: uppercase; color: var(--gray); margin-top: 8px; }}

    /* CATEGORIES */
    .section {{ padding: 80px 24px; max-width: 900px; margin: 0 auto; }}
    .section-label {{ font-size: 11px; letter-spacing: 0.5em; text-transform: uppercase; color: var(--gold); margin-bottom: 16px; }}
    .section-title {{ font-family: 'Playfair Display', serif; font-size: clamp(28px,5vw,48px); font-weight: 700; margin-bottom: 48px; line-height: 1.2; }}
    .categories-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; }}
    .category-card {{
      background: #111; border: 1px solid #1e1e1e; border-top: 2px solid var(--gold-dark);
      padding: 32px 28px; transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .category-card:hover {{ transform: translateY(-4px); border-top-color: var(--gold); }}
    .category-medal {{ font-size: 32px; margin-bottom: 16px; }}
    .category-name {{ font-size: 10px; letter-spacing: 0.4em; text-transform: uppercase; color: var(--gold); margin-bottom: 12px; }}
    .category-winner {{ font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; margin-bottom: 12px; }}
    .category-desc {{ font-size: 14px; line-height: 1.7; color: #aaa; }}

    /* GALLERY */
    .gallery-section {{ padding: 80px 24px; background: var(--darker); }}
    .gallery-inner {{ max-width: 900px; margin: 0 auto; }}
    .gallery-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr);
      grid-template-rows: auto auto; gap: 12px; margin-top: 40px;
    }}
    .gallery-grid .g1 {{ grid-column: 1 / 3; }}
    .gallery-grid .g4 {{ grid-column: 2 / 4; }}
    .photo-frame {{
      background: #1a1a0a; border: 1px solid #2a2a1a; aspect-ratio: 1;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 10px; overflow: hidden; position: relative;
    }}
    .g1, .g4 {{ aspect-ratio: 16/9; }}
    .ph-icon {{ font-size: 28px; opacity: 0.2; }}
    .ph-label {{ font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: #555; }}

    /* DECLARATION */
    .declaration-section {{ padding: 100px 24px; max-width: 700px; margin: 0 auto; text-align: center; }}
    .decl-quote-mark {{
      font-family: 'Playfair Display', serif; font-size: 120px; line-height: 0.5;
      color: var(--gold-dark); opacity: 0.4; margin-bottom: 24px;
    }}
    .decl-text {{
      font-family: 'Playfair Display', serif; font-size: clamp(18px,3vw,26px);
      line-height: 1.8; font-style: italic; color: var(--white); margin-bottom: 48px;
    }}
    .decl-text em {{ color: var(--gold-light); font-style: normal; }}
    .decl-signature {{ font-family: 'Playfair Display', serif; font-size: 28px; color: var(--gold); font-style: italic; }}
    .decl-date {{ font-size: 12px; letter-spacing: 0.3em; text-transform: uppercase; color: var(--gray); margin-top: 12px; }}
    .decl-divider {{ width: 80px; height: 1px; background: var(--gold-dark); margin: 32px auto; }}

    /* FOOTER */
    footer {{ padding: 60px 24px; text-align: center; border-top: 1px solid #1a1a1a; }}
    .footer-logo {{ font-family: 'Playfair Display', serif; font-size: 14px; letter-spacing: 0.4em; text-transform: uppercase; color: var(--gold); margin-bottom: 16px; }}
    .footer-text {{ font-size: 12px; color: #444; letter-spacing: 0.2em; }}

    /* ANIMATIONS */
    @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(24px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
    @keyframes pulse {{ 0%,100% {{ opacity:0.3; }} 50% {{ opacity:0.8; }} }}
    @keyframes floatUp {{
      0%   {{ opacity:0; transform:translateY(0) scale(1); }}
      10%  {{ opacity:1; }}
      90%  {{ opacity:0.5; }}
      100% {{ opacity:0; transform:translateY(-100vh) scale(0.5); }}
    }}
    @keyframes equalizer {{ from {{ transform:scaleY(0.4); }} to {{ transform:scaleY(1); }} }}

    /* REVEAL */
    .reveal {{ opacity:0; transform:translateY(30px); transition:opacity 0.8s ease,transform 0.8s ease; }}
    .reveal.visible {{ opacity:1; transform:none; }}

    @media (max-width:600px) {{
      .gallery-grid {{ grid-template-columns: 1fr 1fr; }}
      .gallery-grid .g1, .gallery-grid .g4 {{ grid-column: 1 / -1; }}
      .music-bar {{ padding: 12px 16px; }}
    }}
  </style>
</head>
<body>

<!-- INTRO -->
<div id="intro" onclick="openCeremony()">
  <div class="gold-particles" id="particles"></div>
  <p class="intro-logo">12 de Junho · Dia dos Namorados</p>
  <h1 class="intro-title">Atleta<br>do Ano</h1>
  <p class="intro-year">2 0 2 6</p>
  <div class="intro-divider"></div>
  <p class="intro-hint">Toque para revelar</p>
</div>

<!-- MAIN -->
<div id="main">

  <!-- HERO -->
  <section class="hero">
    <p class="hero-eyebrow">12 de Junho · Dia dos Namorados 2026</p>
    <div class="trophy-wrap">
      <svg class="trophy-svg" viewBox="0 0 100 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#8B6914"/>
            <stop offset="50%" stop-color="#F0D080"/>
            <stop offset="100%" stop-color="#C9A84C"/>
          </linearGradient>
        </defs>
        <path d="M30 10 H70 L62 60 H38 Z" fill="url(#tg)" opacity="0.9"/>
        <path d="M30 18 Q10 20 12 38 Q14 52 30 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
        <path d="M70 18 Q90 20 88 38 Q86 52 70 50" stroke="url(#tg)" stroke-width="4" fill="none" stroke-linecap="round"/>
        <rect x="44" y="60" width="12" height="22" fill="url(#tg)" rx="2"/>
        <rect x="28" y="82" width="44" height="10" fill="url(#tg)" rx="3"/>
        <rect x="22" y="92" width="56" height="8" fill="url(#tg)" rx="3"/>
        <path d="M50 22 L52.4 29.2 L60 29.2 L54 33.8 L56.4 41 L50 36.4 L43.6 41 L46 33.8 L40 29.2 L47.6 29.2 Z" fill="#fff" opacity="0.9"/>
      </svg>
    </div>
    <p class="hero-eyebrow" style="animation-delay:0.7s">A premiação mais especial de todas as edições</p>
    <h1 class="hero-title">ATLETA<br>DO ANO</h1>
    <p class="hero-name">{NOME_DELA} 🏆</p>
    <p class="hero-edition">Categoria: Coração de Campeã · 12.06.2026</p>
    <div class="hero-line"></div>
    <p class="scroll-hint">↓ Continue descobrindo</p>
  </section>

  <!-- MUSIC -->
  <div class="music-bar">
    {spotify_html if spotify_html else '<p style="text-align:center;color:#888;font-size:12px;letter-spacing:0.2em;">♪ A música de vocês tocando aqui</p>'}
  </div>

  <!-- FOTO PRINCIPAL -->
  <section class="photo-section">
    <div class="photo-frame-main reveal">
      {capa_tag(f_capa)}
    </div>
    <p class="photo-caption reveal">"A campeã que escolhi amar todos os dias — dentro e fora das pistas."</p>
  </section>

  <!-- STATS -->
  <div class="stats-section">
    <div class="stats-inner">
      <div class="reveal">
        <div class="stat-number" data-target="1">0</div>
        <div class="stat-label">namorado sortudo</div>
      </div>
      <div class="reveal">
        <div class="stat-number" data-target="12">0</div>
        <div class="stat-label">de junho · nosso dia</div>
      </div>
      <div class="reveal">
        <div class="stat-number" data-target="100">0</div>
        <div class="stat-label">% do meu coração</div>
      </div>
      <div class="reveal">
        <div class="stat-number">∞</div>
        <div class="stat-label">razões pra te amar</div>
      </div>
    </div>
  </div>

  <!-- CATEGORIES -->
  <section class="section">
    <p class="section-label reveal">Dia dos Namorados · Categorias premiadas</p>
    <h2 class="section-title reveal">Tudo que você é<br>e que me conquistou</h2>
    <div class="categories-grid">

      <div class="category-card reveal">
        <div class="category-medal">🥇</div>
        <div class="category-name">Melhor Performance · 2026</div>
        <div class="category-winner">Força Inabalável</div>
        <div class="category-desc">
          Pela capacidade de acordar antes do sol, honrar cada treino e nunca deixar que o cansaço vença a determinação. Você prova todo dia que ouro é feito de repetição — e eu tenho sorte de ver isso de perto.
        </div>
      </div>

      <div class="category-card reveal">
        <div class="category-medal">❤️</div>
        <div class="category-name">Prêmio do Coração · 12.06</div>
        <div class="category-winner">A Que Me Conquistou</div>
        <div class="category-desc">
          Não foi num dia especial, não teve momento exato. Você foi chegando, do jeito que você é — determinada, intensa, bonita — e eu percebi que não queria mais ninguém ao meu lado.
        </div>
      </div>

      <div class="category-card reveal">
        <div class="category-medal">🏅</div>
        <div class="category-name">Categoria Relacionamento</div>
        <div class="category-winner">Melhor Parceira</div>
        <div class="category-desc">
          Por me mostrar o que é comprometimento de verdade. Por trazer leveza mesmo nos dias pesados. Por celebrar minhas vitórias com a mesma energia que celebra as suas.
        </div>
      </div>

      <div class="category-card reveal">
        <div class="category-medal">⭐</div>
        <div class="category-name">Prêmio Especial · Dia dos Namorados</div>
        <div class="category-winner">Meu Amor</div>
        <div class="category-desc">
          Hoje, 12 de junho, quero que você saiba: não precisa de prova, nem de medalha, nem de plateia pra ser especial pra mim. Você já é, todo dia, só por ser quem você é.
        </div>
      </div>

    </div>
  </section>

  <!-- GALLERY -->
  <section class="gallery-section">
    <div class="gallery-inner">
      <p class="section-label reveal">12 de Junho · Arquivo de memórias</p>
      <h2 class="section-title reveal">Os nossos momentos<br>que ficam pra sempre</h2>
      <div class="gallery-grid">
        <div class="photo-frame g1 reveal">{img_tag(f_foto1, "📸", "foto1.jpg")}</div>
        <div class="photo-frame reveal">{img_tag(f_foto2, "🏃‍♀️", "foto2.jpg")}</div>
        <div class="photo-frame reveal">{img_tag(f_foto3, "🎽", "foto3.jpg")}</div>
        <div class="photo-frame reveal">{img_tag(f_foto4, "💛", "foto4.jpg")}</div>
        <div class="photo-frame g4 reveal">{img_tag(f_foto5, "🌟", "foto5.jpg")}</div>
      </div>
    </div>
  </section>

  <!-- DECLARATION -->
  <section class="declaration-section">
    <p class="decl-quote-mark reveal">"</p>
    <p class="decl-text reveal">
      Hoje é 12 de junho. O dia que o mundo escolheu pra falar de amor.<br/><br/>
      Mas eu não preciso de uma data pra saber o que sinto. Sinto isso quando te vejo acordar cedo pra treinar enquanto eu ainda estou dormindo. Quando você volta cansada e mesmo assim sorri. Quando você fala dos seus objetivos com aquela vontade nos olhos.<br/><br/>
      <em>Você é atleta na pista e no amor — com a mesma entrega, a mesma garra, a mesma presença.</em><br/><br/>
      Nesse dia dos namorados, eu quero que você saiba que torço por você em tudo — nas provas, nos treinos, nos sonhos, na vida. <em>E que não existe prêmio melhor do que poder te chamar de minha.</em>
    </p>
    <div class="decl-divider"></div>
    <p class="decl-signature reveal">Com todo o meu amor,<br/>{SEU_NOME} 💛</p>
    <p class="decl-date reveal">12 de Junho de 2026 · Dia dos Namorados</p>
  </section>

  <!-- FOOTER -->
  <footer>
    <p class="footer-logo">Atleta do Ano · 12 de Junho · 2026</p>
    <p class="footer-text">Edição Especial Dia dos Namorados · Feita com muito amor, só pra você 💛</p>
  </footer>

</div>

<script>
  (function() {{
    const c = document.getElementById('particles');
    for (let i = 0; i < 40; i++) {{
      const p = document.createElement('div');
      p.className = 'particle';
      p.style.left = Math.random() * 100 + '%';
      p.style.bottom = '-4px';
      p.style.animationDuration = (4 + Math.random() * 6) + 's';
      p.style.animationDelay    = (Math.random() * 6) + 's';
      p.style.opacity = Math.random() * 0.6;
      c.appendChild(p);
    }}
  }})();

  function openCeremony() {{
    const intro = document.getElementById('intro');
    const main  = document.getElementById('main');
    intro.classList.add('hidden');
    main.classList.add('visible');

    // Scroll pro topo do conteúdo após a transição
    setTimeout(() => {{
      main.scrollIntoView({{ behavior: 'smooth' }});
    }}, 900);

    const io = new IntersectionObserver((entries) => {{
      entries.forEach(e => {{ if (e.isIntersecting) {{ e.target.classList.add('visible'); io.unobserve(e.target); }} }});
    }}, {{ threshold: 0.12 }});
    document.querySelectorAll('.reveal').forEach(el => io.observe(el));

    const io2 = new IntersectionObserver((entries) => {{
      entries.forEach(e => {{
        if (e.isIntersecting) {{
          const el = e.target, t = parseInt(el.dataset.target);
          if (isNaN(t)) return;
          let cur = 0; const step = Math.ceil(t / 60);
          const timer = setInterval(() => {{
            cur = Math.min(cur + step, t);
            el.textContent = cur + (t === 100 ? '%' : '');
            if (cur >= t) clearInterval(timer);
          }}, 24);
          io2.unobserve(el);
        }}
      }});
    }}, {{ threshold: 0.5 }});
    document.querySelectorAll('[data-target]').forEach(c => io2.observe(c));
  }}
</script>
</body>
</html>
"""

components.html(html, height=5200, scrolling=True)
