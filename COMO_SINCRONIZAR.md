# Como sincronizar dados no PerformanceRun

## Visão geral das fontes de dados

```
Garmin Connect  ──►  métricas de treino (distância, pace, FC, elevação, VO2max, etc.)
Strava          ──►  GPS / mapa (polyline, altitude stream, coordenadas)
```

O `sync.py` é o **único arquivo** que você precisa rodar. Ele busca tudo do Garmin,
depois enriquece com GPS do Strava para as atividades que ainda não têm mapa.

---

## Pré-requisitos (fazer uma vez)

### 1. Autenticação Garmin

Execute uma vez para gerar os tokens locais:

```bash
python garmin_login.py
```

> Se o plugin Garmin já funciona no Cowork, os tokens provavelmente já estão em
> `C:\Users\<você>\.garminconnect\` e você pode pular este passo.
> Para confirmar, basta rodar `python sync.py` — se logar, está ok.

### 2. Autenticação Strava (opcional — só para mapas)

O token do Strava fica em `data/token.json`. Se o arquivo já existe com um
`refresh_token` válido, o sync funciona automaticamente. Caso contrário, você
precisa gerar um novo token via OAuth no Strava Developer Portal.

> Sem Strava: as atividades aparecem normalmente em todas as abas, exceto
> nos **mapas** (tab Mapa), que ficam sem a linha de rota desenhada.

---

## Sincronização do dia a dia

### Passo 1 — Buscar atividades novas

```bash
cd C:\Users\gamsc209\Documents\ProjetosPyGit\strava
python sync.py
```

Isso faz:
- Busca atividades novas no Garmin (desde a última que está no CSV)
- Tenta adicionar GPS/mapa via Strava para as sem mapa
- Atualiza `data/processed/activities_consolidated.csv`
- Atualiza o banco SQLite local
- Atualiza `data/processed/garmin_insights.json` (VO2max, FC repouso, previsões)

### Passo 2 — Publicar no Streamlit Cloud

```bash
git add data/processed/
git commit -m "data: sync $(Get-Date -Format yyyy-MM-dd)"
git push
```

O Streamlit Cloud detecta o push e re-deploya automaticamente com os dados novos.

---

## Opções avançadas

```bash
# Só atualizar VO2max, FC repouso, previsões de prova (sem buscar corridas)
python sync.py --insights-only

# Sincronizar tudo desde o início (backfill completo — lento)
python sync.py --full

# Sincronizar a partir de uma data específica
python sync.py --date 2026-06-01
```

---

## Migrar dados locais para o banco SQLite

Se você adicionou atividades novas via CSV e quer popular o banco:

```bash
python -m src.db.migrate
```

---

## Backfill de polylines (mapas) para atividades antigas

Se você tem atividades sem mapa (importadas só via Garmin):

```bash
python backfill_polylines.py --token SEU_STRAVA_ACCESS_TOKEN
```

> Use sem `--full` para evitar erros 404 em atividades que existem só no Garmin.

---

## Plano de treino

O plano de treino é salvo **automaticamente no GitHub** a cada vez que você
importa ou edita na aba Plano (requer `GITHUB_TOKEN` configurado nos secrets
do Streamlit Cloud).

Para o GitHub Token funcionar, configure em:
**Streamlit Cloud → seu app → Settings → Secrets**

```toml
GITHUB_TOKEN = "ghp_seu_token_aqui"
```

O token precisa da permissão `repo` (leitura e escrita no repositório).

---

## Resumo rápido

| O que fazer | Comando |
|---|---|
| Sincronizar corridas novas | `python sync.py` |
| Só atualizar métricas Garmin | `python sync.py --insights-only` |
| Publicar no Streamlit | `git add data/processed/ && git commit -m "data: sync" && git push` |
| Popular banco SQLite local | `python -m src.db.migrate` |
| Login Garmin (primeira vez) | `python garmin_login.py` |
