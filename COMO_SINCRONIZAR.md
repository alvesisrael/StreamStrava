# Como sincronizar dados no PerformanceRun

## Visão geral das fontes de dados

```
Amazfit T-Rex 3  ──►  Zepp App  ──►  Strava  ──►  sync.py  ──►  dashboard
```

O Amazfit sincroniza automaticamente com o Strava via app Zepp após cada treino.
O `sync.py` busca tudo direto do Strava — é o **único arquivo** que você precisa rodar.

> **Dados históricos do Garmin** estão preservados no CSV e em `garmin_insights.json`.
> Eles não são mais atualizados, mas continuam visíveis no dashboard.

---

## Pré-requisitos (fazer uma vez)

### Autenticação Strava

O token do Strava fica em `data/token.json`. Se o arquivo já existe com um
`refresh_token` válido, o sync funciona automaticamente.

Para verificar:
```bash
python -c "import json; d=json.load(open('data/token.json')); print('OK' if d.get('refresh_token') else 'PRECISA RENOVAR')"
```

Se precisar renovar ou configurar pela primeira vez, siga o fluxo OAuth no
Strava Developer Portal e salve o resultado em `data/token.json`:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_at": 9999999999
}
```

---

## Sincronização do dia a dia

### Passo 1 — Buscar atividades novas

```bash
cd C:\Users\gamsc209\Documents\ProjetosPyGit\strava
python sync.py
```

Isso faz:
- Busca atividades novas no Strava (desde a última que está no CSV)
- Calcula TRIMP e índice de eficiência aeróbica automaticamente
- Atualiza `data/processed/activities_consolidated.csv`
- Atualiza `data/processed/activity_laps_consolidated.csv` (laps por atividade)
- Atualiza o banco SQLite local (se disponível)

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
# Backfill completo desde STRAVA_START (primeira vez ou recuperação)
python sync.py --full

# Sincronizar a partir de uma data específica
python sync.py --date 2026-06-01

# Só atualizar laps sem buscar atividades novas
python sync.py --laps-only
```

---

## Backfill de polylines (mapas) para atividades antigas

Se você tem atividades sem rota no mapa:

```bash
python backfill_polylines.py --token SEU_STRAVA_ACCESS_TOKEN
```

Para incluir também altitude GPS (stream detalhado):

```bash
python backfill_polylines.py --token SEU_STRAVA_ACCESS_TOKEN --altitude
```

---

## Migrar dados locais para o banco SQLite

Se você adicionou atividades novas via CSV e quer popular o banco:

```bash
python -m src.db.migrate
```

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
| Backfill completo | `python sync.py --full` |
| Só laps | `python sync.py --laps-only` |
| Publicar no Streamlit | `git add data/processed/ && git commit -m "data: sync" && git push` |
| Popular banco SQLite local | `python -m src.db.migrate` |
| Backfill de mapas | `python backfill_polylines.py --token SEU_TOKEN` |
