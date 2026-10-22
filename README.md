# Opportunity Scanner

Where could AI save time in your organisation? An exploratory scan, built on task-level AI exposure.

Pick your industry and headcount. The scanner starts from the typical mix of roles in that UK industry; you then add, remove and resize roles to match your organisation. For each role, it splits the work into three kinds of task:

| | Label | What it means | What to do |
|---|---|---|---|
| **Buy and train** | E1 | An off-the-shelf AI assistant (ChatGPT, Copilot) could halve the time at equal quality | Licences and training. The quick wins |
| **Build** | E2 | It would take software or an agent built on AI and connected to your systems | A project |
| **People stay central** | E0 | AI can't halve the time | Leave it |

It then ranks roles by where the buy-and-train and build tasks sit, weighted by headcount, and lists each role's tasks under each heading.

It is a starting point for a conversation, not a forecast: it says where to look, not how much time or money you'll save.

## Run it

```bash
uv sync
uv run uvicorn scanner.app:app --reload      # http://127.0.0.1:8000
uv run pytest -q
```

Rebuild the data from `data/raw/` with `uv run python pipeline/build_data.py`.

## How it works

1. **Industry mix.** ONS ad hoc table 2978 gives UK employment by occupation (SOC 2020, 4-digit) and industry (SIC 2007 division), from the Annual Population Survey, 2022–2024. The three years are pooled. ONS blanks out small cells, so an industry whose published cells cover less than 70% of its workers borrows the mix of its SIC section; the page says when that happens.
2. **Roles to O*NET.** Each UK occupation maps to one or more O*NET occupations through Stable Future's SOC 2020 → O*NET-SOC 2019 crosswalk. Job-title search uses the ONS SOC 2020 coding index (about 30,000 titles).
3. **Exposure.** Each O*NET task carries an exposure label (E0, E1 or E2; E3 merged into E2) under the rubric of Eloundou et al., ["GPTs are GPTs"](https://arxiv.org/abs/2303.10130). A role's shares are the mean over its tasks, and over its linked O*NET occupations, each counted equally.
4. **The scan** weights each role's shares by its headcount. Plain code; no LLM.

**Index.** The current index is Eloundou et al.'s 2023 human labels. It will switch to the 2026 re-rating ([gpts-are-gpts-2026](https://github.com/benjamingrime)), three models rating all 18,838 O*NET 31.0 tasks with an updated rubric, once that run is complete. The build script takes either.

## Limits

- O*NET describes US jobs; UK roles map to them through a crosswalk.
- The industry mix is typical for the UK, not yours. Edit it.
- Shares are of tasks, not of working time, and exposure is potential, not adoption.
- No assessment of data, systems or readiness.
- Says nothing about jobs lost, and isn't meant to.

## Data and licences

| Source | Licence |
|---|---|
| ONS ad hoc 2978, *Employment by occupation and industry division, UK, 2022 to 2024* | Open Government Licence v3.0. Contains public sector information licensed under the Open Government Licence v3.0 |
| ONS SOC 2020 coding index | Open Government Licence v3.0 |
| O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration (USDOL/ETA) | CC BY 4.0. Includes information from O*NET 31.0 by USDOL/ETA, used under CC BY 4.0 |
| Eloundou, Manning, Mishkin & Rock, *GPTs are GPTs*, label set | MIT (see `data/raw/gpts-are-gpts/LICENSE`) |
| SOC 2020 → O*NET crosswalk | Stable Future |

Checksums of the raw files are in `data/raw/SHA256SUMS`. Code: MIT.
