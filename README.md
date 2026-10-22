# Opportunity Scanner

Where could AI save time in your organisation? An exploratory scan, built on task-level AI exposure.

Pick your industry and headcount. The scanner starts from the typical mix of roles in that UK industry; you then add, remove and resize roles to match your organisation. For each role, it splits working time across three kinds of task:

| | Label | What it means | What to do |
|---|---|---|---|
| **Buy and train** | E1 | An off-the-shelf AI assistant (ChatGPT, Copilot) could halve the time at equal quality | Licences and training. The quick wins |
| **Build** | E2 | It would take software or an agent built on AI and connected to your systems | A project |
| **People stay central** | E0 | AI can't halve the time | Leave it |

It then ranks roles by how much staff time goes on buy-and-train and build tasks, and lists each role's tasks under each heading with estimated hours a week.

Or upload your own roles: an agent maps each job title to a UK occupation, pausing to ask you when it isn't sure, then scans them. You can ask questions about the results, and any MCP client (Claude, ChatGPT, Cursor) can use the same tools.

It is a starting point for a conversation, not a forecast: it says where staff time goes and where to look, not how much time or money you'll save.

## Run it

```bash
uv sync
uv run uvicorn scanner.app:app --reload      # http://127.0.0.1:8000
uv run pytest -q
```

Rebuild the data from `data/raw/` with `uv run python pipeline/build_data.py`.

The agent (needs `OPENROUTER_API_KEY`, or `SCANNER_LLM=bedrock:<model id>` with AWS credentials):

```bash
uv run python -m scanner.assess examples/roles.csv            # map, review the unsure ones, scan
uv run python -m scanner.ask "which is more exposed, payroll clerks or receptionists?"
uv run python -m evals.mapping --n 100                         # mapping eval; --replay for recorded replies
```

MCP: the web app serves Streamable HTTP at `/mcp`. Locally over stdio, add this to your MCP client's config:

```json
{"mcpServers": {"opportunity-scanner": {"command": "uv", "args": ["--directory", "/path/to/opportunity-scanner", "run", "python", "-m", "scanner.mcp_server"]}}}
```

Docker: `docker build -t opportunity-scanner . && docker run -p 8080:8080 -e OPENROUTER_API_KEY opportunity-scanner`.

Time per task (needs `OPENROUTER_API_KEY` for `compare`; replies are cached, so reruns are free):

```bash
uv run python pipeline/time_shares.py compare --cap 25   # model judgements, about $15
uv run python pipeline/time_shares.py export             # -> data/time/judgments.csv.gz
uv run python pipeline/time_shares.py solve              # -> data/time/time_shares.csv
```

## How it works

1. **Industry mix.** ONS ad hoc table 2978 gives UK employment by occupation (SOC 2020, 4-digit) and industry (SIC 2007 division), from the Annual Population Survey, 2022–2024. The three years are pooled. ONS blanks out small cells, so an industry whose published cells cover less than 70% of its workers borrows the mix of its SIC section; the page says when that happens.
2. **Roles to O*NET.** Each UK occupation maps to one or more O*NET occupations through Stable Future's SOC 2020 → O*NET-SOC 2019 crosswalk. Job-title search uses the ONS SOC 2020 coding index (about 30,000 titles).
3. **Exposure.** Each O*NET task carries an exposure label (E0, E1 or E2; E3 merged into E2) under the rubric of Eloundou et al., ["GPTs are GPTs"](https://arxiv.org/abs/2303.10130).
4. **Time per task.** We use the method of Hatgis-Kessell, Aguirre, Wan and Bommasani, ["Estimating time spent on work tasks"](https://arxiv.org/abs/2608.05172) (2026), reimplemented in `pipeline/time_shares.py`. Time on a task = how often it's done (O*NET's worker survey) × how long one go takes. For the second, a model judges which of two tasks in an occupation takes longer; the judgements give a ranking, and a linear program finds durations that keep the ranking and fill a 7-hour day. We differ from the paper in five ways: Claude Haiku 5.5 at high reasoning (paper: GPT-5.2); one call per task, comparing it with all the others, so each pair is judged twice and counts only if both agree (paper: one call per pair, kept if the token probability is at least 0.7); the midpoint of the paper's two objectives (paper: the max-span one, with the other as a range); fixed frequency constants rather than US CPS hours; and the minimum per go scaled down where it overflows the day (below). For 17 occupations whose tasks are so frequent that the paper's 6-minute minimum per go overflows the day (waiters, air traffic controllers, radiographers), the minimum is scaled down until it fits. Across the 875 occupations both cover, our time shares rank tasks much as the paper's do (median Spearman 0.78; middle half 0.65–0.87). Run details: `data/time/run.json` and `solve.json`.
5. **A role's shares** are its tasks' labels weighted by time, then averaged over its linked O*NET occupations, each counted equally. Where the labelled tasks cover less than half of an O*NET occupation's estimated day, its tasks count equally instead.
6. **The scan** weights each role's shares by its headcount, and turns them into hours a week for full-time staff (37.5 hours, about 33 of them on listed tasks). Plain code; no LLM.

### The agent

Tools (`scanner/tools.py`) are plain read-only functions shared by the workflow, the questions agent and the MCP server: `grep_titles`, `get_role`, `compare_roles`, `scan`, `list_roles`.

**Assessment workflow** (`scanner/assess.py`, a LangGraph graph). Code sets the order; the LLM does only the steps that need judgement.
1. `parse`: code reads a CSV of roles (title, optional headcount, department, description) and drops columns that look like personal data. A job description goes to the LLM, which extracts the title and responsibilities.
2. `map_role`, once per role, in parallel: the LLM sees all 412 SOC 2020 unit groups (cached), searches the ONS job-title index with `grep_titles` as often as it needs (index titles are written back to front: "Officer, personnel"), reads close candidates with `get_role`, and submits a code with its confidence and up to 3 alternatives.
3. `review`: if any role is below 0.7 confidence, the run pauses (a LangGraph interrupt) and a person picks from the candidates. Paused runs are checkpointed, so a review can come hours later.
4. `score`: code only, the scan above.

**Questions agent** (`scanner/ask.py`): the LLM calls tools in a loop (at most 8 calls) until it can answer. A faithfulness check then requires every number in the answer to come from a tool result or the question; if one doesn't, the agent gets one chance to correct it. It won't advise on whom to dismiss.

**Mapping eval** (`evals/mapping.py`): 100 job titles from the ONS index, one from each of 100 random unit groups, rewritten as people write them ("Accounts clerk", not "Clerk, accounts") and hidden from `grep_titles` so the model can't look up its own answers. 22 October 2026:

| Model | Exact | Right code in top 3 | Marked sure | Right when sure | $ per role |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (OpenRouter) | 67% | 84% | 46% | 91% | 0.001 |
| DeepSeek V3.2 (Bedrock) | 66% | 68% | 83% | 78% | 0.024 |
| gpt-oss-120b (Bedrock) | 57% | 63% | 81% | 70% | 0.003 |
| Qwen3 235B (Bedrock) | 43% | 54% | 66% | 64% | 0.007 |

"Right when sure" matters most: a mapping marked sure skips review. Haiku is the deployed model. Amazon Nova 2 Lite couldn't finish within its 20-requests-a-minute quota. Many misses are near-misses a reviewer would settle ("Brake examiner": vehicle or rail?). Tests and CI replay recorded replies, so they run offline at no cost.

### Deployment

One Docker image runs on a laptop, on AWS Lambda (through the Lambda Web Adapter) and on Kubernetes. On AWS, in eu-north-1, all in Terraform (`infra/`): Lambda behind a Function URL; DynamoDB for paused runs and a daily run counter; S3 for large checkpoints; Secrets Manager for the OpenRouter key; CloudWatch logs and X-Ray traces; an AWS Budgets alarm. Every push to `main` goes through CodePipeline and CodeBuild (`buildspec.yml`): tests, image to ECR, `terraform apply`, smoke test. AWS pulls from GitHub through a CodeConnections app, so no AWS keys are stored anywhere. Cost is held by a daily cap on LLM runs, the account's concurrency limit and the budget alarm.

**Index.** The current index is Eloundou et al.'s 2023 human labels. It will switch to the 2026 re-rating ([gpts-are-gpts-2026](https://github.com/benjamingrime)), three models rating all 18,838 O*NET 31.0 tasks with an updated rubric, once that run is complete. The build script takes either.

## Limits

- O*NET describes US jobs; UK roles map to them through a crosswalk.
- The industry mix is typical for the UK, not yours. Edit it.
- Time per task is a model's estimate. Beyond the paper's checks against real workers, ours is checked only against the paper's estimates.
- Hours assume full-time staff. They are time spent on tasks, not time AI would save. Exposure is potential, not adoption.
- No assessment of data, systems or readiness.
- Says nothing about jobs lost, and isn't meant to.
- Mapping a job title to an occupation is a judgement. About 1 in 10 mappings the model marks as sure is wrong; check the report's mappings.

## Data and licences

| Source | Licence |
|---|---|
| ONS ad hoc 2978, *Employment by occupation and industry division, UK, 2022 to 2024* | Open Government Licence v3.0. Contains public sector information licensed under the Open Government Licence v3.0 |
| ONS SOC 2020 coding index | Open Government Licence v3.0 |
| O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration (USDOL/ETA) | CC BY 4.0. Includes information from O*NET 31.0 by USDOL/ETA, used under CC BY 4.0 |
| Eloundou, Manning, Mishkin & Rock, *GPTs are GPTs*, label set | MIT (see `data/raw/gpts-are-gpts/LICENSE`) |
| SOC 2020 → O*NET crosswalk | Stable Future |
| Time per task (`data/time/`): our estimates, using the method of Hatgis-Kessell et al. (2026), arXiv:2608.05172 (paper CC BY 4.0) | MIT |

Checksums of the raw files are in `data/raw/SHA256SUMS`. Code: MIT.
