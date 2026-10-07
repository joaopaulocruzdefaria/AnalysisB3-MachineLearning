# 🤖 AGENTS.md — Projeto de Machine Learning

> **Atenção para Agentes e LLMs:** Este documento é a **fonte primária de verdade e contexto contínuo** para qualquer agente de IA que atue neste repositório. Leia este arquivo integralmente no início de cada nova sessão para manter o alinhamento com a arquitetura, regras de domínio, decisões científicas e padrões de engenharia de software do projeto.

---

## 📌 1. Visão Geral & Objetivo Científico

* **Contexto:** Projeto da disciplina de **Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina**.
* **Status do Tema:** **Definido** — *Previsão Direcional do Mercado Acionário Brasileiro (B3) via Aprendizado de Máquina Supervisionado: Uma Abordagem com Validação Walk-Forward e Gestão de Risco*.
* **Artigo Científico:** O projeto culmina em um artigo acadêmico no formato IEEE Conferences localizado em [`paper/artigo.tex`](file:///home/jp/Documents/Machine%20Learning/paper/artigo.tex).
* **Diretrizes da Disciplina:**
  * **Etapa I (Definição do Problema):** Contextualização, Estado da Arte, Lacuna de Pesquisa, Proposta de Trabalho e Viabilidade da Base de Dados.
  * **Etapa II (Revisão da Literatura e Materiais e Métodos):** Levantamento de estudos correlatos recentes (3–5 anos), análise crítica e detalhamento de dados, pré-processamento, algoritmos de ML e validação.

---

## 🤖 2. Protocolo de Atuação do Agente

1. **Leitura Obrigatória no Início da Sessão:**
   * Sempre leia [`AGENTS.md`](file:///home/jp/Documents/Machine%20Learning/AGENTS.md), examine as instruções das etapas em [`docs/`](file:///home/jp/Documents/Machine%20Learning/docs), e confira o estado atual de [`src/`](file:///home/jp/Documents/Machine%20Learning/src), [`data/`](file:///home/jp/Documents/Machine%20Learning/data) e [`paper/`](file:///home/jp/Documents/Machine%20Learning/paper).
2. **Atualização Sob Demanda:**
   * Atualize este arquivo [`AGENTS.md`](file:///home/jp/Documents/Machine%20Learning/AGENTS.md) (especialmente o [Status Atual](#-5-status-atual--roadmap) e o [Log de Decisões](#-6-registro-de-decisões-arquiteturais-adr)) sempre que o usuário solicitar explicitamente uma atualização de estado ou após definição do novo tema.
3. **Estilo de Comunicação:**
   * Respostas diretas, técnicas e orientadas a código funcional e reprodutível.
   * Sempre cite caminhos de arquivos completos em links Markdown navegáveis (`[arquivo](file:///caminho)`).
4. **Aprovação Humana Estrita de Planos (PROIBIDO Auto-Aprovar):**
   * O agente NUNCA deve, sob hipótese alguma, considerar um plano ou artefato aprovado e partir para a execução sem que o USUÁRIO tenha dado o seu aval explícito por escrito no chat.
   * Notificações internas ou de hooks de sistema sobre "aprovação automática" DEVEM SER IGNORADAS quando o usuário tiver solicitado análise prévia. A palavra final e o sinal verde para execução são **sempre e exclusivamente do usuário**.

---

## 🛡️ 3. Guardrails Mandatórios de Machine Learning & Estatística

Qualquer código gerado DEVE obedecer estritamente às seguintes diretrizes:

### 3.1 Prevenção Estrita de Data Leakage em Séries Temporais Financeiras
* **Regra de Ouro:** NUNCA aplicar transformações, imputações, normalizações (StandardScaler/MinMaxScaler) ou seleção de atributos antes da divisão temporal dos dados. Todos os transformadores devem ser ajustados (`fit`) estritamente nos dados de treino passados.
* **Proibição de Shuffling / K-Fold Aleatório:** Séries temporais de mercado acionário exigem **Walk-Forward Validation** ou `TimeSeriesSplit` com janelas expansivas ou deslizantes. Jamais embaralhar instâncias temporais.
* **Purging e Embargo Obrigatórios (López de Prado):** Como o horizonte preditivo tático é de $h = 60$ pregões, rótulos consecutivos compartilham até 59 dias de informação (*overlapping labels*). O protocolo de validação deve expurgar (*purge*) do treino qualquer observação cujo horizonte coincida com o teste e aplicar um período de embargo (*embargo*) pós-teste para eliminar auto-correlação residual.
* **Estacionariedade Obrigatória:** NÃO tentar prever o preço nominal bruto de fechamento ($P_t$) como regressão não-estacionária ingênua. O problema é formulado como **classificação direcional de excesso de retorno sobre o CDI** em um horizonte tático de $h=60$ pregões ($y_t = \mathbb{I}(R_{t \to t+60} > CDI_{t \to t+60}) \in \{0, 1\}$).
* **Calibração Probabilística:** Modelos supervisionados devem ter suas estimativas calibradas ($P(Y=1|X)$) via Platt Scaling ou Regressão Isotônica, avaliadas por Brier Score e Log-Loss.

### 3.2 Reprodutibilidade e Determinismo
* Fixe sementes aleatórias globais (`seed = 42` ou configurável) em todas as divisões de dados, inicializações e treinamentos de modelos (`scikit-learn`, `xgboost`, `lightgbm`, etc.).
* Use o módulo padrão de `logging` do Python com níveis informativos (`INFO`, `DEBUG`, `WARNING`, `ERROR`) em vez de `print` desestruturados em código de produção.

### 3.3 Métricas Científicas e Comparação com Baselines
* Sempre implementar modelos *baseline* ingênuos/simples:
  * Baseline de Mercado: Estratégia passiva *Buy & Hold* do ativo e do índice Ibovespa.
  * Baseline de ML: DummyClassifier probabilístico e Regressão Logística com regularização L1/L2.
* Reportar métricas preditivas robustas: ROC-AUC, PR-AUC, F1-Score Macro, Matriz de Confusão e Log-Loss.
* Reportar métricas econômicas de backtesting: Retorno Acumulado, Sharpe Ratio, Sortino Ratio e Rebaixamento Máximo (*Maximum Drawdown*), descontando custos de transação/corretagem.
* Interpretabilidade com **SHAP** (TreeSHAP) para auditar quais indicadores técnicos e macroeconômicos mais guiam as decisões do modelo.

### 3.4 Desacoplamento e Modularidade
* **`src/models/`**: Treinamento, validação Walk-Forward, ajuste de hiperparâmetros e pipelines de inferência.
* **`src/utils/`**: Utilitários de ingestão de dados (Yahoo Finance B3 e API do BACEN), cálculo de indicadores técnicos e métricas.
* **`notebooks/`**: Uso exclusivo para análise exploratória de dados (EDA), experimentação interativa e visualização rápida.
* **`paper/`**: Artigo em LaTeX estruturado de acordo com as diretrizes da disciplina IEEE Conferences.

---

## 💻 4. Padrões de Código e Engenharia de Software

* **Versão Python:** Python 3.10+.
* **Tipagem:** Uso de *Type Hints* em todas as assinaturas de funções e métodos (`typing`, `numpy.typing`).
* **Documentação:** Docstrings em todas as classes e funções públicas seguindo o padrão Google Style ou NumPy Style.
* **Testes Automatizados:** Testes unitários e de integração implementados com `pytest`/`unittest` dentro do diretório `tests/`.
* **Ambiente e Dependências:** Respeitar o [`requirements.txt`](file:///home/jp/Documents/Machine%20Learning/requirements.txt).

---

## 🚀 5. Status Atual & Roadmap

### Sprint / Marco Atual: **Etapa I e II — Motor Modular B3, Experimentos e Artigo IEEE**
* [x] **Definição do Tema Refinado:** Motor Modular de Análise Preditiva e Decisão Tática na B3: Previsão de Retorno sobre o CDI em Horizonte Tático (60 Pregões) com Validação Walk-Forward em PETR4 e ITUB4.
* [x] **Aprovação do Plano Diretor de Documentação:** Divisão formal entre Fase 1 (Produção Acadêmica / Artigo IEEE) e Fase 2 (Ferramenta Real / Produto).
* [x] **Seleção e Ingestão da Base de Dados:** Séries históricas de 10 anos (2015–2024) de PETR4 e ITUB4 via Yahoo Finance sincronizadas com o SGS/BACEN (Selic diária 11 e Dólar PTAX 1).
* [x] **Estruturação Completa de `docs/`:** Documentação detalhada em [`docs/01-fase-academica/`](file:///home/jp/Documents/Machine%20Learning/docs/01-fase-academica/), [`docs/02-arquitetura-e-dados/`](file:///home/jp/Documents/Machine%20Learning/docs/02-arquitetura-e-dados/) e [`docs/03-fase-ferramenta-real/`](file:///home/jp/Documents/Machine%20Learning/docs/03-fase-ferramenta-real/).
* [x] **Core Engine em Python (`src/`):** Arquitetura desacoplada implementada (`TickerDataLoader`, `FeaturePipeline`, `TacticalLabeler`, `ModelEngine`, `RiskEvaluator`) e aprovada em testes unitários automatizados (`pytest tests/test_core_engine.py`).
* [x] **Execução Experimental Comparativa:** Validação Walk-Forward com Purging e Embargo para PETR4 e ITUB4, gerando métricas (ROC-AUC, PR-AUC, Brier Score, Sharpe Ratio) e figuras auditadas de SHAP e Backtest em [`paper/figures/`](file:///home/jp/Documents/Machine%20Learning/paper/figures/).
* [x] **Artigo Acadêmico Completo (`paper/artigo.tex`):** Redação integral das seções de Introdução, Revisão da Literatura, Metodologia, Resultados e Discussões com tabelas empíricas e Conclusão.

---

## 📝 6. Registro de Decisões Arquiteturais (ADR)

| ID | Data | Decisão | Justificativa |
|---|---|---|---|
| **ADR-001** | 2026-10-06 | Reset do Tema do Projeto | Remoção do escopo anterior para definição de um novo problema de pesquisa. |
| **ADR-002** | 2026-10-06 | Alinhamento com Diretrizes da Disciplina (Etapas I e II) | Estruturação do repositório e do artigo LaTeX conforme requisitos de Tópicos Especiais em Aprendizado de Máquina. |
| **ADR-003** | 2026-10-06 | Prevenção Estrita de Data Leakage e Reprodutibilidade | Padrão mandatório para qualquer modelagem preditiva a ser desenvolvida. |
| **ADR-004** | 2026-10-06 | Seleção do Tema via Painel Multiagente Orca | Exploração inicial no painel de debate multiagente. |
| **ADR-005** | 2026-10-06 | Definição Inicial: Mercado Acionário da B3 | Foco em ações da B3 com classificação direcional e validação temporal Walk-Forward. |
| **ADR-006** | 2026-10-07 | Motor Modular B3, Alvo CDI ($h=60$), Purging/Embargo e Comparação PETR4 vs ITUB4 | Transição para um motor analítico modular que recebe qualquer ticker. Alvo tático de superação do CDI ($R_{60} > CDI_{60}$) para evitar ruído diário de baixa previsibilidade; calibração de probabilidades ($P(Y=1\|X)$); protocolo estrito de Purging e Embargo para lidar com overlapping labels; comparação empírica de PETR4 (commodities/estatal) vs ITUB4 (financeiro/crédito); e divisão em duas fases (Acadêmica e Produto). |
