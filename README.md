# Motor Modular de Análise Preditiva e Decisão Tática na B3 🤖📈

> **Projeto:** Desenvolvimento e Validação Experimental de uma Ferramenta Analítica em Python para Suporte à Decisão do Investidor na Bolsa Brasileira (B3).  
> **Disciplina:** Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina  
> **Instituição:** Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)  
> **Formato de Entrega:** Artigo científico padrão IEEE Conferences ([`paper/artigo.tex`](paper/artigo.tex))

---

## 💡 Visão Geral do Projeto

Diferente de scripts rígidos focados em uma única ação ou em predições ingênuas de curto prazo (onde o ruído de mercado domina), este projeto desenvolve um **motor analítico modular em Python** capaz de:

1. **Receber o ticker de qualquer ação líquida da B3** (ex.: `PETR4`, `ITUB4`, `VALE3`).
2. **Calcular automaticamente métricas de desconto histórico** do ativo frente aos seus próprios ciclos e ao custo de oportunidade da economia brasileira (CDI / Selic via BACEN).
3. **Estimar a probabilidade calibrada** $P(Y=1|X)$ de o ativo superar a taxa livre de risco (CDI acumulado) em um horizonte tático de **60 pregões** (~3 meses):
   $$y_t = \mathbb{I}(R_{t \to t+60} > CDI_{t \to t+60})$$
4. **Validar experimentalmente o modelo** através de um protocolo estrito de **Walk-Forward Validation com Purging e Embargo** (López de Prado), testando comparativamente dois ativos com regimes operacionais distintos:
   * **PETR4:** Setor de commodities cíclicas, estatal, sensível ao petróleo Brent e câmbio USD/BRL.
   * **ITUB4:** Setor financeiro/bancário, sensível ao ciclo doméstico de crédito, inadimplência e taxa Selic.

---

## 🧭 Estrutura em Duas Fases

* **Fase 1 (Produção Acadêmica / Artigo IEEE):** Foco rigoroso no núcleo analítico (`src/`), levantamento bibliográfico (2020–2025), mitigação estrita de *data leakage*, calibração de modelos supervisionados (LightGBM, XGBoost, CatBoost, Logistic Regression), explicabilidade via SHAP e redação do artigo no padrão IEEE Conferences.
* **Fase 2 (Ferramenta Real / Pós-Disciplina):** Envelopamento do *Core Engine* em API (FastAPI) ou Dashboard interativo (Streamlit), automação de coleta diária de fechamentos e dimensionamento de posição via Critério de Kelly Fracionário.

---

## 📁 Estrutura do Repositório

```text
├── data/                      # Gestão de dados do projeto
│   ├── raw/                   # Cotações brutas B3 (Yahoo Finance) e séries BACEN (SGS)
│   ├── processed/             # Datasets alinhados, tratados e com features de desconto
│   └── metadata/              # Dicionários de atributos e mapeamentos
│
├── src/                       # Código-fonte modular (Core Engine)
│   ├── data/                  # Ingestão e alinhamento de dados (TickerDataLoader)
│   ├── features/              # Engenharia de atributos e métricas de desconto (FeaturePipeline)
│   ├── labeling/              # Rotulagem tática e Purging/Embargo (TacticalLabeler)
│   ├── models/                # Modelos, Walk-Forward e Calibração (ModelEngine)
│   └── evaluation/            # Métricas estatísticas, SHAP e Backtest (RiskEvaluator)
│
├── paper/                     # Artigo científico em LaTeX (IEEE Conferences)
│   ├── artigo.tex             # Documento LaTeX principal
│   ├── references.bib         # Referências bibliográficas BibTeX
│   └── figures/               # Gráficos de calibração, curvas ROC/PR e SHAP
│
├── docs/                      # Documentação completa e diretrizes
│   ├── 00_plano_mestre.md     # Plano diretor do projeto
│   ├── 01-fase-academica/     # Definição do problema, revisão da literatura e protocolo experimental
│   ├── 02-arquitetura-e-dados/# Especificação de classes e catálogo de features
│   └── 03-fase-ferramenta-real/# Roadmap de produto e gestão de risco Kelly
│
├── notebooks/                 # Análise exploratória de dados (EDA) e prototipação
├── tests/                     # Testes unitários e de integração (pytest)
├── requirements.txt           # Dependências Python
└── AGENTS.md                  # Fonte de verdade contínua para agentes de IA
```

---

## 🚀 Como Começar

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd <diretorio-do-repositorio>
   ```

2. **Crie e ative um ambiente virtual Python:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Consulte a documentação:**
   * Diretrizes do artigo e prazos: [`docs/01-fase-academica/`](docs/01-fase-academica/)
   * Arquitetura de software: [`docs/02-arquitetura-e-dados/`](docs/02-arquitetura-e-dados/)
   * Artigo em LaTeX: [`paper/artigo.tex`](paper/artigo.tex)
