# 🏀 AGENTS.md — Quiet Eye Proxy & Machine Learning no Basquete

> **Atenção para Agentes e LLMs:** Este documento é a **fonte primária de verdade e contexto contínuo** para qualquer agente de IA que atue neste repositório. Leia este arquivo integralmente no início de cada nova sessão para manter o alinhamento com a arquitetura, regras de domínio, decisões científicas e padrões de engenharia de software do projeto.

---

## 📌 1. Visão Geral & Objetivo Científico

* **Projeto:** Estimativa Não-Invasiva de Métricas de *Quiet Eye* (QE) a partir de Vídeos de Lances Livres da NBA e Predição de Desempenho com Machine Learning e Modelos Estatísticos.
* **Problema Científico:** O *Quiet Eye* tradicional requer equipamentos invasivos (óculos de *eye-tracking* a >200 Hz). Este projeto desenvolve e valida um **Quiet Eye Proxy** não-invasivo baseado em Visão Computacional (MediaPipe Face Mesh + Pose + Ball Tracking) em vídeos de transmissão pública (720p/1080p a 30/60 FPS) para quantificar se e quanto a duração/estabilidade do foco visual impacta a probabilidade de acerto do lance livre.
* **Artigo Científico:** O projeto culmina em um artigo acadêmico em LaTeX localizado em [`paper/artigo.tex`](file:///home/jp/Documents/Machine%20Learning/paper/artigo.tex).

---

## 🤖 2. Protocolo de Atuação do Agente

1. **Leitura Obrigatória no Início da Sessão:**
   * Sempre leia [`AGENTS.md`](file:///home/jp/Documents/Machine%20Learning/AGENTS.md), examine [`docs/GUIA_PROJETO_QUIET_EYE.md`](file:///home/jp/Documents/Machine%20Learning/docs/GUIA_PROJETO_QUIET_EYE.md) quando necessário para detalhes teóricos, e confira o estado atual de [`src/`](file:///home/jp/Documents/Machine%20Learning/src) e [`data/`](file:///home/jp/Documents/Machine%20Learning/data).
2. **Atualização Sob Demanda:**
   * Atualize este arquivo [`AGENTS.md`](file:///home/jp/Documents/Machine%20Learning/AGENTS.md) (especialmente o [Status Atual](#-7-status-atual--roadmap) e o [Log de Decisões](#-8-registro-de-decisões-arquiteturais-adr)) sempre que o usuário solicitar explicitamente uma atualização de estado ou após marcos relevantes combinados.
3. **Estilo de Comunicação:**
   * Respostas diretas, técnicas e orientadas a código funcional e reprodutível.
   * Sempre cite caminhos de arquivos completos em links Markdown navegáveis (`[arquivo](file:///caminho)`).

---

## 🛡️ 3. Guardrails Mandatórios de ML, Estatística & Visão Computacional

Qualquer código gerado DEVE obedecer estritamente às seguintes diretrizes:

### 3.1 Prevenção Estrita de Data Leakage & Modelagem Multinível
* **Regra de Ouro:** NUNCA utilize `KFold` aleatório ou `train_test_split` simples em dados com múltiplos atletas.
* **Estratégia de Validação:** Utilize SEMPRE **`GroupKFold`** ou **`LeaveOneGroupOut`** agrupando pela coluna `player_name` para avaliar generalização out-of-player.
* **Decomposição Intra vs. Inter-Atleta (Within vs. Between-Player):**
  * Para responder se o QE afeta o desempenho, é mandatório separar o efeito *entre-jogadores* do efeito *dentro-do-jogador*.
  * Implementar **Modelos Lineares Generalizados de Efeitos Mistos (GLMM - Mixed-Effects Logistic Regression)** com intercepto aleatório por atleta via `statsmodels` ou `lme4`.
  * Computar features padronizadas intra-jogador: $z_{QE} = \frac{t_{QE} - \mu_{QE, player}}{\sigma_{QE, player}}$.

### 3.2 Controle de Variável de Confusão (Duração da Rotina)
* **Regra Crítica:** NUNCA avaliar a duração absoluta do QE isoladamente sem controlar pelo tempo total da rotina de arremesso.
* **Features Mandatórias:**
  * `routine_duration_s`: Tempo total desde o início da rotina (recebimento da bola/posicionamento) até o *release*.
  * `qe_ratio_pct`: Proporção da rotina gasta em Quiet Eye $\left(\frac{t_{QE}}{\text{routine\_duration}} \times 100\right)$.
  * A `routine_duration_s` DEVE entrar como covariável em todos os modelos estatísticos e de ML.

### 3.3 Métricas Científicas e Tamanho de Efeito (Effect Size)
* Além de métricas de classificação (ROC-AUC, PR-AUC, F1-Score), o pipeline DEVE reportar **Tamanho de Efeito com Intervalo de Confiança (IC 95%)**:
  * **Odds Ratios (OR):** $\text{OR} = \exp(\beta)$ com IC 95% para cada incremento de 100 ms em $t_{QE}$ e para $z_{QE}$.
  * **Ablation / Baseline Test:** Comparar o modelo completo contra um modelo baseline contendo apenas `career_ft_pct` (via *Likelihood Ratio Test*, $\Delta\text{AIC}$ e $\Delta\text{AUC}$).

### 3.4 Robustez na Visão Computacional & Release Híbrido
* **Interpolação de Dropout:** Trate landmarks nulos ou com baixa acurácia no MediaPipe usando interpolação linear temporal (`pandas.interpolate` ou `numpy`).
* **Filtros de Suavização:** Aplique filtros temporais (como Média Móvel com janela de 3–5 frames ou Filtro de Savitzky-Golay) nas séries temporais de ângulos da cabeça (*pitch*, *yaw*, *roll*) antes de calcular derivadas e janelas de estabilidade.
* **Detecção de Release Híbrida:** 
  1. O *MediaPipe Pose* (cinemática de extensão de pulso/ombro) define uma janela restrita candidata ao release (~5 frames).
  2. O rastreamento do centroide da bola (descolamento físico dos dedos via HSV/mask ou YOLO) crava o frame exato do término do QE (*Offset*).
* **Validação do Proxy (Ground-Truth Subset):**
  * Manter um subconjunto de validação (~20 clipes) anotado manualmente frame a frame para reportar Correlação de Pearson, Erro Absoluto Médio (MAE) e Coeficiente de Correlação Intraclasse (ICC) do proxy de QE e do release.

### 3.5 Reprodutibilidade e Determinismo
* Fixe sementes aleatórias globais (`seed = 42` ou configurável) em todos os scripts de divisão de dados, treinamento de modelos (`scikit-learn`, `xgboost`, `statsmodels`) e pipelines.
* Use o módulo padrão de `logging` do Python com níveis informativos (`INFO`, `DEBUG`, `WARNING`, `ERROR`) em vez de `print` espalhados pelo código de produção.

### 3.6 Desacoplamento e Modularidade
* **`src/vision/`**: Scripts e classes exclusivas para extração e processamento de vídeo (OpenCV, MediaPipe FaceMesh/Pose, rastreador de bola e estimador de QE).
* **`src/models/`**: Scripts de GLMM (efeitos mistos), modelos de ML (XGBoost/RF), ablações, métricas estatísticas e SHAP.
* **`src/utils/`**: Utilitários auxiliares de I/O, manipulação de arquivos, métricas e geração de plots.
* **`notebooks/`**: Uso exclusivo para análise exploratória de dados (EDA), testes rápidos e prototipagem visual.

---

## 💻 4. Padrões de Código e Engenharia de Software

* **Versão Python:** Python 3.10+.
* **Tipagem:** Uso obrigatório de *Type Hints* em todas as assinaturas de funções e métodos (`typing`, `numpy.typing`).
* **Documentação:** Docstrings em todas as classes e funções públicas seguindo o padrão Google Style ou NumPy Style.
* **Testes Automatizados:** Testes unitários e de integração implementados com `pytest` dentro do diretório `tests/`.
* **Ambiente e Dependências:** Respeitar o [`requirements.txt`](file:///home/jp/Documents/Machine%20Learning/requirements.txt).

---

## 📁 5. Arquitetura do Repositório & Estrutura de Dados

```text
├── data/
│   ├── raw/                   # Vídeos brutos coletados (.mp4, .avi)
│   ├── processed/             # Features extraídas consolidadas (.csv, .parquet)
│   └── metadata/              # Planilha de anotação manual (dataset_metadata.csv)
├── src/
│   ├── vision/                # Extração facial/pose, ball tracking e cálculo de QE
│   ├── models/                # GLMM (Efeitos Mistos), ML, avaliação e SHAP
│   └── utils/                 # I/O, logs, constantes e plots
├── models/                    # Pesos de modelos treinados (.pkl, .joblib, .json)
├── notebooks/                 # EDA e prototipagem interativa
├── paper/                     # Artigo LaTeX, figuras e referências BibTeX
│   ├── artigo.tex
│   ├── references.bib
│   └── figures/
├── docs/                      # Documentação teórica e guias de referência
├── tests/                     # Testes automatizados com pytest
├── requirements.txt           # Dependências do projeto
├── README.md                  # Apresentação do repositório
└── AGENTS.md                  # [ESTE ARQUIVO] Contexto e regras para Agentes
```

---

## 📐 6. Definição do Pipeline Matemático do Quiet Eye Proxy

1. **Ângulos da Cabeça:** A partir dos pontos 3D do `MediaPipe FaceMesh` (nariz, queixo, cantos dos olhos e boca), computar o vetor de rotação (*Pitch*, *Yaw*, *Roll*) usando `cv2.solvePnP`.
2. **Cone de Atenção Visual:** Considera-se foco no alvo (cesta) quando $\Delta Pitch \le \epsilon$ e $\Delta Yaw \le \epsilon$.
3. **Início do QE (*Onset*):** Primeiro frame em que o vetor angular permanece estável dentro do cone por tempo $\ge 100\text{ ms}$.
4. **Término do QE (*Offset* / Release):** Instante em que a bola se descola fisicamente das mãos, detectado pela fusão entre a cinemática de extensão de braço (`MediaPipe Pose`) e o rastreamento da trajetória da bola.
5. **Duração e Proporção do QE ($t_{QE}$ e $\%_{QE}$):**
   $$\text{QE\_duration\_ms} = \frac{\text{Frame}_{\text{Release}} - \text{Frame}_{\text{Onset}}}{\text{FPS}} \times 1000$$
   $$\text{QE\_ratio\_pct} = \frac{\text{QE\_duration\_ms}}{\text{Routine\_duration\_ms}} \times 100$$
   $$\text{QE\_zscore} = \frac{\text{QE\_duration\_ms} - \mu_{\text{player}}}{\sigma_{\text{player}}}$$

---

## 🚀 7. Status Atual & Roadmap

### Sprint / Marco Atual: **Fases 1 & 2 (Infraestrutura de Visão Computacional + Amostragem Balanceada Multi-Shot)**
* [ ] **Anotação de Dados:** Estruturar `data/metadata/dataset_metadata.csv` com protocolo de amostragem balanceada (8–15 lances livres por atleta com contraste acerto × erro em ~15 atletas, totalizando ~150–200 clipes).
* [ ] **Subconjunto de Validação Manual:** Separar ~20 clipes para anotação frame a frame de Onset/Release manual para validação cruzada do proxy (ICC/MAE).
* [ ] **Módulo de Visão Computacional:** Implementar extratores modulares em `src/vision/` (`face_geometry.py`, `pose_tracker.py`, `ball_tracker.py`, `quiet_eye_extractor.py`, `validation.py`).
* [ ] **Testes de Visão:** Criar suíte de testes unitários em `tests/` para validar a extração de ângulos, tracking de bola e consistência temporal.
* [ ] **Pipeline de Features:** Gerar dataset consolidado em `data/processed/features_extracted.csv` contendo métricas absolutas, relativas (`qe_ratio_pct`), padronizadas intra-atleta (`qe_zscore`) e covariáveis de controle (`routine_duration_s`).
* [ ] **Modelagem Estatística & ML:**
  * [ ] Modelo Baseline com `career_ft_pct`.
  * [ ] Modelo Linear Generalizado de Efeitos Mistos (GLMM) para isolar efeitos *within* vs. *between* athlete.
  * [ ] Modelos de Machine Learning (XGBoost, Random Forest) com `GroupKFold`.
* [ ] **Interpretabilidade e Artigo:** Gerar gráficos SHAP, tabelas de Odds Ratios com IC 95% e preencher a metodologia e resultados no [`paper/artigo.tex`](file:///home/jp/Documents/Machine%20Learning/paper/artigo.tex).

---

## 📝 8. Registro de Decisões Arquiteturais (ADR)

| ID | Data | Decisão | Justificativa |
|---|---|---|---|
| **ADR-001** | 2026-08-26 | Adoção de `GroupKFold` por `player_name` | Evita Data Leakage e garante que os modelos de ML aprendam métricas neuromotoras de Quiet Eye, não a identidade dos atletas. |
| **ADR-002** | 2026-08-26 | Separação estrita `src/vision` e `src/models` | Garante que o pipeline de Visão Computacional possa ser reutilizado e testado independentemente da modelagem estatística e de ML. |
| **ADR-003** | 2026-08-26 | Interpolação e Suavização de Landmarks | Corrige quedas de tracking nos frames de vídeo (baixa resolução ou oclusão) e estabiliza o cálculo das derivadas temporais de QE. |
| **ADR-004** | 2026-08-26 | Atualização do `AGENTS.md` Sob Demanda | O agente mantém leitura obrigatória na inicialização e atualiza o estado/decisões conforme solicitado pelo usuário. |
| **ADR-005** | 2026-08-26 | Protocolo Multi-Shot Balanceado (8–15 arremessos/atleta, ~150–200 clipes) | Permite estabelecer linha de base intra-atleta e contraste suficiente de acertos vs. erros para poder estatístico. |
| **ADR-006** | 2026-08-26 | Controle Obrigatório da Duração da Rotina (`qe_ratio_pct` e `routine_duration_s`) | Elimina variável de confusão fundamental: impede que arremessos mais lentos no geral sejam interpretados falsamente como QE intencional. |
| **ADR-007** | 2026-08-26 | Modelagem com Efeitos Mistos (GLMM) e Decomposição Within vs. Between | Padrão ouro em biomecânica para isolar o efeito do QE do próprio atleta em relação à sua média individual. |
| **ADR-008** | 2026-08-26 | Detecção Híbrida de Release (Pose + Ball Tracking) | Garante precisão no término exato do QE (*offset*), evitando que o erro de tracking achate o tamanho do efeito medido. |
| **ADR-009** | 2026-08-26 | Validação de Proxy em Subconjunto Ground-Truth (~20 clipes) | Blinda o trabalho contra revisores acadêmicos através de métricas formais de concordância (ICC, MAE, Pearson). |
