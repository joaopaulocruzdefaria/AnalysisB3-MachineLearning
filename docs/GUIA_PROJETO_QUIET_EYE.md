# 🏀 Guia Completo do Projeto: Quiet Eye Proxy & Machine Learning no Basquete

> **Título Sugerido:** *Estimativa Não Invasiva de Quiet Eye via Visão Computacional e Predição de Desempenho em Lances Livres da NBA usando Machine Learning*  
> **Objetivo:** Extrair métricas temporais de atenção visual (*Quiet Eye Proxy*) a partir de vídeos públicos de lances livres da NBA e treinar modelos de Machine Learning para prever a eficácia e o padrão de acerto/erro do arremesso.

---

## 📑 Sumário
1. [Visão Geral e Arquitetura do Sistema](#1-visão-geral-e-arquitetura-do-sistema)
2. [Fase 1: Coleta e Protocolo de Anotação dos Dados](#fase-1-coleta-e-protocolo-de-anotação-dos-dados)
3. [Fase 2: Pipeline de Visão Computacional (Extração do QE)](#fase-2-pipeline-de-visão-computacional-extração-do-qe)
4. [Fase 3: Engenharia de Features & Estruturação da Tabela](#fase-3-engenharia-de-features--estruturação-da-tabela)
5. [Fase 4: Treinamento e Avaliação dos Modelos de ML](#fase-4-treinamento-e-avaliação-dos-modelos-de-ml)
6. [Fase 5: Estrutura do Artigo Científico (`artigo.tex`)](#fase-5-estrutura-do-artigo-científico-artigotex)
7. [Boas Práticas e Como Evitar Armadilhas](#boas-práticas-e-como-evitar-armadilhas)

---

## 1. Visão Geral e Arquitetura do Sistema

```
[ Vídeo do Lance Livre ]
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. VISÃO COMPUTACIONAL (OpenCV + MediaPipe)                 │
│  - MediaPipe FaceMesh -> Ângulos da Cabeça (Pitch, Yaw, Roll)│
│  - MediaPipe Pose     -> Detecção do Set-point e Release    │
│  - Algoritmo de QE    -> Janela de Estabilidade Pré-Release │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TABELA DE FEATURES (Pandas Dataset)                      │
│  - qe_duration_ms, qe_stability, total_routine_s, etc.      │
│  - Rótulos: outcome_binary (0/1), shot_category             │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MACHINE LEARNING & ANÁLISE (Scikit-Learn / XGBoost)     │
│  - Modelos: Regressão Logística, Random Forest, XGBoost     │
│  - Validação: Stratified K-Fold / GroupKFold                │
│  - Interpretabilidade: SHAP Values / Feature Importance    │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
[ Artigo Científico Final em LaTeX ]
```

---

## Fase 1: Coleta e Protocolo de Anotação dos Dados

### 1.1 Critérios de Seleção dos Vídeos
* **Fonte:** YouTube (vídeos de jogos completos da NBA ou compilações de lances livres em 720p/1080p a 30 ou 60 FPS).
* **Enquadramento:** Câmera frontal ou diagonal baixa (*baseline broadcast camera*), sem cortes bruscos de câmera entre a entrega da bola e o arremesso.
* **Volume Ideal:** 60 a 120 arremessos (mescla de atletas de elite no lance livre, ex: Stephen Curry, Damian Lillard, e atletas com menor aproveitamento/estilo diferente, ex: Giannis Antetokounmpo, Shaquille O'Neal, Rudy Gobert).

### 1.2 Recorte do Clipe
* **Início:** O momento exato em que o árbitro entrega a bola nas mãos do jogador.
* **Fim:** O momento em que a bola entra na cesta ou bate no aro e o desfecho é concluído.
* **Duração média:** 3 a 6 segundos por clipe.

### 1.3 Planilha de Anotação Manual (`dataset_metadata.csv`)
Crie uma planilha com as seguintes colunas:
* `shot_id`: Identificador único (ex: `curry_ft_001.mp4`).
* `player_name`: Nome do atleta.
* `career_ft_pct`: Porcentagem histórica de lance livre do atleta (opcional, para controle).
* `outcome_binary`: `1` (Acertou) ou `0` (Errou).
* `shot_category`: 
  * `Swish` (Sem tocar no aro / Perfeito)
  * `Rim-In` (Tocou no aro e entrou)
  * `Front-Rim` (Curto / Bateu na frente do aro)
  * `Back-Rim` (Longo / Bateu atrás no aro ou tabela)
  * `Airball` (Não tocou no aro)

---

## Fase 2: Pipeline de Visão Computacional (Extração do QE)

### 2.1 Módulos Utilizados
* `cv2` (OpenCV) para leitura dos frames e cálculo de FPS.
* `mediapipe.solutions.face_mesh` para estimar os pontos faciais 3D e derivar a orientação angular da cabeça ($Pitch, Yaw, Roll$).
* `mediapipe.solutions.pose` para rastrear os pulsos e ombros e detectar o momento exato do *Release* (quando a bola deixa as mãos).

### 2.2 Algoritmo de Detecção do Quiet Eye Proxy
1. **Ângulo da Cabeça no Alvo:** Definir um cone de alinhamento com a cesta (ex: $\Delta Pitch \le \epsilon$ e $\Delta Yaw \le \epsilon$).
2. **Detecção do Início (*Onset*):** Primeiro frame em que o vetor de direção da cabeça estabiliza no cone do aro por mais de 100 ms.
3. **Detecção do Término (*Offset*):** Momento do *Release* detectado pela cinemática do pulso no MediaPipe Pose.
4. **Cálculo da Duração ($t_{QE}$):**
   $$\text{QE\_duration\_ms} = \frac{\text{Frame}_{\text{Release}} - \text{Frame}_{\text{Onset}}}{\text{FPS}} \times 1000$$

---

## Fase 3: Engenharia de Features & Estruturação da Tabela

Para cada arremesso processado, o script gera uma linha com as seguintes variáveis:

| Variável | Descrição | Tipo |
| :--- | :--- | :--- |
| `qe_duration_ms` | Duração total da fixação no aro antes da soltura da bola | Numérica contínua (ms) |
| `qe_onset_relative` | Momento em que a fixação iniciou em relação ao início do movimento do braço | Numérica contínua (%) |
| `head_stability_var` | Variância angular dos eixos Pitch/Yaw durante o período de QE | Numérica contínua |
| `total_routine_duration_s` | Tempo total desde que o árbitro deu a bola até o arremesso | Numérica contínua (s) |
| `player_category` | Elite ($\ge 85\%$), Médio ($70-84\%$), Baixo ($< 70\%$) | Categórica |
| **`target_made`** | **Rótulo principal de acerto (1) ou erro (0)** | **Binário (Alvo)** |
| **`target_shot_category`** | **Rótulo detalhado do desfecho do arremesso** | **Multiclasse (Análise)** |

---

## Fase 4: Treinamento e Avaliação dos Modelos de ML

### 4.1 Experimentos de Modelagem
1. **Baseline:** Regressão Logística (para provar correlação linear/logística direta entre tempo de QE e taxa de acerto).
2. **Modelos de Árvore:** Random Forest e XGBoost / LightGBM (para capturar relações não-lineares entre estabilidade, tempo de rotina e acerto).
3. **Classificador de Margem:** SVM (com kernel RBF) ou MLP simples.

### 4.2 Esquema de Validação
* **Stratified 5-Fold Cross-Validation:** Mantendo a proporção de Acertos/Erros equilibrada em cada fold.
* **GroupKFold (por jogador):** Para testar se o modelo consegue generalizar para jogadores nunca vistos durante o treino (evita *data leakage* de estilo individual).

### 4.3 Métricas de Avaliação
* **Acurácia**, **F1-Score Ponderado**, **Precisão**, **Revocação (Recall)**.
* **Curva ROC e AUC (Área sob a Curva)**.
* **SHAP (SHapley Additive exPlanations):** Gráfico de importância das variáveis comprovando o peso do Quiet Eye na decisão do modelo.

### 4.4 Experimento Bônus (Desfecho Detalhado)
* Análise estatística (ANOVA / Teste de Kruskal-Wallis) comparando a distribuição do tempo de QE entre arremessos `Swish`, `Rim-In`, `Front-Rim` e `Airball`.

---

## Fase 5: Estrutura do Artigo Científico (`artigo.tex`)

1. **Título & Resumo (Abstract):** Contexto do lance livre, a lacuna de medir QE sem sensores invasivos, metodologia proposta com MediaPipe + ML e principais resultados alcançados.
2. **1. Introdução:** A importância do lance livre no basquete de alto rendimento, a teoria motora do Quiet Eye (Joan Vickers, 1996) e o papel recente de IA/Visão Computacional no esporte (SkillSight, 2025).
3. **2. Trabalhos Relacionados:**
   - Ciências do Esporte e Atenção Visual (*Quiet Eye in Sports*).
   - Visão Computacional Aplicada a Esportes (*Markerless Pose & Gaze Tracking*).
   - Machine Learning em Análise de Desempenho Esportivo.
4. **3. Metodologia:**
   - Coleta e padronização do dataset de lances livres da NBA.
   - Pipeline de extração de pose e estimativa de orientação de cabeça (*Proxy QE*).
   - Formulação do problema de Machine Learning e arquitetura dos classificadores.
5. **4. Resultados e Discussão:**
   - Desempenho comparativo dos modelos de ML (Tabela com Acurácia, F1-Score, AUC).
   - Análise de interpretabilidade com SHAP (comprovação da importância de $t_{QE}$).
   - Relação entre duração de fixação e tipos de erro (*Front-Rim vs Back-Rim*).
6. **5. Conclusão e Trabalhos Futuros:** Resumo das contribuições e aplicação prática para treinamento esportivo assistido por IA.

---

## Boas Práticas e Como Evitar Armadilhas

> [!TIP]
> **Comece Pequeno:** Teste o código de extração do MediaPipe em apenas **3 vídeos** (um acerto perfeito, um erro frontal e um erro longo) antes de baixar o lote completo.

> [!IMPORTANT]
> **Defesa Científica do "Proxy":** No artigo e na apresentação, sempre use o termo **"Head-Gaze Orientation Proxy"** ou **"Proxy de Quiet Eye"**. Explique claramente que, por ser vídeo broadcast de TV, a orientação da cabeça é um estimador não invasivo robusto e validado pela literatura quando não há eye-trackers físicos.

> [!NOTE]
> **Reprodutibilidade:** Mantenha os scripts modulares (`extract_features.py`, `train_models.py`, `evaluate.py`) e disponibilize o link do repositório no paper.
