# Quiet Eye Proxy & Machine Learning no Basquete 🏀👁️

Estimativa não-invasiva de métricas de Quiet Eye (QE) a partir de vídeos de lances livres da NBA utilizando Visão Computacional e predição de desempenho com Machine Learning.

---

## 📁 Estrutura do Projeto

```text
├── data/                      # Gestão de dados do projeto
│   ├── raw/                   # Vídeos brutos coletados (.mp4, .avi)
│   ├── processed/             # Datasets com features extraídas (.csv, .parquet)
│   └── metadata/              # Planilhas de anotação/labels dos arremessos
│
├── src/                       # Código-fonte modular em Python
│   ├── vision/                # Extração de pose, face mesh e cálculo de Quiet Eye
│   ├── models/                # Treinamento, validação e pipelines de ML
│   └── utils/                 # Funções auxiliares (I/O, métricas, plots)
│
├── models/                    # Modelos treinados salvos (.pkl, .joblib, .pt, .onnx)
│
├── notebooks/                 # Jupyter Notebooks para análise exploratória e testes
│
├── paper/                     # Escrita do artigo científico em LaTeX
│   ├── artigo.tex             # Documento LaTeX principal
│   ├── references.bib         # Referências bibliográficas BibTeX
│   └── figures/               # Gráficos e figuras incluídas no artigo
│
├── docs/                      # Documentações, guias e resumos teóricos
│   ├── GUIA_PROJETO_QUIET_EYE.md
│   └── resumo_skillsight.txt
│
├── requirements.txt           # Dependências do ambiente Python
├── .gitignore                 # Arquivos e diretórios ignorados pelo Git
└── README.md                  # Apresentação e guia do repositório
```

---

## 🚀 Como Começar

1. **Clone o repositório:**
   ```bash
   git clone git@github.com:joaopaulocruzdefaria/quietEye-MachineLearning.git
   cd quietEye-MachineLearning
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
