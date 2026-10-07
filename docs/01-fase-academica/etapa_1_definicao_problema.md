# 📑 Etapa I: Definição do Problema, Hipóteses e Viabilidade da Base de Dados

> **Disciplina:** Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina  
> **Tema:** Motor Modular de Análise Preditiva e Decisão Tática para Ações da B3 com Validação Walk-Forward  
> **Entregável:** Fundamentação da Seção I (Introdução) do artigo científico IEEE Conferences.

---

## 1. Contextualização e Relevância

O mercado de capitais no Brasil, centralizado na **B3 (Brasil, Bolsa, Balcão)**, constitui um ambiente singular no cenário financeiro global de mercados emergentes. Caracteriza-se por:
* **Elevada Taxa Livre de Risco Histórica:** A taxa Selic média e o CDI praticados no Brasil impõem um custo de oportunidade extremamente alto para os alocadores de capital. Investir em renda variável brasileira apenas faz sentido racional quando há expectativa de prêmio de risco substancial sobre a taxa de juros real.
* **Sensibilidade Macroeconômica:** As ações brasileiras não flutuam de modo isolado; sua precificação é fortemente modulada por choques na taxa Selic, variações cambiais (USD/BRL) e pelo ciclo internacional de commodities.
* **Comportamento Cíclico de Longo Prazo:** Os papéis da bolsa exibem fortes oscilações em torno de suas médias e múltiplos históricos de valuation, alternando períodos de euforia com descontos expressivos de preço.

---

## 2. Hipótese Científica e Formulação Matemática

### 2.1 Por que o Horizonte Tático de 60 Pregões ($h=60$)?
A grande maioria dos trabalhos acadêmicos ingênuos foca em prever o retorno do pregão seguinte ($h=1$). Contudo, em finanças quantitativas, o ruído intradiário da microestrutura de mercado possui uma razão sinal-ruído (*Signal-to-Noise Ratio* - SNR) tendendo a zero, o que torna as previsões diárias estatisticamente indistinguíveis de passeios aleatórios (*Random Walk*).

Em contrapartida, em um **horizonte tático de 60 pregões úteis** (~3 meses civis, correspondente a um ciclo de divulgação de resultados trimestrais):
* Métricas de **desconto histórico de preço** e **reversão à média** exercem força preditiva acumulada.
* O custo de oportunidade (taxa CDI acumulada no trimestre) atua como um limiar econômico nítido.
* Fricções operacionais e custos de transação (corretagens e emolumentos) têm impacto percentual diluído frente ao horizonte temporal.

### 2.2 Formulação do Alvo Preditivo (Target)
O problema é modelado como **classificação binária supervisionada**. Para um instante temporal $t$, calculamos o retorno acumulado do ativo no horizonte futuro de 60 pregões ($R_{t \to t+60}$) e o retorno acumulado da taxa livre de risco CDI no mesmo intervalo ($CDI_{t \to t+60}$):

$$R_{t \to t+60} = \frac{P_{t+60}}{P_t} - 1$$

$$CDI_{t \to t+60} = \prod_{k=1}^{60} \left(1 + r_{\text{CDI}, t+k}\right) - 1$$

O rótulo binário $y_t \in \{0, 1\}$ é formalmente definido como:

$$y_t = \begin{cases} 
1, & \text{se } R_{t \to t+60} > CDI_{t \to t+60} \quad (\text{Ativo gerou alpha sobre a taxa livre de risco}) \\
0, & \text{se } R_{t \to t+60} \le CDI_{t \to t+60} \quad (\text{Custo de oportunidade não superado})
\end{cases}$$

O objetivo do modelo de Aprendizado de Máquina é estimar a probabilidade a posteriori calibrada:

$$\hat{p}_t = \hat{P}(y_t = 1 \mid \mathbf{x}_t)$$

onde $\mathbf{x}_t$ é o vetor de atributos puramente passados conhecidos até o fechamento do dia $t$.

---

## 3. Estado da Arte e Lacunas de Pesquisa

| Lacuna Identificada na Literatura | Abordagem Comum (Falha) | Proposta Deste Projeto |
|---|---|---|
| **Formulação de Target Não-Estacionário** | Tentar prever o preço nominal bruto ($P_{t+1}$) via regressão linear ou LSTM, sofrendo de alta acurácia espúria defasada. | Classificação supervisionada de excesso de retorno sobre o CDI em horizonte tático ($h=60$), com séries estritamente estacionárias. |
| **Vazamento Temporal de Dados** | K-fold aleatório ou padronização de dados (`StandardScaler`) no dataset inteiro. | Validação Walk-Forward temporal rigorosa com janelas expansivas e protocolo de **Purging e Embargo** para eliminar sobreposição. |
| **Desconexão do Custo de Oportunidade** | Alvos simplistas de valorização nominal ($R > 0$), ignorando a taxa de juros do país. | Benchmark real e audital da economia brasileira: superação do CDI acumulado do período. |
| **Predições Não Calibradas** | Uso cego de limiares rígidos ($p > 0.5$) sem avaliar a confiabilidade probabilística. | Calibração de probabilidades (Platt Scaling / Isotonic) avaliadas por Brier Score e Log-Loss, permitindo dimensionamento de capital. |
| **Falta de Teste de Generalização Cruzada** | Testar o modelo em um único ativo e extrapolar conclusões. | Teste comparativo sistemático entre dois setores ortogonais: commodities cíclicas (**PETR4**) vs. crédito/bancário (**ITUB4**). |

---

## 4. Viabilidade Empírica da Base de Dados

A viabilidade técnica e a reprodutibilidade integral do projeto são garantidas pelo acesso a bases de dados financeiras e governamentais oficiais e abertas:

1. **Cotações Históricas dos Ativos (B3 via Yahoo Finance):**
   * Séries temporais de cotações diárias ajustadas por proventos (dividendos e desdobramentos): Abertura, Máxima, Mínima, Fechamento e Volume (OHLCV).
   * Período histórico de coleta: Mais de 10 anos (2014 a 2024), proporcionando mais de 2.500 pregões por ativo e cobrindo múltiplos regimes de mercado (impeachment, corte de juros, crise pandêmica de 2020 e ciclo de alta da Selic).
   * Coleta automatizada via biblioteca `yfinance` em Python.

2. **Indicadores Macroeconômicos Oficiais (SGS - Banco Central do Brasil):**
   * **Taxa Selic Over diária (Série SGS 11):** Permite calcular com precisão exata o CDI diário e o retorno acumulado da taxa livre de risco em qualquer janela temporal.
   * **Câmbio Comercial USD/BRL PTAX (Série SGS 1):** Cotação média oficial diária do dólar norte-americano apurada pelo BACEN, fundamental para capturar a exposição cambial dos ativos da B3.
   * Coleta automatizada via API REST pública do Sistema Gerenciador de Séries Temporais (SGS/BACEN).

3. **Integridade e Tratamento de Feriados:**
   * Alinhamento estrito dos calendários de negociação da B3 e do BACEN através de junção interna (*inner join*) nas datas úteis, garantindo ausência de valores ausentes (*NaN*) e sincronia temporal perfeita.
