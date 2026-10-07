# 📊 Dicionário de Atributos e Engenharia de Features

> **Princípio Mandatório:** Todas as variáveis de entrada $\mathbf{x}_t$ são estritamente **estacionárias** e calculadas utilizando exclusivamente dados disponíveis até o fechamento do pregão $t$.

---

## 1. Família 1: Métricas de Desconto Histórico e Ciclos de Preço

Em vez de utilizar múltiplos de balanço contábil que sofrem de atraso de reporte (*lookahead bias*), modela-se o desconto e a extensão de ciclo através de métricas quantitativas contínuas de preço ajustado:

| Atributo | Fórmula Matemática | Interpretação Financeira |
|---|---|---|
| `zscore_sma_200` | $\frac{P_t - \text{SMA}_{200}(P)_t}{\sigma_{200}(P)_t}$ | Desvio-padrão do preço atual em relação à média móvel de 200 pregões (~1 ano). Valores negativos acentuados indicam desconto histórico de ciclo. |
| `zscore_sma_500` | $\frac{P_t - \text{SMA}_{500}(P)_t}{\sigma_{500}(P)_t}$ | Desvio-padrão do preço atual em relação ao ciclo bienal (500 pregões). Captura fundos cíclicos plurianuais. |
| `drawdown_52w` | $\frac{P_t - \max_{\tau \in [t-252, t]} P_\tau}{\max_{\tau \in [t-252, t]} P_\tau}$ | Distância percentual em relação à máxima observada nas últimas 52 semanas (1 ano). Mensura o tamanho da correção vigente. |
| `bollinger_pct_b_long` | $\frac{P_t - \text{BB}_{\text{lower}, 200, 2\sigma}}{\text{BB}_{\text{upper}, 200, 2\sigma} - \text{BB}_{\text{lower}, 200, 2\sigma}}$ | Posição relativa do preço dentro do canal de volatilidade de 200 pregões ($<0$ indica ativo negociado abaixo da banda inferior extrema). |
| `spread_asset_cdi_past60` | $R_{t-60 \to t} - CDI_{t-60 \to t}$ | Excesso de retorno observado no ativo contra o CDI nos 60 pregões anteriores. Avalia se o papel está em momento de sobrecompra ou sobrevenda frente à taxa básica. |

---

## 2. Família 2: Momentum, Osciladores e Volatilidade

Indicadores técnicos clássicos normalizados para garantir estacionariedade:

| Atributo | Descrição Matemática | Faixa / Normalização |
|---|---|---|
| `rsi_14` | Índice de Força Relativa de 14 períodos com médias de Wilder. | Normalizado em $[0, 1]$ dividindo por 100. |
| `stoch_k_14` | Oscilador Estocástico rápido $\%K$: $\frac{P_t - \min_{14} L}{\max_{14} H - \min_{14} L}$. | Limitado em $[0, 1]$. |
| `macd_normalized` | $\frac{\text{MACD Line}_{12,26} - \text{Signal Line}_9}{P_t}$ | Linha MACD normalizada pelo preço de fechamento para conferir comparabilidade temporal. |
| `atr_ratio_14` | $\frac{\text{ATR}_{14}(P)_t}{P_t}$ | *Average True Range* de 14 períodos normalizado pelo preço. Mensura a volatilidade recente do papel. |
| `volume_zscore_20` | $\frac{V_t - \text{SMA}_{20}(V)_t}{\sigma_{20}(V)_t}$ | Z-score do volume financeiro negociado nos últimos 20 pregões, identificando anomalias de liquidez e fluxo institucional. |

---

## 3. Família 3: Covariáveis Macroeconômicas (BACEN)

Séries oficiais do Banco Central do Brasil alinhadas temporalmente:

| Atributo | Origem dos Dados | Fórmula / Transformação |
|---|---|---|
| `selic_annualized` | SGS Série 11 (Selic Over) | Taxa Selic anualizada vigente no dia $t$: $(1 + r_{\text{selic}, t})^{252} - 1$. |
| `selic_delta_60` | SGS Série 11 (Selic Over) | Variação percentual na taxa Selic acumulada nos últimos 60 pregões úteis ($\Delta$ de ciclo monetário). |
| `usdbRL_return_60` | SGS Série 1 (Dólar PTAX) | Retorno percentual do câmbio comercial nos últimos 60 pregões: $\frac{\text{PTAX}_t}{\text{PTAX}_{t-60}} - 1$. |
| `usdbRL_vol_20` | SGS Série 1 (Dólar PTAX) | Volatilidade realizada (desvio-padrão dos retornos logarítmicos diários) da moeda em 20 dias úteis. |

---

## 4. Matriz de Tratamento e Pré-Processamento

1. **Eliminação de NaN Inicial:** Como o atributo mais longo requer 500 pregões para estabilização de cálculo (`zscore_sma_500`), os primeiros 504 pregões do dataset histórico (2014–2015) atuam como período de aquecimento (*warm-up period*).
2. **Escalonamento Robusto:** Para os modelos sensíveis à escala (Regressão Logística), utiliza-se o `RobustScaler`, que subtrai a mediana e divide pelo intervalo interquartil (IQR), mitigando a sensibilidade a *outliers* de crises financeiras. O ajuste (`fit`) é restrito aos dados de treino de cada janela Walk-Forward.
