# 📐 Gestão de Risco e Dimensionamento de Posição via Critério de Kelly Fracionário

> **Escopo:** Fase 2 (Produto / Ferramenta Real) — Metodologia matemática para converter a probabilidade calibrada $\hat{p}_t = P(Y=1|\mathbf{x}_t)$ em alocação percentual de capital da carteira.

---

## 1. Por que Probabilidades Calibradas são Indispensáveis?

Na maioria das ferramentas amadoras de investimento, o modelo de Machine Learning apenas fornece um sinal binário (Comprar / Vender). Essa abordagem é falha porque:
1. Trata uma probabilidade de $51\%$ com a mesma agressividade de uma de $85\%$.
2. Ignora a assimetria entre ganhos esperados e perdas potenciais.
3. Não oferece critérios matemáticos para dimensionamento de lote ou proteção contra ruína.

Ao garantir que o modelo esteja calibrado (via *Platt Scaling* ou Regressão Isotônica), o valor $\hat{p}_t$ representa a frequência empírica real de superação do CDI.

---

## 2. Formulação Matemática do Critério de Kelly

O Critério de Kelly (*Kelly Criterion*) determina a fração ótima de capital $f^*$ a ser alocada em um ativo para maximizar a taxa de crescimento geométrico do patrimônio no longo prazo:

$$f^* = \frac{p \cdot b - q}{b} = p - \frac{q}{b}$$

onde:
* $p = \hat{p}_t$: Probabilidade calibrada estimada pelo modelo de o ativo superar o CDI no horizonte de 60 pregões.
* $q = 1 - p$: Probabilidade de o ativo não superar a taxa livre de risco.
* $b$: Razão de *payoff* (relação entre o ganho médio nas operações vitoriosas e a perda média nas operações perdedoras):
  $$b = \frac{\mathbb{E}[\text{Excesso de Retorno} \mid y=1]}{\mathbb{E}[|\text{Sub-rendimento}| \mid y=0]}$$

---

## 3. Atenuação via Critério de Kelly Fracionário (*Fractional Kelly*)

Em séries financeiras reais, estimativas pontuais de $p$ e $b$ possuem incerteza amostral. A aplicação cega do Kelly pleno ($f^*$) gera alocações excessivamente voláteis e rebaixamentos (*drawdowns*) severos.

Portanto, o motor adota o **Half-Kelly** ou **Quarter-Kelly** (fração prudencial $\lambda \in [0.25, 0.50]$):

$$f_{\text{tático}} = \lambda \cdot f^*$$

### Regras de Salvaguarda (Risk Guardrails):
1. **Piso Mínimo de Convicção:** Se $\hat{p}_t \le 0.52$, o modelo define $f = 0$ (alocação 100% no CDI com risco zero).
2. **Teto Máximo por Ativo:** Jamais alocar mais de $25\%$ do patrimônio total em um único ticker ($f \le 0.25$), garantindo diversificação compulsória.
3. **Reserva de Emergência em Renda Fixa:** O capital não alocado em ações permanece automaticamente rendendo 100% do CDI pós-fixado.
