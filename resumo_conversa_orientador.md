# Resumo para conversa com o orientador
## Da pergunta sobre ordem de extração ao estado atual da tese

---

## 1. A pergunta original do senhor

> *"Pega o primeiro neurônio de saída e exporta todos os seus pesos de entrada... A sequência linear de números segue [C_out, C_in, H, W] em ordem row-major. Se o seu método assume pesos agrupados por neurônio de entrada, você precisará transpor o tensor antes."*

O ponto central era: **eu sabia que tinha escolhido essa ordem, ou ela tinha sido escolhida por mim pelo PyTorch?** E qual a consequência disso para a análise.

---

## 2. O que assumimos como compromisso metodológico

Em resposta, ficou registrado na metodologia da tese (Seção 3.5, "Extração ordenada dos pesos") a **pirâmide de decisões**:

**Camada inferior (herdada — não negociável):**
- Ordem entre módulos do `state_dict` (determinística, herdada da arquitetura)
- Layout row-major `[C_out, C_in, H, W]` para Conv2d, `[Out, In]` para Linear (convenção C-contiguous do PyTorch)

**Camada superior (escolhas analíticas defendidas):**
- **D1**: agrupamento em 4 categorias (`conv`, `dense`, `bn_gamma`, `bn_beta`)
- **D2**: Conv2d agrupada por **neurônio de saída** (sem transposição). Bloco contíguo de $C_{in} \cdot H \cdot W$ valores = um filtro inteiro = unidade interpretável aprendida.
- **D3**: subamostragem **sistemática** (passo fixo) preservando a ordem do flatten
- **D4**: cálculo da LMC e SampEn **por tensor** com agregação posterior por grupo
- **D5**: inclusão de γ e β na análise; exclusão de bias de Conv2d/Linear

Adicionalmente, montei um **Apêndice A** com exemplo numérico rastreando elemento a elemento o mapeamento `[2,2,2,2]` → vetor flat, incluindo o contraste com a versão transposta — exatamente como o senhor sugeriu.

---

## 3. O que rodou depois (4 fases experimentais)

Com a metodologia decidida, executei 4 sweeps controlados para testar a hipótese H1 ("a complexidade dos pesos é marcador de overfitting"):

### Fase 1 — Subamostragem do treino (induzir overfit)
- PathMNIST a 5%, 10%, 25% × 3 sementes × 100 épocas = 9 runs
- **Achado**: LMC pica antes do mínimo de val_loss em todos os 3 regimes (lead time +12 a +17 épocas)

### Fase 2 — Weight decay (modular regularização)
- PathMNIST × WD ∈ {0, 10⁻⁵, 10⁻⁴, 10⁻³} × 3 sementes = 12 runs
- **Achado**: pico LMC em ~7 épocas independente de WD; o mínimo de val_loss varia de 42 a 93 épocas

### Fase 3 — Arquitetura (testar invariância de família)
- PathMNIST × {ResNet-18, ResNet-34, DenseNet-121} × 3 sementes = 9 runs
- **Achado**: pico LMC em janela [4, 9] épocas para as 3 arquiteturas

### Análise cross-dataset (gratuita, usando baseline existente)
- Aplicar mesmo detector a PathMNIST + OrganAMNIST + BloodMNIST, 100 épocas, 3 sementes = 9 runs já existentes
- **Achado surpresa**: a invariância intra-dataset **NÃO se generaliza** entre datasets:
  - PathMNIST: pico LMC em 6 épocas ✓
  - BloodMNIST: pico LMC em 6 épocas ✓
  - **OrganAMNIST: pico LMC em 24 épocas** (fator 4× de deslocamento)
- A decomposição por grupo (Seção 4.7.3) localizou o anomalismo: ele aparece sincronicamente em `bn_gamma` e `bn_beta`, mas não em `conv`/`dense` — consistente com adaptação mais lenta das estatísticas de ativação em OrganAMNIST.

---

## 4. O que isso fez com a hipótese H1

A hipótese inicial era *"LMC dos pesos é marcador objetivo do início do sobreajuste."*

Os resultados a refinaram para uma versão mais defensável:

> *A trajetória da LMC dos pesos de `bn_gamma` atinge seu extremo **antes** do mínimo de val_loss. O instante exato e a forma da trajetória dependem do dataset, mas o **ordenamento temporal** (pico-LMC → mínimo-val_loss) é robusto entre subamostragens do treino, intensidades de weight decay e famílias arquiteturais em um mesmo dataset.*

Operacionalmente: o pico da LMC marca uma **transição estrutural intrínseca** ("saída do regime de inicialização gaussiana"), cuja escala temporal é controlada pelo regime de treino e modulada pelo dataset, mas que sempre precede a generalização ótima.

---

## 5. Análises com SampEn 2D (sem refazer experimentos)

Como o senhor apontou que a SampEn 2D opera direto na matriz **sem flatten**, ela contorna o problema da ordem de extração. Apliquei a SampEn 2D ao mosaico 56×56 dos 64 filtros 7×7 da `conv1`:

### Resultado simples (Seção 4.8)
SampEn 2D no checkpoint final, $r = 0{,}5\sigma$:
- PathMNIST: **3,98** [CI 95% 3,67–4,23]
- OrganAMNIST: **5,77** [5,66–5,89]
- BloodMNIST: **6,18** [6,08–6,30]

**Os CIs não se sobrepõem entre datasets** — a SampEn 2D discrimina os três datasets. A SampEn 1D do grupo `conv` no mesmo checkpoint é quase idêntica entre eles (~2,07–2,10), ou seja, o flatten encobre a diferença.

**Achado bônus interpretável**: PathMNIST mostra assimetria de canais — canal vermelho com SampEn 2D = 3,36 vs canais verde/azul = 4,28. Consistente com padrões espaciais característicos do corante H&E.

### Extensão multiescala MSE 2D (Seção 4.9)
Aplicando coarse-graining 2D em escalas τ = 1 a 10:
- **Em τ = 1** (escala fina): PathMNIST tem a MENOR SampEn 2D
- **Em τ = 4–7** (escala intermediária): PathMNIST tem a MAIOR SampEn 2D
- **Cruzamento das curvas em τ ≈ 3**

Implicação metodológica: a escolha de escala é crítica — reportar SampEn 2D com τ=1 fixo pode dar discriminação enganosa. A assinatura multiescala é mais robusta.

---

## 6. Material para defender se o senhor questionar

### "Quantos runs?" 
**60+ runs** totais: 27 baseline + 9 subamostragem + 12 WD + 6 arquitetura = 54 runs controlados; mais análises post-hoc cross-dataset, SampEn 2D e MSE 2D.

### "Por que apenas ResNet-18 nas Fases 1 e 2?"
Por compatibilidade com o baseline validado pelo grupo. A Fase 3 estende para ResNet-34 e DenseNet-121, e mostra invariância intra-família.

### "Como detectaram o pico?"
Detector robusto descrito na Seção 4.4.3.1: restrição a `bn_gamma`, exclusão da janela de warm-up (epoch < 0,5), suavização Savitzky-Golay (janela 7, polinômio 3), intervalo de confiança 95% por bootstrap de 1000 reamostragens das sementes.

### "Por que `bn_gamma`?"
Identificado empiricamente na Fase 1 (Figura 8 da tese): o sinal antecipador concentra-se nos parâmetros γ; `conv` é ruidoso, `dense` é monotônico, `bn_beta` segue γ com ruído maior. Justificativa post-hoc, mas internamente consistente.

### "E o problema da ordem do flatten?"
Endereçado por três caminhos:
1. **D2 + Apêndice A**: ordem definida e ilustrada elemento a elemento
2. **D3**: subamostragem sistemática (não embaralha)
3. **Seção 4.8**: SampEn 2D contorna o flatten ao operar direto na matriz; e Seção 4.9 estende para multiescala

### "Tese pode quebrar onde?"
Honestamente:
- O caso OrganAMNIST quebra a invariância clean — está reportado e interpretado como achado, não escondido
- A SampEn 1D **não funciona como marcador antecipador independente** (Seção 4.4.4) — também reportado
- O detector robusto exige escolhas (grupo, janela, suavização) que poderiam ser mais sistematicamente justificadas — listado como trabalho futuro (item 1)

---

## 7. Estado da tese hoje

- **76 páginas**, 1,58 MB
- **5 capítulos** + 1 apêndice
- **7 achados consolidados** na conclusão
- **7 direções de trabalhos futuros**
- Capítulo de Resultados tem **9 seções** (4.1 a 4.9) cobrindo: visão geral, LMC, SampEn 1D, subamostragem (4.4), weight decay (4.5), arquitetura (4.6), cross-dataset (4.7), SampEn 2D (4.8), MSE 2D (4.9)

---

## 8. Perguntas para o senhor

1. A reformulação da H1 (passar de "marcador de overfit" para "marcador da transição estrutural que precede a generalização ótima") está aceitável ou prefere que mantenhamos o framing original?

2. O achado de OrganAMNIST com fator 4× de deslocamento — vale a pena rodar a Fase 1 (subsampling) também em OrganAMNIST para investigar o mecanismo, ou ficamos com a interpretação atual?

3. O detector robusto exige 3 escolhas combinadas (bn_gamma, warm-up cutoff, SavGol). Devo investir em detectores alternativos (regressão segmentada, change-point bayesiano) para fechar o item 1 dos trabalhos futuros?

4. A SampEn 2D + MSE 2D nos mosaicos da `conv1` ficou só na cross-dataset comparison. Vale aplicar também à evolução temporal (durante o treinamento) — mas isso exigiria salvar checkpoints completos a cada época, o que aumentaria o custo de disco significativamente.

5. O Apêndice A com o exemplo numérico que o senhor sugeriu ficou OK ou ele expande mais (mais um exemplo, talvez com a transposição visualizada)?
