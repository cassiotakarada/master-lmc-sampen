# Plano de Trabalho — Complexidade LMC e SampEn como Indicadores de Treinamento

**Data:** 2026-09-11
**Repositório:** `~/Área de trabalho/Masters Claude`
**Documento do projeto (3 pilares):** a ser escrito/atualizado — modelo de referência em `projeto_exemplo.pdf`
**Dissertação (texto):** `~/Área de trabalho/Claude/thesis-complexity-nn/tese/`

---

## 0. Decisões tomadas na reunião com o orientador

| # | Decisão | Impacto |
|---|---------|---------|
| D1 | **Foco exclusivo na camada densa.** Extrair os pesos apenas dela. | Elimina conv/batchnorm da análise; simplifica muito o pipeline |
| D2 | **Descrever o "caminho"** (entrada→neurônio ou neurônio→entrada) e fixar a ordem do achatamento (*flatten*). | Preferência do orientador: ordem por neurônio (`n1x1, n1x2, ...`) |
| D3 | **Justificativa voltada ao desenvolvedor** da rede/treinamento, nos 3 pilares. | Reescrita do texto do projeto |
| D4 | **LMC + SampEn como 2 indicadores** de convergência / aprendizado / overfitting. | Objetivo central dos resultados |
| D5 | **Particionamento treino/validação/teste** bem descrito no texto E aplicado no código. | Hoje o código **não tem split de teste** |
| D6 | **Dados após o overfitting não devem ser utilizados.** | Precisa de definição operacional de "início do overfitting" + corte |
| D7 | **Cálculo ao vivo durante o treinamento**, devolvendo o *status* da máquina. | Novo módulo no `Trainer` |
| D8 | **Entregar os códigos de cálculo de LMC e SampEn** ao orientador. | Pacote autocontido e documentado |

### Ponto a confirmar com o orientador
- ~~"Metodologia Grok e Grep"~~ — **RESOLVIDO (2026-09-11):** era somente **grokking** (generalização tardia). A §3.3 e a F7 seguem com essa leitura.

---

## 1. Diagnóstico do estado atual do código

### O que já existe e funciona
- `src/main_train.py` + `src/training/trainer.py` — treino MONAI (MedNIST), ResNet-18 / DenseNet-121, salva `epoch_XXX.pth` por época, `history.csv`, TensorBoard.
- `monai_lmc_sampen_plot.py` — calcula LMC e SampEn por época, **média sobre todas as camadas**.
- `aggregate_lmc_sampen_plot.py` — agrega várias sementes.
- `data_analysis_summary.py`, `summary_complexity_plot.py`, `monai_wieghts_plot.py` — gráficos por camada.
- `run_sweep.py` — roda experimentos em sequência.
- `best_epochs.py` — acha a melhor época pelo `val_loss`.
- Runs já existentes: `monai_weights/run_mednist_resnet18_64{,_seed1,_seed7}` (ResNet-18, 64×64, 20 épocas, 3 sementes).

### Problemas identificados (que este plano corrige)

| ID | Problema | Onde | Gravidade |
|----|----------|------|-----------|
| P1 | LMC/SampEn **duplicados em 4 arquivos** (copy-paste). Qualquer correção precisa ser feita 4 vezes. | `data_analysis_summary.py`, `monai_lmc_sampen_plot.py`, `summary_complexity_plot.py`, `monai_wieghts_plot.py` | Alta |
| P2 | **Não existe split de teste.** `main_train.py` usa só as seções `training` e `validation` do MedNIST; `cfg.val_split` é declarado mas **nunca usado**. | `src/main_train.py:prepare_mednist_datasets` | Alta (D5) |
| P3 | **Não há parada / corte por overfitting.** O `Trainer` salva `aft_weights` na **primeira** subida do `val_loss` — critério frágil (ruído de uma época dispara). Não há *patience*, não há corte das épocas posteriores. | `src/training/trainer.py:fit` | Alta (D6) |
| P4 | **Subamostragem aleatória** (`rng.choice(..., replace=False)`) embaralha o vetor antes da SampEn, que é sensível à ordem. | `monai_lmc_sampen_plot.py:66` e gêmeos | Média — **desaparece com D1**, pois `fc.weight` tem 3.072 pesos (< 10.000, nunca subamostra) |
| P5 | A análise **tira a média de LMC/SampEn sobre todas as camadas**, misturando conv, batchnorm e densa. | `compute_metrics_for_weights` | Média — **resolvido por D1** |
| P6 | `train_val_split` usa `random.shuffle` **sem semente** → não reprodutível. | `src/data/split.py:11` | Média |
| P7 | O valor absoluto da LMC depende fortemente do nº de *bins* (0,65 / 0,52 / 0,36 / 0,23 para 20/50/100/200 bins). Falta normalização e justificativa do `bins`. | `lmc_complexity` | Média |
| P8 | A `sample_entropy` local nunca foi validada contra implementação de referência. | idem | Média |
| P9 | Nenhum cálculo de complexidade **durante** o treino — tudo é *post-mortem*. | `src/training/trainer.py` | Alta (D7) |

---

## 2. Fundamento técnico da decisão D2 — o "caminho" e a ordem do flatten

### 2.1 Formalização

Seja a camada densa com $M$ entradas $x_1 \dots x_M$ e $N$ neurônios $n_1 \dots n_N$.
A matriz de pesos é $W \in \mathbb{R}^{N \times M}$, e $w_{ji}$ é o peso do **caminho** $x_i \rightarrow n_j$.

No PyTorch, `nn.Linear.weight` tem exatamente a forma `[out_features, in_features]` = $[N, M]$.
No nosso ResNet-18 com 6 classes: `fc.weight` tem forma **[6, 512]** → **3.072 pesos**.

### 2.2 As duas ordens de percurso

**(A) Ordem por neurônio — "n-major" (preferência do orientador)**

$$v_A = (w_{11}, w_{12}, \dots, w_{1M},\; w_{21}, w_{22}, \dots, w_{2M},\; \dots,\; w_{NM})$$

Lê-se: *todos os pesos que chegam em $n_1$, depois todos os que chegam em $n_2$...* — ou seja, `n1x1, n1x2, n1x3, ..., n2x1, ...`
No código: `W.reshape(-1)` — é o achatamento **nativo** do PyTorch, sem transposição.

**(B) Ordem por entrada — "x-major"**

$$v_B = (w_{11}, w_{21}, \dots, w_{N1},\; w_{12}, w_{22}, \dots, w_{N2},\; \dots)$$

Lê-se: *todos os pesos que saem de $x_1$, depois os que saem de $x_2$...* — ou seja, `x1n1, x1n2, ..., x2n1, ...`
No código: `W.t().reshape(-1)`.

### 2.3 Por que isso importa

- **A LMC é invariante à ordem.** Ela só usa o histograma dos valores → $\text{LMC}(v_A) = \text{LMC}(v_B)$ exatamente.
- **A SampEn NÃO é invariante à ordem.** Ela compara janelas consecutivas do vetor → as duas ordens dão números diferentes, e a diferença carrega significado (a ordem A mede regularidade *dentro do receptivo de cada neurônio*; a ordem B mede regularidade *na distribuição de uma entrada entre os neurônios*).

Por isso: **implementar as duas como opção** (`--flatten_order {n_major,x_major}`), **default `n_major`**, e **reportar a ablação** comparando as duas. Isso vira uma subseção de metodologia com figura.

### 2.4 Convenções fixadas
- **Excluir o bias** (`fc.bias`) do vetor — o projeto analisa *caminhos*, e o bias não é um caminho. (O código atual já exclui.)
- Camada densa alvo: `fc.weight` (ResNet-18) e `class_layers.out.weight` (DenseNet-121) — resolver por `isinstance(module, nn.Linear)`, não por nome fixo.
- Se houver mais de uma camada densa, analisar cada uma separadamente (nunca concatenar).

---

## 3. Definição operacional de overfitting (D6) e particionamento (D5)

### 3.1 Os três conjuntos
| Conjunto | Para que serve | Regra de uso |
|----------|----------------|--------------|
| **Treino** | ajustar os pesos | único que entra no `backward()` |
| **Validação** | escolher a época e detectar overfitting | **nunca** entra no gradiente; olhado toda época |
| **Teste** | medir o desempenho final | **olhado uma única vez**, no fim, na época selecionada |

Analogia: treino = lista de exercícios; validação = simulado; teste = a prova de verdade. Quem usa a prova para estudar não sabe se aprendeu.

### 3.2 Definição de "início do overfitting"

1. $e^\* = \arg\min_e \; \text{val\_loss}(e)$ — **melhor época**.
2. **Início do overfitting** $e^{of}$ = a primeira época $e > e^\*$ tal que
   $\text{val\_loss}(e) > \text{val\_loss}(e^\*) + \delta$ por **`patience` épocas consecutivas**.
   Defaults propostos: $\delta = 0{,}0$ (só a subida basta) e `patience = 3`.
3. **Regra de corte (D6):** toda análise de LMC/SampEn usa apenas as épocas $e \le e^{of}$. As épocas posteriores ficam gravadas em disco (para a figura de "o que acontece depois"), mas **marcadas como `post_overfit=True`** e **excluídas** das conclusões.
4. Registrar tudo em `overfit_report.json` por run: `best_epoch`, `overfit_onset_epoch`, `patience`, `delta`, `n_epochs_usadas`, `n_epochs_descartadas`.

### 3.3 Grokking
Grokking é o caso em que a acurácia de validação fica parada por muito tempo e **dispara tarde**. Isso quebra o critério ingênuo de parada: parar cedo demais mata o grokking. Por isso o plano prevê:
- `patience` configurável e um modo `--no_early_stop` que treina até o fim e só **marca** o corte a posteriori (é o modo que usaremos nos experimentos, para ter a trajetória completa);
- uma hipótese testável: **a LMC/SampEn da camada densa muda antes da acurácia de validação** → serviria de aviso antecipado, inclusive em regime de grokking.

---

## 4. Os 2 indicadores e o status ao vivo (D7)

### 4.1 O que o monitor calcula a cada época
Sobre o vetor $v$ (camada densa achatada na ordem escolhida), a cada fim de época:
- `lmc` — complexidade LMC
- `sampen` — entropia amostral
- `d_lmc`, `d_sampen` — variação em relação à época anterior
- média móvel de 3 épocas de ambos (para reduzir ruído)

Custo: 3.072 pesos → a SampEn faz uma matriz de distâncias ~3.070×3.070 ≈ 9,4 M pares. **Menos de 1 s por época em CPU** — desprezível frente ao treino. (Validar no passo F4.)

### 4.2 Máquina de estados do *status*

| Status | Condição (proposta inicial, a calibrar com os dados) |
|--------|------------------------------------------------------|
| `INICIALIZANDO` | épocas 1–2, sem histórico suficiente |
| `APRENDENDO` | `val_loss` caindo **e** LMC/SampEn com deriva monotônica (a rede está se auto-organizando) |
| `CONVERGINDO` | `val_loss` ainda caindo, mas `|d_lmc|` e `|d_sampen|` abaixo de um limiar → os pesos estão se estabilizando |
| `ESTAGNADO` | `val_loss` plano **e** complexidade plana por ≥ `patience` épocas |
| `ALERTA_OVERFIT` | complexidade inverte a tendência **antes** de o `val_loss` subir (hipótese central do trabalho) |
| `OVERFITTING` | critério da §3.2 satisfeito |

> **Importante (honestidade científica):** os limiares só podem ser fixados **depois** de olhar as trajetórias dos runs existentes. O status é uma *ferramenta de leitura*, não um resultado — o texto deve deixar isso explícito, e o ajuste dos limiares deve ser feito num conjunto de runs e **verificado** em runs que não foram usados para ajustá-los.

### 4.3 Saídas do monitor
- colunas novas em `history.csv`: `lmc_dense`, `sampen_dense`, `status`
- escalares no TensorBoard: `Complexity/LMC_dense`, `Complexity/SampEn_dense`
- linha no log a cada época: `Epoch 7 | ... | LMC=0.3412 | SampEn=1.8871 | status=APRENDENDO`
- `complexity_live.csv` e `overfit_report.json` no diretório do run

---

## 5. Fases de execução

> As fases **F1–F6 são prioritárias** (decisões da reunião). **F7–F9 ficam para depois.**

---

### ✅ F1 — Módulo único de complexidade + pacote de entrega (D8) — **CONCLUÍDA em 2026-09-11**
**Por quê:** resolve P1 (4 cópias do mesmo código) e produz o material que o orientador pediu.

**Fazer:**
1. Criar `src/complexity.py` — **fonte única de verdade**:
   - `shannon_entropy_from_hist(p)` — com opção de normalização por $\log_2 N_{bins}$
   - `disequilibrium_from_hist(p)` — distância euclidiana ao equilíbrio, com opção normalizada
   - `lmc_complexity(x, bins=100, normalize=True)`
   - `sample_entropy(x, m=2, r=None)` — `r = 0.2·std` por padrão, com `r` explícito permitido
   - `flatten_dense(W, order="n_major")` — implementa a §2.2
   - `find_dense_layers(model_or_state_dict, param_types)` — localiza as camadas densas
   - docstrings com as fórmulas em LaTeX e as referências (López-Ruiz–Mancini–Calbet 1995; Richman & Moorman 2000)
2. Trocar os 4 scripts para **importarem** de `src/complexity.py` (apagar as cópias).
3. Alinhar com a implementação da tese (`thesis-complexity-nn/src/complexity.py`) — se divergirem, decidir uma e propagar.
4. Criar `deliverables/complexity_reference/` para enviar ao orientador:
   - `complexity.py` (cópia autocontida, só depende de numpy/scipy)
   - `README.md` com as fórmulas, parâmetros, exemplo de uso e limitações
   - `exemplo_uso.py` — roda sobre um `fc.weight` real e imprime os dois números

**Critério de aceite:** `grep -rn "def lmc_complexity" *.py src/` retorna **uma** definição; os 4 scripts continuam rodando e produzem os mesmos números de antes (comparar CSVs).

#### Resultado da F1

- `src/complexity.py` criado como fonte única. Em vez de consolidar as 4 cópias (que eram
  as piores versões), **portou-se a implementação já validada da dissertação**
  (`thesis-complexity-nn/src/complexity.py`), que traz de brinde `sample_entropy_2d`,
  `mse_1d` e `mse_2d` — necessários na F9. As convenções numéricas dos dois repositórios
  ficam assim idênticas por construção.
- Acrescentado ao módulo o que a §2 deste plano exige: `flatten_dense(W, order)` com as
  ordens `n_major` / `x_major`, `find_dense_layers(state_dict, param_types)` e
  `dense_complexity(W, order)`. O módulo depende só de numpy/scipy (sem torch), o que
  permite entregá-lo autocontido.
- Os 4 scripts (`data_analysis_summary.py`, `monai_lmc_sampen_plot.py`,
  `summary_complexity_plot.py`, `monai_wieghts_plot.py`) passaram a importar do módulo;
  168 linhas duplicadas removidas.
- **Armadilha encontrada e neutralizada:** o default adaptativo de bins
  (`min(100, max(10, n//5))`) diverge muito do `bins=100` fixo que os scripts usavam, em
  camadas pequenas — um BatchNorm de 64 pesos daria 12 bins e LMC ~40 % diferente
  (0,098 vs 0,135). Os wrappers nos scripts passam `n_bins=100` **explicitamente**.
  Na camada densa (3.072 pesos) as duas regras coincidem em 100 bins, então D1 não é afetada.
- **Verificação:** recálculo das épocas 1–3 de `run_mednist_resnet18_64` (41 tensores/época)
  contra o `epoch_metrics.csv` já existente → diferença máxima **2,8 × 10⁻¹⁷** na LMC e
  **0** na SampEn. Suíte de testes existente continua passando.
- Confirmado empiricamente o ponto da §2.3: `LMC(v_A) == LMC(v_B)` exatamente, enquanto
  `SampEn` difere (2,195628 vs 2,188732 no `fc.weight` da época 5).
- Pacote de entrega em `deliverables/complexity_reference/`: `complexity.py` (artefato
  **gerado** por `deliverables/build_deliverable.py`, para não recriar o problema das cópias),
  `README.md` com fórmulas/parâmetros/limitações/referências, `exemplo_uso.py` executável e
  `fc_weight_exemplo.npy` (matriz real [6, 512]) — roda só com numpy + scipy.

**Fora de escopo desta fase (de propósito):** os scripts continuam com o comportamento
antigo de média sobre *todas* as camadas. A troca para **somente camada densa** (D1) e a
exposição do flag `--flatten_order` acontecem na **F4/F6** — assim o critério de aceite da
F1 ("mesmos números") permanece verificável.

---

### ✅ F2 — Validação numérica das duas medidas (resolve P7, P8) — **CONCLUÍDA em 2026-09-11**
**Por quê:** antes de usar como "indicador", os números precisam estar certos.

**Fazer:**
1. `tests/test_complexity.py` com casos de referência:
   - LMC de distribuição uniforme → disequilíbrio ≈ 0 → LMC ≈ 0
   - LMC de uma delta (todo peso num bin) → entropia ≈ 0 → LMC ≈ 0
   - LMC máxima em distribuição intermediária (o "morro" característico)
   - SampEn de sinal periódico puro → ≈ 0; de ruído branco → alto
   - SampEn comparada a implementação de referência (`antropy` ou `nolds`) com tolerância declarada
   - `flatten_dense`: verificar que $v_A$ e $v_B$ são permutações uma da outra e que `LMC(v_A) == LMC(v_B)`
2. **Estudo de sensibilidade ao `bins`**: rodar LMC da camada densa com bins ∈ {20, 50, 100, 200, Freedman–Diaconis} ao longo das épocas. Decidir e **justificar no texto** o valor fixo. O que importa é a **posição do pico/da inflexão**, não o valor absoluto — mostrar se ela é estável.
3. Estudo de sensibilidade ao `r` da SampEn: r ∈ {0,1·σ; 0,15·σ; 0,2·σ; 0,25·σ}.

**Critério de aceite:** `pytest tests/test_complexity.py` verde; figura `sensibilidade_bins.png` mostrando que a época do pico não depende do `bins`.

#### Resultado da F2

**Testes — `tests/test_complexity.py`: 61 passando, 1 pulado.**
Nenhuma biblioteca externa estava instalada (`antropy`, `nolds`, `EntropyHub`…), então em vez
de acrescentar dependência escrevi **implementações ingênuas de referência** dentro do teste —
laços explícitos, escritas direto das definições dos artigos. Isso valida duas coisas de uma vez:
que a fórmula implementada é a fórmula publicada, e que as otimizações (`cdist`, stride tricks)
não introduziram erro. Concordância de `rel=1e-6` em SampEn (m ∈ {1,2,3} × 4 tipos de sinal,
incluindo os pesos reais do projeto) e `rel=1e-12` em LMC (3 valores de bins × 4 distribuições).
O teste opcional contra `antropy` fica pulado e passa a rodar sozinho se um dia for instalado.

Cobertura: o "morro" da LMC (cristal ≈ 0 < intermediário > gás ≈ 0, com folga de 20×), H máxima
= log₂(N), D = 0 exato na uniforme, SampEn de seno (0,24) vs ruído branco (2,17), monotonicidade
em r, invariância da LMC a permutação **vs** dependência da SampEn à ordem, as duas ordens de
flatten sobre pesos reais, `find_dense_layers` isolando `fc.weight` no checkpoint real, e
sanidade de SampEn2D/MSE.

**Bug corrigido no caminho:** `H` e `SampEn` retornavam **`-0.0`** nos casos degenerados
(distribuição delta e rampa linear), o que apareceria como "complexidade negativa" em CSVs e
gráficos. Corrigido no módulo, com teste de regressão.

**Sensibilidade (`scripts/sensibilidade_parametros.py`, 3 sementes × 20 épocas, só `fc.weight`):**

| | LMC vs nº de bins | SampEn vs tolerância r |
|---|---|---|
| Spearman entre parametrizações | **≥ 0,991** (todas as combinações) | **0,43** entre r=0,10 e r=0,25 |
| Época do pico/mínimo | **idêntica** com bins 20/50/100/200, nas 3 sementes | **instável**: {16, 12, 12, 18} numa semente; {11, 18, 12, 12} noutra |
| Amplitude do sinal ao longo do treino | ~130 % da média | apenas **3–8 %** da média |

➜ **Decisão: `n_bins = 100` fixo.** O valor absoluto da LMC depende dos bins, mas a *forma* da
trajetória e a época do pico não — a escolha é uma **convenção**, não um grau de liberdade que
permitiria escolher o resultado. Isso é exatamente o que o critério de aceite pedia.

➜ **A regra de Freedman-Diaconis foi descartada**, e por um motivo que só apareceu ao rodar: ela
escolhe um número de bins **diferente a cada época** (19 a 40 ao longo de um mesmo run), o que
mistura variação de parâmetro com variação de sinal. É a única curva que desvia das demais.

➜ **Decisão: `r = 0,20·σ` fixo** — mas com uma ressalva honesta, abaixo.

#### ⚠️ Achado que muda o plano: a SampEn 1D está fraca nestes dados

A LMC se comporta como um indicador: varia ~130 % ao longo do treino, com pico bem definido e
robusto ao parâmetro. **A SampEn 1D não.** Ela varia só 3–8 % da média, a trajetória parece
ruído (ver painel B de `sensibilidade_r_*.png`) e a época do mínimo muda conforme o r escolhido.
A amplitude dentro de um run é ~4× o espalhamento entre sementes, então *existe* sinal — mas é
fraco demais para sustentar sozinho a afirmação "segundo indicador de overfitting".

Três causas possíveis, a separar na F6:
1. **O treino é fácil demais** — MedNIST converge rápido e o overfitting em 20 épocas é fraco.
   O run de overfitting induzido (5 % dos dados) da F6 testa isso diretamente.
2. **20 épocas é pouco** — daí a mudança para 40 épocas já prevista.
3. **A SampEn 1D pode não ser a medida certa para uma matriz 6×512.** Achatar uma matriz tão
   assimétrica em uma sequência pode destruir a estrutura. A **SampEn2D** (já implementada e
   testada) opera sobre a matriz e **não exige escolher ordem de achatamento** — é candidata
   natural a segundo indicador.

➜ **Ação incorporada à F6:** reportar LMC, SampEn 1D (nas duas ordens) **e SampEn2D**, e deixar
o dado decidir qual é o segundo indicador. Se a SampEn 1D continuar fraca com 40 épocas e
overfitting induzido, isso é um **resultado a reportar**, não um fracasso a esconder.

#### Nota para as figuras da F6
As figuras atuais de LMC × SampEn usam **dois eixos y** (`twinx()` em `monai_lmc_sampen_plot.py`
e `summary_complexity_plot.py`). Isso é o anti-padrão nº 1 de visualização de dados: com duas
escalas independentes, dá para fazer as curvas se cruzarem onde se quiser, e a "coincidência"
vira artefato da escolha de escala. Na F6 essas figuras viram **painéis lado a lado** (small
multiples), que é como o estudo de sensibilidade já foi feito.

---

### ✅ F3 — Split treino/validação/teste (D5, resolve P2, P6) — **CONCLUÍDA em 2026-09-11**
**Fazer:**
1. `prepare_mednist_datasets`: passar a carregar também `section="test"` do MedNIST e devolver `test_loader`.
2. `create_loaders` passa a devolver 3 loaders.
3. `Trainer` ganha `test_loader` e um método `evaluate_test()` — chamado **uma única vez**, no fim, com os pesos da época selecionada (`bef_weights.pth`).
4. `src/data/split.py`: receber `seed` e usar `random.Random(seed).shuffle(...)` (P6). Se a função não for mais usada, remover.
5. Registrar os tamanhos dos 3 conjuntos no log e em `config.json`.
6. Documentar a regra "teste olhado uma vez" no README e no texto do projeto.

**Critério de aceite:** um run curto (`--epochs 3`) imprime `train=N1 val=N2 test=N3`, e `test_metrics.json` aparece no diretório do run com uma única avaliação.

#### Resultado da F3

- **Partição oficial do MedNIST**: `training` / `validation` / `test` = 47.164 / 5.895 / 5.895
  (80 / 10 / 10 %). Usar a partição oficial, em vez de recortar à mão, torna o experimento
  reprodutível por terceiros.
- **Separação entre semente da partição e semente do treino.** Introduzido `data_seed`
  (default 0, **fixa entre runs**), distinto de `seed` (pesos e ordem dos lotes, varia entre
  repetições). Se as duas fossem a mesma, cada semente veria uma partição diferente, os runs
  deixariam de ser comparáveis e uma amostra que foi teste num run seria treino no outro.
- **`evaluate_test()`**: avalia o teste **uma única vez**, depois do treino, recarregando os
  pesos da época de menor `val_loss` — nunca os da última época, que já pode estar em
  overfitting. Grava `test_metrics.json` com `epoca_avaliada` registrada. Validação e teste
  passam pelo mesmo `_evaluate()`, para que as métricas sejam calculadas de forma idêntica.
- **P2 resolvido:** `val_split` era declarado no config, passado pelo `run_sweep.py` e
  **não fazia absolutamente nada**. Removido. No lugar entraram `data_seed`, `val_frac`,
  `test_frac` e `train_fraction` — todos com efeito real.
- **P6 resolvido:** `train_val_split` usava `random.shuffle` sem semente e **nunca era
  chamada**. Removida. `split.py` agora tem `stratified_subset_indices()`, determinística e
  estratificada por classe (sortear sem estratificar poderia zerar a classe rara e trocar o
  fenômeno estudado — overfitting — por outro: desbalanceamento).
- **Testes:** `tests/test_split.py` com 15 casos. Suíte total: **79 passando, 1 pulado.**
- Partição gravada em `config.json` e nos logs, com a disciplina dos três conjuntos escrita
  por extenso a cada run.

#### 🚨 Achado da F3 que invalida o run de overfitting planejado para a F6

Ao rodar o smoke test com `train_fraction=0.02`, o resultado parecia overfitting perfeito:
`train_loss=0,13` e `val_acc=0,17` — exatamente 1/6, acaso puro em 6 classes. Seria o run de
controle dos sonhos. **Não era.**

Diagnóstico (avaliando o próprio conjunto de treino nos dois modos do BatchNorm):

| conjunto | modo do BatchNorm | loss | acurácia |
|---|---|---|---|
| treino | eval (estatísticas móveis) | 16,51 | **0,164** |
| treino | train (estatísticas do lote) | 0,139 | 0,971 |
| validação | eval (estatísticas móveis) | 5,68 | 0,139 |
| validação | train (estatísticas do lote) | 0,214 | **0,945** |

O modelo **generaliza bem** (94,5 % na validação) e nem sequer classifica o *próprio treino*
em modo `eval()`. A causa é o BatchNorm: suas estatísticas móveis partem de média 0 /
variância 1 e, com 944 amostras em 2 épocas, receberam só ~16 atualizações. Ficaram ruins, e
é isso — não overfitting — que derruba a inferência.

**Por que isso importa muito:** se a F6 induzisse overfitting encolhendo o treino, mediríamos
LMC e SampEn de um artefato de normalização e chamaríamos de overfitting. A conclusão central
do trabalho ficaria construída sobre o fenômeno errado, e o resultado pareceria *ótimo* —
justamente o tipo de erro que não se anuncia.

**Mudanças adotadas:**
1. **Trava no código**: `main_train.py` agora loga o número de passos de treino e emite
   `WARNING` quando o total fica abaixo de 200 atualizações do BatchNorm, dizendo o que
   conferir e o que fazer.
2. **A F6 muda a forma de induzir overfitting** — de `train_fraction` pequeno para
   **ruído de rótulo** (ver F6 abaixo).
3. Lembrete permanente: antes de chamar qualquer coisa de overfitting, **checar a acurácia do
   próprio treino em `eval()`**. Overfitting real = treino alto, validação baixa. Artefato de
   BatchNorm = os dois baixos.

---

### ✅ F4 — Monitor de complexidade ao vivo (D7, D1, D2) — **CONCLUÍDA em 2026-09-21**
**Fazer:**
1. `src/training/complexity_monitor.py`:
   - `class ComplexityMonitor` com `on_epoch_end(model, epoch, history) -> dict`
   - localiza as camadas densas, achata na ordem configurada, chama `src/complexity.py`
   - mantém o histórico e devolve `{lmc, sampen, d_lmc, d_sampen, status}`
   - `class TrainingStatus(Enum)` com os estados da §4.2
2. Integrar no `Trainer.fit()` — atrás de uma flag `--live_complexity` (default ligado), para poder desligar em benchmark de tempo.
3. Escrever em `history.csv`, TensorBoard, `complexity_live.csv`, e na linha de log da época.
4. Medir o overhead: rodar 5 épocas com e sem o monitor e reportar o % de tempo extra.
5. Novos flags em `config.py` / CLI: `--flatten_order`, `--lmc_bins`, `--sampen_r_factor`, `--live_complexity/--no_live_complexity`.

**Critério de aceite:** `--epochs 5` produz log com `LMC=... SampEn=... status=...` em toda época; overhead medido e < 5 % do tempo de época.

#### Resultado da F4

**Critério de aceite atingido com folga.** Linha de log por época:
`Epoch 1 | ... | LMC=0.01080 | SampEn=2.1828 | SampEn2D=6.5840 | status=INICIALIZANDO`

| | valor |
|---|---|
| Overhead do monitor | **113 ms/época** |
| Fração de uma época de treino (dados completos, CPU) | **0,02 %** |
| Limite do critério de aceite | 5 % |

Entregues: `src/training/complexity_monitor.py` (`ComplexityMonitor`, `TrainingStatus`,
`LimiaresStatus`) e `src/training/overfit.py` (`detect_overfit_onset`, `epocas_usaveis`) —
este último antecipado da F5 porque o monitor precisa dele, e duplicar a regra recriaria o
problema P1. Saídas: `history.csv` (colunas `lmc_dense`, `sampen_dense`, `status`),
`complexity_live.csv`, `overfit_report.json`, TensorBoard (`Complexity/*`) e log por época.
Flags: `--flatten_order`, `--lmc_bins`, `--sampen_r_factor`, `--sampen2d_m`, `--patience`,
`--overfit_delta`, `--no_live_complexity`, `--no_live_sampen2d`. **110 testes passando.**

#### ⚠️ Achado: a SampEn2D com m=2 é inutilizável em pesos de camada densa

Com o padrão `m=2`, a SampEn2D ficou **indefinida em 10 de 10 épocas** de um run real.

A primeira explicação que levantei — "a matriz só tem 6 linhas" — estava **errada**, e o
teste a derrubou. A causa real: a janela `(m+1)×(m+1)` tem 9 elementos e **todos** precisam
casar dentro de `r`; duas vizinhanças 3×3 só casam se a matriz tiver **estrutura espacial**.
Pesos de camada densa são aproximadamente independentes — cada peso liga um par
(entrada, neurônio) e não guarda relação com o vizinho na matriz — então não há nada que se
repita. Verificado que **aumentar a matriz não resolve**: `(32,48)`, `(64,64)`, `(64,128)`,
`(64,256)` e `(6,512)` falham todas igualmente. Numa matriz *com* estrutura (padrão
ladrilhado, ruído suavizado) a medida volta a ser finita.

➜ **Decisão: `sampen2d_m = 1`** (janelas 1×1 contra 2×2), onde as contagens ficam robustas
(B≈7×10⁵, A≈10³). O monitor emite `WARNING` explicando a causa se a medida ficar indefinida.

➜ **Consequência para a interpretação (registrar no texto):** a SampEn2D desta camada mede
o quanto vizinhanças 2×2 de pesos se repetem — **não** "textura" no sentido de imagem, que
uma camada densa não possui. Não vender a medida como algo que ela não é.

**Força relativa dos três candidatos** (run real, 20 épocas, amplitude/média):

| indicador | amplitude ao longo do treino |
|---|---|
| **LMC** | **96,2 %** |
| SampEn2D (m=1) | 4,4 % |
| SampEn 1D | 1,8 % |

A SampEn2D é ~2,4× mais forte que a SampEn 1D — justifica tê-la incluído — mas as duas
seguem muito atrás da LMC. A F6 decide com dados de overfitting real.

#### Sobre a máquina de estados: funciona, mas ainda não diz nada neste run

A linha do tempo de status foi gerada reproduzindo os 20 checkpoints reais pelo monitor.
O status oscila (`OVERFITTING` nas épocas 14–16, `CONVERGINDO` na 17) porque **neste run o
`val_loss` não tem para onde ir**: ele vive entre 1×10⁻⁴ e 5×10⁻³ e salta por um fator de 10
entre épocas consecutivas. O detector, corretamente, conclui `confirmado=False` —
**este run nunca entra em overfitting**, o mínimo global é a última época.

Ou seja: a máquina está sendo obrigada a descrever um fenômeno que não existe nestes dados.
Isso **não valida nem invalida** os limiares; só confirma que a calibração depende do run de
ruído de rótulo da F6. Item de calibração anotado: suavizar o `val_loss` (média móvel) antes
de alimentar o status, para reduzir a oscilação — mas ajustar isso agora, sem dados de
overfitting, seria chutar.

**Distinção documentada no código:** o status online é um juízo *provisório*, feito sem
conhecer o futuro, e pode ser retirado (é o que permite sobreviver ao grokking). Quem manda
no corte dos dados (D6) é o `overfit_report.json`, calculado sobre a trajetória completa ao
fim do treino. Divergir é esperado, e o `Trainer` agora registra quando isso acontece.

---

### ✅ F5 — Detecção e corte de overfitting (D6) — **CONCLUÍDA em 2026-09-21**
**Fazer:**
1. `src/training/overfit.py` — implementa a §3.2: `detect_overfit_onset(history, patience, delta)`.
2. `Trainer` grava `overfit_report.json` ao fim, com `best_epoch` e `overfit_onset_epoch`.
3. Flags `--patience`, `--overfit_delta`, `--early_stop/--no_early_stop`.
4. **Todos** os scripts de análise passam a ler `overfit_report.json` e a filtrar as épocas pós-overfitting, sinalizando no gráfico (região sombreada + legenda "descartado").
5. Atualizar `best_epochs.py` para reportar também o `overfit_onset_epoch`.

**Critério de aceite:** os gráficos de LMC/SampEn ao longo das épocas mostram a região descartada sombreada; nenhum número de conclusão vem dessa região.

#### Resultado da F5

`src/training/overfit.py` e o `overfit_report.json` já tinham saído na F4. O que esta fase
acrescentou:

- **`src/analysis/overfit_cut.py`** — fonte única do corte para os consumidores:
  `carregar_relatorio`, `marcar_pos_overfit`, `apenas_usaveis`, `sombrear_regiao_descartada`,
  `resumo_texto`.
- **Fallback essencial:** se o run não tem `overfit_report.json` (todos os runs anteriores à
  F4), o relatório é **recalculado do `history.csv`**. Exigir re-treino só para aplicar o
  corte a runs antigos seria absurdo.
- **Sombreado, não corte de eixo.** As épocas descartadas continuam no gráfico, em vermelho
  claro, com legenda "descartado (pós-overfitting)" e linha tracejada na melhor época. O
  leitor vê o que aconteceu depois e confere que a linha está no lugar certo. Elas não entram
  em nenhuma média, correlação ou tabela.
- **Nunca sombreia um run sem overfitting confirmado** — seria mentir sobre o dado.
- `--early_stop` implementado, **desligado por padrão**: parar cedo truncaria a trajetória e
  impediria observar grokking. Liga só para economizar máquina.
- `best_epochs.py` reescrito: tabela com `melhor_epoca`, `inicio_overfit`, `epoca_deteccao`,
  `usaveis`, `descartadas` e a origem do relatório.
- **130 testes passando.**

**Corte aplicado aos runs existentes:**

| run | melhor época | overfitting | épocas usáveis | descartadas |
|---|---|---|---|---|
| `run_mednist_resnet18_64` | 20 | não confirmado | 20 | 0 |
| `run_mednist_resnet18_64_seed1` | 4 | **confirmado** | 4 | **16** |
| `run_mednist_resnet18_64_seed7` | 14 | **confirmado** | 14 | 6 |

#### 🚨 Bug que eu mesmo introduzi, e a correção

Remover a subamostragem dos scripts legados (pedido do autor, para não descartar pesos)
os deixou **inexecutáveis**: eles percorriam *todas* as camadas, e a SampEn exata é O(n²) —
uma conv de 2.359.296 pesos exigiria uma matriz de distâncias de **22 TB**. A camada densa,
com 3.072 pesos, precisa de 38 MB.

**Correção**, alinhada com D1: os quatro scripts legados passaram a analisar **somente a
camada densa**, via `find_dense_layers`. Assim nenhum peso é descartado *e* o cálculo é
viável. Antecipou parte da F6, mas a alternativa era deixar scripts quebrados no repositório.
Efeito colateral bom: ficaram ~40× mais rápidos.

#### ⚠️ Consequência incômoda: D6 aplicada a estes runs não deixa quase nada

Com o corte, a semente 1 mantém **4 de 20 épocas**. Pior: o **pico da LMC dessa semente
está na época 10 — dentro da região descartada**. A "época do pico" que a F2 reportou para
ela não poderia entrar em nenhuma conclusão sob a regra D6.

A causa é a mesma de sempre: o MedNIST satura na primeira época. A `val_accuracy` na melhor
época da semente 1 é **1,00000** e o `val_loss` é 0,0004 — declarar "overfitting" a partir da
época 4 é ler ruído numa métrica já saturada, não um fenômeno real. O detector está certo
segundo sua própria regra; o problema é que a regra está sendo aplicada a um treino que não
tem o que detectar.

➜ **Isto reforça, pela terceira vez, que o run de ruído de rótulo da F6 não é um extra:
é o único run em que a metodologia inteira (corte D6, status, comparação entre indicadores)
pode ser exercitada de verdade.** Com os dados atuais, aplicar D6 estritamente descarta
justamente a parte da trajetória que a F2 usou para mostrar a robustez da LMC.

➜ **Item para a F6:** considerar um `overfit_delta` maior que zero, calibrado à escala do
`val_loss` do problema, para que flutuações de 4ª casa decimal numa métrica saturada não
disparem o corte.

---

### ✅ F6 — Re-execução dos experimentos e ablação da ordem (D1, D2) — **CONCLUÍDA em 2026-09-21**
**Fazer:**
1. Atualizar `run_sweep.py`: grid mínimo = **3 sementes × 2 ordens de flatten** (a ordem não exige re-treino — é só recalcular sobre os mesmos checkpoints; então na prática são **3 treinos** e 2 análises cada).
   - Grid: ResNet-18 @64, 3 sementes, **40 épocas** (mais que as 20 atuais, para ver o overfitting acontecer de verdade em vez de só suspeitar).
   - **Run de overfitting induzido por RUÍDO DE RÓTULO** (não por `train_fraction` — ver o
     achado da F3). Embaralhar o rótulo de **30 %** das amostras de treino (*confirmado pelo autor em 2026-09-21*)
     mantém as 47.164 amostras, mantém as estatísticas do BatchNorm saudáveis e força
     memorização genuína. É a técnica padrão da literatura de generalização
     (Zhang et al., 2017, *Understanding deep learning requires rethinking generalization*).
   - **Verificação obrigatória** desse run: acurácia do próprio treino em `eval()` tem de
     ficar **alta** enquanto a de validação cai. Se as duas caírem, é artefato, não overfitting.
2. Reprocessar todos os runs com os scripts novos (só camada densa), computando **três**
   candidatos a indicador por época:
   - **LMC** (invariante à ordem, `n_bins=100` fixo pela F2);
   - **SampEn 1D** nas duas ordens (`n_major` e `x_major`);
   - **SampEn2D** sobre a matriz 6×512 — *aprovado pelo autor em 2026-09-11*. É a candidata
     mais forte a segundo indicador porque **não exige escolher ordem de achatamento**,
     eliminando um grau de liberdade metodológico, e porque a SampEn 1D se mostrou fraca na F2.
3. Gerar as figuras:
   - `lmc_sampen_densa_por_epoca.png` (com região pós-overfit sombreada)
   - `ablacao_ordem_flatten.png` (n-major vs x-major, SampEn — a LMC deve coincidir; isso serve de sanidade)
   - `status_timeline.png` (faixa colorida do status por época, sob a curva de val_loss)
4. Calibrar os limiares do status (§4.2) em 2 sementes e **verificar** na terceira.
5. Tabela final: para cada run, `best_epoch`, `overfit_onset`, época do pico de LMC, época da inflexão da SampEn, e o **atraso/antecipação** entre o sinal de complexidade e o sinal de val_loss. ← **este é o resultado central do trabalho.**

**Critério de aceite:** tabela com as 3 sementes + run de overfitting induzido; afirmação quantificada do tipo "a LMC da camada densa inflete X ± Y épocas antes da subida do val_loss" (ou a constatação honesta de que não inflete).

#### Resultado da F6

**Overfitting inequívoco obtido.** Run `f6_ruido30`: 30 % de ruído de rótulo (24,98 %
efetivamente errados), sem aumento de dados e sem weight decay — os dois combatem
memorização. A acurácia de treino em `eval()` subiu de **74,8 % para 97,7 %** enquanto a de
validação caiu de **99,9 % para 84,6 %**. Treino alto **e** validação baixa: a checagem
anti-BatchNorm da F3 confirma memorização real, não artefato.

**Cinco** sementes de controle (40 épocas, treino normal) não apresentam degradação sustentada —
o critério de acurácia devolve `confirmado=False` para as cinco, como deveria.

#### 🚨 A LMC não discrimina overfitting

Desenho final: **5 runs com ruído × 5 controles** (sementes 42, 1, 7, 13, 23 dos dois lados).

| indicador | amplitude com ruído | amplitude nos controles | razão | p (Welch) | dispara em runs SEM overfitting |
|---|---|---|---|---|---|
| **LMC** | 120,3 % | 118,9 % | **1,0×** | 0,85 | 4 de 5 |
| SampEn 1D | 19,4 % | 5,5 % | 3,5× | < 0,0001 | **5 de 5** |
| **SampEn2D** | **30,0 %** | 5,7 % | **5,3×** | **< 0,0001** | 3 de 5 |

**A LMC varia ~110–125 % em todos os runs, com ou sem overfitting.** Ela mede a organização
progressiva dos pesos ao longo do treino — não o overfitting. É uma medida robusta e bem
comportada (a F2 provou isso), mas **não serve como indicador de overfitting**.

Isto corrige um erro de leitura meu que vinha desde a F2: eu tratei **amplitude** como se
fosse **poder de discriminação**. São coisas diferentes. Um indicador que varia muito nos
dois regimes não avisa nada. O critério correto é o CONTRASTE entre um run com overfitting e
controles sem — e por isso os runs de controle não são um detalhe do experimento, são metade dele.

#### Antecedência nos runs com overfitting — replicado em 5 sementes (2026-09-22)

Âncora: época em que a acurácia de validação cai 1 ponto percentual de forma sustentada.
Positivo = o indicador disparou antes.

| run | âncora | LMC | SampEn 1D | SampEn2D |
|---|---|---|---|---|
| `f6_ruido30` (semente 42) | 15 | +2 | +3 | **+7** |
| `f6_ruido30_seed1` | 15 | +5 | +4 | **+8** |
| `f6_ruido30_seed7` | 14 | +1 | +5 | **+7** |
| `f6_ruido30_seed13` | 14 | +1 | +4 | **+7** |
| `f6_ruido30_seed23` | 15 | +2 | +2 | **+8** |
| **média ± desvio** | | +2,2 ± 1,6 | +3,6 ± 1,1 | **+7,4 ± 0,5** |
| **IC 95 %** | | [0,2; 4,2] | [2,2; 5,0] | **[6,7; 8,1]** |

**Afirmação central do trabalho, agora quantificada com n = 5:** a SampEn2D da camada densa
inflete **7,4 ± 0,5 épocas antes** da degradação da acurácia de validação (IC 95 %: 6,7 a 8,1
épocas). Os cinco runs deram apenas dois valores, 7 ou 8 — é a mais estável das três medidas.
A LMC é errática justamente no run onde deveria avisar (IC de 0,2 a 4,2 épocas: chega a
encostar no zero, ou seja, não se pode descartar que não antecipe nada).

Poder de discriminação: ver a tabela 5 × 5 acima — LMC 1,0× (p = 0,85, **não discrimina**),
SampEn 1D 3,5×, SampEn2D 5,3× (ambas p < 0,0001).

#### A separação limpa: QUANDO o indicador dispara

A taxa de falso positivo sozinha engana. Com 5 controles a SampEn2D dispara em 3 deles — mas
**sempre tarde**, e os runs com ruído **sempre cedo**:

| indicador | época do sinal — ruído (n=5) | época do sinal — controles | separa? |
|---|---|---|---|
| LMC | 10, 13, 13, 13, 13 | 10, 12, 18, 21 | **não** (sobrepõe) |
| SampEn 1D | 9, 10, 11, 12, 13 | 17, 17, 19, 25, 26 | sim, margem de 4 épocas |
| **SampEn2D** | **7, 7, 7, 7, 8** | **25, 27, 31** | **sim, margem de 17 épocas** |

**A SampEn2D não tem sobreposição nenhuma**: qualquer limiar entre as épocas 9 e 24 classifica
corretamente os 10 runs. O "falso positivo" dos controles é um sinal tardio, na cauda do treino,
que uma regra de decisão com janela temporal descarta. A SampEn 1D também separa, com margem bem
menor. A LMC sobrepõe e não separa nem por amplitude nem por época.

Ressalva: os controles têm 40 épocas e os runs de ruído, 60 — comparar a *época* só é legítimo
porque todos compartilham dataset, arquitetura e schedule, e porque os sinais dos controles
(25–31) caem dentro das 40 épocas disponíveis, não são truncamento.

#### Ablação da ordem do achatamento (D2) — feita em 2026-09-22

`scripts/ablacao_flatten.py`. O monitor da F4 já gravava **as duas ordens** a cada época
(`sampen` e `sampen_x_major`), então a ablação não custou GPU nenhuma — só faltava analisar.
Mesmo detector da F6 aplicado às duas, importado de `analise_f6` e não copiado.

**Sanidade OK:** a LMC dá o mesmo número nas duas ordens até o último bit (diferença 0,0e+00
em 3 épocas conferidas direto dos checkpoints). Era o esperado — a LMC só olha o histograma —
e confirma que `flatten_dense` não perde nem duplica peso em nenhuma das ordens.

| ordem | amplitude ruído | controles | razão | antecedência | falsos + | margem |
|---|---|---|---|---|---|---|
| `n_major` (padrão) | 19,4 % | 5,5 % | 3,5× | +3,6 ± 1,1 | 5/5 | +4 épocas |
| **`x_major`** | **56,1 %** | 7,3 % | **7,7×** | **+7,4 ± 0,5** | 5/5 | **+2 épocas** |

**A ordem importa, e muito — e a escolhida não é a melhor.** Trocar a ordem de leitura dos
*mesmos* 3.072 pesos dobra a antecedência (3,6 → 7,4 épocas) e mais que dobra o poder de
discriminação (3,5× → 7,7×, a maior razão medida no projeto, acima dos 5,3× da SampEn2D).

**Mas a margem de separação piora:** o último sinal dos runs de ruído (época 8) e o primeiro
dos controles (época 10) ficam a 2 épocas de distância, contra 4 do `n_major` e **17 da
SampEn2D**. Ou seja: `x_major` avisa mais cedo e reage mais forte, porém um limiar de época
construído sobre ele tem pouquíssima folga — e é a folga que sobrevive a um dataset novo.
Amplitude e antecedência não são o critério final; a margem é.

**`x_major` é quase uma redundância da SampEn2D:** correlação de +0,88 a +0,97 entre as duas
séries nos 5 runs de ruído (contra +0,81 a +0,94 do `n_major`). Faz sentido — `x_major`
percorre a matriz na direção curta (6 neurônios), aproximando a vizinhança que a SampEn2D lê
de forma bidimensional. **A antecedência das duas é idêntica: +7,4 ± 0,5, com sinal nas épocas
7-7-7-7-8.** Isto *reforça* o resultado central em vez de competir com ele: duas medidas com
formulações diferentes, ao olharem a mesma vizinhança, chegam à mesma época.

#### Linha do tempo do status (§4.2) — feita em 2026-09-22

`scripts/status_timeline.py` → `status_timeline.png`. A faixa colorida por época, sob a
curva de `val_loss`, mostra o que o monitor **dizia naquele momento** — um juízo online,
sem conhecer o futuro.

**O status, como está calibrado, não discrimina overfitting.**

| grupo | OVERFITTING em… | 1ª vez (média) |
|---|---|---|
| 5 runs com ruído | 88 % das épocas | época 7,6 |
| 5 controles saudáveis | **51 % das épocas** | época 8,8 |
| razão | **1,7×** | — |

1,7× contra os 5,3× da SampEn2D. O status dispara em **10 de 10 runs**, e a época em que
dispara também não separa (7,6 × 8,8). Não é bug: é a definição em uso — `OVERFITTING` ali
significa "o `val_loss` não melhora há `patience` épocas", e num problema que satura na 1ª
época isso acontece cedo mesmo com a rede saudável. O `ALERTA_OVERFIT` aparece em 3 dos 5
controles e em **nenhum** dos runs com ruído — exatamente o oposto do pretendido.

Consequência para o texto: o status é **leitura auxiliar durante o treino**, não resultado,
e o achado da F6 não se apoia nele — apoia-se na SampEn2D e no `overfit_report.json`,
calculado sobre a trajetória completa. O aviso já escrito em `complexity_monitor.py`
("os limiares são PROVISÓRIOS") fica assim confirmado pelos dados.

#### ✅ Status recalibrado para a acurácia — 2026-09-22

`scripts/calibrar_status.py`. **Resolve o item da F6 que pedia calibrar em algumas sementes
e verificar em outras.** Duas mudanças na máquina de estados (§4.2):

1. **`OVERFITTING` deixou de olhar `val_loss` e passa a olhar a acurácia** — "ficou
   `queda_acc` abaixo do pico corrente por `patience` épocas". Detector online novo em
   `overfit.py` (`epocas_abaixo_do_pico`), irmão do `detect_degradacao_acuracia` offline.
2. **`ALERTA_OVERFIT` deixou de olhar a inversão da LMC e passa a olhar o afastamento da
   SampEn2D** do seu platô inicial — a LMC ocupava esse lugar mas a própria F6 mostrou que
   ela não discrimina (1,0× contra 5,3×). O detector `epoca_de_afastamento` saiu de
   `scripts/analise_f6.py` para `src/training/overfit.py`: o status online e a análise
   offline passam a usar **a mesma função**, não duas cópias.

**Protocolo seguido à risca.** A busca em grade (3×3×3×3 = 81 combinações) enxergou apenas
3 runs de ruído + 3 controles. As sementes 13 e 23, dos dois grupos, ficaram de fora e só
foram usadas para verificar. Critério de escolha, nesta ordem: margem de separação maior,
depois aviso mais cedo **em relação à degradação real** — não em relação ao próprio rótulo
`OVERFITTING`, que se move quando os limiares mudam e premiaria limiar frouxo.

Limiares escolhidos: `queda_acc=0,005`, `patience=3`, `afast_k=5,0`, `afast_p=3`.

| conjunto | detectou | falsos positivos | antecedência do alerta |
|---|---|---|---|
| calibração (3+3) | 3/3 | 0/3 | +5,3 épocas |
| **verificação (2+2)** | **2/2** | **0/2** | **+5,0 épocas** |

**O desempenho na verificação é igual ao da calibração** — o limiar não decorou. Contraste
antes × depois, nos mesmos 10 runs:

| | com `val_loss` (antes) | com acurácia (depois) |
|---|---|---|
| runs com ruído acusados | 5/5 | 5/5 |
| **controles acusados** | **5/5** | **0/5** |
| OVERFITTING nos controles | 51 % das épocas | **0 %** |

O `ALERTA_OVERFIT` dispara nos 5 runs com ruído (épocas 9–10) e em **1** dos 5 controles,
na época 31 — 21 épocas depois do último alerta legítimo. Antes disso ele fazia o oposto:
disparava em 3 controles e em nenhum run com ruído.

Figuras: `status_timeline.png` (como estava, com `--fonte csv`) e
`status_timeline_recalibrado.png` (com o classificador atual). O script ganhou `--fonte`
justamente para que o "antes" continue reproduzível.

**Ressalvas.** A antecedência do status (+5,0) é menor que a da SampEn2D crua (+7,4) porque
`afast_k=5,0` é mais exigente que o `k=4,0` da análise — a calibração trocou 2 épocas de
aviso por quase nenhum falso positivo. Com 6 runs de calibração e 4 de verificação, isto é
uma verificação honesta, **não** uma estimativa estável: 54 das 81 combinações passavam a
restrição dura, o que indica que o problema é fácil neste conjunto, não que o limiar é ótimo.

#### Teste prospectivo — semente 99, dados que não existiam na calibração (2026-09-22)

A verificação acima usou as sementes 13 e 23: elas ficaram fora da busca, mas os dados
delas **já existiam** quando a grade foi percorrida. A semente 99 é diferente — os dois runs
foram treinados **depois** de os limiares serem fixados, e o status saiu ao vivo no log, pelo
caminho de produção, não por replay.

| critério | resultado |
|---|---|
| overfitting detectado no run de ruído | ✅ época 14 |
| falso positivo no controle | ✅ nenhum, nas 40 épocas |
| memorização confirmada (anti-BatchNorm) | ✅ treino `eval()` 74,7 % → 97,5 %, validação 99,9 % → 84,3 % |
| separação do alerta | ✅ época 12 (ruído) × 26 (controle), margem 14 épocas |
| **antecedência do alerta** | ⚠️ **+3 épocas** |

**A restrição dura sobreviveu intacta em dados novos; a antecedência não.**

| run | conjunto | alerta | âncora | antecedência |
|---|---|---|---|---|
| `f6_ruido30` | calibração | 10 | 15 | +5 |
| `f6_ruido30_seed1` | calibração | 9 | 15 | +6 |
| `f6_ruido30_seed7` | calibração | 9 | 14 | +5 |
| `f6_ruido30_seed13` | verificação | 10 | 14 | +4 |
| `f6_ruido30_seed23` | verificação | 9 | 15 | +6 |
| **`f6_ruido30_seed99`** | **prospectivo** | **12** | 15 | **+3** |

+3 ficou abaixo de todos os cinco runs anteriores (que variavam de +4 a +6). Com n = 1 não
dava para saber se era a régua real ou azar de uma semente — daí as duas prospectivas
seguintes.

#### Fecho: 3 sementes prospectivas (99, 101, 202) — 2026-09-23

| run | conjunto | alerta | overfit | âncora | antecedência |
|---|---|---|---|---|---|
| `f6_ruido30` | calibração | 10 | 17 | 15 | +5 |
| `f6_ruido30_seed1` | calibração | 9 | 14 | 15 | +6 |
| `f6_ruido30_seed7` | calibração | 9 | 14 | 14 | +5 |
| `f6_ruido30_seed13` | verificação | 10 | 14 | 14 | +4 |
| `f6_ruido30_seed23` | verificação | 9 | 15 | 15 | +6 |
| `f6_ruido30_seed99` | prospectivo | 12 | 14 | 15 | **+3** |
| `f6_ruido30_seed101` | prospectivo | 9 | 16 | 16 | **+7** |
| `f6_ruido30_seed202` | prospectivo | 9 | 15 | 15 | **+6** |

**O +3 da semente 99 foi azar, não viés.** A média prospectiva (5,3) é praticamente idêntica
à da calibração (5,2) — Welch p = 0,92. O que a calibração subestimou foi a **dispersão**:
desvio 0,8 nas 5 primeiras contra **2,1** nas 3 prospectivas.

Reunindo os 8 runs de ruído: **+5,3 ± 1,3 épocas (IC 95 %: 4,2 a 6,3), faixa observada de 3
a 7**. É esta a afirmação a levar para o texto — não o "+5,0 ± 0,5" que a calibração sugeria.
A lição vale além deste número: *o desvio medido onde se calibrou mede o ajuste, não o
fenômeno.*

**O critério que interessa na prática não degradou em nada:**

| | runs de ruído | controles |
|---|---|---|
| calibração + verificação | 5/5 detectados | 0/5 falsos |
| **prospectivo** | **3/3 detectados** | **0/3 falsos** |
| **total** | **8/8** | **0/8** |

Os 3 controles prospectivos alertaram tarde (épocas 26, 28, 29) e nenhum chegou a
`OVERFITTING`. Margem de separação do alerta em todos os 16 runs: **14 épocas** (último
alerta num run de ruído: 12; primeiro num controle: 26).

**Nota de reprodutibilidade.** A primeira tentativa destes 4 runs abortou com
`CUDA error: out of memory` — a GPU 0 da máquina é compartilhada e estava com 31,9 GB de
32,8 GB tomados por processos de outros usuários. Refeitos com `CUDA_VISIBLE_DEVICES=1`.
O treino usa 190 MB de GPU; o problema é de convivência na máquina, não do projeto.

#### 🎯 Níveis de ruído: 15 %, 30 % e 50 % — 2026-09-23

`scripts/niveis_de_ruido.py` → `niveis_de_ruido.png`. Fecha a última lacuna da F6: até aqui
**todo** o resultado vinha de um único nível, o que tornava a conclusão uma afirmação sobre
"30 % de rótulos sorteados", não sobre overfitting. As sementes 42, 1 e 7 foram **reusadas**
nos três níveis, então a única coisa que muda entre eles é o ruído.

| ruído sorteado | errados de fato | época do **alerta** | época da degradação | antecedência | detectou |
|---|---|---|---|---|---|
| 15 % | 12,5 % | 10,0 ± 1,0 | 17,7 ± 0,6 | **+7,7 ± 1,5** | 3/3 |
| 30 % | 25,0 % | 9,3 ± 0,6 | 14,7 ± 0,6 | **+5,3 ± 0,6** | 3/3 |
| 50 % | 41,7 % | 10,0 ± 0,0 | 13,0 ± 1,0 | **+3,0 ± 1,0** | 3/3 |

**O achado não é que a antecedência encolhe — é POR QUE ela encolhe.**

| grandeza | inclinação por ponto de ruído | r | p | leitura |
|---|---|---|---|---|
| época do **alerta** | +0,002 | +0,04 | **0,92** | **não depende do ruído** |
| época da degradação | −0,157 | −0,93 | 0,0004 | depende |
| antecedência | −0,159 | −0,90 | 0,0009 | depende |

A época em que a SampEn2D dispara é **estatisticamente indistinguível entre os três níveis**
(ANOVA p = 0,42): ~10 em todos. Quem se move é a degradação visível, que chega cada vez mais
cedo. A antecedência encolhe como consequência aritmética disso — **não porque o indicador
piore com ruído alto**.

**Interpretação.** A SampEn2D parece marcar o **início da memorização**, um evento que
acontece na mesma altura do treino independentemente de quanto ruído existe. O que o ruído
controla é a velocidade com que essa memorização vira dano mensurável na validação. Isso é
mais forte do que "o indicador antecipa o overfitting": sugere que ele mede o *mecanismo*, e
a validação mede o *sintoma*.

Detecção: **9/9 runs** nos três níveis, sem falso positivo nos 8 controles. Memorização
confirmada em todos (treino em `eval()` 96–99 % contra validação de 93 % a 63 %, conforme o
nível). O nível de 15 % é o teste mais duro — metade do ruído original — e é justamente
onde a antecedência é **maior**.

**Ressalva:** 3 sementes por nível, 3 níveis, uma arquitetura, um dataset. A tendência é
nítida e monotônica, mas a afirmação "o alerta não depende do nível de ruído" apoia-se em
9 runs. E o `p = 0,92` do alerta é ausência de evidência de dependência, **não** evidência
de independência — com n = 9 o teste não teria força para achar um efeito pequeno.

#### ❌ DenseNet-121: a MEDIDA generaliza, o DETECTOR não — 2026-09-23

6 runs (3 ruído 30 % + 3 controles), mesmas sementes dos runs de referência, prefixo
`dn121_` para não contaminar as varreduras `f6_*`. Camada densa 6×**1024** em vez de 6×512.

**O que generalizou:**

| critério | ResNet-18 | DenseNet-121 |
|---|---|---|
| overfitting detectado (`OVERFITTING`) | 3/3 | **3/3** |
| falso positivo nos controles | 0/3 | **0/3** |
| amplitude da SampEn2D, ruído × controle | 5,5× | **2,8×** |

O critério de acurácia atravessou a mudança de arquitetura sem arranhão, e a SampEn2D
**continua discriminando** — com contraste menor (2,8× contra 5,5×), mas longe do 1,0× da LMC.

**O que NÃO generalizou: o detector de afastamento.** O `ALERTA_OVERFIT` disparou em **1 de 3**
runs de ruído, e um controle (`dn121_seed1`) alertou na **época 8** — antes do único alerta
legítimo (época 21). A margem de separação, que era +14 épocas na ResNet, fica **negativa**.

**A causa é diagnosticável, e é do detector, não da medida.** O critério é
"afastou-se mais de `k`·σ do platô das épocas 1–5". Esse σ varia **47×** entre runs:

| arquitetura | grupo | σ do platô (épocas 1–5) |
|---|---|---|
| ResNet-18 | ruído | 0,048 · 0,058 · 0,038 |
| ResNet-18 | controle | 0,050 · 0,057 · 0,037 |
| DenseNet-121 | ruído | **0,297 · 0,278 · 0,298** |
| DenseNet-121 | controle | **0,024 · 0,006 · 0,027** |

Duas falhas opostas, pelo mesmo motivo:
1. **Nos runs de ruído da DenseNet o "platô" não é platô.** A dinâmica dela é ~2× mais lenta
   (degradação na época 28–32 contra 14–15) e a série ainda está se movendo nas épocas 1–5.
   O σ fica 6× maior, a faixa `5σ` fica larguíssima, e o afastamento posterior nunca a cruza.
2. **No controle `dn121_seed1` o σ é 0,006** — quase zero. Qualquer flutuação ultrapassa
   `5σ`, e ele alerta na época 6 com *qualquer* `k` testado (2, 3, 4 ou 5).

Ou seja: normalizar pelo desvio das 5 primeiras épocas pressupõe que elas sejam um platô
estável. Na ResNet isso valia; na DenseNet, não — nas duas pontas.

**O que NÃO fiz, de propósito.** Re-ajustar `afast_base`/`afast_k` nestes 6 runs faria o
alerta "funcionar" na DenseNet — e seria exatamente a circularidade contra a qual este plano
avisa desde a F4. Um detector re-calibrado em cada arquitetura que encontra não é um
indicador, é um ajuste de curva. Corrigi-lo exige repensar a normalização (escala relativa à
própria série, ou platô detectado em vez de fixado nas épocas 1–5) e **validar em
arquitetura nova**, não nestas.

**Consequência para o texto.** A afirmação defensável passa a ser mais estreita e mais
honesta: *a SampEn2D da camada densa discrimina memorização em duas arquiteturas; a regra
de decisão que a transforma em alerta antecipado foi validada apenas na ResNet-18 e não
transfere como está.* Separar a medida do detector é o que permite reportar os dois
resultados sem que o negativo apague o positivo.

#### ✅ Detector redesenhado: queda relativa ao máximo corrente — 2026-09-24

**A troca.** Saiu "afastou-se `k`·σ do platô das épocas 1–5"; entrou **"caiu `queda_rel`
abaixo do seu máximo corrente por `queda_p` épocas"** (`epoca_de_queda_relativa`).

A escolha não foi por tentativa: é a **mesma forma** do critério de acurácia
(`epocas_abaixo_do_pico`) — e aquele foi justamente o único que atravessou a troca de
arquitetura sem ajuste. O que muda é a escala de normalização: o próprio máximo da série,
em vez de um σ estimado com 5 pontos que variava 47× entre runs.

Pressuposto declarado: **a SampEn2D cai durante a memorização.** Verificado nos 12 runs das
duas arquiteturas (queda de 24–29 % nos runs com ruído contra 5–16 % nos controles).

**Protocolo.** Limiares ajustados **só em runs de ResNet-18** (14 com ruído cobrindo os 3
níveis, 8 controles) e aplicados **sem retoque** à DenseNet-121. Escolhidos:
`queda_rel = 10 %`, `queda_p = 2`.

| | ResNet-18 (ajuste) | DenseNet-121 (**teste cego**) |
|---|---|---|
| runs de ruído detectados | 14/14 | **3/3** |
| runs de ruído que alertaram | 14/14 | **3/3** (época 6) |
| controles com `OVERFITTING` | 0/8 | 0/3 |
| controles que alertaram | **0/8** | 2/3, tarde (épocas 23 e 40) |
| margem de separação | ∞ | **+17 épocas** |

Contra o detector antigo na mesma DenseNet: 1/3 runs detectados e margem **negativa**.

**O que a troca custou — e está no texto.** O critério novo detecta *acúmulo*, não *início*,
então chega mais tarde: época 11–14 contra 7–9 na ResNet.

| nível de ruído (ResNet) | antecedência antiga | antecedência nova |
|---|---|---|
| 15 % | +8 a +11 | +4 a +7 |
| 30 % | +7 a +8 | +1 a +4 |
| 50 % | +4 a +6 | **+1, +1, −1** |

**Com 50 % de ruído o aviso deixa de ser aviso** — num dos runs ele chega depois da
degradação. É o preço da robustez, e é um limite a declarar, não a esconder.

**O terceiro desenho, medido e descartado.** Testei também "a série caiu em `p` épocas
consecutivas" — só o sinal da variação, sem σ e sem nível. Detecta *início*: antecedência
média **8,6** na ResNet (mínimo +6) e **24–28** na DenseNet, melhor que os outros dois em
todos os runs. Foi descartado pela especificidade: dispara em **7 dos 8 controles** da
ResNet, com margem de **2 épocas** (3 na DenseNet). Margem fina foi exatamente o que quebrou
na primeira troca de arquitetura; trocar 8 épocas de aviso por uma margem que provavelmente
não sobrevive a um dataset novo seria repetir o mesmo erro com outro nome.

**Ressalva de honestidade.** Os totais de queda da DenseNet foram inspecionados **antes** de
eu escolher a *forma* do critério — então o teste cego vale para os limiares e para as
grandezas avaliadas (época do alerta, margem, antecedência), que não foram ajustadas nela,
mas **não** é um cego perfeito no nível da escolha do desenho. O teste que falta é uma
terceira arquitetura, nunca vista.

**Decisão:** manter `n_major` como padrão — é a preferência declarada do orientador (D2) e tem
margem melhor —, e **reportar a ablação como subseção de metodologia**, porque a sensibilidade
à ordem é um resultado, não um detalhe de implementação. A SampEn2D segue como indicador
principal: mesma antecedência do melhor `x_major`, margem 8× maior e sem depender de escolher
ordem alguma — que era exatamente o argumento da F2 para adotá-la.

Memorização confirmada nas cinco (checagem anti-BatchNorm da F3): acurácia de treino em `eval()`
sobe de ~74 % para 97–98 % enquanto a de validação cai de ~99,9 % para 83–85 %. Comportamento
notavelmente uniforme entre sementes.

**Semente do ruído separada da semente da partição.** A corrupção dos rótulos usava `data_seed`,
que é fixa de propósito (congela quem cai em treino/validação/teste). Repetir trocando só `seed`
reproduziria **exatamente os mesmos rótulos errados** — mediria sensibilidade à inicialização,
não replicação. Criado `--noise_seed` (padrão = `data_seed`, então `f6_ruido30` continua
reproduzível); nas quatro sementes novas variam a inicialização, a ordem dos lotes e **quais**
rótulos são corrompidos, com a partição intacta. Taxa efetiva de erro estável: 24,8–25,1 % nas cinco.

#### Âncora corrigida: `val_loss` é frágil em métrica saturada

O mínimo do `val_loss` do run de ruído cai na **época 4**, quando a acurácia ainda era
99,75 % e o modelo estava saudável — as "pioras" eram flutuações de terceira casa. Ancorar
ali descartaria **56 de 60 épocas** por ruído, e o sombreado da figura contradizia a própria
análise.

Acrescentado `detect_degradacao_acuracia()` em `src/training/overfit.py` (mesma fonte única,
mesmo formato de retorno): a degradação é a primeira época em que a acurácia fica 1 ponto
abaixo do pico por 3 épocas consecutivas. `carregar_relatorio(..., criterio="val_accuracy")`
seleciona o critério. Resolve o item deixado em aberto na F5.

#### Ressalvas honestas

- ~~**n = 1 run com overfitting.**~~ **Resolvido em 2026-09-22:** replicado em 5 sementes
  (42, 1, 7, 13, 23), com realizações de ruído independentes. Antecedência da SampEn2D:
  +7,4 ± 0,5 épocas (IC 95 %: 6,7 a 8,1). Os 5 runs compartilham arquitetura, dataset e nível
  de ruído — o IC mede variação entre sementes, **não** generalização para outros cenários.
- Os parâmetros do detector de afastamento (plateau de 5 épocas, 4 desvios, 3 épocas
  consecutivas) foram escolhidos olhando os controles. A verificação independente — calibrar
  em 2 sementes e testar na 3ª — ainda não foi feita com runs de overfitting.
- **O detector dispara em 3 dos 5 controles** (SampEn2D). A separação só aparece ao olhar
  *quando* ele dispara. Falta transformar isso numa regra de decisão explícita, com limiar de
  época declarado, e validá-la fora das sementes usadas para escolhê-lo.
- A margem de separação (2 a 17 épocas, conforme a medida) foi medida nas MESMAS 5+5 sementes
  que calibraram o detector. Uma margem medida onde se calibrou é otimista por construção.
- Só uma arquitetura (ResNet-18), um dataset (MedNIST), um nível de ruído (30 %).
- O MedNIST satura em 1 época; o efeito só aparece porque o ruído foi induzido artificialmente.

---

### 🔴 F7 — Texto do projeto: os 3 pilares com foco no desenvolvedor (D3)
**Fazer:** escrever/atualizar o documento do projeto com três seções:

1. **Conteúdo teórico**
   - Complexidade LMC: entropia de Shannon × desequilíbrio; por que é baixa tanto no cristal (ordem total) quanto no gás (desordem total) e alta no meio.
   - SampEn: irregularidade/previsibilidade de uma sequência; papel de `m` e `r`.
   - A camada densa como objeto: por que ela concentra a decisão da rede e por que é o lugar natural para ler o estado do aprendizado.
   - A formalização do caminho e da ordem do flatten (§2 deste plano).

2. **Justificativa — voltada ao desenvolvedor** *(este é o pedido explícito do orientador)*
   - Hoje, para saber se a rede aprendeu, o desenvolvedor **depende inteiramente dos dados**: precisa de conjuntos de treino, validação e teste bem construídos, rotulados e representativos. Isso é caro, lento, e em imagem médica muitas vezes inviável.
   - Os dois indicadores são calculados **só a partir dos pesos** — não precisam de rótulo nenhum. Se funcionarem, dão ao desenvolvedor uma leitura do treinamento **independente do conjunto de validação**.
   - Casos de uso concretos: (a) decidir quando parar; (b) detectar que a rede parou de aprender antes de gastar mais GPU; (c) comparar arquiteturas sem precisar de mais dados rotulados; (d) diagnosticar overfitting quando o conjunto de validação é pequeno demais para ser confiável.
   - Entrega prática: o monitor da F4, que qualquer pessoa pluga no próprio laço de treino.

3. **Estado da arte**
   - Métricas de generalização a partir dos pesos: Bartlett et al. 2017 (normas/margens espectrais), Dziugaite & Roy 2017 (PAC-Bayes), Jiang et al. 2020 (*Fantastic Generalization Measures*), Martin & Mahoney 2021 (heavy-tailed self-regularization / WeightWatcher).
   - **O nicho deste trabalho:** leitura *informação-teórica da distribuição de valores* dos pesos (não da estrutura espectral) e **foco temporal** — a trajetória ao longo do treino, não um estado único.
   - Grokking e a literatura de early stopping.

**Critério de aceite:** texto revisado, cada afirmação de estado da arte com referência verificada (usar o Scite para não inventar citação).

---

---

### ✅ F3b — Remoção do MONAI: PyTorch/torchvision puro — **CONCLUÍDA em 2026-09-21**

**Decisão do autor (2026-09-21):** abandonar o MONAI, em código **e** na tese. Não constava
do `plan.md` nem da lista de decisões inicial — verificado em quatro lugares antes de agir.

#### O que se descobriu antes de mexer

- **O código da dissertação já era PyTorch puro** (`torchvision.models`, `torch.utils.data`).
  Nenhum import de MONAI. Então alinhei este repositório com o stack que a tese já usava —
  mesmo raciocínio da F1, onde portei o `complexity.py` dela.
- **A tese mencionava MONAI numa única linha** (`metodologia.tex:160`), e a menção era
  **factualmente errada em três níveis**: o código dela nunca usou MONAI; o
  `requirements.txt` dela declarava `monai==1.3.2`, versão diferente da citada (1.4.0); e o
  pacote **nem estava instalado** no venv da tese.

#### Partição congelada — o ponto delicado

Quem definia treino/validação/teste era o `MedNISTDataset` do MONAI, com embaralhamento
semeado interno. Removê-lo mudaria a partição e quebraria a comparabilidade.

`scripts/congelar_particao.py` gravou a lista exata de arquivos de cada conjunto em
`data/mednist_split.json` (6,5 MB, sha256 `cffabf711c22…`, 47.164/5.895/5.895), **antes** da
remoção, com verificação de que nenhuma amostra aparece em dois conjuntos. A partição deixou
de depender da implementação interna de uma biblioteca e virou **dado versionado do projeto**
— ganho líquido de reprodutibilidade.

#### Substituições

| antes | agora |
|---|---|
| `monai.apps.MedNISTDataset` | `src/data/mednist.py` + partição congelada |
| `monai.networks.nets` | `torchvision.models` |
| `monai.transforms` | `torchvision.transforms` |
| `monai.data.{Dataset,CacheDataset,DataLoader}` | `torch.utils.data` |

**A camada densa analisada não mudou: `fc` com forma [6, 512] = 3.072 pesos.** Toda a
análise das fases anteriores continua válida.

Imagens em escala de cinza passam a ser replicadas para 3 canais (as redes do torchvision
esperam RGB), espelhando o que a dissertação já fazia. `in_channels != 3` agora levanta erro
em vez de divergir em silêncio. **136 testes passando.**

#### Efeito colateral inesperado: 3× mais rápido

| | época (47.164 amostras, 64×64) |
|---|---|
| MONAI, CPU | 568 s |
| MONAI, GPU | 24,9 s |
| **torchvision, GPU** | **7,6 s** |

Ganho total de **75×** sobre o ponto de partida. O sweep da F6 passa de ~19 h para poucos
minutos.

#### ⚠️ Runs anteriores formam uma linha de base separada

A ResNet-18 do torchvision **não** é numericamente idêntica à do MONAI (que deriva de uma
ResNet 3D médica, com tronco diferente). Os três runs existentes não são comparáveis com os
novos. Como a F6 ia re-treinar tudo de qualquer forma, o custo é zero — mas precisa estar
dito no texto.

#### Também corrigido

- `requirements.txt` deste repo: MONAI removido; **torch fixado como `+cu124`**, com aviso
  de que a ausência do sufixo instala a build CPU-only (o que manteve a GPU ociosa até hoje).
- `requirements.txt` da tese: bloco MONAI removido; comentário do numpy, que justificava o
  pin por causa do MONAI, reescrito.
- `metodologia.tex`: "MONAI 1.4.0" → "torchvision 0.19.1". Tese recompilada: **78 páginas,
  0 citações não resolvidas, 0 ocorrências de MONAI no PDF**.
- README com a nova stack e o aviso sobre os wrappers quebrados do `myenv/bin/`.

#### 🚨 Achado não relacionado, mas urgente: resposta de chat impressa no PDF da tese

`tese/capitulos/resultados.tex` (modificado em **2026-08-25**, não por mim) contém um bloco
de resposta conversacional de IA colado dentro do LaTeX: uma cerca ```` ```latex ````, o
fechamento, e os cabeçalhos `### Resumo` e `### O que fazer` com bullets.

Isso causa **8 erros de compilação** e — pior — **é impresso no PDF, página 34, Capítulo 4**,
em primeira pessoa: *"Escrevi três parágrafos em LaTeX, em português formal, para a
dissertação…"*, com o texto embolado por entrar em modo matemático.

Conserto mapeado: remover as linhas **16, 22 e 24–32** de `resultados.tex`. Os três
parágrafos legítimos (linhas 17–21) ficam. **Aguardando autorização do autor** — o
repositório da tese **não está sob git**, então farei backup antes.

### 🟡 F8 — Depois: consolidação e limpeza
- Unificar `monai_wieghts/` (com erro de digitação) e `monai_weights/`.
- Aposentar os scripts legados de MedMNIST (`medmnist_weights_extraction.py`, `data_analysis_summary.py`) ou movê-los para `legacy/`.
- README reescrito refletindo o pipeline novo.
- `requirements.txt`: fixar `antropy`/`nolds` se forem usados na validação.
- Reconciliar as discrepâncias D3/D4 já registradas entre a metodologia da tese e o código (subamostragem aleatória e cálculo por grupo) — **grande parte delas some** com o foco na camada densa, mas o texto da tese precisa ser corrigido.

### 🟡 F9 — Depois: extensões
- SampEn2D sobre a matriz $W$ (sem achatar) — o projeto original do orientador prevê; com a camada densa 6×512 é barato.
- Multiscale entropy (MSE).
- DenseNet-121 além do ResNet-18, para mostrar que o indicador não é específico da arquitetura.
- Mais sementes (3 é pouco para afirmação estatística) e teste de significância.
- Dashboard ao vivo (TensorBoard custom scalar ou página HTML).

---

## 6. Ordem de execução recomendada

```
F1 (módulo único)  ──►  F2 (validação)  ──►  F3 (split)  ──►  F4 (monitor ao vivo)
                                                                    │
                                                                    ▼
                                                  F5 (corte overfit)  ──►  F6 (experimentos)
                                                                                │
                                                                                ▼
                                                                          F7 (texto)
```

F1 → F2 → F3 → F4 → F5 são **código**, cada uma testável isoladamente.
F6 é **tempo de máquina** (3 treinos de 40 épocas + 1 de overfitting induzido).
F7 é **escrita**, e depende dos números da F6 para a seção de justificativa ficar concreta.

**Estimativa grosseira:** F1–F5 ≈ 1 sessão de trabalho cada; F6 ≈ algumas horas de GPU/CPU; F7 ≈ 2 sessões.

---

## 7. Riscos

| Risco | Mitigação |
|-------|-----------|
| O indicador **não** antecipa o overfitting nos dados do MedNIST | Reportar honestamente; o run de overfitting induzido garante ao menos um caso limpo para caracterizar |
| Limiares do status parecem escolhidos *a posteriori* (overfitting da metodologia sobre si mesma) | Calibrar em 2 sementes, verificar na 3ª; declarar isso no texto |
| Overhead do cálculo ao vivo atrapalha treinos grandes | Já é barato para a camada densa (3.072 pesos); flag para desligar; medir e reportar |
| 3 sementes é pouco para significância | Reconhecer como limitação; F9 prevê ampliar |
| MedNIST é fácil demais (converge rápido, overfitting fraco) | Run com 5 % dos dados; se preciso, voltar ao PathMNIST |

