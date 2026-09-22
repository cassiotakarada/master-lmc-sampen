# Códigos de cálculo — Complexidade LMC e Entropia Amostral (SampEn)

Pacote autocontido com as implementações usadas no projeto **"Complexidade estatística
dos pesos de redes neurais como indicador de treinamento"**.

**Dependências:** apenas `numpy` e `scipy`. Não precisa de PyTorch nem do restante do repositório.

```
complexity.py             módulo com todas as medidas
exemplo_uso.py            exemplo executável sobre uma camada densa real
fc_weight_exemplo.npy     matriz fc.weight [6, 512] de uma ResNet-18 treinada em MedNIST
fc_weight_exemplo.json    metadados da matriz acima
```

Para rodar:
```bash
pip install numpy scipy
python exemplo_uso.py
```

> `complexity.py` é um **artefato gerado** a partir de `src/complexity.py` do repositório
> (via `python deliverables/build_deliverable.py`). Não deve ser editado aqui — existe
> uma única implementação de cada medida em todo o projeto.

---

## 1. Complexidade estatística LMC

*López-Ruiz, Mancini & Calbet (1995), Phys. Lett. A 209:321–326.*

Dado o vetor de pesos, monta-se o histograma normalizado `p = (p_1, …, p_N)` sobre `N` bins:

```
H = − Σ p_i · log₂(p_i)          entropia de Shannon (bits)
D =   Σ (p_i − 1/N)²             desequilíbrio (distância L2 da uniforme)
C =   H · D                      complexidade LMC
```

**Intuição.** `C` é baixa nos dois extremos e alta no meio:

| Regime | H | D | C |
|--------|---|---|---|
| "Cristal" — todos os pesos iguais (ordem total) | ≈ 0 | alto | ≈ 0 |
| "Gás" — pesos uniformemente espalhados (desordem total) | máximo | ≈ 0 | ≈ 0 |
| Estrutura intermediária | médio | médio | **alta** |

É exatamente essa não-monotonicidade que torna a LMC interessante como indicador: uma rede
no início do treino está perto do "gás" (pesos aleatórios) e vai adquirindo estrutura.

**Função:**
```python
lmc_complexity(weights, n_bins=None) -> dict(entropy, disequilibrium, complexity, n_bins)
```

**Parâmetros e convenções**

| Parâmetro | Default | Observação |
|-----------|---------|------------|
| `n_bins` | `min(100, max(10, n // 5))` | regra adaptativa validada pelo grupo |
| base do log | 2 (bits) | |
| normalização | **nenhuma** — `C = H · D` cru | ver limitação L1 abaixo |

Para a camada densa analisada no projeto (3.072 pesos), a regra adaptativa resulta em
**exatamente 100 bins**, que é o valor usado em todas as comparações.

---

## 2. Entropia amostral 1D (SampEn)

*Richman & Moorman (2000), Am. J. Physiol. 278:H2039–H2049; Pincus (1991).*

```
SampEn(m, r) = − ln( A / B )

B = #{ pares ordenados (i,j), i≠j, com dist_Chebyshev entre templates de tamanho m   ≤ r }
A = #{ pares ordenados (i,j), i≠j, com dist_Chebyshev entre templates de tamanho m+1 ≤ r }
```

Ambos os contadores percorrem o **mesmo** conjunto de `N − m` posições iniciais (convenção
padrão de Richman & Moorman). A distância de Chebyshev entre dois templates é o maior
valor absoluto de diferença elemento a elemento.

**Intuição.** SampEn responde: *"quando dois trechos da sequência se parecem por `m` passos,
com que frequência eles continuam se parecendo no passo seguinte?"* Se quase sempre
continuam → sequência previsível → SampEn baixa. Se raramente → SampEn alta.

**Função:**
```python
sample_entropy_1d(series, m=2, r=None, r_scale=0.2) -> float
```

| Parâmetro | Default | Observação |
|-----------|---------|------------|
| `m` | 2 | comprimento do template |
| `r` | `0.2 × SD(série)` (`ddof=0`) | tolerância de "parecido" |

**Casos de retorno:**

| Retorno | Quando |
|---------|--------|
| `0.0` | série constante (perfeitamente regular) |
| `np.nan` | série curta demais (`N < m+2`) ou `B = 0` (indefinido) |
| `np.inf` | `B > 0` mas `A = 0` (irregularidade máxima) |

---

## 3. A camada densa e a ordem do achatamento

Seja a camada densa com `M` entradas `x_1…x_M` e `N` neurônios `n_1…n_N`.
A matriz de pesos é `W` com forma `[N, M]`, e `w_ji` é o peso do **caminho `x_i → n_j`**.
Essa é a forma nativa de `nn.Linear.weight` no PyTorch (`[out_features, in_features]`).

A SampEn opera sobre uma **sequência**, então é preciso achatar `W` em um vetor — e há duas
ordens possíveis:

| Ordem | Sequência | Leitura | Código |
|-------|-----------|---------|--------|
| **`n_major`** (adotada) | `w₁₁, w₁₂, …, w₁M, w₂₁, …` | `n1x1, n1x2, …` — todos os pesos que **chegam** em n1, depois em n2… | `W.reshape(-1)` |
| `x_major` | `w₁₁, w₂₁, …, w_N1, w₁₂, …` | `x1n1, x1n2, …` — todos os pesos que **saem** de x1, depois de x2… | `W.T.reshape(-1)` |

**Consequência fundamental:**

- A **LMC é invariante à ordem** — usa só o histograma dos valores. `LMC(v_A) == LMC(v_B)` exatamente.
- A **SampEn NÃO é invariante** — compara janelas consecutivas. Os dois valores diferem, e a
  diferença tem significado: `n_major` mede a regularidade **dentro do campo receptivo de cada
  neurônio**; `x_major` mede a regularidade de **como uma entrada se distribui entre os neurônios**.

Por isso a ordem é um parâmetro explícito e declarado, nunca implícito.

**Funções:**
```python
flatten_dense(weight, order="n_major")             -> np.ndarray 1D
find_dense_layers(state_dict, param_types=None)    -> list[str]
dense_complexity(weight, order="n_major", ...)     -> dict(lmc, entropy, disequilibrium, sampen, ...)
```

O **bias é excluído** da análise: o projeto mede *caminhos* entre unidades, e o bias não é um caminho.

---

## 4. Também incluídos (para trabalho futuro)

| Função | Referência |
|--------|-----------|
| `sample_entropy_2d(matrix, m=2, r=None, max_dim=64)` | Silva et al. (2018) — trata `W` como estrutura 2D, **sem** precisar escolher ordem de achatamento |
| `mse_1d(series, max_scale=10)` | Costa, Goldberger & Peng (2002, 2005) — entropia multiescala |
| `mse_2d(matrix, max_scale=10)` | versão 2D da multiescala |

---

## 5. Limitações conhecidas (declaradas de propósito)

**L1 — A LMC não é normalizada.** `H` e `D` são valores crus, então a escala de `C` depende
do número de bins. Medições com `n_bins` diferentes **não são comparáveis**. Em estudo de
sensibilidade sobre pesos de BatchNorm, a LMC variou de 0,65 a 0,23 ao ir de 20 para 200 bins.
*Mitigação:* usar sempre o mesmo `n_bins` (100) e reportar a **posição do pico/inflexão ao longo
das épocas**, que é o que interessa — não o valor absoluto.

**L2 — A SampEn depende da ordem.** Ver seção 3. Não é um defeito, mas exige que a ordem
seja declarada em qualquer resultado reportado.

**L3 — Custo computacional.** A SampEn constrói uma matriz de distâncias `(N−m)²`. Para a
camada densa (3.072 pesos) isso é ≈ 9,4 M pares — menos de 1 s. Para vetores de 10.000+
elementos passa a custar centenas de MB de RAM.

**L4 — `r = 0.2 × SD` é uma convenção**, não um ótimo. O valor usual na literatura fica entre
0,1 e 0,25 do desvio padrão.

---

## 6. Verificação

A implementação foi validada contra os resultados já publicados do projeto: recalculando
LMC e SampEn sobre checkpoints reais (ResNet-18 / MedNIST, 41 tensores por época), os valores
reproduzem os do pipeline anterior com diferença máxima de **3 × 10⁻¹⁷** (épsilon de máquina).

---

## Referências

- López-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). A statistical measure of complexity. *Physics Letters A*, 209(5–6), 321–326.
- Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology*, 278(6), H2039–H2049.
- Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. *PNAS*, 88(6), 2297–2301.
- Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of complex physiologic time series. *Physical Review Letters*, 89(6), 068102.
- Silva, L. E. V., Senra Filho, A. C. S., Fazan, V. P. S., Felipe, J. C., & Murta Jr., L. O. (2018). Two-dimensional sample entropy analysis of rat sural nerve aging. *Medical & Biological Engineering & Computing*.
