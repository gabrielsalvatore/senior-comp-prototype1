# Who Bore the 2021–2024 Inflation Surge?

**Senior Comprehensive Project Proposal (DOUBL-600)** · Gabriel Saccoccio · October 2026

This senior comp measures inflation for each household in the BLS Consumer Expenditure Survey, then asks which household types faced higher inflation before and after COVID-19, why, and how much the answer depends on the method used. It replaces the PIE prototype's mock transactions with real household microdata and turns its weighted-index code into an engine whose algorithms are compared and analyzed.

---

## 1. Research question and contribution

**Main question:** How did the inflation experienced by U.S. households differ across socioeconomic characteristics, and did those differences change from before COVID-19 to after it?

Sub-questions:

1. Which household types (income, urban/rural, sex of reference person and family type, age, race, education, housing tenure, region) faced inflation above the CPI-U before COVID (2017–2019) and after it (2021–2024)?
2. Which characteristics remain statistically significant once the others are held constant?
3. How much of each gap comes from what households buy versus the prices they face, and how sensitive is it to the choice of weights and index formula?
4. What is the cheapest correct way to compute thousands of household indexes with standard errors, and how do candidate algorithms scale?

**Contribution.** Inflation heterogeneity is an established literature (Kaplan and Schulhofer-Wohl; Jaravel; BLS research indexes), so this project does not claim a new phenomenon. It contributes:

- a replication of income-group results, benchmarked against BLS's official research CPI by income quintile;
- an extension to urban versus rural households, which the official CPI does not cover;
- a systematic sensitivity analysis across index methods;
- an open, documented computational engine with measured complexity and survey-based standard errors.

## 2. Reader feedback map

Every question from the spring readers' meeting and the October advisor meeting has a section below.

| Reader | Question raised | Answered in |
| --- | --- | --- |
| Economics | How are weights calculated and what do they represent? | §4 Economic framework |
| Economics | What economic theory underlies the calculation? Which assumptions? | §4 Economic framework |
| Economics | Where does household data come from? Is it accessible? Which BLS variables exist? | §3 Data |
| Economics | Rural vs urban; differences across groups; which effects drive them | §5 Comparisons and decomposition |
| Economics | Alternative methods; sensitivity to assumptions | §6 Alternative methods and sensitivity |
| Economics | Which time period, and why; does the data cover it? | §7 Time period |
| Computer science | Runtime, complexity, link to computational theory | §8 Computational component |
| Computer science | Compare different algorithms; expand performance analysis | §8 Computational component |
| Advisor (Oct) | Why these weights? What do alternative weights look like? | §4 Economic framework: Why these weights |
| Advisor (Oct) | More household types, e.g. male- vs female-led; age and sex of the household head | §5 Comparisons and decomposition |
| Advisor (Oct) | Compare before and after COVID (e.g. were rural areas better off before?) | §7 Time period |
| Advisor (Oct) | Statistical significance of the variables | §5 Comparisons: Statistical significance |
| Advisor (Oct) | Simpler main question about socioeconomic variables | §1 Research question |

## 3. Data

All inputs are free, public BLS releases. No personal or private data is used, which removes the privacy issue the PIE prototype dodged with mock records.

| Source | What it provides | Coverage | Role in the project |
| --- | --- | --- | --- |
| [CE Public-Use Microdata](https://www.bls.gov/cex/pumd_data.htm) (Interview + Diary) | Spending by detailed item code (UCC) per household, plus income, age, family size, region, urban/rural | Released through 2024; CSV, SAS, Stata | Household expenditure weights and group characteristics |
| CPI-U item indexes | Monthly price index per CPI item, nationally and for 4 regions | Monthly, decades of history | Price change per category |
| [CE–CPI concordance](https://www.bls.gov/cpi/additional-resources/ce-cpi-concordance.htm) | Maps each CE item code (UCC) to a CPI entry-level item (ELI) | Published by BLS | Links spending to prices |
| [R-CPI-I](https://www.bls.gov/cpi/research-series/r-cpi-i.htm) | BLS research CPI by equivalized income quintile | Monthly, Dec 2005 – Dec 2025 | External benchmark for validation |

**Key variables** (from the [PUMD Getting Started Guide](https://www.bls.gov/cex/pumd-getting-started-guide.htm)):

- `FINLWT21`: each household's population weight.
- `WTREP01`–`WTREP44`: 44 replicate weights for computing standard errors.
- `FINCBTXM`: household income before taxes.
- `MTBI` file: monthly expenditures by `UCC`, with `REF_MO` and `REF_YR`.
- `BLS_URBN` (urban/rural) and `REGION`.
- `RENTEQVX`: owners' rental equivalence of their home, for valuing owner housing as the CPI does (definition to confirm in the data dictionary).
- Reference-person and household characteristics: `SEX_REF`, `AGE_REF`, `FAM_TYPE` (e.g. married couple, single mother, single father, single person), `EDUC_REF`, `REF_RACE`, `HISP_REF`, `CUTENURE` (owner with or without mortgage, renter), `NO_EARNR`.

**Feasibility check (2023 Interview package, run October 1, 2026).** Every variable above is present. The package holds 18,871 household-quarter records from 10,344 households, including 1,218 rural records (about 300 per quarter, 6.3% of the weighted population). The expenditure file has 1.5 million rows across 492 distinct UCC codes and covers all 12 months of 2023.

**Known data caveats.**

- Since 2020 each Interview package holds Q2–Q4 of its year plus Q1 of the next, so a calendar year needs two packages.
- The CPI-U covers over 90% of the population but [excludes rural nonmetropolitan households](https://www.bls.gov/opub/hom/cpi/concepts.htm). No rural price index exists, so rural households can differ only through their basket. That limit is stated, not hidden, and is itself a finding about how the official statistics work.

## 4. Economic framework

The target is each household's cost-of-living change: how much more it must spend to keep the same standard of living. Price indexes approximate that (Konüs cost-of-living theory; Diewert's index-number theory).

**Weights.** A household's weight on CPI item *i* is its share of total spending on that item in the base period. It represents how exposed that household is to price changes in item *i*.

$$
w_{h,i} = \frac{e_{h,i}}{\sum_{j} e_{h,j}}
$$

Here *e* is spending by household *h* on item *i*, summed over the household's CE quarters and mapped from UCC codes to CPI items via the concordance.

**Why these weights.** The question is about differences between households, so the weights must vary by household. The CPI's single basket cannot show a gap by construction. The base year is fixed at the start of each window, so the index moves only because prices changed, not because households reacted to them. The alternatives below are all computed too, and the sensitivity section reports how much the results move.

| Weight scheme | How it is built | What it represents | How it differs from the chosen weights |
| --- | --- | --- | --- |
| **Household base-year shares (chosen)** | One household's spending on each item ÷ its total spending, base year | That household's exposure to each price | Shows full heterogeneity; noisy for one household (lumpy purchases such as a car) |
| Official CPI-U relative importance | BLS aggregate spending shares of all urban consumers | The average urban dollar | One basket for everyone, so no gaps; weighted toward big spenders |
| Group aggregate shares (plutocratic) | Group's total spending per item ÷ group's total spending | The group's average dollar | The CPI and R-CPI-I method; high spenders dominate the group |
| Group average shares (democratic) | Mean of household shares within the group | The typical household in the group | Every household counts once; smooths individual noise |
| Current-period or chained shares | Shares updated each year | The basket after households adjust | Allows substitution; mixes price change with behavior change |
| Smoothed shares | Multi-year average, or shrunk toward the group mean | A stable long-run basket | Less noise, less individual detail; BLS smooths R-CPI-I weights this way |

**Household index (baseline, Laspeyres).** Fixed base-period basket, valued at current prices:

$$
\pi_{h,t} = \sum_{i} w_{h,i} \left( \frac{P_{i,t}}{P_{i,0}} \right) - 1
$$

Laspeyres assumes households do not substitute when relative prices move, so it is an upper bound on the true cost-of-living change. Paasche, using current-period shares, is a lower bound. Fisher and Törnqvist sit between the two and approximate the true index closely (Diewert's "superlative" indexes).

**Group index.** A group's inflation is either the average of its households' rates ("democratic", every household counts once) or its aggregate spending weights ("plutocratic", how the CPI works, so big spenders count more). Both are reported because they answer different questions.

**Assumptions to justify in the thesis.**

1. Within an item and region, all households face the same price change. Jaravel shows that products bought by high-income households can inflate more slowly, so this likely understates gaps.
2. CE spending shares reflect true consumption. CE underreports some items, notably alcohol and tobacco.
3. Owner housing is valued as rental equivalence, as the CPI does, not as mortgage payments, using CE's `RENTEQVX`.
4. Spending shares are stable within the base year.

## 5. Comparisons and decomposition

Households are compared on eight characteristics, each against the CPI-U and against each other. All come from the CE household file.

| Characteristic | Groups compared | CE variable |
| --- | --- | --- |
| Income | Quintiles of equivalized income (income ÷ √household size), matching R-CPI-I | `FINCBTXM`, `FAM_SIZE` |
| Urban vs rural | Urban, rural | `BLS_URBN` |
| Sex of reference person and family type | Male vs female reference person; single mothers, single fathers, married couples, single men, single women | `SEX_REF`, `FAM_TYPE` |
| Age of reference person | Under 35, 35–64, 65+ | `AGE_REF` |
| Race and Hispanic origin | CE race categories; Hispanic vs not | `REF_RACE`, `HISP_REF` |
| Education of reference person | High school or less, some college, bachelor's or more | `EDUC_REF` |
| Housing tenure | Owner with mortgage, owner outright, renter | `CUTENURE` |
| Region | Northeast, Midwest, South, West | `REGION` |

**The household head.** CE's closest concept to a household head is the **reference person**: the person who owns or rents the home. For single-adult households, reference-person sex cleanly means male-led or female-led. For married couples it mostly reflects who was listed first, so sex comparisons rely mainly on `FAM_TYPE`: single mothers vs single fathers, single women vs single men.

**Where gaps come from.** The gap between group *g* and a reference *r* (the CPI-U basket, or another group) splits into three effects:

$$
\pi_g - \pi_r = \underbrace{\sum_i (w_{g,i} - w_{r,i})\,\pi_{r,i}}_{\text{basket effect}} + \underbrace{\sum_i w_{r,i}\,(\pi_{g,i} - \pi_{r,i})}_{\text{price effect}} + \underbrace{\sum_i (w_{g,i} - w_{r,i})(\pi_{g,i} - \pi_{r,i})}_{\text{interaction}}
$$

- **Basket effect:** the group buys more of what inflated fastest (e.g., gasoline and food for rural households).
- **Price effect:** the group lives where prices rose faster, measured with regional CPI indexes. For rural households this term is zero by construction, which the thesis states explicitly.
- **Substitution effect:** measured separately as Laspeyres minus Törnqvist, showing which groups could and could not shift spending to cheaper goods.

Each effect is also broken down by category, so results read like "two-thirds of the rural gap comes from motor fuel and food at home."

**Statistical significance.** Raw group gaps can mislead. Rural households may look worse off only because they earn less. A survey-weighted regression of each household's inflation rate on all characteristics at once separates the two:

$$
\pi_h = \alpha + \beta_1 \text{Rural}_h + \beta_2 \text{IncomeQuintile}_h + \beta_3 \text{FemaleRef}_h + \beta_4 \text{Age}_h + \dots + \varepsilon_h
$$

- Each coefficient is the gap for that characteristic with the others held constant.
- Standard errors come from the 44 replicate weights, so tests respect the survey design.
- Every gap is reported with a 95% confidence interval and a significance test. A correction for multiple comparisons applies because many groups are tested at once.
- Adding an after-COVID indicator and its interactions (e.g. Rural × Post) tests whether a gap changed between the two periods.

## 6. Alternative methods and sensitivity

Each main result is recomputed under every variant below. A finding counts as robust only if its sign and rough size survive all of them.

| Choice | Baseline | Alternatives tested | What it reveals |
| --- | --- | --- | --- |
| Index formula | Laspeyres | Paasche, Fisher, Törnqvist | Size of substitution bias by group |
| Weight updating | Fixed base-year weights | Chained annual weights | Whether stale weights distort the surge period |
| Group aggregation | Plutocratic | Democratic | Whether high spenders mask lower-income experience |
| Category detail | Detailed CPI items (a few hundred) | Coarse 8 major groups | How much aggregation hides (PIE used 6) |
| Prices | National item indexes | Regional item indexes | Size of the price effect |
| Housing | Rental equivalence | Excluding shelter | How much shelter drives the gaps |

The category-detail test also fixes the PIE prototype's flaw: a 6-category basket put even the national-average profile 0.7 points above the CPI, which is basket-truncation bias.

## 7. Time period

The thesis compares two windows of equal design: before COVID (2017–2019) and after it (2021–2024). 2020 is left out as a transition year. Each window uses spending weights from the year before it starts: 2016 for the pre-COVID window, 2019 for the post-COVID window.

| Window | Prices | Base-year weights | Why |
| --- | --- | --- | --- |
| Before COVID | Jan 2017 – Dec 2019 | 2016 | Low, stable inflation; shows whether group gaps are normal |
| After COVID | Jan 2021 – Dec 2024 | 2019 | The surge (CPI-U peaked near 9% in mid-2022) and its cooling |

- **The key test** is whether each group's gap changed between the windows, measured as gap after minus gap before with a replicate-weight standard error.
- **Example hypothesis:** rural households were at or below average before COVID, when gasoline was cheap, then above average after, when gasoline and food spiked. The data decides whether that holds.
- **Why 2019 for the post window:** the last full pre-pandemic year. 2020 spending was distorted by lockdowns and its CE collection was disrupted.
- **Coverage confirmed:** CE microdata is released through 2024, CPI item indexes are monthly and current, and R-CPI-I runs through December 2025.

## 8. Computational component

The engine treats index construction as linear algebra. Stack household weights into a matrix **W** (*H* households × *I* items) and price relatives into **R** (*I* items × *T* months). Every household's index at every month is then one product:

$$
\Pi = W R - \mathbf{1}
$$

That framing lets the thesis compare algorithms that produce identical numbers at very different cost.

| Algorithm | Idea | Time complexity | Expected trade-off |
| --- | --- | --- | --- |
| A1 Naive loops | Household by household, item by item (PIE's current approach) | O(H · I · T) | Simple; slowest constant |
| A2 Dense matrix product | One BLAS call (NumPy) | O(H · I · T), far lower constant | Fast, but stores every zero |
| A3 Sparse matrix product | Store only items a household bought | O(nnz(W) · T) | Wins when baskets are sparse |
| A4 Aggregate first | Collapse W to G group rows, then multiply (associativity) | O(H · I + G · I · T) | Huge saving for group results; linear indexes only |
| A5 Incremental chaining | Update the index by one month instead of recomputing | O(nnz(W)) per new month | Suits monthly updates; must manage chain drift |

**Link to theory.** A4 is the matrix-chain ordering problem: (A·W)·R and A·(W·R) give the same answer at different costs. It works for linear indexes (Laspeyres, Paasche) but not for Fisher or Törnqvist, which are geometric. The thesis proves which formulas permit it and measures the gap.

**Uncertainty.** Standard errors use BLS's balanced repeated replication: recompute every estimate with each of the 44 replicate weights. Naively that is 45× the work. With A4 each replicate only changes the aggregation step, so the cost grows far more slowly. A bootstrap is compared as an alternative.

**Evaluation.**

- Correctness: all algorithms must agree to floating-point tolerance on every output.
- Scaling: runtime and peak memory as *H*, *I* and *T* grow, with log-log slopes compared to the predicted complexity.
- Validity: income-quintile results compared against R-CPI-I (correlation and mean absolute gap).

## 9. Optional final chapter: US vs Brazil

If the core is finished by early spring, a final chapter applies the same engine to Brazil and asks whether the surge hit low-income households harder there than in the US. It is optional, and the thesis stands without it.

- **Data:** IBGE's household budget survey microdata (POF 2017–2018) for weights, and IPCA subitem price indexes for prices.
- **Benchmark:** Ipea's [inflation indicator by income band](https://www.ipea.gov.br/cartadeconjuntura/index.php/tag/inflacao-por-faixa-de-renda/), which covers six income classes and uses POF 2017–2018 weights. It plays the role R-CPI-I plays for the US.
- **Why it is interesting:** Brazil had its own 2021–2022 surge, and food is a larger share of low-income budgets there, so basket effects should be larger.
- **Risks:** a second data system, Portuguese documentation, older weights (2017–2018), and categories that do not line up one-to-one with the US. The comparison is limited to income groups and broad categories to keep it tractable.

## 10. Work plan, risks and open questions

Target months are provisional until the DOUBL-600 milestone dates are confirmed.

1. **October:** reader approval of this proposal; finish the feasibility check (household variables confirmed for 2023; still to do: 2016 and 2019 packages, UCC-to-CPI concordance coverage, CPI item series).
2. **November:** engine with algorithms A1–A3; first household indexes for 2021–2024; validation against R-CPI-I.
3. **December:** group comparisons, decomposition, first results chapter draft.
4. **January–February:** sensitivity analysis, A4–A5, replicate-weight standard errors, scaling experiments.
5. **March:** full draft; optional Brazil chapter only if the core is complete.
6. **April:** revisions and defense.

**Risks.**

- **Rural sample size:** about 300 rural records per quarter is enough for a rural vs urban comparison, but rural × income quintile cells drop to roughly 60. Mitigation: report confidence intervals and pool years for fine subgroups.
- **Scope creep:** eight characteristics × two windows × several methods is a lot of results. Mitigation: income, urban/rural, sex and family type, and age are the core; race, education and tenure enter through the regression and get full tables only if time allows.
- **Interview vs Diary:** combining the two surveys is complex. Mitigation: start Interview-only, which covers most spending, and add Diary food items if time allows.
- **Concordance changes:** CPI item codes are revised over time, including between 2016 and 2019. Mitigation: map to a stable item level and document every manual mapping.

**Open questions.**

- [ ] What are the DOUBL-600 proposal, draft and defense deadlines?
- [ ] Do both readers accept this scope as a replacement for the PIE proposal?
- [ ] Is Interview-only acceptable for the first version?

## Sources

Data documentation (checked October 2026):

- [BLS: CE Public-Use Microdata downloads](https://www.bls.gov/cex/pumd_data.htm)
- [BLS: CE PUMD Getting Started Guide](https://www.bls.gov/cex/pumd-getting-started-guide.htm)
- [BLS: CE UCC to CPI ELI concordance](https://www.bls.gov/cpi/additional-resources/ce-cpi-concordance.htm)
- [BLS: Research CPI by equivalized income quintile (R-CPI-I)](https://www.bls.gov/cpi/research-series/r-cpi-i.htm)
- [BLS Handbook of Methods: CPI concepts and population coverage](https://www.bls.gov/opub/hom/cpi/concepts.htm)
- [Ipea: inflation by income band](https://www.ipea.gov.br/cartadeconjuntura/index.php/tag/inflacao-por-faixa-de-renda/)

Literature (carried over from the junior seminar bibliography): Kaplan and Schulhofer-Wohl on household inflation from scanner data; Jaravel on inflation inequality; Hobijn et al. and Levell and Skingle on group price indexes; Diewert on index-number theory and superlative indexes.
