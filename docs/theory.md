# When can a group price index be computed aggregate first?

This note states and proves when a group's price index can be computed from the group's aggregate spending shares ("aggregate first", algorithm A4) rather than by computing every household's index and then averaging ("household first", A2/A3). It also bounds the error when the two differ and measures that error in the 2023 CE data. The code is in `pie/theory.py`, the property tests are in `tests/test_theory.py`, and section 6 of `python -m pie intrvw23` prints the measurements.

It corrects a claim in the proposal. The proposal said aggregate first works for Laspeyres and Paasche but not for Fisher or Törnqvist. The accurate statement is below. Aggregate first is exact for every index in the generalized-mean family *if the group index is built from aggregate shares*. The *democratic mean of household indexes* matches it only for Laspeyres.

## 1. Setup

Household $h$ has spending shares $s_{h1},\dots,s_{hI}$ (nonnegative, summing to 1). Each item has a price relative $r_i = p_{i1}/p_{i0} > 0$. A group of households has weights $\omega_h \ge 0$ with $\sum_h \omega_h = 1$. These are population weights for a democratic index, or population weight × total spending for a plutocratic (CPI-style) index.

**Generalized-mean indexes.** Let $g$ be continuous and strictly monotone on $(0,\infty)$ and let $\varphi = g^{-1}$. The household index is

$$P_h = \varphi\Big(\sum_i s_{hi}\, g(r_i)\Big) = \varphi(x_h), \qquad x_h := \sum_i s_{hi}\, g(r_i).$$

| $g(r)$ | $\varphi(x)$ | shares | index |
|---|---|---|---|
| $r$ | $x$ | base period $s^0$ | Laspeyres |
| $1/r$ | $1/x$ | current period $s^1$ | Paasche |
| $\log r$ | $e^x$ | average $(s^0+s^1)/2$ | Törnqvist |
| $\log r$ | $e^x$ | base period $s^0$ | geometric Laspeyres |

The Paasche row holds because $\sum_i s^1_i / r_i = \sum_i p_{i0} q_{i1} / \sum_i p_{i1} q_{i1}$, the reciprocal of Paasche. So Paasche is linear in its reciprocal, and Törnqvist is linear in logs. `test_family_reproduces_named_indexes` checks all three against the direct formulas. Fisher, $\sqrt{L\cdot P}$, is not of this form; see Remark 3.

**Two group indexes.**

- *Aggregate-share index:* $P^{\text{agg}}_G = \varphi\big(\sum_i S_{Gi}\, g(r_i)\big)$ with $S_G = \sum_h \omega_h s_h$.
- *Democratic mean:* $D_G = \sum_h \omega_h P_h$.

In matrix form, with $W$ the $H\times I$ share matrix, $A$ the $G\times H$ averaging matrix and $g(R)$ applied elementwise to the $I\times T$ relatives:

$$P^{\text{agg}} = \varphi\big((A W)\, g(R)\big), \qquad D = A\, \varphi\big(W\, g(R)\big).$$

## 2. Aggregate-share indexes commute with aggregation

**Proposition 1.** For every $g$, $P^{\text{agg}}_G = \varphi\big(\sum_h \omega_h\, \varphi^{-1}(P_h)\big)$. It can be computed from $S_G = (AW)_G$ alone, in $O(HI + GIT)$ time with sparse $A$, and the result is identical to computing $x_h$ for every household first.

*Proof.* By linearity of the inner sum,
$\sum_i S_{Gi}\, g(r_i) = \sum_i \sum_h \omega_h s_{hi}\, g(r_i) = \sum_h \omega_h x_h$,
and $x_h = \varphi^{-1}(P_h)$. Applying $\varphi$ to both sides gives the identity. In matrix form, $(AW)\,g(R) = A\,(W\,g(R))$ by associativity, so $\varphi$ of either side is the same. Computing $AW$ costs $O(\operatorname{nnz}(A)\, I) = O(HI)$, and multiplying the $G\times I$ result by $g(R)$ costs $O(GIT)$. $\square$

The same argument works for nested groups. The aggregate shares of a union of groups are the weighted average of the groups' aggregate shares, so building group indexes from aggregate shares is consistent in aggregation for every $\varphi$.

## 3. The democratic mean commutes only when φ is affine

**Proposition 2.** $D_G = P^{\text{agg}}_G$ for every group (all shares, weights and relatives) if and only if $\varphi$ is affine. When $\varphi$ is convex, $D_G \ge P^{\text{agg}}_G$. When it is concave, $D_G \le P^{\text{agg}}_G$.

*Proof.* ($\Leftarrow$) If $\varphi(x) = a + bx$, then $D_G = \sum_h \omega_h (a + b x_h) = a + b \sum_h \omega_h x_h = P^{\text{agg}}_G$.

($\Rightarrow$) Take two items and two households with $s_1 = (1,0)$ and $s_2 = (0,1)$, so $x_1 = g(r_1)$ and $x_2 = g(r_2)$. As $r_1$ and $r_2$ range over $(0,\infty)$, $x_1$ and $x_2$ range over the whole interval $J = g((0,\infty))$. With weights $(\lambda, 1-\lambda)$, equality says $\varphi(\lambda x_1 + (1-\lambda)x_2) = \lambda\varphi(x_1) + (1-\lambda)\varphi(x_2)$ for all $x_1, x_2 \in J$ and $\lambda \in [0,1]$. So $\varphi$ is both convex and concave on $J$, which means it is affine.

The inequalities are Jensen's inequality, $\sum_h \omega_h \varphi(x_h) \ge \varphi(\sum_h \omega_h x_h)$ for convex $\varphi$. $\square$

**Corollary.** Laspeyres ($\varphi(x)=x$) is the only index in the table whose democratic mean can be computed aggregate first. For Paasche ($\varphi(x) = 1/x$, convex on $x>0$) and for Törnqvist and geometric Laspeyres ($\varphi = \exp$, convex), the democratic mean is at least the aggregate-share index. In both cases the difference is an AM–HM or AM–GM gap of the household indexes:

- geometric: $D_G - P^{\text{agg}}_G = \text{AM}(P_h) - \text{GM}(P_h)$;
- harmonic: $D_G - P^{\text{agg}}_G = \text{AM}(P_h) - \text{HM}(P_h)$.

## 4. Bounding the gap

**Proposition 3.** Let $\bar x = \sum_h \omega_h x_h$, $V = \sum_h \omega_h (x_h - \bar x)^2$, and let $\varphi$ be twice differentiable on $[x_{\min}, x_{\max}]$ (the range of the $x_h$), with $m \le \varphi'' \le M$ there. Then

$$\tfrac12\, m\, V \;\le\; D_G - P^{\text{agg}}_G \;\le\; \tfrac12\, M\, V.$$

*Proof.* Taylor's theorem with the Lagrange remainder gives, for each $h$, some $\xi_h$ between $x_h$ and $\bar x$ with
$\varphi(x_h) = \varphi(\bar x) + \varphi'(\bar x)(x_h - \bar x) + \tfrac12 \varphi''(\xi_h)(x_h - \bar x)^2$.
Multiply by $\omega_h$ and sum. The linear term vanishes because $\sum_h \omega_h (x_h - \bar x) = 0$, leaving $D_G - \varphi(\bar x) = \tfrac12 \sum_h \omega_h \varphi''(\xi_h)(x_h-\bar x)^2$. Every $\xi_h$ lies in $[x_{\min}, x_{\max}]$, so $m \le \varphi''(\xi_h) \le M$. $\square$

For the two nonlinear cases:

- geometric ($\varphi = \exp$, $\varphi'' = e^x$ increasing): $\tfrac12 e^{x_{\min}} V \le \text{gap} \le \tfrac12 e^{x_{\max}} V$, with $V = \operatorname{Var}_\omega(\log P_h)$;
- harmonic ($\varphi = 1/x$, $\varphi'' = 2/x^3$ decreasing): $V / x_{\max}^3 \le \text{gap} \le V / x_{\min}^3$, with $V = \operatorname{Var}_\omega(1/P_h)$.

The second-order approximation $\tfrac12 \varphi''(\bar x)\, V$ lies between the bounds.

**The correction can also be computed aggregate first.** $V$ seems to require every $x_h$, but
$V = g(r)^\top M_G\, g(r) - (S_G^\top g(r))^2$, where $M_G = \sum_h \omega_h s_h s_h^\top = W^\top \operatorname{diag}(\omega) W$ restricted to group $G$.
So $P^{\text{agg}}_G + \tfrac12 \varphi''(\bar x) V$ needs only $S_G$ ($I$ numbers) and $M_G$ ($I^2$ numbers) per group. That costs $O(HI^2)$ once, then $O(GI^2)$ per price vector, instead of $O(HI)$ per price vector. This is worthwhile when there are many price vectors (a long $T$ or a large sensitivity grid) and $I$ is moderate. `test_variance_from_second_moments` checks the identity.

**Remark 3 (Fisher).** Fisher is $\sqrt{L_h P_h}$, not a quasi-arithmetic mean, so Proposition 2 does not apply directly. By AM–GM, $F_h \le (L_h + P_h)/2$, so the democratic mean of Fisher indexes is at most the average of the democratic Laspeyres and Paasche means. Its aggregate first form needs the current-period shares, which the prototype's single CE year does not provide.

## 5. Measured gap in the 2023 CE data

The prototype has one year of shares, so the measurement uses base-period shares for both nonlinear means: geometric Laspeyres (the form BLS uses inside CPI item strata) and the harmonic mean of relatives. Paasche and Törnqvist need a second year of shares and are planned for the thesis. Window: June 2021 → June 2022. Units: percentage points of inflation. Source: `python -m pie intrvw23`, section 6.

| Group | Geometric gap | Lower | Upper | Harmonic gap | Lower | Upper |
|---|---|---|---|---|---|---|
| Rural | 0.022 | 0.021 | 0.023 | 0.035 | 0.030 | 0.042 |
| Urban | 0.017 | 0.016 | 0.020 | 0.028 | 0.023 | 0.043 |
| Q1 (lowest income) | 0.026 | 0.024 | 0.031 | 0.043 | 0.035 | 0.066 |
| Q2 | 0.019 | 0.018 | 0.021 | 0.031 | 0.026 | 0.040 |
| Q3 | 0.017 | 0.016 | 0.019 | 0.027 | 0.022 | 0.037 |
| Q4 | 0.012 | 0.011 | 0.013 | 0.019 | 0.016 | 0.024 |
| Q5 (highest income) | 0.011 | 0.010 | 0.012 | 0.017 | 0.015 | 0.022 |

What the numbers show:

1. **The bounds hold in every group**, and the second-order term matches the exact gap to within 0.001 pp.
2. **The gap is small compared with the differences being studied.** It is at most 0.04 pp, while the rural–urban and Q1–Q5 differences in the same window are about 1 pp. For these groups, using the aggregate-share index in place of the democratic mean (to get O(HI + GIT) cost) changes group means by under 5% of the gaps of interest.
3. **The gap is not constant across groups.** It falls steadily from Q1 to Q5, because lower-income households' baskets differ more from each other ($V$ is larger). An aggregate-first shortcut therefore slightly *understates* the lowest quintile's mean index relative to the highest: by about 0.015 pp (geometric) to 0.026 pp (harmonic), or 1–2% of the measured Q1–Q5 gap of about 1 pp. The thesis should report this when it uses aggregate shares for nonlinear indexes.
4. **The gap grows with how far prices spread** (through $V$). It will be larger with ~250 CPI items than with 16 categories, and in windows such as 2021–22 when relative prices move a lot. The measurement should be repeated with the item-level concordance.
