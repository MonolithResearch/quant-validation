# How the validation statistics are calculated

A high Sharpe can arise from sampling error within one trial or from taking the largest of many noisy estimates. PSR addresses the first question. DSR changes the benchmark to account for the search. CSCV asks whether repeating the selection on different rows identifies a candidate that also ranks well on the remaining rows. [Worked example](../examples/selection_bias.py)

## Returns, moments and units

Supply aligned excess returns at one sampling frequency with the desired costs already deducted. For $n$ observations the report estimates per-period Sharpe as $s=\bar r/\hat\sigma_r$, using the return standard deviation with `ddof=0`. Write $g_3$ for skewness and $g_4$ for Pearson kurtosis (a Gaussian has $g_4=3$). `_moments` uses SciPy's biased central-moment estimators. `litigious_report` requires at least three observations and usable dispersion and moments for Sharpe inference. [Implementation](../monolith_validation/core.py)

Benchmarks and formula inputs below use per-period units. For $q$ periods per year the report displays Sharpe, its standard error and interval endpoints multiplied by $\sqrt q$. This scaling assumes uncorrelated returns with stable moments. Serial dependence changes the variance of aggregated returns. The package supplies no autocovariance adjustment. See `sharpe_ratio`, `litigious_report` and `test_sharpe_ratio_annualises`. [Core tests](../tests/test_core.py) · [Lo (2002), *The Statistics of Sharpe Ratios*](https://doi.org/10.2469/faj.v58.n4.2453)

## Probabilistic Sharpe ratio

For a benchmark $b$ fixed before selection, PSR gives the one-sided normal-approximation confidence that the population Sharpe exceeds $b$. With $\Phi$ the standard normal CDF, the moment adjustment, standard error and score are

$$
V(s)=1-g_3s+\frac{g_4-1}{4}s^2,\qquad
SE(s)=\sqrt{\frac{V(s)}{n-1}},\qquad
\mathrm{PSR}(b)=\Phi\left(\frac{s-b}{SE(s)}\right).
$$

The approximation needs finite, estimable moments through order four and enough effectively independent observations. Treating overlapping returns as independent can overstate confidence. PSR alone makes no allowance for selecting the winner.

`_sr_variance_term`, `sharpe_std_error` and `probabilistic_sharpe_ratio` implement these expressions. The latter two floor $V$ at $10^{-300}$ for numerical safety. In `test_psr_matches_standard_normal_cdf_clean_case` the standardised distance is 1.96 and the expected score is about 0.975. [Core tests](../tests/test_core.py) · [Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier*, §§2–3](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf)

## Deflated Sharpe ratio

Let $N$ be the number of trials and $\tau^2$ the variance of their **per-period Sharpe estimates**. This measures dispersion across candidates. Under a zero-mean null with independent, approximately normal trial Sharpes, the expected maximum for $N>1$ is approximated by

$$
b_N=\tau\left[(1-\gamma)\Phi^{-1}(1-1/N)
+\gamma\Phi^{-1}\left(1-\frac{1}{Ne}\right)\right],
\qquad \mathrm{DSR}=\mathrm{PSR}(b_N).
$$

Here $\gamma$ is Euler's constant. Greater trial dispersion raises the benchmark for a fixed multi-trial search. A larger search raises it at fixed positive dispersion. The approximation targets large $N$. `expected_max_sharpe` uses it for smaller searches too, evaluating the equivalent inverse survival functions. For one trial it returns zero. `deflated_sharpe_ratio` evaluates PSR at the resulting benchmark.

The report takes the largest of the declared count, the supplied Sharpe count and the number of matrix columns. It accepts an explicit per-period standard deviation or divides the sample standard deviation of the **annual** trial Sharpes (`ddof=1`) by $\sqrt q$. Multi-trial deflation requires positive dispersion. The code cannot infer an effective independent count for correlated variants or recover omitted trials. `test_expected_max_sharpe_two_trials_anchor`, `test_expected_max_sharpe_linear_in_std` and `test_dsr_matches_authors_published_numerical_example` exercise the benchmark, scaling and published DSR example. [Core tests](../tests/test_core.py) · [Report tests](../tests/test_gate_evidence.py) · [Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Eq. 2 and Appendix A.1](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

## Minimum track record length

Hold $s$, $g_3$ and $g_4$ fixed. For $s>b$ and a one-sided confidence level $p>1/2$, solving the PSR equation for the number of observations gives

$$
n_{\min}=1+V(s)\left(\frac{\Phi^{-1}(p)}{s-b}\right)^2.
$$

`min_track_record_length` returns the fractional threshold and floors $V$ at zero. Round up to count whole observations. It returns infinity when $s\le b$. The report uses $p=0.95$ and divides the result by $q$ for years. Additional observations can change every estimated moment, so MinTRL cannot predict when a strategy will pass. `test_min_track_record_length_known` checks a finite threshold and `test_min_track_record_length_infinite_below_benchmark` checks the boundary. [Core tests](../tests/test_core.py) · [Bailey & López de Prado (2012), §4](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf)

## Sharpe confidence interval

For central coverage $p$, `sharpe_confidence_interval` uses the same moment-adjusted standard error as PSR and a **two-sided** normal quantile.

$$
\left[s-zSE(s),\,s+zSE(s)\right],\qquad z=\Phi^{-1}\left((1+p)/2\right).
$$

This interval describes sampling uncertainty around the Sharpe of a specified return series. Post-selection coverage and future returns require separate evidence. With $SE=0.1$ and $s=0.3$, `test_sharpe_confidence_interval_95` checks endpoints near 0.104 and 0.496. The Gaussian special case has $V(s)=1+s^2/2$. [Core tests](../tests/test_core.py) · [Bailey & López de Prado (2012), §2.5](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf)

## Probability of backtest overfitting via CSCV

Start from a finite $T\times M$ matrix of aligned returns, one column per candidate in the search. Split time into an even number $S$ of contiguous blocks. For each of the $\binom{S}{S/2}$ choices of training blocks, select the highest training Sharpe and rank **that same column** among all candidates on the complementary blocks. Columns must be fixed before ranking, and block length must suit the dependence in the data.

For split $c$, let $r_c$ be the winner's ascending test rank, with 1 the worst. The code computes

$$
\omega_c=\frac{r_c}{M+1},\qquad
\lambda_c=\log\frac{\omega_c}{1-\omega_c},\qquad
\widehat{\mathrm{PBO}}=\frac{1}{\binom{S}{S/2}}\sum_c \mathbf{1}_{\{\lambda_c<0\}}.
$$

`probability_of_backtest_overfitting` uses `_column_sharpe` on both halves. Before taking logs the code clips the relative rank to the interval $[10^{-12}, 1-10^{-12}]$. The paper specifies equal-sized blocks. The implementation achieves this by dropping trailing remainder rows, so use a divisible row count to retain every observation. Tied training winners resolve to the first column. A stable sort assigns distinct ranks to tied test values in column order. An exact median rank counts as zero failures. Ties can therefore make the result depend on column order.

PBO measures relative rank on these complementary blocks. It cannot give the probability of an absolute trading loss or substitute for a subsequent chronological holdout. `test_pbo_dominant_strategy_is_zero` and `test_pbo_adversarial_strategy_is_one` check both extremes using deliberately constructed matrices. [Core tests](../tests/test_core.py) · [Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*, §§2–3](https://escholarship.org/uc/item/4w1110bb)
