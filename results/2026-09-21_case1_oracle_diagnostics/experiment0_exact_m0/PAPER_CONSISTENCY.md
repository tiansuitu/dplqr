# Consistency check vs Zhong & Wang (2024 JBES)

Source: `Zhong and Wang 2024 JBES.pdf`, Section 5.1 Simulation I (Homoscedastic Errors), Case 1 (linear). Supplement checked for extra DGP detail (none beyond main text tables).

## Matches the paper

| Item | Paper (Sim I / Case 1) | Experiment 0 |
|------|------------------------|--------------|
| Copula | Gaussian copula on [0,2], correlation parameter 0.5 | Equicorrelated latent Gaussians with ρ=0.5, then \(\tilde Z_j=2\Phi(G_j)\) (standard Gaussian-copula construction) |
| Margins | Each \(\tilde Z_j\sim\mathrm{Unif}[0,2]\) | Same |
| \(Z\) | \((\tilde Z_1,\ldots,\tilde Z_8)^\top\) | Same |
| \(X\) | \(X_1=I(\tilde Z_9>1)\), \(X_2=\tilde Z_{10}\) | Same (\(X_1=I(G_9>0)\)) |
| \(\theta\) | \((1,-1)^\top\) | Same |
| Errors | Student-\(t_3\), mean 0, independent of \((X,Z)\) | Same |
| \(m\) (Case 1) | \(m(z)=0.56\sum_{k=1}^8 z_k\) | Same |
| Sample sizes | \(n\in\{1000,2000\}\) | Same |
| Homoscedasticity | Additive error model (15) | Same |
| Asymptotic variance (homo) | Cor. 3.1: \(V=\tau(1-\tau)\,[\mathrm{Var}\{X-E(X\mid Z)\}]^{-1}/f(0)^2\) | Oracle sandwich \(\Sigma_2^{-1}\Sigma_1\Sigma_2^{-1}\) with \(\Sigma_1=\tau(1-\tau)\widehat{\mathrm{E}}[\tilde X\tilde X^\top]\), \(\Sigma_2=f_0\widehat{\mathrm{E}}[\tilde X\tilde X^\top]\) — algebraically the same |

## Intentional / diagnostic differences (not paper bugs)

| Item | Paper | Experiment 0 | Notes |
|------|-------|--------------|-------|
| Replications \(Q\) | 200 | **50** | Specified for this diagnostic pack |
| Quantiles | \(\tau\in\{0.25,0.5,0.75\}\) | **\(\tau=0.5\) only** | Spec for Exp 0 |
| Nuisance \(m\) | Estimated (LQR / PLAQR / DPLQR) | **Exact \(m_0\) fed in** | Oracle benchmark |
| \(f(0)\) | Estimated from residuals (R `density`) | **Exact** \(f_0=2/(\pi\sqrt{3})\) | Oracle |
| \(\varphi^*\) | Estimated by NN projection | **Analytic** \(E[X\mid Z]\) via latent \(G_{1:8}\) | Oracle; paper does not give closed form |
| Train/val split | 80:20 for NN tuning | **Not used** | No NN in Exp 0 |
| Solver | Adam (DPLQR) / QR methods | **`linprog` L1 / median LP** | Global convex optimum for known offset |
| Extra sim features | Test set \(N=5000\), RMSE/ERE for \(\hat m\) | Not applicable | Exp 0 does not estimate \(m\) |

## No inconsistencies found in the DGP coding relative to Section 5.1 Case 1

The latent equicorrelation + \(2\Phi(\cdot)\) map is the natural reading of “Gaussian copula on [0,2] with correlation parameter 0.5.” The paper does not spell out the latent correlation matrix beyond the single parameter 0.5; equicorrelation is the standard choice and matches prior dplqr replications.

Paper Table 1 LQR Case 1 \(\tau=0.5\) biases for \(\hat\theta_1\) are small (order \(10^{-2}\)) with SD \(\approx 0.11\) (\(n=1000\)) and \(\approx 0.08\) (\(n=2000\)). Experiment 0 is an *oracle* median regression with known \(m_0\), so Monte Carlo SD of \(\hat\theta\) should be **no larger** than paper LQR (often similar or slightly smaller). Coverage in paper Table 2 (LQR, Case 1, \(\tau=0.5\)) is 0.975 / 0.945 for \(n=1000/2000\); Exp 0 oracle Wald coverage should be near 95%.
