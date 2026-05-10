# References used in the MAB project

This file lists the scientific references used to justify the formulas, algorithms, and simulations implemented in the project.

## Robbins (1952)

**Reference:** Robbins, H. (1952). *Some aspects of the sequential design of experiments*. Bulletin of the American Mathematical Society.

**Used for:** historical introduction to sequential experimental design and allocation problems.

**Why it matters here:** the multi-armed bandit problem can be understood as a sequential allocation problem. At each step, the learner must decide where to allocate the next observation.

## Robbins and Monro (1951)

**Reference:** Robbins, H., & Monro, S. (1951). *A stochastic approximation method*. Annals of Mathematical Statistics.

**Used for:** online empirical mean updates and stochastic approximation.

**Why it matters here:** in the simulations, empirical means are updated sequentially after each observed reward. This supports the incremental update formula:

\[
\hat{\mu}_a(t+1)
=
\hat{\mu}_a(t)
+
\frac{X_{a,t+1} - \hat{\mu}_a(t)}{N_a(t+1)}.
\]

## Lai and Robbins (1985)

**Reference:** Lai, T. L., & Robbins, H. (1985). *Asymptotically efficient adaptive allocation rules*. Advances in Applied Mathematics.

**Used for:** regret lower bounds and asymptotically efficient adaptive allocation.

**Why it matters here:** this paper explains why regret is a central theoretical object in stochastic bandits. It motivates the search for adaptive policies that do not waste too many samples on suboptimal arms.

## Auer, Cesa-Bianchi and Fischer (2002)

**Reference:** Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*. Machine Learning.

**Used for:** UCB and finite-time regret analysis.

**Why it matters here:** this is the main reference for the UCB algorithm implemented in the project. The UCB score used in the simulation is:

$\[
UCB_a(t)
=
\hat{\mu}_a(t)
+
\sqrt{\frac{2\log(t)}{N_a(t)}}.
\]

The idea is optimism under uncertainty: an arm is selected either because it has a high empirical mean or because it remains uncertain.

## Bubeck and Cesa-Bianchi (2012)

**Reference:** Bubeck, S., & Cesa-Bianchi, N. (2012). *Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems*. Foundations and Trends in Machine Learning.

**Used for:** modern synthesis of regret analysis in stochastic and nonstochastic bandits.

**Why it matters here:** this survey is used to structure the general explanation of bandits, regret, UCB, and the exploration-exploitation trade-off.