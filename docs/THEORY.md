# Certified mode: statement and proof

This note states exactly what certified mode guarantees, under which
assumptions, and why. The code is in `erratica/learning/egate.py` and
`erratica/learning/evalstream.py`.

The individual tools are known results. Betting e-processes for bounded
means are due to Waudby-Smith and Ramdas; Ville's inequality is classical;
the budget recycling rule is the online fallback procedure of Tian and
Ramdas. What is assembled here is their combination into a gate for a
system that modifies itself indefinitely, with the bookkeeping that makes
the combination valid.

## 1. Setting

A *system* maps a prompt to an answer. `M_0` is the starting system. At any
time one system is *deployed*; it starts as `M_0`.

There is a finite set `B` of *protected behaviours*. Behaviour `b` has a
distribution `D_b` over evaluation items and a scorer `r_b(item, answer)`
with values in `[0, 1]`. Write `mu_b(M)` for the mean of `r_b` when `M`
answers items drawn from `D_b`. A route may also have a *task* `t`, with
`D_t`, `r_t`, and `mu_t` defined the same way.

Candidates arrive one at a time. Candidate `k` (the k-th to enter the gate)
proposes replacing the deployed system `I_k` (the *incumbent*) by `C_k`. How
candidates are produced is unrestricted: they may depend on all earlier
data, on earlier gate decisions, and on each other.

The gate tests candidate `k` on a subset `B_k` of `B`, the behaviours it
could affect, and on its route's task if one is registered.

**Bad candidates.** Fix a margin `eps` in `(0, 1)`. Candidate `k` is *bad*
if at least one of these holds:

- (harm) `mu_b(C_k) <= mu_b(M_0) - eps` for some `b` in `B_k`;
- (no gain) a task test is required and `mu_t(C_k) <= mu_t(I_k)`.

Whether a candidate is bad is fixed before its test starts.

## 2. Assumptions

- **A1 (bounded scores).** Every score lies in `[0, 1]`.
- **A2 (fresh samples).** The items used to test candidate `k` are drawn
  independently from the relevant `D_b` or `D_t`, independently of
  everything that happened before the test started, including `C_k`
  itself. The implementation gets this by using every item exactly once.
- **A3 (what is tested is what ships).** If candidate `k` is promoted, the
  deployed system becomes exactly `C_k`.
- **A4 (locality).** For `b` not in `B_k`, `C_k` and `I_k` give the same
  answers on `D_b`. In the implementation a patch is scoped to one route
  and the model is run deterministically, so behaviours on other routes
  see an identical prompt.

Nothing is assumed about dependence between different candidates' tests,
about how candidates are generated, or about the answers being
deterministic within a test.

## 3. The procedure

**Budget.** Fix `delta` in `(0, 1)` and positive numbers `alpha_1,
alpha_2, ...` with sum at most `delta`. The default is

    alpha_k = delta / (C (k+1) ln^2(k+1)),   C = 2.11 >= sum_{n>=2} 1/(n ln^2 n).

**Level.** Candidate `k` is tested at level

    a_k = alpha_k + a_{k-1} * 1{candidate k-1 was promoted},   a_0 = 0.

**Tests.** For each `b` in `B_k`, items `x_1, x_2, ...` are drawn and

    Z_i = r_b(x_i, C_k(x_i)) - r_b(x_i, M_0(x_i)) + eps,   in [eps - 1, eps + 1].

For the task, `Z_i = r_t(x_i, C_k(x_i)) - r_t(x_i, I_k(x_i))`, in `[-1, 1]`.
Each test maintains a *wealth* process

    W_n = (1/2) * (1/K) * sum_j prod_{i<=n} (1 + lam_j Z_i)
        + (1/2) * prod_{i<=n} (1 + L_i Z_i),

where `lam_1 > ... > lam_K > 0` is a fixed grid with `lam_1 = 0.9 / |lower
end of the range|`, and `L_i` is one of the grid values chosen from `Z_1,
..., Z_{i-1}` only.

**Decision.** Candidate `k` is promoted if and only if every one of its
tests reaches `W_n >= 1 / a_k` at some `n`. Sampling may stop at any time
for any reason; in particular a test is abandoned (and the candidate
rejected) at a sample cap, or when a mirror-image test finds evidence that
the candidate is worse than its reference.

## 4. Main result

**Theorem 1.** Under A1 to A3, with probability at least `1 - delta`, no bad
candidate is ever promoted.

**Corollary 2.** Under A1 to A4, with probability at least `1 - delta`,
simultaneously for all times:

1. for every protected behaviour `b`, the deployed system `M` satisfies
   `mu_b(M) > mu_b(M_0) - eps`;
2. on every route with a task, each promotion strictly raised `mu_t` over
   the system it replaced, so `mu_t` of the deployed system is
   nondecreasing in time and never below `mu_t(M_0)`.

### Proof of Theorem 1

*Step 1: one test is valid.* Let candidate `k` be bad, and let `T` be a
test whose null hypothesis is true for it (the harmed behaviour, or the
task). By A1 the `Z_i` are bounded as stated, and by A2, conditionally on
the past `F_{i-1}` (everything before the test, plus `Z_1, ..., Z_{i-1}`),
`E[Z_i | F_{i-1}] <= 0`. For a fixed `lam` in the grid,

    E[1 + lam Z_i | F_{i-1}] <= 1   and   1 + lam Z_i >= 0.1 > 0,

so `prod (1 + lam Z_i)` is a nonnegative supermartingale started at 1. The
same holds with `lam` replaced by the predictable `L_i`. `W_n` is a convex
combination of such processes, hence also a nonnegative supermartingale
with `W_0 = 1`. By Ville's inequality, for any `a` that is determined
before the test starts,

    P( W_n >= 1/a for some n | F_before ) <= a.

Stopping early, for any reason, only removes outcomes from this event.

*Step 2: all tests must pass.* A bad candidate is promoted only if the
test `T` of Step 1 crosses `1 / a_k`. Since `a_k` is determined before the
test starts,

    P( bad candidate k is promoted | F_before ) <= a_k.         (1)

No correction for the number of tests per candidate is needed.

*Step 3: the sequence.* Let `tau` be the index of the first bad candidate
that is promoted (infinite if none). Then

    P(tau < infinity) = sum_k P(tau = k)
                      = sum_k E[ 1{no bad promotion before k} 1{k is bad} P(k promoted | F_before) ]
                     <= E[ sum_k 1{no bad promotion before k} 1{k is bad} a_k ]      by (1).

It remains to bound the sum inside the expectation on every outcome. Let
`j_1 < j_2 < ...` be the bad candidates `j` with no bad promotion before
`j`. By construction none of them except possibly the last was promoted.
Unrolling the recursion,

    a_j = alpha_j + alpha_{j-1} + ... + alpha_{j-m},

where `j-1, ..., j-m` is the maximal run of consecutive promoted candidates
immediately before `j`. For `j = j_i`, every candidate in that run is good
(a promoted bad one would be an earlier bad promotion), and the run cannot
reach back to `j_{i-1}`, which was not promoted. So the index sets
`{j_i - m_i, ..., j_i}` are disjoint, and

    sum_i a_{j_i} <= sum_k alpha_k <= delta.

Hence `P(tau < infinity) <= delta`. QED.

### Proof of Corollary 2

On the event of Theorem 1, every promoted candidate is good. Let `M` be
deployed at some time, and `b` a protected behaviour. If no promoted
candidate ever tested `b`, then by A4 the answers on `D_b` never changed
and `mu_b(M) = mu_b(M_0)`. Otherwise let `k` be the last promoted candidate
with `b` in `B_k`. It was good, so `mu_b(C_k) > mu_b(M_0) - eps`; by A3 the
deployed system became `C_k`, and by A4 no later promotion changed the
answers on `D_b`. Either way (1) holds. (2) is the "no gain" clause applied
to each promotion.

A rollback of the most recent promotion on a route returns that route to a
system that was itself deployed earlier, for which (1) and (2) already
held. Removing an older patch would produce a system that was never tested
as a whole, which is why certified mode refuses it.

**Why the anchor is `M_0`.** Comparing each candidate with its incumbent
instead would allow a loss of nearly `eps` at every step, and the losses
would add up. Comparing with `M_0` makes (1) a statement about the deployed
system directly, with no accumulation over any number of patches.

**Why the margin cannot be zero.** With `eps = 0`, a candidate whose
answers on `D_b` are unchanged has `Z_i = 0` always, so `W_n = 1` forever
and it can never be promoted.

## 5. Cost

**Proposition 3 (how close the bet is to the best one).** For any data
sequence, `log W_n >= max_j sum_i log(1 + lam_j Z_i) - log(2K)`. The bettor
is never more than `log(2K)` behind the best fixed grid bet chosen in
hindsight.

*Proof.* `W_n` is at least half the mixture, and the mixture is at least
`1/K` times its largest term. QED.

**Proposition 4 (the price of a harmless patch).** Suppose a candidate
leaves `b` exactly unchanged, so the paired difference is always 0.

(i) Any test that controls the probability of promoting a candidate with
`mu_b(C) <= mu_b(M_0) - eps` at level `a`, for all score distributions on
`[0, 1]`, and that promotes this candidate with probability one, needs on
average at least

    log(1/a) / log(1 / (1 - eps))

samples.

(ii) The test above crosses after at most
`(log(1/a) + log(2K)) / log(1 + 0.9 eps / (1 - eps))` samples.

For small `eps` both are about `log(1/a) / eps`; the ratio of the rates is
about `1 / 0.9`.

*Proof.* (i) The data distribution is a point mass at difference 0. The
closest distribution of differences on `[-1, 1]` with mean at most `-eps`,
in Kullback-Leibler divergence from the point mass, puts mass `1 - eps` on
0 and `eps` on `-1`, at divergence `log(1/(1 - eps))`. The standard
change-of-measure bound for sequential tests gives `E[N] >= log(1/a)`
divided by that divergence. (ii) With `Z_i = eps` always, the largest grid
bet multiplies wealth by `1 + 0.9 eps / (1 - eps)` per sample; apply
Proposition 3. QED.

**The trade-off.** With the default budget, `log(1/alpha_k) = log(1/delta)
+ log k + O(log log k)`. Certifying candidate `k` therefore costs about

    (log(1/delta) + log k) / g

samples per behaviour, where the growth rate `g` is about `eps` for a quiet
behaviour and about `(eps + shift)^2 / (2 variance)` for one whose paired
differences are noisy with mean `shift`. A budget of `N` fresh samples
certifies roughly `N g / (log(1/delta) + log K)` patches. Safety (`delta`),
tolerance (`eps`), and the number of patches a sample budget buys trade
against each other, and by (i) no valid method escapes that for bounded
scores. `expected_samples` computes these numbers.

**Proposition 5 (early stopping costs little power).** Let the mirror test
for behaviour `b` be a wealth process for the null "`mu_b(C) >= mu_b(M_0)`"
and stop the candidate when it reaches `1/beta`. A candidate that is truly
no worse than baseline on `b` is stopped this way with probability at most
`beta`. The same holds for the task with "no worse than the incumbent".

*Proof.* Ville's inequality for the mirror process. Validity of the main
tests is unaffected because stopping is always allowed. QED.

## 6. After certification

**Proposition 6 (drift alarm).** For each `b`, run one wealth process
forever on `Y_i = r_b(x_i, M_0(x_i)) - r_b(x_i, M(x_i)) - eps`, where `M`
is whatever is deployed when item `i` is drawn, and raise an alarm when it
reaches `|B| / beta`. If at every time the deployed system satisfies
`mu_b(M) >= mu_b(M_0) - eps`, the probability that any alarm is ever
raised is at most `beta`.

*Proof.* Under that condition `E[Y_i | past] <= 0` at every step even
though `M` changes over time, so each process is a nonnegative
supermartingale; apply Ville and a union bound over `B`. QED.

The alarm exists because the assumptions can stop holding: a model file is
replaced, a scorer is edited, the evaluation stream drifts away from real
use. The certificate cannot know. The alarm can.

**Checking a certificate.** A certificate stores `k`, `delta`, the level
`a_k` and how much of it was inherited, the margin, and the raw paired
differences of every test in order. `verify_certificate` recomputes each
wealth process from those differences and checks the decision.
`verify_chain` also checks that `a_k - alpha_k` never exceeds the level of
a promoted immediate predecessor. Anyone with the database can run both.

## 7. Reusing evaluation samples

Assumption A2 makes every candidate pay for its own fresh samples and its
own baseline answers. Most candidates are rejected, so most of that is
spent on candidates that change nothing. This section shows the samples can
be shared.

**Procedure.** Evaluation proceeds in *batches*. A batch is an i.i.d.
sequence of items, extended on demand. Every candidate tested on a batch
reads its items in the same order from the start. A batch is retired when a
candidate tested on it is promoted (or earlier, at will); the next
candidate starts batch `e + 1` with new items. The baseline's and the
incumbent's answers to a batch's items are computed once and shared, which
is sound because neither changes while the batch is in use.

The j-th candidate tested on batch `e` has own budget

    alpha_{e,j} = delta * w_e * g_j,   sum_e w_e <= 1,   sum_j g_j <= 1,

and is tested at level `a_{e,j} = alpha_{e,j}`, plus, for `j = 1` only, the
level of the immediately preceding candidate if that one was promoted. The
implementation uses `w_e = 1 / (C (e+1) ln^2(e+1))` and
`g_j = 1 / (j (j+1))`.

- **A2' (sealed rejections).** A batch is independent of everything that
  happened before it was started. While it is in use, whatever produces
  candidates may depend on it only through the fact that the earlier
  candidates tested on it were rejected.

A2' is a protocol requirement. It is why certified mode with reuse seals
the certificate of a rejected test (which behaviour failed, after how many
samples, with what scores) until the batch is retired. Everything else the
patch writer normally uses, such as recorded feedback, is independent of
the batch and unaffected.

**Theorem 7.** Under A1, A2', and A3, with probability at least
`1 - delta` no bad candidate is ever promoted.

*Proof.* Fix batch `e` and let `G_e` be everything known when it starts.
Because the generator sees only "rejected" from the batch, the candidate
it proposes after `j - 1` rejections on batch `e` is a function of `G_e`
and of randomness independent of the batch. Call it the *phantom*
candidate `C'_{e,j}`: it is what would be proposed at position `j` if every
earlier candidate on the batch were rejected. The whole phantom list
`C'_{e,1}, C'_{e,2}, ...` is independent of the batch given `G_e`, and its
levels `a_{e,1}, a_{e,2}, ...` are known given `G_e`.

A real candidate is tested at position `j` of batch `e` only if the
`j - 1` before it were rejected, and then it *is* `C'_{e,j}`. So

    {a bad candidate is promoted at position j of batch e}
        is contained in  {C'_{e,j} is bad and its test on batch e crosses 1 / a_{e,j}}.

`C'_{e,j}` is independent of the batch, so Step 1 of Theorem 1 applies to
it: the second event has conditional probability at most `a_{e,j}` given
`G_e`. Nothing is needed about how the tests of different phantoms on the
same batch depend on each other.

Let `tau` be the first bad promotion. Summing over batches and positions,

    P(tau < infinity) <= E[ sum_e 1{no bad promotion before batch e}
                              sum_j 1{C'_{e,j} bad} a_{e,j} ].

Bound the double sum on every outcome. The own budgets `alpha_{e,j}` over
all pairs `(e, j)` sum to at most `delta`. Inherited budget appears only at
`j = 1`, for a batch whose first phantom is bad, and consists of the own
budgets of a run of consecutive promoted candidates ending just before
batch `e`. Those candidates are real, were promoted with no bad promotion
before them, and are therefore good, so their own budgets are not counted
anywhere as bad phantoms. Two such runs cannot overlap, by the argument of
Theorem 1: a batch whose first candidate is bad and not promoted contains
at least one more candidate, and its eventual promoted candidate, at
position two or later, inherits nothing, which ends any run passing through
it. Every own budget is therefore counted at most once, and the total is
at most `delta`. QED.

**What it buys.** Without reuse, a run of `r` rejected candidates followed
by a promoted one draws `n_1 + ... + n_{r+1}` items and as many answers
from each reference system. With reuse it draws `max_i n_i` items and
reference answers. Candidate answers are not saved: each candidate must
still answer every item it is tested on.

**What it costs.** The two-level budget replaces `log k` in the required
evidence by `log e + log j` plus lower-order terms, the same order since
`k` is about `e` times the typical `j`; the first candidate pays `log 2`
more than before. And the details of a rejection are not available until
the batch is retired. Retiring early is always allowed and always safe; it
only forgoes the saving.

**What it does not allow.** Reusing a batch after a promotion would let
candidates depend on the batch through *which* candidate was promoted. The
same argument then needs a union over all positions where the promotion
could have happened, a penalty that grows with the batch's length. Retiring
at promotion avoids the penalty altogether.

## 8. What is not covered

- **The scorer.** Everything is a statement about `r_b`. If `r_b` rewards
  the wrong thing, a certified system can be worse in ways no test saw.
- **The evaluation distribution.** `D_b` is your evaluation stream. If real
  use looks different, the guarantee is about the stream.
- **Secrecy of items.** A2 fails if whatever writes patches can see items
  before they are used. Single use protects against reuse, not against
  leaks. With batch reuse (A2') the same goes for the sealed details of
  rejected tests, and side channels such as how long a test took are
  outside the model.
- **Behaviours nobody listed.** Only behaviours in `B` are protected.
- **Losses smaller than the margin.** A promoted patch may lower a
  protected mean by less than `eps`. The loss does not accumulate, but it
  is allowed.
- **Fine-tunes.** A weight update changes every route at once, so A4 gives
  no saving, and scores on the evaluation stream bound nothing off it.
  Weight cycles are refused in certified mode rather than half-covered.
- **Experience recall.** Recalled corrections change answers without a
  test, so certified mode requires recall to be off.
- **Time-dependent tools.** A learned tool that behaves differently on a
  later date violates the fixed-distribution part of A2. The drift alarm is
  the only defence.
