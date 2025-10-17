# On Imperfection, Normality, and Why You're Okay

The **Central Limit Theorem** is one of those mathematical results that feels almost magical when you first encounter it properly. Not the mechanical definition you memorise for exams, but what it actually means about the world and about us.

Individual chaos and imperfection, when aggregated, produces order and normality. The flaws don't disappear—they average out into something stable, predictable, beautiful. This theorem is quite close to my heart because it is one of the reasons why I choose to deeply study the subject, and keep myself engrossed with it till date (despite the obvious occupational hazard of crumbling self-worth, but that’s okay, I guess).

---

## The Formal Proof

The formal proof of the Central Limit Theorem feels almost anti-climatic given how profound its implications are, nevertheless, here it is:

**Given:**

* Random variables $Y_1, Y_2, \ldots$ from some distribution (any distribution, really—that's the beauty of it)
* Common mean $\mu$ and finite variance $\sigma^2$
* These variables are independent

**The Question:** What happens to the average $\bar{X}*n = \frac{1}{n}\sum*{i=1}^{n} Y_i$ as $n$ gets large?

The answer is the **Central Limit Theorem**. Let me work through the proof using **moment generating functions**. It's a bit involved, but stay with me—the machinery matters.

### The Proof

First, we standardise. Define $Y_i^* = (Y_i - \mu)/\sigma$, so each has mean 0 and variance 1. Then:

$$
\frac{\sqrt{n}(\bar{X}*n - \mu)}{\sigma} = \frac{1}{\sqrt{n}}\sum*{i=1}^{n} Y_i^*
$$

Let $M_Y(t)$ denote the moment generating function of $Y^*$. From the properties of MGFs:

$$
M_{\sqrt{n}(\bar{X}*n-\mu)/\sigma}(t) = M*{\sum_{i=1}^{n} Y_i^*/\sqrt{n}}(t) = M_{Y^*}\left(\frac{t}{\sqrt{n}}\right)^n
$$

Now here's where it gets interesting. We expand $M_Y(t/\sqrt{n})$ in a Taylor series around 0:

$$
M_Y\left(\frac{t}{\sqrt{n}}\right) = \sum_{k=0}^{\infty} M_Y^{(k)}(0) \frac{(t/\sqrt{n})^k}{k!}
$$

where $M_Y^{(k)}(0) = \frac{d^k}{dt^k}M_Y(t)|_{t=0}$. Since $Y^*$ has mean 0 and variance 1:

* $M_Y^{(0)}(0) = 1$
* $M_Y^{(1)}(0) = 0$
* $M_Y^{(2)}(0) = 1$

So:

$$
M_Y\left(\frac{t}{\sqrt{n}}\right) = 1 + \frac{(t/\sqrt{n})^2}{2!} + R_Y\left(\frac{t}{\sqrt{n}}\right)
$$

where $R_Y$ is the remainder term. An application of Taylor's Theorem shows that for fixed $t \neq 0$:

$$
\lim_{n \to \infty} \frac{R_Y(t/\sqrt{n})}{(t/\sqrt{n})^2} = 0
$$

Since $t$ is fixed, we also have:

$$
\lim_{n \to \infty} \frac{R_Y(t/\sqrt{n})}{(1/\sqrt{n})^2} = \lim_{n \to \infty} nR_Y\left(\frac{t}{\sqrt{n}}\right) = 0
$$

Therefore:

$$
\lim_{n \to \infty} \left(M_Y\left(\frac{t}{\sqrt{n}}\right)\right)^n = \lim_{n \to \infty} \left[1 + \frac{1}{n}\left(\frac{t^2}{2} + nR_Y\left(\frac{t}{\sqrt{n}}\right)\right)\right]^n
$$

$$
= \lim_{n \to \infty} \left[1 + \frac{1}{n}\left(\frac{t^2}{2}\right)\right]^n = e^{t^2/2}
$$

And $e^{t^2/2}$ is precisely the moment generating function (MGF) of a $N(0,1)$ random variable.

**The theorem is proved.**

---

## What the Mathematics Is Actually Saying

We started with individual observations that could follow any distribution—uniform, exponential, some bizarre bimodal monstrosity, it doesn't matter. Each $Y_i$ is its own chaotic, unpredictable thing.

But then we averaged them. And in that averaging, something extraordinary occurred. All the idiosyncrasies, all the particular features that made each distribution unique, all the sharp edges and strange shapes, they washed away. By the time we reached the limit, all that remained was $e^{t^2/2}$, the MGF of the normal distribution.

The chaos collapsed into normality. Through aggregation.

Individual imperfection, when summed, becomes collective normality. Your particular flaws don't determine the aggregate outcome. In the limit, they vanish entirely.

---

## On Imperfection and (Ab)normality

Why is the normal distribution everywhere? Heights, test scores, measurement errors and pretty much all these disparate phenomena follow roughly the same bell curve. The answer is almost anticlimactic: because so many real-world phenomena are sums of many small, independent effects.

Your height isn't determined by a single gene. It's the cumulative result of thousands of genetic variants, nutritional factors, hormonal influences, random cellular events. Each factor contributes a small amount, and when you sum them all... normal distribution.

Measurement errors? Slight variations in temperature, tiny vibrations, imperfect calibration, observer reaction time, a hundred other factors. Sum of many small effects. Normal distribution.

The normal distribution isn't normal because the world is fundamentally Gaussian. It's normal because the world is fundamentally **additive**. We are sums of influences, products of accumulated small effects, and the mathematics of summation inevitably produces normality.

This has implications for how we think about ourselves. When you focus on your particular flaws and shortcomings, you're seeing individual random variables—the specific ways you deviate from some imagined ideal. But you're not an individual random variable. You're a sum of thousands of influences, experiences, genetic factors, environmental conditions, random events, and deliberate choices.

The mathematics says: sums converge to normal. Your particular configuration of flaws, when you step back and see yourself in the aggregate, places you somewhere on a distribution that encompasses everyone. You're not an outlier. You can't be, because the human distribution is itself the result of summing countless individual variations.

To be fair, the theorem doesn't say *you're perfect*. It says something more profound: you're **normal**, in the most literal mathematical sense. And normal is exactly where the bulk of the distribution lives.

---

## Literature and Philosophy: Variance and Humanity

Dostoevsky's novels are full of characters who try to eliminate their variance—to become perfectly rational, perfectly good, perfectly controlled. Raskolnikov tries to be a superman above moral law and ends up a murderer haunted by guilt. The Grand Inquisitor tries to perfect humanity and ends up imprisoning Christ himself. Prince Myshkin believes he can be perfectly good and ends up causing more harm than help, unable to function in a world that requires accepting one's own capacity for imperfection.

They all attempt to reach **zero variance**, and in doing so, they lose the very thing that makes love and change and growth possible. They become isolated, brittle, and destructive.

Meanwhile, it's the flawed characters who carry the capacity for redemption. Sonya the prostitute, Father Zosima with his mysterious past, Dmitri Karamazov with his wild passions—they've accepted their own imperfection. They're not trying to transcend the human condition; they're trying to live fully within it. They understood that you cannot eliminate variance and imperfections and still have something worth calling human. You need the variance. Not infinite, not zero, but **real**. Bounded but present.

Endless pursuit of perfection, of life without any change or variance, makes it rather bleak and dull. Which is exactly what happens to Grand in *Camus's The Plague*, who spends the entire novel trying to write the perfect first sentence. He rewrites it endlessly, changing a word here, adjusting the rhythm there, never satisfied, never moving forward.

> "Evenings, whole weeks, spent on one word, just think! Sometimes on a mere conjunction!"

His first sentence will never be perfect because perfection is a moving target, a phantom. And in chasing it, he never writes the second sentence. He refuses to aggregate, refuses to accept that the individual observation is flawed, insists that perfection must exist at the level of the single data point. And so he produces nothing.

In the final episode of *BoJack Horseman*, after six seasons of watching someone drown in self-destruction, unable to forgive himself for the ways he's hurt everyone around him, there's a moment on a rooftop where Diane says something simple:

> "It's going to be okay."

Not *you're perfect.* Not *you didn't do those terrible things.* Not even *you're forgiven.* Just: *it's going to be okay.*

---

## Asymptotic Nature of the Proof

The theorem is **asymptotic**. It tells you what happens as $n \to \infty$, but you never actually get there. In practice, you're always working with finite $n$.

For small $n$, the individual character of the distribution matters a lot. The convergence is slow. When you're young, when you have fewer experiences to average over, each individual event weighs heavily. Every failure feels catastrophic because you don't yet have enough data points to see the larger pattern.

But as you accumulate more experiences, more observations, more chances to see yourself in different contexts—the averaging starts to work. The pattern begins to emerge. Individual fluctuations matter less because you can see them in context of the larger distribution.

Maybe that's what maturity is: reaching a large enough $n$ that convergence begins. Not that you stop having flaws or struggles, but that you can see them as variance around a mean rather than as fundamental truths about who you are.

---

## On Being Human

The theorem offers a framework for something we intuitively sense but struggle to articulate: that imperfection is not just acceptable but **necessary**.

You cannot get to normality without variance. You cannot reach the limit without the individual fluctuations. The beauty of the bell curve emerges from the chaos of the individual observations.

When someone tells you *you're okay*, they might mean you're not as bad as you think. But the CLT says something stronger: you're **normal**, in the most profound sense. You're exactly what you should be—a bounded random variable contributing to a larger pattern, a single term in a sum that converges to something stable and predictable and, in its way, beautiful.

The Central Limit Theorem isn't just about probability distributions. It's about the mathematics of being human in a world that demands perfection while being fundamentally built from imperfection.

> We must imagine Sisyphus happy. The random variables will converge—the mathematics guarantees it. And it's going to be okay.

**You are okay.**
