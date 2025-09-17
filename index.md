---
layout: default
title: Reinforcement Learning: Fundamentals of Policy Gradients
---

## Reinforcement Learning: Fundamentals of Policy Gradients

For the past few months, I've been diving deep into reinforcement learning, and it's hard not to feel like we're in the middle of a renaissance — thanks in huge part to DeepSeek and its release of R1. And the algorithm that underpins the stellar success of R1 is called **Group Relative Policy Optimization (GRPO)**. However, the roots of this success can be traced back to a much older and simpler idea: the **policy gradient**. This post is my attempt to demystify why the equations look the way they do. I promise I'll try my best to explain things naturally, step by step, without getting lost in symbols. ;)

---

### The Hill-Climber’s Dilemma

Let’s think of our agent as a climber, blindfolded, standing somewhere in a vast landscape of possible decisions. Each action is like taking a step, and the height of the hill corresponds to the expected long-term reward. The idea (sort of) is to climb in the direction of the peak without ever actually “seeing” the mountain.

The climber’s instincts—the **policy**, denoted as $\pi_\theta$—are parameterised by $\theta$. These parameters are the dials we can tune so that, over time, the climber “learns” to take better steps. To do this, we employ a strategy called **gradient ascent**. At any given spot, the best thing to do is to move uphill, in the direction of the gradient of the expected reward:

$$
\nabla_\theta J(\theta).
$$

But here’s the snag: calculating this gradient directly involves an integral over all possible trajectories $\tau$:

$$
J(\theta) = \int \pi_\theta(\tau) R(\tau) \, d\tau.
$$

This integral is just a fancy way of saying “we need to account for *every possible future*,” and that’s impossible to compute explicitly. We need a trick.

---

### The Log-Derivative Trick: A Bit of Mathematical Magic

The hard part is that we need to differentiate the probability of a trajectory, $\nabla_\theta \pi_\theta(\tau)$. That’s not something we can handle directly. But here’s where a simple identity from calculus saves the day:

$$
\nabla_\theta \log f(x) = \frac{\nabla_\theta f(x)}{f(x)}.
$$

Rearranging gives:

$$
\nabla_\theta f(x) = f(x) \, \nabla_\theta \log f(x).
$$

Apply this to our policy:

$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \, \nabla_\theta \log \pi_\theta(\tau).
$$

This is the **log-derivative trick**, and it’s the key. Suddenly, the nasty derivative we couldn’t deal with turns into something much friendlier: the log of the policy.

Plugging this back into our expectation:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) R(\tau)].
$$

And there it is. Instead of summing over all possible futures, we can just **sample** trajectories, compute their rewards, and average. The integral has been tamed into something Monte Carlo can handle.

---

### The Variance Problem: When Good Actions Get Unfairly Rewarded

So we’ve got a formula, but it comes with baggage. The estimates have **huge variance**. Imagine a football match that ends 1–0. Every player on the winning team gets a “thumbs up,” including the striker who scored and the defender who almost cost the game. They all get rewarded equally, even though some clearly contributed more than others. The signal is noisy, and the agent can’t figure out which actions were genuinely good. Learning becomes slow and unstable.

We need a way to clean this up.

---

### The Baseline Trick: Making Things Fair

The idea is simple: don’t just ask “was the final outcome good?”—ask “was it *better than average*?” To do this, we introduce a baseline $b$ and subtract it from the reward:

$$
\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(\tau) (R(\tau) - b)].
$$

This subtraction doesn’t bias our estimate—because subtracting a constant doesn’t change the expectation. But choosing the right $b$ can slash variance.

Let’s find the optimal $b$. Define the variance function:

$$
L(b) = \mathbb{E}[(R - b)^2].
$$

Expand it:

$$
L(b) = \mathbb{E}[R^2] - 2b\mathbb{E}[R] + b^2.
$$

Differentiate w.r.t. $b$:

$$
\frac{dL}{db} = -2\mathbb{E}[R] + 2b.
$$

Set this to zero:

$$
b = \mathbb{E}[R].
$$

So the best baseline is just the **expected reward**, or in RL terms, the **Value Function** $V(s)$.

---

### The Advantage: A Cleaner Signal

Once we subtract this baseline, what we’re left with is the **advantage**:

$$
A(s,a) = R(s,a) - V(s).
$$

If the action does better than expected, we make it more likely. If worse, we make it less likely. The learning signal is now sharper, variance is lower, and our blindfolded climber finally has a reliable sense of direction.

---

### Wrapping Up

The story of policy gradients is a neat little arc:

* Start with the goal: maximise expected reward.
* Hit a wall: an impossible integral.
* Use a trick: the log-derivative identity.
* Discover a flaw: high variance.
* Fix it: subtract a baseline.

The result is an elegant learning algorithm that underpins not just GRPO, but most of modern reinforcement learning. What starts as a blind climber fumbling around becomes a guided ascent, all thanks to a few clever mathematical steps.

This is the magic: not the equations themselves, but the *reasoning* that turns dead-ends into breakthroughs.
