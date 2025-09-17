---
layout: default
title: Reinforcement Learning: Fundamentals of Policy Gradients
---

## Reinforcement Learning: Fundamentals of Policy Gradients

For the past few months, I've been diving deep into reinforcement learning, and it's hard not to feel like we're in the middle of a renaissance — thanks in huge part to DeepSeek and its release of R1. The algorithm that underpins the stellar success of R1 is called **Group Relative Policy Optimization (GRPO)**. However, the roots of this success can be traced back to a much older and simpler idea: the **policy gradient**. This post is my attempt to demystify why the equations look the way they do, as it often behooves me to understand the "why" behind everything hehehe.

---

### The Hill-Climber's Dilemma

Let's think of our agent as a climber, blindfolded, standing somewhere in a vast landscape of possible decisions. Each action is like taking a step, and the height of the hill corresponds to the expected long-term reward. The goal is to climb toward the peak without ever actually "seeing" the mountain.

The climber's instincts - the **policy**, denoted as $\pi_\theta$ are parameterized by $\theta$. These parameters are the dials we can tune so that, over time, the climber "learns" to take better steps. To do this, we employ a "technique" called **gradient ascent**. At any given spot, the best thing to do is to move uphill, in the direction of the gradient of the expected reward:

$$
\nabla_\theta J(\theta).
$$

But here's where folks might have kittens: calculating this gradient directly involves an integral over all possible trajectories $\tau$:

$$
J(\theta) = \int \pi_\theta(\tau) R(\tau) \, d\tau.
$$

This integral is just a fancy way of saying "we need to account for *every possible future*," and that's computationally tedious. We need a trick.

---

### Mathematical Foundations: Understanding Expectations

Before we dive into the solution, let's establish the mathematical foundation that will be crucial for understanding the trick. 

**Definition of Expectation**: For a continuous random variable $X$ with probability density function $f(x)$, the expectation of some function $g(X)$ is defined as:

$$
\mathbb{E}[g(X)] = \int g(x) f(x) \, dx
$$

Notice the structure carefully:
- $f(x)$ is the **probability density function** (it tells us how likely each value of $x$ is)
- $g(x)$ is the **function we care about** (what we want to compute the average of)
- The integral computes a weighted average, where each value $g(x)$ is weighted by its probability $f(x)$

This structure will be absolutely crucial for understanding the log-derivative trick that follows.

In our reinforcement learning context:
- $\pi_\theta(\tau)$ plays the role of $f(x)$ (probability density over trajectories)
- $R(\tau)$ plays the role of $g(x)$ (the reward function we care about)
- So our problem becomes: $\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int \pi_\theta(\tau) R(\tau) \, d\tau$

---

### The Log-Derivative Trick: Elementary calculus at the rescue!!

Now we can tackle the core challenge: how do we differentiate an expectation with respect to the parameters $\theta$ when $\theta$ appears inside the probability density?

The fundamental problem is that we need to compute:
$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta \int \pi_\theta(\tau) R(\tau) \, d\tau
$$

The difficulty arises because $\theta$ appears inside $\pi_\theta(\tau)$, making the derivative $\nabla_\theta \pi_\theta(\tau)$ challenging to work with directly.

Here's where a simple identity from calculus saves the day. For any positive function $f(x)$:

$$
\nabla_\theta \log f(x) = \frac{\nabla_\theta f(x)}{f(x)}
$$

Rearranging gives us:
$$
\nabla_\theta f(x) = f(x) \, \nabla_\theta \log f(x)
$$

**Why is this powerful?** Because it transforms a derivative of a probability (which is messy) into a probability times a derivative of a log-probability (which is much more manageable).

Applying this identity to our policy:
$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \, \nabla_\theta \log \pi_\theta(\tau)
$$

Now, let's carefully apply this to our expectation. We start with:
$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta \int \pi_\theta(\tau) R(\tau) \, d\tau
$$

Assuming we can interchange the gradient and integral (a technical condition that holds in most practical cases):
$$
= \int \nabla_\theta \pi_\theta(\tau) R(\tau) \, d\tau
$$

Substituting our log-derivative identity:
$$
= \int \pi_\theta(\tau) \, \nabla_\theta \log \pi_\theta(\tau) R(\tau) \, d\tau
$$

Recognizing this as an expectation again (remember our definition: $\mathbb{E}[g(X)] = \int g(x) f(x) dx$):
$$
= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \log \pi_\theta(\tau)]
$$

**This is the key insight**: Instead of trying to differentiate a complex integral over all possible trajectories, we can sample trajectories from our current policy and compute a much simpler expression involving the log-derivative of the policy.

---

### Computing the Log-Derivative of a Trajectory

Now we need to understand what $\nabla_\theta \log \pi_\theta(\tau)$ actually looks like. A trajectory $\tau$ consists of a sequence of states and actions: $(s_0, a_0, s_1, a_1, \ldots, s_T)$.

The probability of this entire trajectory under our policy decomposes as:
$$
\pi_\theta(\tau) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$

where:
- $\mu(s_0)$ is the probability of the initial state
- $\pi_\theta(a_t|s_t)$ is our policy (probability of action $a_t$ given state $s_t$)
- $P(s_{t+1}|s_t, a_t)$ is the environment dynamics (probability of next state)

Taking the logarithm:
$$
\log \pi_\theta(\tau) = \log \mu(s_0) + \sum_{t=0}^{T-1} [\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t)]
$$

When we differentiate with respect to $\theta$:
$$
\nabla_\theta \log \pi_\theta(\tau) = \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t)
$$

**Notice something remarkable**: The environment dynamics $P(s_{t+1}|s_t, a_t)$ completely disappear! This is because they don't depend on our policy parameters $\theta$. We only need to differentiate the parts of the trajectory probability that actually involve our policy.

This gives us our final policy gradient formula (plug the value of the gradient of the log of our policy w.r.t 'tau' in the first formula):
$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$

---

### The Variance Problem: When Good Actions Get Unfairly Rewarded

We now have a formula we can actually compute: sample some trajectories, calculate their total rewards, and update our policy. But there's a serious practical problem: **the estimates have enormous variance**.

Imagine a football match that ends 1–0. Using our current formula, every player on the winning team gets the exact same "reward signal" (the final score), including both the striker who scored the winning goal and the defender who almost caused an own goal. They all get rewarded equally, even though their individual contributions were vastly different.

More technically, suppose we have two trajectories:
- Trajectory A: Gets reward 100
- Trajectory B: Gets reward 0

Even if most actions in both trajectories were actually reasonable, Trajectory A will have *all* its actions reinforced strongly, while Trajectory B will have *all* its actions discouraged. This creates a very noisy learning signal where the agent can't distinguish between genuinely good actions and actions that just happened to be in a lucky trajectory.

---

### The Baseline: Making Comparisons Fair

The solution is conceptually simple: instead of asking "was the outcome good?", we ask "was the outcome *better than expected*?" We introduce a baseline $b$ and modify our gradient:

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[(R(\tau) - b) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$

A lot of you at this point might ask, won't this baseline make our gradient biased? And if there's anything which upsets a statistician (aside from distributions not being normal), are unbiased estimators. 

Good news, not quite. As we'll see soon :)

---

### Why the Baseline Doesn't Introduce Bias

Let's prove that subtracting a baseline **b** doesn't bias our gradient estimate. We need to show that:

$$
\mathbb{E}{\tau \sim \pi\theta}\left[b \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = 0
$$

I'll prove this step-by-step using a fundamental property of probability distributions.
Step 1: Start with a fundamental fact about probability distributions. Since $$\pi_\theta(\tau)$$ is a probability distribution over all possible trajectories, it must satisfy:

$$
\int \pi_\theta(\tau) d\tau = 1
$$

This is just saying "the probabilities of all possible trajectories sum to 1."
**Step 2**: Since the right side is a constant (1), its gradient with respect to θ\theta
θ must be zero:
$$
\nabla_\theta \left(\int \pi_\theta(\tau) d\tau\right) = \nabla_\theta (1) = 0
$$

**Step 3**: Assuming we can interchange the gradient and integral operations (a standard assumption in most practical cases):
$$
\int \nabla_\theta \pi_\theta(\tau) d\tau = 0
$$

**Step 4**: Now comes the key insight. Apply our log-derivative trick. Remember that:
$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)
$$

Substituting this into Step 3:
$$
\int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) d\tau = 0
$$

**Step 5**: Recognizing this as an expectation (using our definition: E[g(X)]=∫g(x)f(x)dx\mathbb{E}[g(X)] = \int g(x) f(x) dx
E[g(X)]=∫g(x)f(x)dx):
$$
\mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau)] = 0
$$

**Step 6**: From our earlier work on trajectory probabilities, we know that:
$$
\nabla_\theta \log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$
Therefore:
$$
\mathbb{E}{\tau \sim \pi\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = 0
$$

**Step 7**: Finally, since the expectation of this sum is zero, multiplying by any constant baseline bb
b still gives zero:
$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[b \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = b \cdot 0 = 0
$$

The intuition: The gradient of the log-policy, when averaged over all possible trajectories, is zero. This is because probability distributions are normalized—they always sum to 1, and the gradient of a constant is zero

---

### Finding the Optimal Baseline

Now that we know any baseline is unbiased, which one should we choose? We want to minimize the variance of our gradient estimate.

The variance of our gradient estimate (focusing on the reward term) is:
$$
\text{Var}[R(\tau) - b] = \mathbb{E}[(R(\tau) - b)^2] - (\mathbb{E}[R(\tau) - b])^2
$$

Since our estimate is unbiased, $\mathbb{E}[R(\tau) - b] = \mathbb{E}[R(\tau)]$, so:
$$
\text{Var}[R(\tau) - b] = \mathbb{E}[(R(\tau) - b)^2] - (\mathbb{E}[R(\tau)])^2
$$

To minimize this, we need to minimize $\mathbb{E}[(R(\tau) - b)^2]$. Expanding:
$$
\mathbb{E}[(R(\tau) - b)^2] = \mathbb{E}[R(\tau)^2] - 2b\mathbb{E}[R(\tau)] + b^2
$$

Taking the derivative with respect to $b$ and setting it to zero:
$$
\frac{d}{db}\mathbb{E}[(R(\tau) - b)^2] = -2\mathbb{E}[R(\tau)] + 2b = 0
$$

Solving: $b^* = \mathbb{E}[R(\tau)]$

**The optimal baseline is the expected reward!** In reinforcement learning terms, this is the **value function** $V(s)$—the expected total reward starting from state $s$.

---

### From Baselines to Advantages

Once we use the optimal baseline, our gradient becomes:
$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[(R(\tau) - V(s)) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$

The term $(R(\tau) - V(s))$ has a special name: the **advantage function** $A(s,a)$. It answers the question: "How much better is this action compared to the average action in this state?"

- If $A(s,a) > 0$: This action was better than expected → increase its probability
- If $A(s,a) < 0$: This action was worse than expected → decrease its probability
- If $A(s,a) = 0$: This action was exactly as expected → no change

This gives us much cleaner learning signals. Instead of rewarding all actions in good trajectories equally, we specifically reinforce actions that led to better-than-expected outcomes.

---

### Connecting to Modern Algorithms

The advantage function is central to many modern RL algorithms:

- **Actor-Critic methods**: Use separate networks to estimate the policy (actor) and value function (critic)
- **Proximal Policy Optimization (PPO)**: Uses advantages with additional constraints to prevent too-large policy updates
- **Group Relative Policy Optimization (GRPO)**: The algorithm behind DeepSeek R1, which does away with the critic model altogether, to compute empirical advantages

All of these build upon the fundamental insight we've just developed!

If you've made it this far, I sincerely thank you for reading.
---

### Further Reading

For those interested in diving deeper:
- [Sutton & Barto's "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/RLbook2020-5.pdf) for the broader context (Strongly recommend!)
- [Schulman et al.'s "High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438) for more sophisticated advantage estimation