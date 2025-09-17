---
layout: default
title: Reinforcement Learning: Fundamentals of Policy Gradients
---

For the past few months, I've been diving deep into reinforcement learning, and it's hard not to feel like we're in the middle of a renaissance — thanks in huge part to DeepSeek and its release of R1. The algorithm that underpins the stellar success of R1 is called **Group Relative Policy Optimization (GRPO)**. However, the roots of this success can be traced back to a much older and simpler idea: the **policy gradient**. This post is my attempt to demystify why the equations look the way they do, as it often behooves me to understand the "why" behind everything hehehe.

---

### The Hill-Climber's Dilemma

Let's think of our agent as a climber, blindfolded, standing somewhere in a vast landscape of possible decisions. Each action is like taking a step, and the height of the hill corresponds to the expected long-term reward. The goal is to climb toward the peak without ever actually "seeing" the mountain.

The climber's instincts — the **policy**, denoted as $\pi_\theta$ — are parameterized by $\theta$. These parameters are the dials we can tune so that, over time, the climber "learns" to take better steps. To do this, we employ a "technique" called **gradient ascent**. At any given spot, the best thing to do is to move uphill, in the direction of the gradient of the expected reward:

$$
\nabla_\theta J(\theta).
$$

But here's where folks might have kittens: calculating this gradient directly involves an integral over all possible trajectories $\tau$:

$$
J(\theta) = \int \pi_\theta(\tau) R(\tau) \, d\tau.
$$

This integral is just a way of saying "we need to account for *every possible future*," and that's computationally tedious. We need a trick.

---

### Mathematical Foundations: Understanding Expectations

Before we dive into the solution, let's establish certain mathematical foundations that will come in handy. 

**Definition of Expectation**: For a continuous random variable $X$ with probability density function $f(x)$, the expectation of some function $g(X)$ is defined as:

$$
\mathbb{E}[g(X)] = \int g(x) f(x) \, dx
$$

Where:
- $f(x)$ is the **probability density function** (it tells us how likely each value of $x$ is)
- $g(x)$ is the **function we care about** (what we want to compute the average of)
- The integral computes a weighted average, where each value $g(x)$ is weighted by its probability $f(x)$


In reinforcement learning:
- $\pi_\theta(\tau)$ plays the role of $f(x)$ (probability density over trajectories)
- $R(\tau)$ plays the role of $g(x)$ (the reward function we care about)
- So our problem becomes:

$$
\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int \pi_\theta(\tau) R(\tau) \, d\tau
$$

---

### The Log-Derivative Trick: Elementary Calculus to the Rescue

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

**Why is this powerful?** Because it transforms a derivative of a probability (which is messy) into a probability times a derivative of a log-probability (which is much more manageable and numerically convenient).

Applying this identity to our policy:

$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \, \nabla_\theta \log \pi_\theta(\tau)
$$

Now, let's carefully apply this to our expectation. We start with:

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta \int \pi_\theta(\tau) R(\tau) \, d\tau
$$

Assuming we can interchange the gradient and integral (Careful: You can do this operation only where the the integral is finite (courtest of Leibniz rule), since both differentiation and integration are limit operations, you **cannot** apply this slight on infinite limit operations):

$$
= \int \nabla_\theta \pi_\theta(\tau) R(\tau) \, d\tau
$$

Substituting our log-derivative identity:

$$
= \int \pi_\theta(\tau) \, \nabla_\theta \log \pi_\theta(\tau) R(\tau) \, d\tau
$$

Recognizing this as an expectation again (remember our definition):

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
- $\pi_\theta(a_t \mid s_t)$ is our policy (probability of action $a_t$ given state $s_t$)
- $P(s_{t+1} \mid s_t, a_t)$ is the environment dynamics (probability of next state)

Taking the logarithm:

$$
\log \pi_\theta(\tau) = \log \mu(s_0) + \sum_{t=0}^{T-1} [\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t)]
$$

When we differentiate with respect to $\theta$:

$$
\nabla_\theta \log \pi_\theta(\tau) = \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)
$$

**Notice something rather cool**: The environment dynamics $P(s_{t+1}\mid s_t, a_t)$ completely disappear! This is because they don't depend on our policy parameters $\theta$. We only need to differentiate the parts of the trajectory probability that actually involve our policy.

This gives us our final policy gradient formula:

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
$$

---

### The Variance Problem: When Good Actions Get Unfairly Rewarded

We now have a formula we can actually compute: sample some trajectories, calculate their total rewards, and update our policy. But there's a serious practical problem: **the estimates have enormous variance**.

Take for instance, a football match that ends 1–0. Using our current formula, every player on the winning team gets the exact same "reward signal" (the final score), including both the striker who scored the winning goal and the defender who almost caused an own goal. They all get rewarded equally, even though their individual contributions were vastly different.

---

### The Baseline: Making Comparisons Fair

The solution is conceptually simple: instead of asking "was the outcome good?", we ask "was the outcome *better than expected*?" We introduce a baseline $b$ and modify our gradient:

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[(R(\tau) - b) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
$$

A lot of you at this point might ask, won't this baseline make our gradient biased? (Afterall, the only thing that makes a statistician upset, aside from not having IID assumptions, are unbiased estimators) The answer is no. Let me show why.

---

### Why the Baseline Doesn't Introduce Bias

We need to show that:

$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[b \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right] = 0
$$

**Step 1**: Start with a fundamental fact about probability distributions. Since $\pi_\theta(\tau)$ is a probability distribution over all possible trajectories:

$$
\int \pi_\theta(\tau) d\tau = 1
$$

**Step 2**: Differentiate both sides:

$$
\nabla_\theta \left(\int \pi_\theta(\tau) d\tau\right) = \nabla_\theta (1) = 0
$$

**Step 3**: Swap gradient and integral:

$$
\int \nabla_\theta \pi_\theta(\tau) d\tau = 0
$$

**Step 4**: Use the log-derivative trick:

$$
\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)
$$

So:

$$
\int \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau) d\tau = 0
$$

**Step 5**: Recognize this as an expectation:

$$
\mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau)] = 0
$$

**Step 6**: And recall:

$$
\nabla_\theta \log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

So:

$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = 0
$$

**Step 7**: Multiplying by any constant baseline $b$ still gives zero:

$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[b \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = b \cdot 0 = 0
$$

So subtracting a baseline does not bias the estimate.

---

### Finding the Optimal Baseline

Now that we know any baseline is unbiased, which one should we choose? The best baseline is the one that minimizes variance.

Define:

$$
\text{Var}[R(\tau) - b] = \mathbb{E}[(R(\tau) - b)^2] - (\mathbb{E}[R(\tau) - b])^2
$$

Recall that this directly follows from the definition of Variance:

$$
\text{Var}(\X) = \mathbb{E(\X)^2} -(\mathbb{E[\X]})^2
$$

After simplifying, minimizing variance boils down to minimizing:

$$
\mathbb{E}[(R(\tau) - b)^2]
$$

Expanding:

$$
\mathbb{E}[R(\tau)^2] - 2b \mathbb{E}[R(\tau)] + b^2
$$

Differentiate with respect to $b$ and set to zero:

$$
-2\mathbb{E}[R(\tau)] + 2b = 0 \implies b^* = \mathbb{E}[R(\tau)]
$$

So the optimal baseline is the **expected reward**, i.e. the **value function** $V(s)$.

---

### From Baselines to Advantages

Once we use the optimal baseline, our gradient becomes:

$$
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[(R(\tau) - V(s)) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
$$

The term $(R(\tau) - V(s))$ is the **advantage function** $A(s,a)$.

- If $A(s,a) > 0$: action better than expected → increase its probability  
- If $A(s,a) < 0$: action worse than expected → decrease its probability  
- If $A(s,a) = 0$: exactly as expected → no change  

---

### Connecting to Modern Algorithms

The advantage function is central to many modern RL algorithms:

- **Actor-Critic methods**: separate networks for policy (actor) and value function (critic)  
- **Proximal Policy Optimization (PPO)**: uses advantages with constraints on update size  
- **Group Relative Policy Optimization (GRPO)**: the algorithm behind DeepSeek R1, which eliminates the critic entirely and computes empirical advantages  

All of these build upon the fundamental insight we've just developed!

---

### Further Reading

- [Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/RLbook2020-5.pdf)  
- [Schulman et al., *Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438)  

If you've made it this far, thank you for reading.