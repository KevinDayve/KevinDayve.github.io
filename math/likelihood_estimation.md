---
layout: default
title: A (hopefully) rigorous introduction to maximum likelihood estimation
permalink: /math/likelihood-estimation
math: true
---


## 1. The need for Likelihood

Statistical inference is fundamentally the art of reasoning from observed data back to the unobserved process that generated it. To achieve this, a formal mechanism for "reasoning backwards" is needed.

### 1.1 Inferring parameters
The standard setup for estimation is defined as follows:
* We observe a set of $n$ independent and identically distributed (iid) data points, $\mathbf{X} = (X_1, \ldots, X_n)$.
* We assume this data was generated from an unknown probability distribution belonging to a specific family $f(x \mid \theta)$.
* This family is indexed by a set of parameters, $\theta$, which represents the unknown "truth" governing the process.

For example, in a sequence of coin flips, $f$ may represent the Bernoulli distribution, where $\theta$ corresponds to the unknown probability of heads, $p$.

### 1.2 Probability vs Inference
To understand likelihood, one must distinguish between two opposing inquiries regarding the function $f(x \mid \theta)$:

1. The Forward Problem (Probability): If the parameter $\theta$ is known, what data behaviors do we expect?Query: "Given a fair coin ($p=0.5$), what is the probability of observing 7 heads in 10 flips?"Here, we treat $P(\mathbf{x} \mid \theta)$ as a function of the data $\mathbf{x}$, with $\theta$ held fixed.

2. The Backward Problem (Inference): If the data $\mathbf{x}$ is observed, what parameter $\theta$ is the most plausible cause?Query: "Given that 7 heads were observed in 10 flips, what is the most reasonable estimate for $p$?"This requires viewing the relationship from the perspective of $\theta$, treating the data $\mathbf{x}$ as fixed.

### 1.3 Formal definition
The Likelihood Function, denoted $L(\theta \mid \mathbf{x})$, is the formal tool used to address the inference problem. It is defined as the joint probability (or probability density) of the observed data, viewed strictly as a function of the parameter $\theta$.

For an iid sample $\mathbf{X} = (X_1, \ldots, X_n)$ drawn from a density $f(x\mid\theta)$, the likelihood function is given by:
    
$$L(\theta \mid \mathbf{x}) = f(\mathbf{x} \mid \theta) = \prod_{i=1}^n f(x_i \mid \theta)$$

Crucial Distinction:While $f(\mathbf{x} \mid \theta)$ and $L(\theta \mid \mathbf{x})$ are represented by the same mathematical expression, their interpretations differ significantly based on the variable of interest:As a Probability Density: 
* When $\theta$ is fixed, $f(\mathbf{x} \mid \theta)$ is a function of $\mathbf{x}$ and sums (or integrates) to 1 over the sample space.
* As a Likelihood: When $\mathbf{x}$ is fixed, $L(\theta \mid \mathbf{x})$ is a function of $\theta$. Note that $L(\theta \mid \mathbf{x})$ does not necessarily integrate to 1 over the parameter space $\Theta$.

The likelihood function provides a measure of plausibility. If $L(\theta_A \mid \mathbf{x}) > L(\theta_B \mid \mathbf{x})$, the observed data $\mathbf{x}$ provides greater support for the parameter value $\theta_A$ than for $\theta_B$


## 2. The Log-Likelihood Function
While the likelihood function $L(\theta \mid \mathbf{x})$ serves as the conceptual foundation for parameter estimation, it is often analytically intractable. In practice, inference is almost exclusively conducted using the Log-Likelihood Function, denoted as $\ell(\theta \mid \mathbf{x})$

### 2.1 Definition and Transformation

The log-likelihood function is defined as the natural logarithm of the likelihood function. For a sample of $n$ iid observations, the transformation converts the product of densities into a summation:

$$\ell(\theta \mid \mathbf{x}) = \log L(\theta \mid \mathbf{x}) = \log \left( \prod_{i=1}^n f(x_i \mid \theta) \right) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

### 2.2 Why bother with the `log`?
The transition from $L(\theta \mid \mathbf{x})$ to $\ell(\theta \mid \mathbf{x})$ is not merely a matter of convenience; it is motivated by three distinct mathematical and computational properties:

1. **Monotonicity of the Logarithm**: The natural logarithm is a strictly monotonically increasing function. That is, for any $u, v > 0$, if $u > v$, then $\log(u) > \log(v)$. Consequently, the value of $\theta$ that maximizes $L(\theta \mid \mathbf{x})$ is identical to the value that maximizes $\ell(\theta \mid \mathbf{x})$.$$\arg \max_{\theta} L(\theta \mid \mathbf{x}) = \arg \max_{\theta} \ell(\theta \mid \mathbf{x})$$

This property ensures that optimizing the transformed function yields the correct parameter estimate.

2. **Analytical Simplification**: Differentiation is required to locate maxima. The derivative of a product (required for $L$) becomes practically unmanageable as $n$ increases. By converting the product to a sum, the log-likelihood allows for term-by-term differentiation, significantly simplifying the calculation of the score function (the gradient with respect to $\theta$).

3. **Numerical Stability**: Probabilities are values in the interval $[0,1]$. The product of many such values rapidly approaches zero, leading to arithmetic underflow in computational systems. The log-likelihood transforms these small products into a sum of negative numbers, maintaining numerical precision during optimization.

## 3. The Principle of Maximum Likelihood Estimation
Having established the log-likelihood function $\ell(\theta \mid \mathbf{x})$, we now define the primary method for parameter estimation.

### 3.1 The Maximum Likelihood Estimator

The Maximum Likelihood Estimator (MLE), denoted by $\hat{\theta}$, is defined as the parameter value in the parameter space $\Theta$ that maximizes the likelihood function (and by extension, the log-likelihood function) for the observed data $\mathbf{x}$.

Formally, the MLE is the solution to the maximization problem:

$$\hat{\theta} = \arg \max_{\theta \in \Theta} \ell(\theta \mid \mathbf{x})$$

Intuitively, $\hat{\theta}$ represents the parameter value that renders the observed data most plausible

### 3.2 How to Derive?
For models satisfying regularity conditions, more specifically, where the log-likelihood is differentiable and the support of the distribution does not depend on $\theta$ (we'll talk about these shortly), the MLE is found using standard calculus techniques

1. Form the Log-Likelihood:
Construct $\ell(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)$.

2. Calculate the Score Function:
The Score Function, $S(\theta)$, is the gradient of the log-likelihood with respect to the parameter $\theta$:$$S(\theta) = \nabla_\theta \ell(\theta \mid \mathbf{x}) = \frac{\partial}{\partial \theta} \ell(\theta \mid \mathbf{x})$$

3. Solve the Likelihood Equation:

Set the score function to zero to identify critical points. The solution to this equation is the candidate for the MLE:

$$S(\theta) = 0$$

**Note** (Second-Order Condition): To verify that the solution corresponds to a local maximum rather than a minimum or saddle point, one must verify that the second derivative (the Hessian matrix in the multivariate case) is negative definite:

$$\frac{\partial^2}{\partial \theta^2} \ell(\theta \mid \mathbf{x}) < 0$$

### 3.3 Canonical examples (You are urged to attempt these yourself)

#### Example 3.3.1: Bernoulli Distribution (Discrete case)

Consider an iid sample $X_1, \ldots, X_n$ from a Bernoulli distribution with parameter $p$, where $P(X=1) = p$ and $P(X=0) = 1-p$. We seek the MLE $\hat{p}$.

The probability mass function is $f(x \mid p) = p^x(1-p)^{1-x}$ for $x \in \{0, 1\}$.

The log-likelihood for the full sample is:
$$\ell(p \mid \mathbf{x}) = \sum_{i=1}^n \left[ x_i \log p + (1-x_i) \log(1-p) \right]$$

Let $y = \sum_{i=1}^n x_i$ denote the total number of successes. The expression simplifies to:$$\ell(p \mid \mathbf{x}) = y \log p + (n-y) \log(1-p)$$

Differentiating with respect to $p$ yields the score function:$$S(p) = \frac{\partial \ell}{\partial p} = \frac{y}{p} - \frac{n-y}{1-p}$$Setting $S(p) = 0$ and solving for $p$:

$$\frac{y}{p} = \frac{n-y}{1-p} \implies y(1-p) = p(n-y) \implies y = np$$

$$\hat{p} = \frac{y}{n} = \bar{X}$$

Thus, the MLE for the Bernoulli parameter is the sample mean

#### Example 3.3.2: Normal Distribution (Continuous case)

Consider an iid sample $X_1, \ldots, X_n$ from a Normal distribution $N(\mu, \sigma^2)$. We seek the MLEs for both $\mu$ and $\sigma^2$.The log-likelihood function is given by:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2$$

We compute the partial derivatives (score functions) for each parameter:
1. For $\mu$:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu)$$

2. For $\sigma^2$:

$$\frac{\partial \ell}{\partial (\sigma^2)} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^n (x_i - \mu)^2$$

Setting the first equation to zero yields the estimator for the mean:

$$\sum_{i=1}^n (x_i - \mu) = 0 \implies \sum_{i=1}^n x_i = n\mu \implies \hat{\mu} = \bar{X}$$

Setting the second equation to zero and substituting $\hat{\mu} = \bar{X}$:

$$\frac{n}{2\sigma^2} = \frac{1}{2\sigma^4} \sum_{i=1}^n (x_i - \bar{X})^2$$$$n\sigma^2 = \sum_{i=1}^n (x_i - \bar{X})^2$$$$\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{X})^2$$

Thus, the MLE for the variance is the biased sample variance (divided by $n$, not $n-1$)


## 4. Properties of Maximum Likelihood Estimators

Maximum Likelihood Estimators are the standard in statistical inference not merely due to intuitive appeal, but because they possess rigorous theoretical properties. These properties ensure that, particularly as the sample size grows, MLEs behave optimally.

### 4.1 The Invariance Property

One of the most practical features of the MLE is its invariance under parameter transformation.Theorem (Invariance Property of MLE):If $\hat{\theta}$ is the maximum likelihood estimator for a parameter $\theta$, and if $\tau(\theta)$ is a function of $\theta$, then the maximum likelihood estimator for $\tau(\theta)$ is simply $\tau(\hat{\theta})$.

Example:In Section 3.3.2, we derived the MLE for the variance $\sigma^2$ as $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$. By the Invariance Property, the MLE for the standard deviation $\sigma = \sqrt{\sigma^2}$ is:

$$\hat{\sigma} = \sqrt{\hat{\sigma}^2} = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2}$$

Theoretical Justification: 
This result is derived by defining an "induced likelihood" function, $$L^*(\eta)$$, for the transformed parameter $$\eta = \tau(\theta)$$. The function $$L^*(\eta)$$ is defined as the supremum of the original likelihood over the set of $\theta$ values that map to $\eta$. Maximizing this induced likelihood yields 

$$\hat{\eta} = \tau(\hat{\theta})$$.

### 4.2 Asymptotic Properties

While MLEs are not guaranteed to be unbiased for finite samples (as seen with the Normal variance estimator), they exhibit optimal behavior as the sample size $n \to \infty$.
1. **Consistency**: The MLE is a consistent estimator. As the sample size increases, the estimator converges in probability to the true parameter value $\theta_0$.$$\hat{\theta}_n \xrightarrow{p} \theta_0 \quad \text{as} \quad n \to \infty$$

2. **Asymptotic Normality**: The sampling distribution of the MLE converges to a Normal distribution. This property allows for the construction of approximate confidence intervals and hypothesis tests for large samples.
$$\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{I(\theta_0)}\right)$$
where $I(\theta_0)$ is the Fisher Information.

3. **Asymptotic Efficiency**: Among all consistent, asymptotically normal estimators, the MLE has the smallest possible variance. It achieves the theoretical lower bound for variance, known as the Cramér-Rao Lower Bound

### 4.3 Fisher Information and Cramer-Rao Lower Bound

To understand "efficiency," we must quantify the limit of precision for any estimator. The Cramér-Rao Lower Bound provides a theoretical lower limit for the variance of an unbiased estimator.

### 4.3.1. Preliminary: The Score Function and Fisher Information

First, recall the Score Function, $S(\theta \mid \mathbf{x}) = \frac{\partial}{\partial \theta} \log f(\mathbf{x}\mid\theta)$.The Fisher Information, denoted $I(\theta)$, is defined as the variance of the score function:

$$I(\theta) = E_\theta \left[ \left( \frac{\partial}{\partial \theta} \log f(\mathbf{X}\mid\theta) \right)^2 \right] = \text{Var}(S(\theta \mid \mathbf{X}))$$

Note: Under standard regularity conditions (specifically, if we can differentiate twice under the integral), the Fisher Information can also be computed as the expected negative curvature (Hessian):

$$I(\theta) = -E_\theta \left[ \frac{\partial^2}{\partial \theta^2} \log f(\mathbf{X}\mid\theta) \right]$$

### 4.3.2. The Proof (Cramér-Rao Inequality):

Let $X_1, \ldots, X_n$ be a sample with joint pdf $f(\mathbf{x} \mid \theta)$, and let $W(\mathbf{X}) = W(X_1, \ldots, X_n)$ be any estimator with finite variance.Under regularity conditions that allow the interchange of differentiation and integration, the variance of $W(\mathbf{X})$ satisfies:

$$\text{Var}(W(\mathbf{X})) \ge \frac{\left( \frac{d}{d\theta} E_\theta[W(\mathbf{X})] \right)^2}{I(\theta)}$$

In the specific case where $W(\mathbf{X})$ is an unbiased estimator of $\theta$ (i.e., $E[W] = \theta$), the inequality simplifies to:

$$\text{Var}(W(\mathbf{X})) \ge \frac{1}{I(\theta)}$$Proof:The proof relies on the covariance between the estimator $W$ and the score function $S$.

Step 1: First, observe that the expected value of the score function is zero:$$E[S(\theta\mid\mathbf{X})] = \int \frac{\partial \log f(\mathbf{x}\mid\theta)}{\partial \theta} f(\mathbf{x}\mid\theta) \, d\mathbf{x} = \int \frac{1}{f(\mathbf{x}\mid\theta)} \frac{\partial f(\mathbf{x}\mid\theta)}{\partial \theta} f(\mathbf{x}\mid\theta) \, d\mathbf{x} = \int \frac{\partial f(\mathbf{x}\mid\theta)}{\partial \theta} \, d\mathbf{x}$$

Assuming we can swap the derivative and integral (a strict regularity condition):

$$= \frac{d}{d\theta} \int f(\mathbf{x}\mid\theta) \, d\mathbf{x} = \frac{d}{d\theta}(1) = 0$$

Step 2: Covariance of the Estimator and the Score

Consider the covariance between the estimator $W(\mathbf{X})$ and the score $S(\theta\mid\mathbf{X})$. Since $E[S]=0$, the covariance is simply the expectation of their product:$$\text{Cov}(W, S) = E[W(\mathbf{X}) S(\theta\mid\mathbf{X})] - E[W]E[S] = E[W(\mathbf{X}) S(\theta\mid\mathbf{X})]$$

Expanding the expectation:

$$E[W S] = \int W(\mathbf{x}) \left( \frac{\partial}{\partial \theta} \log f(\mathbf{x}\mid\theta) \right) f(\mathbf{x}\mid\theta) \, d\mathbf{x}$$

Applying the "Log-Derivative Trick" (multiplying and dividing by the density):

$$\frac{\partial}{\partial \theta} \log f(\mathbf{x}\mid\theta) = \frac{1}{f(\mathbf{x}\mid\theta)} \frac{\partial f(\mathbf{x}\mid\theta)}{\partial \theta}$$

Substituting this back into the integral:$$E[W S] = \int W(\mathbf{x}) \left[ \frac{1}{f(\mathbf{x}\mid\theta)} \frac{\partial f(\mathbf{x}\mid\theta)}{\partial \theta} \right] f(\mathbf{x}\mid\theta) \, d\mathbf{x} = \int W(\mathbf{x}) \frac{\partial f(\mathbf{x}\mid\theta)}{\partial \theta} \, d\mathbf{x}$$

Again, utilizing the regularity condition to swap the derivative and integral:

$$E[W S] = \frac{d}{d\theta} \int W(\mathbf{x}) f(\mathbf{x}\mid\theta) \, d\mathbf{x} = \frac{d}{d\theta} E_\theta[W(\mathbf{X})]$$

If $W$ is an unbiased estimator of $\theta$, then $E[W] = \theta$, and thus $\frac{d}{d\theta}(\theta) = 1$.

Step 3: The Cauchy-Schwarz inequality for random variables states that $[\text{Cov}(X, Y)]^2 \le \text{Var}(X)\text{Var}(Y)$. Applying this to $W$ and $S$:

$$\left( \text{Cov}(W, S) \right)^2 \le \text{Var}(W) \text{Var}(S)$$

Substituting our results ($\text{Cov}(W, S) = 1$ and $\text{Var}(S) = I(\theta)$):

$$1^2 \le \text{Var}(W) \cdot I(\theta)$$

Rearranging the terms yields the lower bound:

$$\text{Var}(W) \ge \frac{1}{I(\theta)}$$$\blacksquare$

### 4.3.3. The IID Case

The theorem above applies to the joint density $f(\mathbf{x}\mid\theta)$. If the data $\mathbf{X} = (X_1, \ldots, X_n)$ are independent and identically distributed (iid), the Fisher Information of the entire sample, $I_n(\theta)$, is simply $n$ times the Fisher Information of a single observation, $I_1(\theta)$.

Thus, for iid samples, the variance of any unbiased estimator is bounded by:$$\text{Var}(W) \ge \frac{1}{n I_1(\theta)}$$

## 5. Challenges and "Non-regular" cases

The derivation of the MLE via the score function $S(\theta) = 0$ relies heavily on a set of Regularity Conditions. The most critical of these conditions requires that the support of the distribution—the set of values for which $f(x\mid\theta) > 0$ - does not depend on the parameter $\theta$. When the domain of the data is dependent on the parameter, the likelihood function often exhibits discontinuities. In such cases, the likelihood function is not differentiable everywhere, and the maximum must be found through logical inspection rather than calculus.

### 5.1 When Calculus doesn't work

Recall that finding the MLE typically involves solving $\frac{\partial}{\partial \theta} \ell(\theta \mid \mathbf{x}) = 0$. This method implicitly assumes that the maximum lies in the interior of the parameter space and that the function is smooth.If the support of $X$ depends on $\theta$, the parameter $\theta$ usually acts as a hard boundary (a truncation point). The likelihood function becomes strictly increasing or decreasing up to this boundary, meaning the derivative will never be zero. The maximum occurs at the boundary itself.

### 5.2 A canonical example (Uniform distribution)

Consider an iid sample $X_1, \ldots, X_n$ drawn from a Uniform distribution on the interval $(0, \theta)$, denoted as $X_i \sim U(0, \theta)$. We wish to estimate the upper bound $\theta$.

The Likelihood FunctionThe probability density function for a single observation is:

$$f(x \mid \theta) = \begin{cases} \frac{1}{\theta} & \text{if } 0 \le x \le \theta \\ 0 & \text{otherwise} \end{cases}$$

Using the indicator function notation $I(\cdot)$, this can be written as:

$$f(x \mid \theta) = \frac{1}{\theta} I(0 \le x \le \theta)$$

The likelihood function is the product of these densities:$$L(\theta \mid \mathbf{x}) = \prod_{i=1}^n \frac{1}{\theta} I(0 \le x_i \le \theta) = \frac{1}{\theta^n} \prod_{i=1}^n I(0 \le x_i \le \theta)$$

The product of the indicator functions $\prod I(0 \le x_i \le \theta)$ is equal to 1 if and only if all observed values $x_i$ are less than or equal to $\theta$. This is equivalent to saying that the maximum observed value must be less than or equal to $\theta$.

Let $X_{(n)} = \max(X_1, \ldots, X_n)$ denote the $n$-th order statistic (the sample maximum). The likelihood function simplifies to:

$$L(\theta \mid \mathbf{x}) = \begin{cases} \frac{1}{\theta^n} & \text{if } \theta \ge x_{(n)} \\ 0 & \text{if } \theta < x_{(n)} \end{cases}$$

If we attempt to take the derivative of $\ell(\theta) = -n \log \theta$, we get $-\frac{n}{\theta}$, which is always negative. This indicates that the function is strictly decreasing.
* For $\theta < x_{(n)}$, the likelihood is exactly 0 (impossible).
* For $\theta \ge x_{(n)}$, the likelihood is $1/\theta^n$, which is a decreasing function of $\theta$.To maximize a decreasing function, we must choose the smallest possible value for $\theta$ that satisfies the constraint $\theta \ge x_{(n)}$.

Therefore, the MLE is the sample maximum:$$\hat{\theta} = X_{(n)}$$

Note on Bias: It is worth noting that while $\hat{\theta} = X_{(n)}$ is the MLE, it is a biased estimator. Since $X_i < \theta$ with probability 1, the maximum $X_{(n)}$ will strictly be less than $\theta$ on average. An unbiased estimator for this problem is actually $\frac{n+1}{n} X_{(n)}$


## 6. Connection to Machine Learning
The Principle of Maximum Likelihood is not merely a historical artifact of classical statistics; it remains the primary engine driving modern machine learning and deep learning algorithms. In this final section, lets explore how MLE serves as the theoretical justification for loss functions, Bayesian inference, and optimization in latent variable models.

### 6.1 Connection to Bayesian Inference (MAP)

In the frequentist framework of MLE, the parameter $\theta$ is considered a fixed, albeit unknown, constant. Bayesian statistics, conversely, treats $\theta$ as a random variable described by a prior distribution, $\pi(\theta)$.

Inference is conducted on the posterior distribution, $\pi(\theta \mid \mathbf{x})$, computed via Bayes' Theorem:

$$\pi(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta) \pi(\theta)}{\int f(\mathbf{x} \mid \theta') \pi(\theta') \, d\theta'} \propto L(\theta \mid \mathbf{x}) \pi(\theta)$$

The Bayesian equivalent of the MLE is the Maximum A Posteriori (MAP) estimator, which selects the mode of the posterior distribution:

$$\hat{\theta}_{MAP} = \arg \max_{\theta} \left[ \log L(\theta \mid \mathbf{x}) + \log \pi(\theta) \right]$$

Theorem (Equivalence of MLE and MAP):If the prior distribution $\pi(\theta)$ is uniform (i.e., $\pi(\theta) \propto c$ for all $\theta$ in the domain), then the MAP estimate is identical to the MLE.

$$\hat{\theta}_{MAP} = \arg \max_{\theta} \log L(\theta \mid \mathbf{x}) = \hat{\theta}_{MLE}$$

Thus, MLE can be viewed as a special case of Bayesian inference where one assumes a non-informative (flat) prior.

### 6.2 Connection to Loss functions

In machine learning, models are trained by minimizing a "Loss Function" (or "Cost Function"). In nearly all supervised learning cases, these loss functions are derived directly from the negative log-likelihood. Maximizing the likelihood is mathematically equivalent to minimizing the negative log-likelihood.

#### 6.2.1. Regression and Mean Squared Error (MSE)

Consider a regression model where the target $y_i$ is a deterministic function of the input $x_i$ plus Gaussian noise:

$$y_i = f(x_i; \theta) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)$$

This implies that conditional on $x_i$, $y_i \sim N(f(x_i; \theta), \sigma^2)$.

The log-likelihood for the dataset is:

$$\ell(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

ignoring constant terms with respect to $\theta$, maximizing $\ell(\theta)$ is equivalent to maximizing:

$$-\sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

which is equivalent to minimizing:

$$J(\theta) = \sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

This demonstrates that minimizing the Mean Squared Error (MSE) is formally equivalent to performing MLE under the assumption of Gaussian noise.

#### 6.2.2 Classification

Consider a binary classification model where $y_i \in \{0, 1\}$. The model predicts a probability $p_i = f(x_i; \theta)$ (e.g., via a sigmoid output). The distribution is Bernoulli.
The log-likelihood is:

$$\ell(\theta) = \sum_{i=1}^n \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

In information theory, the negative of this quantity is known as the Binary Cross-Entropy loss.

$$J(\theta) = -\sum_{i=1}^n \left[ y_i \log f(x_i; \theta) + (1-y_i) \log(1-f(x_i; \theta)) \right]$$

Thus, minimizing Cross-Entropy is exactly equivalent to MLE for Bernoulli-distributed data.

### 6.3 Numerical MLE
In complex models (e.g., Neural Networks), the likelihood equation $S(\theta) = 0$ rarely yields a closed-form algebraic solution. We rely on iterative numerical optimization.

1. Gradient Ascent
We iteratively update $\theta$ in the direction of the gradient (the score function):

$$\theta^{(t+1)} = \theta^{(t)} + \eta \nabla_\theta \ell(\theta^{(t)})$$

where $\eta$ is the learning rate. (In ML, we typically use Gradient Descent on the negative log-likelihood).

2. Newton-Raphson Method. 
This second-order method utilizes the curvature of the likelihood surface to converge faster.$$\theta^{(t+1)} = \theta^{(t)} - [H(\theta^{(t)})]^{-1} \nabla_\theta \ell(\theta^{(t)})$$

where $H$ is the Hessian matrix of second derivatives. Recalling Section 4.3, since $E[-H(\theta)] = I(\theta)$ (Fisher Information), this method effectively scales the step size by the inverse of the Fisher Information. This variant is known as Fisher Scoring.

## 7. Conclusion
The transition from observing data to inferring the underlying mechanisms of the world is the central challenge of statistics. As we have just seen, the Principle of Maximum Likelihood provides is a rigorously optimal framework for this task.

By starting from first principles by distinguishing the forward probability from the backward likelihood, I hope that a logical foundation for estimation has been established..

**Acknowledgements and Further Reading**

The structure and rigorous approach of this guide have been heavily influenced by the seminal text "**Statistical Inference**" by George Casella and Roger L. Berger.

For the reader seeking to deepen their understanding of the measure-theoretic foundations of these properties, or to explore hypothesis testing and interval estimation with equal rigor, Casella & Berger remains the definitive reference.

It is my sincere hope that this guide has succeeded in presenting these complex mathematical concepts with the necessary rigor, while retaining the intuitive clarity required to apply them effectively.

Thank you so much for reading.