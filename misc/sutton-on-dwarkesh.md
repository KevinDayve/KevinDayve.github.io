---
layout: default
title: My thoughts on Richard Sutton’s podcast with Dwarkesh Patel
permalink: /misc/sutton-on-dwarkesh
---

So I recently watched the [podcast](https://www.youtube.com/watch?v=21EYKqUsPfg) Dwarkesh had with Rich Sutton - considered to be the father of RL and I will try to pen my thoughts about that conversation here

What I love about Sutton is how he’s always been brutally clear on what he thinks intelligence requires, and his thoughts on LLMs are no exception. Perhaps the entirety of his talk, and as a consequence, this article can be summarised as: LLM’s are not doing what we’d call “real-learning” because they’re not engaging with the world in any meaningful way.

## The Fundamental Difference: Mimicry v Experience

Sutton draws a sharp line between learning from mimicry and learning from experience. LLMs learn by ingesting vast amounts of text - essentially learning to predict "what a person would say" based on static training data. This is fundamentally different from how intelligent agents (humans, animals, even flippin’ squirrels) actually learn.

Real learning happens through trial and error. You do something, the world responds, and you update your understanding based on what actually happened. The key bit here is that the world pushes back in ways you can't predict from a dataset. It surprises you. And those surprises are what drive actual learning.

LLMs don't get surprised. They're trained on a fixed corpus that "will never be available during its normal life," as Sutton puts it. They're not continually updating based on what happens when they act, they're merely getting better at predicting pre-existing text.

## LLM’s world model (Or lack thereof)

Sutton pushes back hard on the idea that LLMs have genuine "world models." Sure, they can predict sequences of tokens impressively well, but that's not the same as predicting what will actually happen in the environment.

I think this is broadly true, though I'd add a small caveat from my own training experience: LLMs do respond to surprise in a limited way. If a token leads to higher prediction error, the model will adjust to avoid that token in future training. But, and this is crucial, this behavior doesn't extend to long horizons.

A token choice that seems fine right now might be deleterious several sentences down the line, but the model has no mechanism to learn from those delayed consequences. It's optimising locally, not globally. Compare this to humans: dilly-dallying (bunking college for a quick movie sure is inviting, but does bear significantly morbid long-term consequence, unless you come from generational affluence or power - in which case you can coast through life without practically learning anything a la Rahul Gandhi) might not pose immediate danger, but it can wreck you ten years later when you're unemployed and directionless. We learn from those long-horizon consequences. LLMs fundamentally can't.

There's also the emotional learning dimension, which LLMs are completely incapable of. So much of human interaction involves predicting what others want, adjusting based on subtle social feedback, learning from relational dynamics. That entire domain is just absent from the LLM paradigm. But this is perhaps too abstract and further down the line so I’ll offer no additional comments regarding this.

## Goals, and how analogous it is to intelligence

According to Sutton LLMs lack a substantive goal. Their objective is the next token prediction and it "doesn't change the world." There's no external reward or consequence that defines success or failure in a meaningful sense. In Reinforcement Learning, the "right thing to do" is whatever gets you reward. It's grounded in outcomes, not just patterns.

I wanted to understand what Sutton meant by actions that "change the world," because it seemed like a strong claim, because how else are you supposed to change the world? But actually, it's quite “out there”: he means actions that alter the state of your environment in ways that influence future sensations, opportunities, and outcomes.

When you, say, learn to code, you're not just processing information - you're changing your world. Your internal state shifts (you have a new skill, and the way you interact with information in the future might have fundamentally shifted), your external possibilities expand (career paths open up), and the world provides real feedback (compilers tell you if your code works, job applications succeed or fail, salaries materialise or don't).

The action creates genuine consequences that loop back and shape future learning. That's what "changing the world" means in this context.

Compare that to an LLM predicting the next token during training. The prediction doesn't alter the dataset. The text from Wikipedia or Reddit stays exactly the same regardless of whether the prediction was correct. The feedback is just a loss signal - a comparison to a fixed answer sheet, not a consequence from a dynamic environment.

## In Conclusion

LLMs are remarkably good at mimicking human reasoning across an absurd range of domains. But Sutton's point is that mimicry, however sophisticated, isn't the same as the kind of adaptive, experiential learning that characterises natural intelligence.

The apple doesn't fall far from the tree with LLMs. If I told you I'm quite different from my dad, you wouldn't question it, that is because my learning has been empirical, shaped by environments and experiences my dad never encountered. My actions can diverge significantly from what he'd do in the same situation (like for example buying 12,000 rupee shoes with a paltry bank balance).

But with LLMs, even models with stronger priors will make choices that aren't completely orthogonal to their predecessors. They're sophisticated interpolators over their training distribution, but they're not engaging in the open-ended, goal-directed exploration that lets organisms genuinely transcend their origins.
