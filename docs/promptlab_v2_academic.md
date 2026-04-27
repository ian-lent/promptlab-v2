# PromptLab v2: Few-Shot Co-Training for AI Text Deslopification

**Ian Lent · STAT 4830 · University of Pennsylvania · Spring 2026**

---

## Abstract

This document describes the design and reasoning behind PromptLab v2, a two-stage system for reducing the detectability of AI-generated essays. The system combines an evolutionary prompt optimizer with a distilled sequence-to-sequence rewriter, trained on contrastive pairs accumulated through a co-training loop. The central contribution is few-shot injection: a mechanism by which the optimizer accumulates its past successful outputs and presents them as in-context examples to future generation calls, making the search self-improving without gradient updates. A T5-base model fine-tuned with Low-Rank Adaptation (LoRA) then distills the optimizer's discoveries into a single forward pass at inference time. This document focuses on the machine learning decisions made throughout the project — what was tried, what the theory says should happen, what actually happened, and what the results suggest about directions for future work.

---

## 1. The Problem and Why It Is Hard

Before describing the system, it is worth being precise about what makes this problem difficult. AI detectors like EditLens (Pangram Labs, ICLR 2026) do not look for any single pattern. They are neural classifiers trained on large corpora of human-written and AI-generated text, and they learn whatever statistical regularities best separate the two distributions. The patterns they find are not necessarily interpretable. It is not simply that AI essays use certain words or phrases — it is that the entire joint distribution over sentence lengths, transition types, hedge frequencies, argument structure, and lexical diversity differs between human and AI text in a way that a sufficiently powerful classifier can detect.

This means that naive approaches to reducing detectability fail in predictable ways. Injecting random lexical variation reduces cosine similarity to the original but does not shift the essay out of the AI distribution — it just adds noise. Paraphrasing with a smaller model tends to preserve the structural patterns that the detector uses. The challenge is to produce text that genuinely moves from the AI stylistic distribution toward the human one, while preserving the semantic content of the original essay. These two goals are in tension: the transformations most likely to fool the detector are also the most likely to change what the essay is saying. The drift penalty described below is a direct architectural response to this tension — it prevents the optimizer from solving a trivially easier proxy problem rather than the actual one.

---

## 2. Stage 1: Evolutionary Search in Prompt Space

### 2.1 Why Not Gradient Descent

The natural first instinct for an optimization problem is to use gradient descent. If we want to minimize the detector score of an essay, why not backpropagate through the detector and update the prompt accordingly? The answer is that the path from a prompt to a detector score passes through a discrete sampling step — the language model generating the essay token by token. This step is not differentiable. Gradients cannot flow through it in any direct sense. Techniques like REINFORCE exist precisely to handle this case, but policy gradient methods on this problem tend toward reward hacking — the model learns shortcuts to reduce the detector score that do not represent genuine deslopification. The approach taken here sidesteps the gradient problem entirely by treating the search as a black-box optimization over prompt space. The optimizer only needs a scalar score for each candidate prompt; it does not need to know how that score was computed. This is computationally more expensive — each evaluation requires generating a full essay and running it through the detector — but it avoids the instabilities that arise from direct gradient-based optimization against the detector.

### 2.2 The Evolutionary Loop

The optimizer maintains a population of candidate prompts, typically ten to twenty at a time. At each generation, a small and fast language model — `llama-3.1-8b-instant` — applies two operations to the current population. Mutation takes a single prompt and produces a variant: it might rephrase the prompt, add a specific instruction, change the tone, or emphasize different aspects of the topic. Crossover takes two prompts and produces a hybrid that combines elements of both. The mutator model is small by design; it runs once per population member per generation, so latency compounds quickly, and quality here is less important than speed. Each candidate prompt is then passed to `llama-3.3-70b-versatile` to generate a full essay. The asymmetry between the mutator and the generator is intentional — the mutator needs to be fast while the generator needs to be capable enough to actually implement the prompt's intent at high quality. A weak generator would produce detectable essays regardless of how well-crafted the prompt is, creating a ceiling on what the optimizer can discover. Each essay is scored, the bottom half of the population is replaced by the best new variants, and this continues for many rounds across many topics.

### 2.3 The Drift Penalty as a Constraint

Left unconstrained, the evolutionary optimizer converges almost immediately to a degenerate solution: incoherent, repetitive, or off-topic text also receives low detector scores. This is precisely what you would expect from any optimizer given a single objective with no constraints on the output space. The drift penalty adds three terms to the raw detector score before selection:

```
optimization_slop = detector_slop
                  + α · (1 − cosine_similarity)
                  + β · (1 − ROUGE-L)
                  + γ · BERTScore_penalty
```

Each term penalizes a different kind of semantic departure from the original essay. Cosine similarity between sentence embeddings catches global topic drift — the essay has wandered onto a different subject. ROUGE-L, which measures the longest common subsequence of n-grams between the original and the candidate, catches structural and argumentative drift — key claims have disappeared even if the topic is nominally preserved. BERTScore uses contextual token-level embeddings to measure meaning preservation at the sentence level, catching paraphrases that preserved surface wording but changed the underlying claim. All three are needed because they operate at different granularities: cosine similarity is relatively insensitive to local changes, ROUGE-L captures intermediate-scale structural change, and BERTScore catches fine-grained meaning shifts that the other two miss.

From a theoretical perspective, this is a constrained optimization problem solved via penalty method: the constraints (preserve meaning) are softened into penalty terms and added to the objective (minimize slop). The weights α, β, γ determine the trade-off between evasion and coherence. A tighter constraint keeps essays more faithful but limits how far the optimizer can move in the slop landscape. These weights were set through manual tuning, which represents a legitimate direction for future work — a systematic Pareto search over the evasion-coherence trade-off could characterize the full frontier rather than selecting a single operating point.

---

## 3. Few-Shot Injection: How the Search Learns from Itself

### 3.1 The Core Idea

Few-shot injection is the mechanism that makes the evolutionary search self-improving over time, and it is the most important contribution of this project to understand carefully — because it demonstrates something general about how large language models can function as optimization components rather than just generators.

The core observation is this: the evolutionary optimizer, after many rounds on many topics, has discovered a library of essays that successfully fool the detector. These essays encode, implicitly, the stylistic transformations that work — particular kinds of sentence structures, specific ways of expressing hedges, the relationship between concrete detail and generic claim that the detector penalizes. This knowledge does not exist anywhere as explicit rules or parameters. It exists only as example texts. And that is precisely the format a large language model needs in order to learn from it. Before each essay generation call, the optimizer retrieves the current top-k lowest-scoring essays from the accumulated library and prepends them to the generation prompt as in-context examples. The language model then generates a new essay with those examples in its context window. It learns implicitly what properties a low-slop essay should have, without ever being told what those properties are.

### 3.2 Why This Works: LLMs as In-Context Learners

This mechanism exploits one of the most well-documented properties of large language models: in-context learning. Given examples of a task in the prompt, a sufficiently capable model can perform that task without any gradient updates to its weights. The examples shift the model's generation distribution toward the demonstrated pattern at inference time, using only the context window. In the context of few-shot injection, this means that round one of the optimizer — when there are no accumulated examples — sees the model generating from its prior: essays that look like whatever it was trained to produce, which is detectable AI text. By round ten, when there are nine examples of low-slop essays in the context, the model's generation is conditioned on those examples. By round fifty, the library contains many such examples and the model has access to a rich demonstration of the target distribution.

The critical point is that this improvement requires no training. The model's weights do not change. No gradient is computed. The entire effect is produced by changing what is in the context window. This makes few-shot injection extremely lightweight relative to the improvement it produces.

### 3.3 Few-Shot Injection as a Curriculum

It is instructive to think of the sequence of rounds as a curriculum. A curriculum in machine learning refers to an ordering of experiences that progresses in a way that guides learning more effectively than random ordering — from less to more informative, from simpler to more complex, from clearer to more ambiguous. Few-shot injection produces a natural curriculum without any explicit design. Early rounds have no examples and the optimizer must discover useful transformations from scratch. Later rounds have rich libraries of demonstrated transformations and the optimizer builds on those discoveries rather than rediscovering them from scratch each time.

This is structurally analogous to how a student learns from worked examples. The first time a student sees a type of problem, they must reason from first principles. After seeing many worked examples, they develop an implicit sense of strategy — what kinds of moves work in what kinds of situations — and apply that intuition to new problems without consciously reconstructing the underlying reasoning. The optimizer is doing something structurally similar: it is building an implicit model of the target distribution from accumulated examples, and that implicit model guides each successive generation call. What makes this particularly elegant is that the curriculum is adaptive. The examples in the library are not fixed in advance; they are the actual outputs of the search process, accumulated dynamically as better solutions are found. As the optimizer improves, the examples it adds to the library are better, which makes the context for future rounds richer, which produces even better outputs. This feedback loop — better examples lead to better search, which produces better examples — is the mechanism responsible for the compounding improvement across rounds.

---

## 4. Stage 2: Distilling the Search into a Rewriter

### 4.1 The Role of the Rewriter

The evolutionary optimizer, even with few-shot injection, is expensive at inference time. Each essay requires many rounds of generation, scoring, and selection before a low-slop output emerges. For a deployed system this is impractical. The rewriter's purpose is to distill the knowledge the optimizer has accumulated — encoded in the library of contrastive pairs — into a model that can produce a deslopified essay in a single forward pass. This is an instance of knowledge distillation, a broad family of techniques in which a powerful but slow system (the teacher) generates training data that a smaller and faster system (the student) learns from. The teacher here is the entire optimizer pipeline — evolutionary search, few-shot injection, and drift penalty. The student is a T5-base model with a LoRA adapter. The student never runs the optimizer at inference time; it has learned to approximate the optimizer's best outputs directly.

### 4.2 Why Essay-to-Essay, Not Prompt-to-Prompt

An earlier version of the rewriter operated in prompt space: given a generic essay prompt, produce a rewritten prompt, then pass it to a language model to generate an essay. This seemed reasonable because the optimizer itself operates in prompt space. It did not work as expected, and the reason is instructive. The rewriter's training signal — the slop score of the generated essay — reaches it through two sequential steps: rewriter produces a prompt, language model generates an essay from that prompt, essay is scored. The language model in the middle is a source of variance the rewriter cannot control. The same rewritten prompt does not always produce the same essay. This variance dilutes the training signal: the rewriter cannot cleanly attribute a good slop score to the quality of its rewritten prompt, because the language model's contribution is confounded with it.

Switching to essay-to-essay removes this intermediate step entirely. The rewriter directly maps the thing the detector scores — the essay — to a better version of itself. There is no intermediate generative step introducing variance. If the rewritten essay has lower slop, the rewriter's weights are updated to reinforce that transformation, clearly and without confounding. This architectural change, holding everything else constant, produced a 52% improvement in the validation metric. One step of indirection versus two turns out to make an enormous difference in how much of the training signal actually reaches the model — a result that generalizes well beyond this specific project.

### 4.3 T5-Base as the Backbone

T5 (Text-to-Text Transfer Transformer, Raffel et al. 2020) was selected as the rewriter backbone because its pretraining objective is precisely sequence-to-sequence transformation: given input text, produce output text. This matches the essay-to-essay task formulation exactly, and it means the model arrives at fine-tuning with strong priors about producing coherent text transformations. T5-base, with 220 million parameters, was selected over larger variants for practical reasons given the dataset size of 40 training pairs. A larger model would overfit more aggressively and would require more elaborate regularization. T5-base provides enough capacity to learn the stylistic transformation without so much capacity that the model memorizes the training essays rather than learning the underlying pattern.

The limitation introduced by this choice is worth naming directly. T5 was further fine-tuned on summarization tasks during its supervised phase, introducing a strong bias toward producing outputs shorter than their inputs. Essay rewriting should preserve length while changing style, not compress. The consequence is that the rewriter tends to produce shorter essays and occasionally generates repetitive loops when beam search loses confidence. The slop reduction is real and the mechanism is sound; the output quality as a standalone readable essay suffers from this pretraining mismatch. Replacing T5 with a generative decoder architecture — even a small one — would likely resolve this while preserving the evasion benefit.

### 4.4 Low-Rank Adaptation

LoRA (Hu et al. 2022) constrains fine-tuning to a low-dimensional subspace of the full parameter space. For each attention weight matrix W in T5, LoRA adds the product of two small matrices BA, where the rank r is much smaller than the dimensions of W. Only A and B are trained; the original weights W are frozen. With r=16 and T5-base's hidden dimension of 768, each module introduces approximately 24,576 trainable parameters, yielding roughly 800,000 total trainable parameters out of 220 million — a reduction by a factor of 275. With only 40 training pairs, a model with 220 million trainable parameters would fit the training set almost exactly within a handful of gradient steps and generalize to nothing. LoRA prevents this by restricting the adaptation to a subspace small enough that 40 examples can provide meaningful signal. The rank ablation — r=8 versus r=16 — confirmed that r=16 is appropriately sized: doubling the rank improved the validation metric by only 0.006, indicating that the bottleneck is not model capacity but something upstream (dataset size or pretraining mismatch).

---

## 5. Training Decisions and the Noise Floor

### 5.1 Learning Rate and the Shape of the Training Landscape

The most consequential single hyperparameter decision in Stage 2 was the learning rate. A sweep comparing LR=3×10⁻⁴ to LR=1×10⁻⁴ produced a twofold improvement in the validation metric. Understanding why requires thinking about what happens when you take gradient steps in a noisy landscape with a very small dataset. At a high learning rate, the optimizer takes large steps in a landscape where the gradient estimate contains substantial sampling noise from the small batch — it moves quickly but often overshoots the regions of parameter space where the task metric is good, landing somewhere worse and oscillating. At a low learning rate, it moves more carefully, tracking the signal in the gradient more closely relative to the noise. The step-50 best checkpoint is not a coincidence; it reflects the point at which the useful signal in the gradients has been extracted and subsequent updates are dominated by noise. This is a direct empirical instance of the noise-limited training regime: the effective number of useful gradient steps is determined by the ratio of dataset signal to model capacity, and once that budget is exhausted, further training degrades rather than improves the task metric. Cosine learning rate decay was used alongside the base rate to smoothly reduce the update size over the course of training, allowing the optimizer to settle into a good region of parameter space rather than continuing to oscillate near convergence.

### 5.2 Curriculum Ordering

The training pairs were ordered by descending slop gap — the difference between the baseline essay's detector score and the best essay's score — before being presented to the trainer. Pairs with the largest gap are the clearest supervision examples: the two essays differ substantially in detectability, and the transformation between them is a strong signal about what the rewriter should learn. Pairs with a small gap provide weak supervision — the baseline and best essays are nearly equally detectable, and the model cannot extract a clear learning signal from the difference between them. Presenting the strongest-signal examples first gives the model's initial weight updates the clearest direction. This is the curriculum learning principle (Bengio et al. 2009) applied to a ranking problem: examples with clearer labels come before examples with ambiguous ones. The model builds a strong foundation from high-quality signal before encountering cases that might otherwise pull the weights in contradictory directions.

### 5.3 Stopping on the Right Metric

The most important training decision — and the one most easily overlooked — was monitoring validation slop mean rather than validation cross-entropy loss for early stopping. After step 50, these two metrics diverge: validation loss continues to decline (the model is becoming more confident in predicting the training essay tokens) while validation slop mean rises (the essays it produces are becoming more detectable). This divergence is the signature of overfitting in a small-data regime: the model is memorizing the training essays rather than learning the stylistic transformation. Monitoring cross-entropy loss and stopping when it plateaus would have terminated training at approximately the right time — but for the wrong reason and unreliably. Monitoring slop mean directly stops training when the actual task metric stops improving. The custom callback runs the full inference pipeline every few training steps: generate rewritten essays from validation inputs, score them with the detector, compute mean. This is expensive compared to computing loss on a held-out set, but it is the only honest measure of whether the model is actually learning to do the task.

The lesson generalizes cleanly: whenever the training loss is a proxy for the true task metric rather than the metric itself, there is a risk that the two diverge during training. In reinforcement learning from human feedback, this manifests as reward hacking — the policy learns to maximize the reward model without improving on the underlying task the reward model was meant to approximate. Here it manifests more gently: the model learns to predict essay tokens well without learning to reduce detectability. The solution in both cases is to evaluate on the true metric, not the proxy.

---

## 6. What This Project Points Toward

### 6.1 Few-Shot Injection as a General Mechanism

The few-shot injection mechanism is not specific to the slop reduction task. Any evolutionary search process that generates natural language outputs could potentially be improved by maintaining a library of its best outputs and including them as in-context examples in future generation calls. This applies to other prompt optimization problems, to automated red-teaming where examples of successful attacks inform future attacks, and broadly to any setting where a language model is a component in a black-box optimization loop. The mechanism is lightweight — no training, no gradient computation — and the main open design questions are how many examples to include, how to select them (lowest score, most diverse, most recent, or some combination), and when the library size starts to hurt through context dilution. These questions have clean theoretical formulations in the in-context learning and retrieval-augmented generation literature and are worth investigating systematically.

### 6.2 The Essay-to-Essay Architecture as a Distillation Template

The Stage 2 architecture — accumulate contrastive pairs from an expensive search process, then distill into a cheap sequence-to-sequence model — is a general template for situations where you have a powerful but slow oracle and want to approximate it with a fast model. The design decisions that made it work here — removal of intermediate generative steps, LoRA for regularization, curriculum ordering, and stopping on the true task metric — are each applicable to other distillation problems. A natural extension is to scale the backbone to a generative decoder to resolve the length compression problem introduced by T5's summarization pretraining. A decoder would not carry this bias and would be more natural for a rewriting task that should preserve length. The rest of the training pipeline could remain identical.

### 6.3 Closing the Loop

The observation that the best checkpoint is always at step 50 — regardless of learning rate or LoRA rank — is a data point in a larger pattern. With 40 training pairs and 800,000 trainable parameters, the noise-limited regime is reached quickly. More high-quality contrastive pairs would push the noise floor out and allow training to run longer before the signal collapses. The evolutionary optimizer, with more compute, could produce those pairs. Future work could close this loop deliberately: run the optimizer longer, accumulate more pairs, retrain the rewriter, use the rewriter's outputs to seed the optimizer's few-shot library, and iterate. Each stage informs the next in a way the current single-pass pipeline does not fully exploit. The rewriter's outputs — which are themselves lower-slop than the originals — could become part of the few-shot library, creating a compounding cycle where the distilled model feeds back into the search that produced it.

---

## 7. Conclusion

PromptLab v2 is, at its core, a study in how to make a small amount of hard-won data go as far as possible. The evolutionary optimizer, constrained by the drift penalty, produces high-quality contrastive pairs that a gradient-based method could not easily generate. Few-shot injection makes the optimizer self-improving without any training cost, by exploiting the in-context learning capacity of the language models it uses as generation components. LoRA constrains the rewriter to a parameter subspace where 40 examples are sufficient to learn a meaningful transformation. Curriculum ordering and slop-based early stopping extract the maximum signal from those 40 examples without overfitting. Each decision reflects the same underlying constraint — a small dataset, a powerful pretrained model, and a gap between the proxy loss and the true task metric — and each decision addresses that constraint from a different angle.

The mechanisms developed here — few-shot injection as a self-improving search primitive, essay-to-essay distillation as a way to make expensive search practical at inference time, and the discipline of stopping on the true task metric throughout training — are each independently interesting and each points toward directions that future work could productively extend.
