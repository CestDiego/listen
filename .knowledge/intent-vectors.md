# Intent Vectors — Research Notes

Research into continuously building a user intent representation from streaming
transcripts. The goal: move from point-in-time skill classification to a
persistent, evolving model of what the user wants/feels/is-doing.

## Problem Statement

Today, each transcript chunk goes through per-skill expert classifiers
independently. After classification, the signal is consumed and discarded.
There is no accumulated representation of user state across time — no
memory of trajectory, no drift detection, no cross-skill awareness.

An **intent vector** would be a persistent, decaying representation that:
- Accumulates signal from each classification cycle
- Decays toward baseline over configurable time windows
- Enables skills to reason about trends, not just moments
- Allows "meta skills" that watch the vector itself (e.g., detect mood decline)

## Approaches (State of the Art)

### 1. EMA over Sentence Embeddings (Simplest)

Embed each transcript chunk with a local sentence encoder (nomic-embed-text,
snowflake-arctic-embed, BGE — all available via MLX). Maintain a running
exponential moving average:

```
intent_vector = α * embed(new_chunk) + (1 - α) * intent_vector
```

α controls decay rate. Cosine similarity against prototype vectors
(computed from labeled training data) gives continuous skill activation.

**Strengths:** Simple, local, no new models needed beyond an embedder.
**Weaknesses:** Sentence embeddings capture semantics, not intent.
"Play some jazz" and "I love jazz music" embed similarly but have
very different intents. Conflates topic with action.

**Candidate embedders for MLX:**
- `nomic-embed-text-v1.5` — 137M params, good balance
- `snowflake-arctic-embed-m` — 110M params, MIT license
- `bge-small-en-v1.5` — 33M params, very fast

### 2. Structured Multi-Dimensional Vector (Recommended Starting Point)

Instead of a single opaque embedding, maintain a vector with interpretable,
independently-updating dimensions:

| Dimension | Range | Decay Half-Life | Signal Source |
|-----------|-------|-----------------|---------------|
| `mood` | [-1, 1] | 120s | Tiny classifier or heuristic |
| `energy` | [0, 1] | 60s | Speech rate, exclamations |
| `task_focus` | [0, 1] | 30s | Command vs. reflection ratio |
| `engagement` | [0, 1] | 90s | Response to system, density |
| `skill_activation[*]` | [0, 1] | 45s per skill | Expert classifier confidence |

Each axis has its own decay rate and update rule. Skill activations come
directly from the existing expert classifiers (the confidence scores are
already there — we just throw them away). Other dimensions can start as
rule-based heuristics and graduate to tiny LoRA classifiers.

**Strengths:**
- Interpretable — you can log it, visualize it, debug it
- Incremental — start with expert outputs only, add dimensions later
- Skills can read it — "user mood declining for 5min" → lower wellbeing threshold
- Drift detection — compute first derivative of any axis over time
- No new models required for v1

**Weaknesses:**
- Requires defining dimensions up front (mitigated by starting small)
- Independent axes miss correlations (mood × task_focus interactions)

**Implementation sketch:**
```typescript
interface IntentVector {
  mood: number;           // [-1, 1]
  energy: number;         // [0, 1]
  taskFocus: number;      // [0, 1]
  engagement: number;     // [0, 1]
  skills: Map<string, number>;  // per-skill [0, 1]
  lastUpdated: Date;
}

// Decay function: value approaches baseline over time
function decay(value: number, baseline: number, halfLifeMs: number, elapsedMs: number): number {
  const factor = Math.pow(0.5, elapsedMs / halfLifeMs);
  return baseline + (value - baseline) * factor;
}
```

### 3. Dialogue State Tracking (DST)

Academic approach from task-oriented dialogue systems. Maintains a probability
distribution over slot-value pairs, updated each turn via Bayesian inference:

```
P(intent_t | utterance_t) ∝ P(utterance_t | intent_t) × P(intent_t | intent_{t-1})
```

Modern variants (2024-2025) use LLMs as belief updaters — feed current state
+ new utterance, output updated state as JSON. Benchmarked on MultiWOZ.

**Relevant models/papers:**
- TripPy (2020) — triple copy strategy for DST
- SOM-DST (2020) — selectively overwrites memory
- IC-DST (2022) — in-context learning for DST with LLMs
- LDST (2024) — lightweight DST fine-tuned on small models

**Strengths:** Principled, handles ambiguity, well-studied.
**Weaknesses:** Requires pre-defined slot ontology. Heavyweight for streaming.
Designed for turn-taking dialogue, not continuous passive listening.

### 4. Activation Engineering / Steering Vectors

From mechanistic interpretability research (Anthropic, Representation
Engineering by Zou et al. 2023). Find **directions** in a model's
hidden-layer activations that correspond to specific intents.

**Process:**
1. Take the existing Qwen 0.8B expert models
2. Run batches of positive/negative examples through the model
3. Extract activations at intermediate layers
4. Compute mean difference → this is the "intent direction"
5. For new utterances, project activations onto these directions
6. Dot product = continuous intent activation score

**Why this is interesting for listen:**
- Operates at the representation level, not the output level
- Captures intent-specific structure that sentence embeddings miss
- Can be extracted from models we already have (Qwen 0.8B experts)
- Gives continuous (not binary) signal per skill
- Multiple intent directions can be tracked simultaneously

**Open questions for MLX:**
- MLX doesn't have native hooks for activation extraction like PyTorch
- Would need to modify the forward pass or use `mlx.nn.Module` internals
- Feasibility TBD — may require custom inference code

**Key references:**
- "Representation Engineering" — Zou et al. 2023
- "Steering Llama 2 via Contrastive Activation Addition" — Turner et al. 2023
- "In-context Vectors" — Liu et al. 2023
- Anthropic's "Scaling Monosemanticity" (2024) — SAE-based feature directions

### 5. Online / Continual Learning

Most aggressive: continuously update expert models as the user interacts.
When the user says "play some music" and then actually uses the music
player → confirmed positive example → update LoRA adapter.

**Approaches:**
- Replay buffer with periodic micro-training
- MAML / Reptile for fast adaptation
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting

**Strengths:** System gets better over time, personalized.
**Weaknesses:** Complex, fragile, risk of model degradation.
Not recommended until the base system is stable.

## Recommended Path for Listen

### Phase 1: Accumulate What You Already Have (No New Models)

The expert classifiers already produce confidence scores on every cycle.
Currently discarded. Instead:
- Store each expert's confidence in a decaying `skill_activation` trace
- Add it to `RouterContext` so skills can read each other's recent activation
- Track first derivative (is music activation rising or falling?)

This alone gives you an intent vector with `N_skills` dimensions.

### Phase 2: Add Interpretable Heuristic Dimensions

Add `mood`, `energy`, `task_focus` as rule-based dimensions:
- `task_focus`: ratio of imperative/command sentences to declarative
- `energy`: speech rate (words per minute from buffer timestamps)
- `mood`: simple sentiment (positive/negative word ratio, or a tiny classifier)

### Phase 3: Graduate to Learned Dimensions

Train tiny LoRA classifiers for dimensions that matter most.
Same pipeline as existing experts — generate data, train, eval.
Each dimension is just another "expert" in the worker pool.

### Phase 4: Explore Steering Vectors (If Phase 1-3 Prove Valuable)

Extract activation directions from the expert models.
Compare against the simpler approaches. Only worth it if the
interpretable dimensions plateau in usefulness.

## Architecture (Where It Fits)

```
                    ┌──────────────────────────────────┐
                    │  IntentVector (new)               │
Transcript ────┐   │                                    │
               │   │  skill_activation: {music: 0.4, …} │
               ▼   │  mood: -0.3                        │
          ┌─────────┤  energy: 0.7                      │
          │ Expert  │  task_focus: 0.9                   │
          │ Server  │                                    │
          │ (MLX)   │  Updated every cycle               │
          └────┬────│  Dimensions decay toward baseline  │
               │    └──────────────┬───────────────────┘
               │                   │
               ▼                   ▼
          RouterResult      Skills read vector:
          (matches)         - Threshold modulation
                            - Proactive triggers
                            - Drift detection
                            - Dashboard display
```

The IntentVector sits between the classifier outputs and the skill handlers.
It's a **default skill** that runs on every cycle (no opt-out), doesn't
produce actions itself, but modifies the context that other skills see.

## Open Questions

1. **Decay rates:** How fast should dimensions decay? Probably different
   per dimension (mood decays slower than task_focus). Needs experimentation.

2. **Cross-dimension interactions:** Does low mood + high task_focus mean
   something different than low mood + low task_focus? Structured vector
   can't capture this without explicit rules.

3. **Calibration:** Expert confidence scores aren't calibrated probabilities.
   May need temperature scaling before feeding into the intent vector.

4. **Evaluation:** How do you eval an intent vector? Need labeled sessions
   with annotated intent trajectories. Expensive to create.

5. **Privacy:** The intent vector is a compact representation of the user's
   mental state over time. Even though everything is local, worth being
   thoughtful about persistence (in-memory only? or logged?).
