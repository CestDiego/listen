# Public Datasets for Listen — Phase 3 Training Data

> Research compiled 2026-03-10. Covers all 6 intent vector dimensions + multi-tool classifier augmentation.

## Quick Reference — Top Priority Datasets

| Dataset | Size | License | Primary Use | HuggingFace ID |
|---------|------|---------|-------------|----------------|
| **CLINC150-OOS** | 23.9k + 1,200 OOS | CC-BY-3.0 | Intent classification + negatives | `clinc/clinc_oos` |
| **MASSIVE** (en-US) | 11.5k train | CC-BY-4.0 | Music/audio/weather intents + slots | `AmazonScience/massive` |
| **Snips NLU** | ~14k (7 intents) | CC0-1.0 | PlayMusic + AddToPlaylist with slots | `snips/snips_built_in_intents` |
| **GoEmotions** | 58k | Apache-2.0 | Mood/wellbeing + negatives (27 emotions) | `google-research-datasets/go_emotions` |
| **EmpatheticDialogues** | 107k utterances | CC-BY-NC-4.0 | Wellbeing/distress detection (32 emotions) | `facebook/empathetic_dialogues` |
| **DailyDialog** | 102k utterances | CC-BY-NC-SA-4.0 | Mood + engagement + negatives | `li2017dailydialog/dailydialog` |
| **MELD** | 13.7k utterances | GPL-3.0 | Mood/arousal in dialogue | `declare-lab/MELD` |
| **EmoBank** | 10k sentences | CC-BY-SA-4.0 | Continuous VAD (valence-arousal-dominance) | JULIELab/EmoBank (GitHub) |
| **NRC VAD Lexicon** | 20k+ words | Research | Word-level arousal lookup (free baseline) | saifmohammad.com |
| **BANKING77** | 13k | CC-BY-4.0 | Pure negatives (banking queries) | `PolyAI/banking77` |
| **ESConv** | 31k utterances | MIT | Emotional support / wellbeing | `thu-coai/esconv` (GitHub) |
| **SLURP** | 72k | CC-BY-SA-4.0 | Spoken intents (music/audio hierarchy) | `qmeeus/slurp` |
| **HWU64** | 25.7k | CC-BY-4.0 | 64 intents including music/audio | HWU64 (GitHub) |

---

## By Dimension

### 1. Multi-Tool Classifier (music.*, wellbeing.check_in, future skills)

**Goal**: Augment current 291-example training set with diverse intents + negatives.

| Priority | Dataset | Usable Examples | Why |
|----------|---------|-----------------|-----|
| ★★★ | CLINC150-OOS | 23.9k + 1,200 OOS | Gold standard for OOS detection. Music intents: `play_music`, `next_song`, `change_volume`. |
| ★★★ | MASSIVE (en-US) | 11.5k | 60 intents: `play_music`, `audio_volume_up/down`, `music_query`, `weather_query`, `alarm_set`. Slot annotations. |
| ★★★ | Snips NLU | ~4k music | `PlayMusic` + `AddToPlaylist` with artist/track/genre slots. CC0 license. |
| ★★☆ | SLURP | ~72k (5k music) | Hierarchical: `music/play_music`, `audio/volume_up`. Has actual audio recordings. |
| ★★☆ | HWU64 | 25.7k | 64 intents, music/audio domain. Home assistant style. |
| ★★☆ | SGD (Schema-Guided Dialogue) | ~48k turns | Music/Media services with intent schemas. CC-BY-SA-4.0. |
| ★★☆ | BANKING77 | 13k | Pure negatives — banking queries that should NOT trigger any tool. |
| ★☆☆ | MultiWOZ | ~115k turns | Travel/booking negatives. Apache-2.0. |
| ★☆☆ | Persona-Chat | ~162k utterances | Chit-chat negatives. MIT license. |
| ★☆☆ | Topical-Chat | ~235k turns | Conversations *about* music (edge case: should NOT trigger play). |
| ★☆☆ | BFCL | ~2k | Function calling eval format. Apache-2.0. |

**Quick-start**: Filter MASSIVE + CLINC + Snips for music intents, map to our tool schema, add OOS examples as negatives.

**Intent mapping**:
```
Our Tool Call     → CLINC150        → MASSIVE           → Snips           → SLURP
music.play        → play_music      → play_music        → PlayMusic       → music/play_music
music.skip        → next_song       → —                 → —               → —
music.volume_up   → change_volume   → audio_volume_up   → —               → audio/volume_up
music.volume_down → change_volume   → audio_volume_down  → —               → audio/volume_down
wellbeing.check_in → —              → —                 → —               → —
(negative/OOS)    → oos (1,200)     → general_quirky    → —               → —
```

### 2. Mood [-1, 1] (sentiment/valence)

**Goal**: Replace word-list heuristic with trained classifier.

| Priority | Dataset | Usable Examples | Why |
|----------|---------|-----------------|-----|
| ★★★ | GoEmotions | 58k | 27 emotions, short text (~12 words avg), Apache-2.0. Map emotions to valence via Russell's circumplex. |
| ★★★ | DailyDialog | 102k utterances | Dialogue with 7 emotion labels. Closest to conversational style. |
| ★★★ | MELD | 13.7k | TV dialogue transcripts, 7 emotions + 3 sentiment. Spoken language with disfluencies. |
| ★★☆ | CMU-MOSEI | 23.5k | **Continuous sentiment (-3 to +3)** on real speech transcripts. 1,000 speakers. |
| ★★☆ | EmoBank | 10k | **Continuous VAD scores** — the only large text dataset with real-valued valence. CC-BY-SA-4.0. |
| ★★☆ | Super Emotion | 553k | Unified from 6 sources. Best for pretraining before fine-tuning. |
| ★★☆ | SemEval-2018 Task 1 | ~10k | Continuous valence/arousal regression on tweets. |
| ★☆☆ | ISEAR | 7.7k | Self-reported emotional situations ("I felt afraid when..."). |

**Key insight**: No single dataset matches our use case (short spoken transcript chunks with continuous valence). Strategy: pretrain on Super Emotion (553k), fine-tune on GoEmotions + DailyDialog + MELD, calibrate regression on EmoBank + CMU-MOSEI.

### 3. Wellbeing / Distress Detection

**Goal**: Augment current training data for `wellbeing.check_in` — catch burnout, self-doubt, imposter syndrome, emotional distress.

| Priority | Dataset | Usable Examples | Why |
|----------|---------|-----------------|-----|
| ★★★ | EmpatheticDialogues | ~20-30k (filtered) | Conversational, 32 emotion labels including *devastated, ashamed, guilty, lonely*. |
| ★★★ | ESConv | ~10-15k (seeker only) | Emotional support conversations. Problem types: depression, anxiety, job crisis, breakup. MIT license. |
| ★★★ | GoEmotions (neg subset) | ~15-20k | Filter to grief, fear, sadness, nervousness, disappointment. Apache-2.0. |
| ★★☆ | Amod Counseling | 3.5k | Real-sounding client statements ("I barely sleep and think about how I'm worthless"). RAIL-D license. |
| ★★☆ | DAIC-WOZ | 189 interviews | **Gold standard for spoken-language depression detection.** PHQ-8 severity scores. Research-only (USC). |
| ★★☆ | CEASE | 1.7k | 5 severity levels for suicidal ideation + cause categories. Graded distress. |
| ★★☆ | Dreaddit | 3.5k | Binary stress + 5 categories (PTSD, social, financial, anxiety). |
| ★☆☆ | Reddit Mental Health | 1.1M posts | 28 subreddit labels. Filter to short posts for spoken-style text. |
| ★☆☆ | AnnoMI | 10k utterances | Real MI therapy transcripts. CC-BY-4.0. |
| ★☆☆ | SAD | 6.8k | Stress/anxiety/depression. Research-only. |

**Ethical notes**:
- Mental health data requires extreme care. Never store identifiable info.
- DAIC-WOZ requires IRB-approved research protocol.
- Reddit mental health data: users did NOT consent to ML training. Use for internal models only.
- If the model detects suicidal ideation, surface crisis resources (988 Lifeline).

### 4. Energy [0, 1] (arousal/activation)

**Goal**: Replace words-per-chunk heuristic with arousal-aware classifier.

| Priority | Dataset | Usable Examples | Why |
|----------|---------|-----------------|-----|
| ★★★ | NRC VAD Lexicon | 20k+ words | **Zero-cost baseline**: word-level arousal lookup. Immediate upgrade. |
| ★★★ | EmoBank | 10k | Continuous arousal annotations on text. CC-BY-SA-4.0. |
| ★★☆ | IEMOCAP | 10k utterances | Gold-standard continuous arousal on conversational transcripts. Research-only (USC). |
| ★★☆ | SemEval-2018 Affect | ~10k | Text arousal regression benchmark. |
| ★★☆ | CMU-MOSEI | 23.5k | Large-scale text + continuous arousal. |
| ★☆☆ | MSP-Podcast | 100h+ | Naturalistic arousal. Podcast speech = close to mic-capture. |
| ★☆☆ | GoEmotions → arousal | 58k | Map 27 emotions to Russell's circumplex arousal dimension. |

**Quick win**: Use NRC VAD Lexicon as word-level arousal lookup to replace the naive words-per-chunk heuristic. No training needed.

### 5. Engagement [0, 1] (participation/activity)

**Goal**: Move beyond chunk-density to dialogue-act-aware engagement scoring.

| Priority | Dataset | Usable Examples | Why |
|----------|---------|-----------------|-----|
| ★★★ | AMI Meeting Corpus | 100h, ~170 meetings | Dialogue acts + emotion for meeting engagement. Free, transcribed, CC-BY-4.0. |
| ★★★ | Switchboard Dialog Act (SwDA) | 205k utterances | ~60 dialogue act tags (backchannels, questions, statements). Engagement proxy. |
| ★★☆ | ICSI Meeting Corpus | 70h, 75 meetings | Real academic meetings. MRDA dialogue acts. CC-BY-4.0. |
| ★★☆ | DailyDialog | 102k utterances | Dialogue act labels (inform, question, directive, commissive). |
| ★☆☆ | CANDOR | 1,656 conversations | Explicit engagement ratings. Research use. |
| ★☆☆ | Fisher English | 1,960h | Massive conversation data for self-supervised features. LDC license. |

**Strategy**: Train a dialogue act classifier on SwDA, then compute engagement as `f(chunk_density, backchannel_ratio, question_rate, mean_utterance_length)`.

---

## Recommended Training Pipeline

```
Phase 3a — Augment multi-tool classifier:
  CLINC-OOS (23k) ──────────┐
  MASSIVE en-US (11.5k) ────┤
  Snips PlayMusic (4k) ──────┼──→ Map to our tool schema ──→ LoRA fine-tune
  GoEmotions negs (58k) ────┤     Target: ~5-10k curated
  BANKING77 negs (13k) ─────┘

Phase 3b — Mood classifier (replaces word-list heuristic):
  Super Emotion (553k) ─────→ Pretrain
  GoEmotions + MELD (72k) ──→ Fine-tune
  EmoBank + CMU-MOSEI ──────→ Calibrate regression to [-1, 1]

Phase 3c — Wellbeing augmentation:
  EmpatheticDialogues ──────┐
  ESConv (seeker) ──────────┤
  GoEmotions neg emotions ──┼──→ Augment existing 291-example set
  Amod Counseling ──────────┘

Phase 3d — Energy quick-win:
  NRC VAD Lexicon ──────────→ Word-level arousal lookup (no training)

Phase 3e — Engagement upgrade:
  SwDA dialogue acts ───────→ Train dialogue act classifier
  AMI Meeting Corpus ───────→ Validate engagement scoring
```

## Quick-Start Code

```python
from datasets import load_dataset

# Tier 1 — immediate use
clinc = load_dataset("clinc/clinc_oos", "plus")
massive = load_dataset("AmazonScience/massive", "en-US")
banking = load_dataset("PolyAI/banking77")
emotions = load_dataset("google-research-datasets/go_emotions")
daily = load_dataset("li2017dailydialog/dailydialog")
meld = load_dataset("declare-lab/MELD")

# Tier 2 — augmentation
empathetic = load_dataset("facebook/empathetic_dialogues")
slurp = load_dataset("qmeeus/slurp")
```

## Papers to Read

- Coppersmith et al. (2018) — NLP for Suicide Risk Screening
- Chancellor & De Choudhury (2020) — Mental Health Prediction Ethics
- Harrigian et al. (2021) — Social Media Data for Mental Health (survey)
- Liu et al. (2021) — ESConv paper
- Rashkin et al. (2019) — EmpatheticDialogues paper
- Mohammad (2018) — NRC VAD Lexicon
- Russell (1980) — Circumplex Model of Affect (maps emotions to valence × arousal)
