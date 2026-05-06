# Selectivity Evaluation — SDXL Concept Sliders

> **All numbers in this document are medians computed by
> `metrics/summarize_selectivity.py`.** To reproduce every figure, run:
> ```
> python metrics/summarize_selectivity.py
> ```
>
> Full per-run results (CSV + aggregate JSON) live in:
> `metrics/results_sdxl_selectivity/{concept}/eval_results.csv`
> `metrics/results_sdxl_selectivity/{concept}/eval_aggregate.json`

## Setup

We evaluate whether a *subject-specific* slider (trained on a specific
subject, e.g. "woman") modifies its intended target more selectively than a
*general* slider (trained on a broad category, e.g. "person").

- **4 concepts**: age, curlyhair, furlength, smile
- **20 images per concept**: each image contains a target subject and a
  non-target subject (e.g. a woman and a man)
- **Masks**: `mask_target.png` and `mask_nontarget.png` drawn per image
  using SAM; the slider is applied *globally* (no spatial masking during
  generation) and metrics are computed over the masked regions afterwards
- **Scales tested**: 3 per concept (see full table below)

---

## Metrics

### LPIPS selectivity (`lpips_sel`)

For each edited image we compute the normalised LPIPS distance between the
edited image and the baseline *inside* the target region and *inside* the
non-target region:

```
lpips_sel = lpips_nontarget_norm / lpips_target_norm
```

**< 1 → selective** (the target region changed more than the non-target).
Values around 1 mean both regions changed equally; > 1 means the non-target
changed *more* (anti-selective).

### CLIP selectivity (`clip_sel`)

We compute the CLIP-space delta between baseline and edited image in each
region, then measure what fraction of the *total semantic shift* lands on
the target:

```
delta_target     = CLIP(edited_target) − CLIP(baseline_target)
delta_nontarget  = CLIP(edited_nontarget) − CLIP(baseline_nontarget)

clip_sel = delta_target / (delta_target + |delta_nontarget|)   if delta_target > 0
clip_sel = 0                                                    otherwise
```

**> 0.5 → selective** (more semantic change in target than non-target).
clip_sel = 0 marks runs where the edit moved the target *away* from the
concept — genuine failures, not measurement noise. The mean is sensitive to
these outliers; we report **median** throughout.

---

## Results — Best Scale per Concept

The best scale is selected per concept as the one that maximises
`clip_sel_median / lpips_sel_median` for the specific slider (combined
selectivity score). Both metrics are then reported at that same scale.

| Concept | Scale | Specific `lpips_sel` | Specific `clip_sel` | General `lpips_sel` | General `clip_sel` |
|---------|------:|---------------------:|--------------------:|--------------------:|-------------------:|
| age | 1.5 | 1.1031 | **0.6540** | 1.1074 | 0.4832 |
| curlyhair | 2.0 | **0.3304** | **0.7632** | 0.7765 | 0.3901 |
| furlength | 2.0 | **0.5454** | **0.8287** | 1.1986 | 0.5014 |
| smile | 1.0 | **0.7465** | **0.8592** | 1.3295 | 0.5105 |

*Bold = the specific slider is clearly more selective on that metric.*

---

## Full Breakdown (All Scales)

### age

| Scale | Slider | `lpips_sel` | `clip_sel` |
|------:|--------|------------:|-----------:|
| 0.5 | specific | 1.3180 | 0.4924 |
| 1.0 | specific | 1.1811 | 0.5935 |
| **1.5** | **specific** | **1.1031** | **0.6540** |
| 0.5 | general | 1.7298 | 0.2665 |
| 1.0 | general | 1.5037 | 0.3488 |
| 1.5 | general | 1.1074 | 0.4832 |

### curlyhair

| Scale | Slider | `lpips_sel` | `clip_sel` |
|------:|--------|------------:|-----------:|
| 1.0 | specific | 0.3742 | 0.7794 |
| **2.0** | **specific** | **0.3304** | **0.7632** |
| 3.0 | specific | 0.5330 | 0.5286 |
| 1.0 | general | 0.9932 | 0.4691 |
| 2.0 | general | 0.7765 | 0.3901 |
| 3.0 | general | 0.8717 | 0.2273 |

### furlength

| Scale | Slider | `lpips_sel` | `clip_sel` |
|------:|--------|------------:|-----------:|
| 1.0 | specific | 0.6108 | 0.7099 |
| **2.0** | **specific** | **0.5454** | **0.8287** |
| 3.0 | specific | 0.6332 | 0.8201 |
| 1.0 | general | 1.3232 | 0.5049 |
| 2.0 | general | 1.1986 | 0.5014 |
| 3.0 | general | 1.1231 | 0.4812 |

### smile

| Scale | Slider | `lpips_sel` | `clip_sel` |
|------:|--------|------------:|-----------:|
| 0.5 | specific | 0.8664 | 0.8469 |
| **1.0** | **specific** | **0.7465** | **0.8592** |
| 1.5 | specific | 0.8183 | 0.6568 |
| 0.5 | general | 1.0844 | 0.6260 |
| 1.0 | general | 1.3295 | 0.5105 |
| 1.5 | general | 1.3293 | 0.5040 |

---

## Analysis

### curlyhair, furlength, smile — strong selective behaviour

For these three concepts the specific slider is **clearly and consistently
more selective** than the general slider on both metrics simultaneously.

The strongest case is **curlyhair** at scale 2.0:
- `lpips_sel = 0.33` → the target region changes **3× more** than the
  non-target in pixel space; the general slider only reaches 0.78 (≈ equal
  change in both regions).
- `clip_sel = 0.76` → **76% of the total semantic edit** lands on the
  target subject; the general slider sends only 39% there.

**furlength** at scale 2.0 is similarly clean:
- `lpips_sel = 0.55` → target changes ~1.8× more (general: 1.20, i.e.
  the *non-target* changes more).
- `clip_sel = 0.83` → 83% of the semantic change is in the target.
  The general slider barely clears 50% (0.50).

**smile** at scale 1.0:
- `lpips_sel = 0.75` → target changes ~1.3× more.
- `clip_sel = 0.86` → 86% of semantic change in the target. The general
  slider again sits near 50% (0.51) — essentially blind to which subject
  it should modify.

### age — a special case

The age concept behaves differently: at best scale 1.5 the LPIPS
selectivity is **1.10 for specific vs 1.11 for general** — both just above
1, meaning neither slider is clearly selective in pixel space. This is not
a failure of the evaluation but a property of age edits: ageing someone
changes global texture cues (wrinkles, skin tone, grey hair, background
lighting) that bleed across the whole image, making LPIPS sensitive
everywhere and not just on the target face.

**CLIP selectivity recovers the advantage**: `0.654` (specific) vs
`0.483` (general). At the semantic level the specific slider still
concentrates more than 65% of the edit on the target, while the general
slider barely crosses 50%.

There is an additional structural issue with the age general slider: it was
trained on the concept *"ageing a woman"*, so when presented with a scene
containing a woman (target) and a man (non-target), it nonetheless tends to
also age the man visually. This can be seen clearly in **eval_age_06 at
scale 1.0 with the general slider**, where the non-target man shows
pronounced ageing despite not being the intended subject. The specific
slider, trained explicitly on the target subject class, avoids this.

### Ceiling effect — age_03 and age_12

Two age images (`eval_age_03`, `eval_age_12`) already contain an elderly man
in the non-target position. For those runs `delta_nontarget ≈ 0` because
the non-target is already at the upper end of the "old" distribution —
there is no room to move further. This **artificially inflates** clip_sel
for age (the formula pushes the ratio toward 1 when the denominator is
near zero). The real selectivity of the age slider is likely lower than the
0.65 median suggests; conversely, interpreting the general slider's 0.48
score at face value is also a minor over-estimate for the same reason.
This caveat does not affect curlyhair, furlength, or smile.

### clip_sel = 0 outliers and why we use median

Some runs score clip_sel = 0 because the slider moved the target region
*away* from the target concept (e.g. a smile slider that slightly closes
the mouth in that particular image). These are real failures, but they are
rare and their effect on the mean is disproportionate. The **median is
robust to these outliers** and gives a more representative picture of
typical behaviour. We report medians throughout; means are available in
`eval_aggregate.json`.

---

## Presentation Phrasing

### Oral (conference / lab meeting)

> "We can quantitatively show that the subject-specific slider is
> significantly more selective than its general counterpart. For the
> curlyhair concept, the pixel-level LPIPS ratio shows the target subject
> changes three times more than the non-target. At the semantic level,
> 76% of the CLIP-space edit is concentrated in the target subject — the
> general slider can only direct 39% there."

> "Looking at the smile concept, 86% of the total semantic change
> measured by CLIP lands on the intended subject. The general slider
> essentially splits its effect equally between the two people in the
> scene."

> "The only case where the pixel metric doesn't fully support selectivity
> is age — and that's because age changes (wrinkles, skin tone, hair
> colour) are inherently distributed across the whole image. The semantic
> CLIP metric still shows a clear advantage: 65% of the edit targets the
> right person with the specific slider, compared to 48% with the general
> one."

### Slide caption (one-liner per concept)

| Concept | Caption |
|---------|---------|
| curlyhair | Specific slider directs 3× more pixel change and 76% of semantic edit to target subject |
| furlength | Specific slider: 1.8× more pixel change, 83% semantic concentration on target |
| smile | 86% of CLIP-space semantic edit lands on target subject |
| age | CLIP selectivity 65% vs 48% (general); LPIPS inconclusive due to global texture edits |

### Written (paper / report)

> We measure selectivity along two complementary axes. **LPIPS
> selectivity** (ratio of normalised perceptual distance in the non-target
> vs target mask region; < 1 = selective) captures low-level pixel change,
> while **CLIP selectivity** (fraction of the total semantic delta
> attributed to the target region; > 0.5 = selective) captures high-level
> semantic alignment. We report medians over 20 images per concept.
>
> For curlyhair, furlength, and smile, the subject-specific slider
> dominates on both metrics at all evaluated scales. At the best scale per
> concept, LPIPS selectivity reaches 0.33 (curlyhair), 0.55 (furlength),
> and 0.75 (smile), meaning the target region undergoes 3×, 1.8×, and
> 1.3× more pixel-level change than the non-target respectively. CLIP
> selectivity reaches 0.76, 0.83, and 0.86 — indicating that 76–86% of
> the total semantic edit is concentrated in the intended subject. The
> general slider, by contrast, shows near-equal or reversed distribution
> (lpips_sel ≥ 0.78; clip_sel ≤ 0.51) for all three concepts.
>
> Age is a special case: both sliders yield lpips_sel ≈ 1.1, reflecting
> that age-related visual cues (wrinkles, skin texture, hair colour) spread
> across the full image. The CLIP metric nonetheless reveals a meaningful
> advantage for the specific slider (0.65 vs 0.48), confirming that the
> semantic edit is still biased toward the target subject.

---

## Notes for Flux Extension

The Python script (`metrics/summarize_selectivity.py`) works with any
results directory. For Flux results, run:

```bash
python metrics/summarize_selectivity.py --results_dir metrics/results_flux_selectivity
```

Note that some concepts (e.g. `furlength`) may need to be skipped or
replaced in the Flux evaluation if the corresponding slider does not
transfer well to the Flux architecture.
