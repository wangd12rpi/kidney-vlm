# Old vs new SFT rationale comparison

Prepared 2026-07-15. These are matched qualitative examples for presentation use.

## Slide-level summary

- Training teachers: 2,672 matched train questions. Old rationales average 163.2 words; new rationales average 129.9 words.
- Old teacher rationales walk through all four choices in 100% of matched rows. New rationales use Observation → Reasoning steps in 100%.
- Test SFTs: 271 shared test questions; 21 are jointly correct. On the shared set, old SFT accuracy is 25.5% and today's SFT accuracy is 26.2%.
- Within the jointly-correct pool, old rationale sections average 149.9 words and use a four-choice walkthrough in 100%; today's rationale sections average 126.7 words and use the two-step format in 100%.

## Checkpoint provenance

- Old SFT: `outputs/oncovlm/kidneyvlm_cot_newrationale_qwen35_9b_2ep/best/lora_adapter`
- Today's SFT: `outputs/oncovlm/sft/qwen3_5_9b_sft_caption_mcq_all_available_pathology_findings_image_step_cot_n2673_r8_projfrozen_20260715_020332_est/best/lora_adapter`
- Both are Qwen3.5-9B LoRA SFT checkpoints; no GRPO checkpoint is used here.

## A. Training-teacher rationales (same question and gold answer)

### A1. TCGA-HNSC / TCGA-CV-7183 — `8582030154938399198`

**Question:** Which pathology finding best matches this case?

- A. Broad pushing to irregular nests with prominent papilliform-exophytic architecture and laminated keratinization, consistent with well-differentiated keratinizing SCC.
- B. High-grade keratinizing squamoid carcinoma with central necrotic debris, anastomosing islands, and heterogeneity between solid-necrotic and stromal-infiltrative regions.
- C. Poorly differentiated solid squamous nests and broad tongues in dense desmoplastic stroma, with focal whorled reaction and skeletal muscle invasion.
- D. Predominantly solid irregular nests and broad tongues permeate desmoplastic stroma, with focal central maturation but no uniform keratinization.

**Gold answer:** Broad pushing to irregular nests with prominent papilliform-exophytic architecture and laminated keratinization, consistent with well-differentiated keratinizing SCC.

**Old GPT rationale (159 words)**

```text
<think>
The tumor shows broad pushing to irregular nests and tongues of squamous epithelium, a prominent papilliform to exophytic surface pattern, and abundant eosinophilic keratinization with laminated keratin and keratin pearl formation. The stroma is fibrous with patchy chronic inflammation, and the overall morphology is well differentiated, in keeping with the recorded G1 designation. The first choice fits because it captures the broad to irregular squamous nests, the exophytic papilliform architecture, and the strong laminated keratinizing, well-differentiated squamous appearance seen here. The second choice is not supported because there is no mention of high-grade features, central necrotic debris, or marked regional heterogeneity. The third choice does not match because poorly differentiated morphology, dense desmoplastic stroma, whorled stromal reaction, and skeletal muscle invasion are not described. The fourth choice is also less suitable because the case emphasizes abundant and well-formed keratinization rather than only focal central maturation, and it does not describe permeative growth through desmoplastic stroma as the dominant pattern.
</think>
<answer>
Broad pushing to irregular nests with prominent papilliform-exophytic architecture and laminated keratinization, consistent with well-differentiated keratinizing SCC.
</answer>
```

**New image-derived GPT rationale (125 words)**

```text
<think>
Step 1 — Observation: The sampled ROIs show a markedly exophytic papilliform/verruciform squamous proliferation with broad bulbous tongues and pushing nests. Tumor islands are composed of mature squamous cells with abundant eosinophilic cytoplasm and widespread laminated keratinization, including whorled keratin pearls. The epithelial-stromal interface is mostly broad and pushing rather than angulated small infiltrative nests, with accompanying chronic inflammatory/desmoplastic stroma. Cytologic maturation is maintained across much of the lesion.

Step 2 — Reasoning: This combination supports a well-differentiated keratinizing squamous carcinoma with papilliform-exophytic growth. The decisive features are the conspicuous papilliform architecture and uniform laminated keratinization within broad nests. A closer alternative would be irregular solid nests in desmoplastic stroma, but that pattern lacks the dominant exophytic papillary contour and orderly keratinizing maturation seen here.
</think>
<answer>
Broad pushing to irregular nests with prominent papilliform-exophytic architecture and laminated keratinization, consistent with well-differentiated keratinizing SCC.
</answer>
```

**Slide takeaway:** Concrete papilliform architecture, pushing nests, and laminated keratinization replace a mechanical four-choice walkthrough.

### A2. TCGA-SARC / TCGA-DX-A1L4 — `8727423578361554004`

**Question:** Which pathology finding best matches this case?

- A. Alternating hypercellular and collagenized/hyalinized zones with geographic pale hyalinization/necrotic areas, indicating greater intratumoral heterogeneity.
- B. Spindle cell sarcoma infiltrates adjacent soft tissue and skeletal muscle, with nodular perivascular lymphoid aggregates in collagenous-focally myxoid stroma.
- C. Monotonous uniform spindle cells in sweeping fascicles and storiform whorls within collagenous-focally myxoid stroma, with scattered chronic inflammatory cells.
- D. Patchy chronic inflammation with conspicuous peripheral and septal lymphoid aggregates in an infiltrative spindle tumor with alternating pale and hyperchromatic zones.

**Gold answer:** Monotonous uniform spindle cells in sweeping fascicles and storiform whorls within collagenous-focally myxoid stroma, with scattered chronic inflammatory cells.

**Old GPT rationale (164 words)**

```text
<think>
The sampled pathology shows a moderately to densely cellular spindle cell neoplasm made of relatively uniform elongated cells, arranged in sweeping fascicles with whorled to storiform areas, set in a collagenous to focally myxoid stroma. It also includes delicate vasculature and scattered chronic inflammatory cells, with a fairly monotonous nonlipogenic sarcomatous appearance. The first choice does not fit because alternating hypercellular and collagenized or hyalinized zones with geographic pale hyalinization or necrosis are not described. The second choice is not supported because infiltration into adjacent soft tissue and skeletal muscle and nodular perivascular lymphoid aggregates are not mentioned. The third choice matches best because it captures the uniform spindle cells, sweeping fascicular and storiform architecture, collagenous-focally myxoid background, and scattered chronic inflammatory cells seen here. The fourth choice is less consistent because conspicuous peripheral and septal lymphoid aggregates, infiltrative growth, and alternating pale and hyperchromatic zones are not described in the sampled fields. Overall, the morphology is most in keeping with the third choice.
</think>
<answer>
Monotonous uniform spindle cells in sweeping fascicles and storiform whorls within collagenous-focally myxoid stroma, with scattered chronic inflammatory cells.
</answer>
```

**New image-derived GPT rationale (136 words)**

```text
<think>
Step 1 — Observation: The sampled ROIs show a spindle cell proliferation composed of fairly uniform slender cells arranged in long sweeping fascicles and frequent storiform/whorled patterns. The background stroma is predominantly collagenous with focally paler looser/myxoid areas. Inflammatory cells are present but mostly as scattered chronic inflammatory infiltrates, with some patchier denser zones. There are hyalinized collagen bundles and small vessels, but no geographic necrotic-appearing pale zones are apparent in these ROIs, and no clear adjacent skeletal muscle interface is shown.

Step 2 — Reasoning: The key combination is architectural monotony—uniform spindle cells in intersecting fascicular and storiform whorls—set in collagenous to focally myxoid stroma with only scattered chronic inflammation. That favors the more orderly spindle-cell pattern rather than a more heterogeneous lesion with geographic hyalinization/necrosis or a pattern dominated by conspicuous peripheral/septal lymphoid aggregates.
</think>
<answer>
Monotonous uniform spindle cells in sweeping fascicles and storiform whorls within collagenous-focally myxoid stroma, with scattered chronic inflammatory cells.
</answer>
```

**Slide takeaway:** The new rationale scopes claims to sampled ROIs and separates spindle-cell architecture from its closest alternative.

### A3. TCGA-GBM / TCGA-26-5133 — `7977529583008979548`

**Question:** Which pathology finding best matches this case?

- A. Necrotic tumor shows cavitary spaces with abundant recent and organizing hemorrhage and irregular thick-walled glomeruloid microvascular proliferation.
- B. Prominent microvascular proliferation with hemorrhagic change amid sheet-like to vaguely fascicular infiltrative glial tumor in fibrillary background.
- C. Microcystic-fibrillary hypercellular glial neoplasm with broad pallor/tissue-dropout necrotic-degenerative areas, adjacent hemorrhage, stromal edema, and no brisk lymphoid infiltrate.
- D. Regional heterogeneity with alternating solid hypercellular fields and necrosis-edema-like pallor; no brisk lymphoid inflammatory infiltrate.

**Gold answer:** Microcystic-fibrillary hypercellular glial neoplasm with broad pallor/tissue-dropout necrotic-degenerative areas, adjacent hemorrhage, stromal edema, and no brisk lymphoid infiltrate.

**Old GPT rationale (167 words)**

```text
<think>
The sections show a highly cellular infiltrative glial neoplasm with a fibrillary to microcystic background, broad areas of pallor and tissue dropout, adjacent hemorrhagic change, focal stromal edema, and no brisk lymphoid infiltrate. The first choice does not fit because cavitary spaces, organizing hemorrhage, and irregular thick-walled glomeruloid vascular change are not described; the vessels are instead delicate to somewhat prominent and thin-walled. The second choice is only partly compatible because hemorrhagic change and a vascular background are present, but sheet-like to vaguely fascicular architecture is not given, and the description emphasizes thin-walled rather than overtly proliferative vessels. The third choice best matches because it captures the hypercellular glial tumor in a microcystic-fibrillary background together with broad pallor/tissue-dropout necrotic-degenerative change, adjacent hemorrhage, stromal edema, and the absence of a brisk lymphoid infiltrate. The fourth choice reflects the noted patch-to-patch heterogeneity and lack of brisk inflammation, but it is less complete because it does not include the microcystic-fibrillary glial background or the adjacent hemorrhagic and edematous vascular-stromal features.
</think>
<answer>
Microcystic-fibrillary hypercellular glial neoplasm with broad pallor/tissue-dropout necrotic-degenerative areas, adjacent hemorrhage, stromal edema, and no brisk lymphoid infiltrate.
</answer>
```

**New image-derived GPT rationale (136 words)**

```text
<think>
Step 1 — Observation: The sampled tissue shows a hypercellular glial-appearing neoplasm with a fibrillary to microcystic background and uneven cellularity. There are broad pale eosinophilic zones with tissue dropout/degenerative change rather than sharply circumscribed cavitary necrosis. Multiple foci of fresh hemorrhage and blood-filled clefts are present along tumor fragments. The stroma looks edematous, and the inflammatory cells are scattered without a brisk dense lymphoid infiltrate in these ROIs.

Step 2 — Reasoning: This combination favors a necrotic-degenerative, microcystic-fibrillary glial tumor with hemorrhagic change and edema. The key is the diffuse pallor/tissue-dropout quality of the necrotic areas within a microcystic glial background, not a predominantly solid tumor simply alternating with pallid zones. A competing pattern would be one centered on conspicuous glomeruloid microvascular proliferation and cavitary hemorrhagic necrosis, which is not apparent in the sampled ROIs.
</think>
<answer>
Microcystic-fibrillary hypercellular glial neoplasm with broad pallor/tissue-dropout necrotic-degenerative areas, adjacent hemorrhage, stromal edema, and no brisk lymphoid infiltrate.
</answer>
```

**Slide takeaway:** The new rationale discriminates diffuse tissue-dropout pallor from cavitary necrosis and glomeruloid vascular proliferation.

## B. SFT test outputs (same question, same correct prediction)

These three cases are jointly correct so the comparison focuses on rationale form and content.

### B1. TCGA-COAD / TCGA-G4-6626 — `59001304306035156`

**Question:** Which pathology finding best matches this case?

- A. Submucosal-adjacent infiltrative moderately differentiated glands with focal cribriform crowding and preserved peripheral non-neoplastic crypt architecture.
- B. Irregular elongated and cribriform glands with variable luminal mucin, deeply traversing bowel wall in prominent desmoplastic stroma.
- C. Crowded irregular angulated-to-fused glands with patchy cribriforming, dirty luminal necrosis, and invasion into fibrotic stroma beside non-neoplastic mucosa.
- D. Irregular angulated-to-cribriform glands with focal papillary-villiform contours, luminal debris, abundant desmoplastic stroma, and patchy inflammation; no mucinous pattern.

**Gold and both predicted answers:** Crowded irregular angulated-to-fused glands with patchy cribriforming, dirty luminal necrosis, and invasion into fibrotic stroma beside non-neoplastic mucosa.

**Old SFT output (136 words)**

```text
The sections show an infiltrative gland-forming epithelial neoplasm with irregular angulated to fused glands, patchy cribriforming, and crowded back-to-back architecture. There is also dirty luminal necrotic debris, abundant desmoplastic stroma, and invasion into fibrotic stroma adjacent to non-neoplastic colonic mucosa. The first choice is less fitting because submucosal adjacency and preserved peripheral crypt architecture are not described. The second choice captures irregular and cribriform glands with desmoplastic stroma, but variable luminal mucin and deeply traversing bowel wall are not supported. The third choice best matches because it includes the crowded irregular angulated-to-fused glands, patchy cribriforming, dirty luminal necrosis, and invasion into fibrotic stroma beside non-neoplastic mucosa, all of which are directly described. The fourth choice is less accurate because papillary-villiform contours and patchy inflammation are not mentioned, even though desmoplastic stroma and luminal debris are present.
</think>

<answer>
Crowded irregular angulated-to-fused glands with patchy cribriforming, dirty luminal necrosis, and invasion into fibrotic stroma beside non-neoplastic mucosa.
</answer>
```

**Today's SFT output (118 words)**

```text
Step 1 — Observation: The sampled ROIs show infiltrative irregular glands extending through fibrotic/desmoplastic stroma, with many glands angulated, fused, and focally cribriform. Several gland lumina contain eosinophilic granular debris consistent with dirty necrosis. The interface includes adjacent non-neoplastic colonic-type mucosa with preserved crypt architecture, while the invasive glands are haphazard and not arranged in a mucinous pattern.

Step 2 — Reasoning: This combination supports an invasive adenocarcinoma with crowded irregular angulated-to-fused glands, patchy cribriforming, and dirty luminal necrosis invading fibrotic stroma next to native mucosa. The decisive feature is the dirty necrotic debris within irregular invasive glands, which is more specific than a generic desmoplastic infiltrative pattern. A mucinous pattern is not apparent in the sampled ROIs.
</think>

<answer>
Crowded irregular angulated-to-fused glands with patchy cribriforming, dirty luminal necrosis, and invasion into fibrotic stroma beside non-neoplastic mucosa.
</answer>
```

**Slide takeaway:** Both SFTs answer correctly; the new SFT separates gland geometry and dirty necrosis from the inference instead of traversing all choices.

### B2. TCGA-HNSC / TCGA-HD-A634 — `6332281543206486651`

**Question:** Which pathology finding best matches this case?

- A. Keratinizing invasive SCC with ulcerated/reactive surface mucosa and adjacent soft tissue-adipose at specimen edge.
- B. Broad infiltrative tongues in dense desmoplasia with patchy pallor and hemorrhagic stromal remodeling suggestive of necrotic-degenerative change.
- C. Irregular nests, cords, and broad trabeculated islands with central keratinization, basaloid peripheral cells, stromal hemorrhage, and lymphoplasmacytic inflammation.
- D. Dense peritumoral lymphoid aggregates with keratinizing nests in desmoplastic stroma, extending near adipose-rich soft tissue.

**Gold and both predicted answers:** Irregular nests, cords, and broad trabeculated islands with central keratinization, basaloid peripheral cells, stromal hemorrhage, and lymphoplasmacytic inflammation.

**Old SFT output (168 words)**

```text
The lesion is an infiltrative epithelial neoplasm with irregular nests, cords, and broad trabeculated islands in a desmoplastic to fibrotic stroma. Several nests show central eosinophilic keratinization, the peripheral cells are basaloid to squamoid, and there is stromal hemorrhage with a lymphoplasmacytic inflammatory background. The first choice is less fitting because ulcerated or reactive surface mucosa is not described, and the emphasis on adjacent soft tissue-adipose at the specimen edge is not the key pattern given. The second choice captures desmoplasia and hemorrhagic stromal change, but its broad infiltrative tongues and necrotic-degenerative change are not the best match to the described nests, cords, trabeculated islands, and keratinization. The third choice aligns best because it includes the irregular nested and cord-like architecture, broad trabeculated islands, central keratinization, basaloid peripheral cells, stromal hemorrhage, and lymphoplasmacytic inflammation that are all specifically described. The fourth choice is less accurate because dense peritumoral lymphoid aggregates and extension near adipose-rich soft tissue are not described, even though keratinizing nests in desmoplastic stroma are present.
</think>

<answer>
Irregular nests, cords, and broad trabeculated islands with central keratinization, basaloid peripheral cells, stromal hemorrhage, and lymphoplasmacytic inflammation.
</answer>
```

**Today's SFT output (136 words)**

```text
Step 1 — Observation: The sampled ROIs show an infiltrative epithelial neoplasm composed of irregular nests, cords, and broad trabeculated islands within a desmoplastic stroma. Several tumor islands have central eosinophilic keratinization/dyskeratotic whorls. The periphery of many nests is darker and more basaloid than the central keratinizing areas. There is conspicuous stromal hemorrhage and a dense lymphoplasmacytic inflammatory infiltrate around and between tumor islands.

Step 2 — Reasoning: This combination supports a keratinizing squamous carcinoma with basaloid peripheral cells and central keratinization, plus hemorrhagic and inflammatory stromal reaction. The decisive feature is the nested/trabeculated architecture with central keratinization rather than broad tongues of tumor in dense desmoplasia. A close alternative would be a keratinizing SCC with ulcerated surface mucosa, but the sampled ROIs emphasize infiltrative islands in hemorrhagic inflamed stroma rather than a reactive surface epithelium.
</think>

<answer>
Irregular nests, cords, and broad trabeculated islands with central keratinization, basaloid peripheral cells, stromal hemorrhage, and lymphoplasmacytic inflammation.
</answer>
```

**Slide takeaway:** Both SFTs answer correctly; the new SFT uses center-to-periphery tumor-island morphology and contrasts only the closest alternative.

### B3. TCGA-MESO / TCGA-SC-A6LP — `6826264933257959015`

**Question:** Which pathology finding best matches this case?

- A. Epithelioid tubulopapillary-trabecular pleural tumor infiltrates through adipose lobules, forming irregular slit-like spaces in dense desmoplastic stroma.
- B. Invasive pleural tumor shows epithelioid-to-spindled trabeculated, nested, focally whorled architecture with irregular elongated slit-like spaces in desmoplastic stroma.
- C. Confluent epithelioid sheets with tubulopapillary-vaguely trabecular areas in broad fibrotic hyalinized stroma and focal hemorrhagic change.
- D. Relatively uniform densely cellular spindle-cell fascicles with sweeping storiform-whorled architecture, broad nodules, delicate stromal clefting, no papillary or glandular elements.

**Gold and both predicted answers:** Relatively uniform densely cellular spindle-cell fascicles with sweeping storiform-whorled architecture, broad nodules, delicate stromal clefting, no papillary or glandular elements.

**Old SFT output (152 words)**

```text
The lesion is described as a densely cellular spindle-cell neoplasm with intersecting fascicles and sweeping storiform to whorled architecture, broad nodular expanses, delicate stromal clefting, and relatively uniform morphology across sampled areas. It also specifically lacks conspicuous gland formation, keratinization, and overt papillary or tubulopapillary architecture. The first choice does not fit because it emphasizes epithelioid tubulopapillary-trabecular growth and adipose infiltration, neither of which is described. The second choice is also unsupported because it calls for epithelioid-to-spindled nested and trabeculated architecture with irregular slit-like spaces, whereas the sampled tumor is predominantly uniform spindle-cell fascicles and whorls without those epithelioid features. The third choice mismatches by focusing on confluent epithelioid sheets, tubulopapillary-vaguely trabecular areas, and hemorrhagic change, none of which are present in the described sections. The fourth choice matches the key findings: dense spindle-cell proliferation in fascicles with sweeping storiform-whorled pattern, broad nodules, delicate clefting, and absence of papillary or glandular elements.
</think>

<answer>
Relatively uniform densely cellular spindle-cell fascicles with sweeping storiform-whorled architecture, broad nodules, delicate stromal clefting, no papillary or glandular elements.
</answer>
```

**Today's SFT output (130 words)**

```text
Step 1 — Observation: The sampled ROIs show a densely cellular spindle-cell proliferation arranged in broad sweeping fascicles with a storiform to whorled appearance. The tumor forms large confluent nodules with relatively uniform cellularity and elongated nuclei. There are delicate stromal clefts/slit-like spaces between fascicles, but no obvious papillary, tubular, or glandular structures. The background stroma is collagenous rather than markedly desmoplastic, and hemorrhagic change is not apparent in the sampled ROIs.

Step 2 — Reasoning: This combination favors a uniform spindle-cell neoplasm with fascicular-storiform architecture and clefting. The decisive point is the lack of epithelioid tubulopapillary or trabeculated nests with slit-like spaces in desmoplastic stroma, which would be required for the closest alternative pleural tumor pattern. Instead, the lesion is dominated by broad spindle-cell nodules with sweeping whorled growth.
</think>

<answer>
Relatively uniform densely cellular spindle-cell fascicles with sweeping storiform-whorled architecture, broad nodules, delicate stromal clefting, no papillary or glandular elements.
</answer>
```

**Slide takeaway:** Both SFTs answer correctly; the new SFT grounds the decision in fascicles, nodules, nuclei, clefts, and stromal quality.

## Interpretation boundaries

- This is a legacy-versus-updated CoT pipeline comparison, not a format-only ablation. The SFT datasets, initialization, and learning rates also differ.
- Existing test outputs use each checkpoint's native inference instruction: the old prompt requests a generic choice comparison, while today's prompt explicitly requests Observation → Reasoning.
- The new teacher dataset is image-conditioned and answer-validated, not independently pathologist-verified.
- The six cases are selected qualitative examples. The aggregate matched-set statistics above should accompany them to avoid implying that three examples establish overall efficacy.
