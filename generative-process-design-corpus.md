# Training Corpus for Generative Process Design

**Source discovery, build plan, and feasibility assessment**
Compiled 2026-08-31. Method: 6 parallel research agents (one per source modality) → adversarial verification pass that fetched every candidate URL. 96 raw candidates → 77 deduplicated → **72 verified** (14 direct, 27 convertible, 31 weak), 5 rejected as unconfirmable.

---

## 1. Bottom line

**Feasibility: moderate. Build it as a design copilot, not an autonomous designer.**

- A corpus of **~1,000–1,200 real semantic flowsheets** plus **unbounded valid-by-construction synthetic data** plus **5k–50k noisy extracted flowsheets** is assemblable from verified public sources in roughly 3–4 months of engineering.
- That scale supports flowsheet **autocompletion, control-structure generation (PFD→P&ID), topology suggestion, and error correction** at the quality the literature already demonstrates (73–89% top-5 accuracy on control-structure tasks).
- It does **not** support open-ended, parameterized, economically-sound novel process design. Public real topologies number ~10³. Molecular generative models (the SMILES analogy this field leans on) had ~10⁸ training molecules. No public source carries operating parameters, stream conditions, or economics at scale.
- The binding constraint is not modeling. It is that every large real-flowsheet corpus ever assembled (Theisen 1,005 PFDs; Hirtreiter 312 P&IDs; Vogel 223 flowsheets) is **unreleased for copyright reasons**. Public data is the residue.

---

## 2. Verified source inventory

### Tier A — Semantic topology, usable now (direct)

| Source | What | Size | Format | License |
|---|---|---|---|---|
| [SFILES2 toolkit](https://github.com/process-intelligence-research/SFILES2) (TU Delft) | Graph↔SFILES 2.0 conversion, augmentation, flowsheet generator, 9 real DWSIM flowsheet graphs | 9 graphs + unbounded generator (papers used ~8k) | NetworkX pickle / SFILES strings | MIT |
| [Hirtreiter PFD→P&ID corpus](https://zenodo.org/records/7658798) | Paired PFD / control-extended P&ID SFILES strings, train/test/augmented splits | ~10k–100k pairs, 24.9 MB JSON | SFILES 2.0 JSON | CC-BY-4.0 |
| [DEXPI TrainingTestCases](https://gitlab.com/dexpi/TrainingTestCases) | Reference P&IDs in Proteus/DEXPI XML incl. BASF, Equinor, Covestro cases | 220 XML files, ~83 case folders (~14 realistic full P&IDs) | Proteus XML | CC-BY-4.0 |
| [PID2Graph](https://zenodo.org/records/14803338) (DLR) | P&ID images + GraphML graph ground truth | 772 P&IDs (72 real, 700 random-topology synthetic), 9.3 GB | GraphML + PNG | CC-BY-SA-4.0 |
| [pyDEXPI](https://github.com/process-intelligence-research/pyDEXPI) | DEXPI 1.3 data model + template-based synthetic P&ID generator → NetworkX | Unbounded generator + sample patterns | DEXPI XML → NetworkX | **AGPL-3.0** |
| [FOSSEE DWSIM flowsheets](https://dwsim.fossee.in/flowsheeting-project/completed-flowsheet) (IIT Bombay) | Curated completed simulations (CO2 capture, extractive distillation, H2 from biogas, …) | 275 flowsheets | .dwxmz (zipped open XML) | unstated (educational; confirm) |
| [DWSIM-Flowsheets mirror](https://github.com/Sashi20/DWSIM-Flowsheets) | FOSSEE snapshot | 194 projects (overlaps FOSSEE) | .dwxmz | none declared |
| [DWSIM built-in samples](https://github.com/DanWBR/dwsim) | Simulator sample library | 186 files | .dwxmz/.dwxml | GPL-3.0 |
| [IDAES examples](https://github.com/IDAES/examples) (US DOE) | Power/chemical flowsheets as Pyomo/IDAES Arcs | ~15–20 flowsheets | Python (graph extractable) | IDAES OSS |
| [WaterTAP](https://github.com/watertap-org/watertap) (US DOE) | Water-treatment trains (RO, MD, ASM, ADM, BSM2, …) | 60 flowsheet modules, ~25–35 distinct trains | Python/Pyomo Arcs | BSD-style |
| [BioSTEAM Bioindustrial-Park](https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park) | Biorefinery designs (cellulosic, sugarcane, biodiesel, …) | ~33–36 packages | Python (runtime graph) | MIT |
| [OpenModelica library index](https://github.com/OpenModelica/OMPackageManager) | 179 open Modelica libraries incl. ClaRa, ThermoSysPro, ThermoPower, TransiEnt | 179 libraries, thousands of schematics | .mo (connect() parseable) | per-library |
| [ThermoSysPro](https://gitlab.pam-retd.fr/thermosysproandco/ThermoSysPro) (EDF) | Power-plant component/flowsheet models (GitHub is a stub; use GitLab) | 16 sub-packages | .mo | Modelica-style |
| [ClaRa](https://github.com/xrg-simulation/ClaRa-official) | Clausius-Rankine plants w/ instrumentation & control | full library + examples | .mo | BSD-3 |

### Tier B — Convertible via parser (simulator files)

| Source | Size | Conversion path |
|---|---|---|
| [COCO/COFE gallery](https://www.cocosimulator.org/index_sample.html) | ~90 flowsheets (.fsd) | COFE COM automation (Windows); covers Petlyuk, reactive/azeotropic distillation, LNG, crackers |
| [PSEcommunity files](https://psecommunity.org/materials/calculationfiles) | ~49 Aspen/HYSYS/ProMax files | Aspen .bkp is ASCII → parseable; .hsc binary needs HYSYS COM |
| [NREL biorefinery Aspen models](https://www.nrel.gov/extranet/biorefinery/aspen-models/) | handful of large .bkp packages | .bkp parser (site DNS-blocked from sandbox; confirmed via index) |
| [GitHub .bkp sweep](https://github.com/search?q=extension%3Abkp+aspen&type=code) | 313 indexed .bkp files | .bkp parser; per-repo licenses mostly absent |
| [HYSYS repos on GitHub](https://github.com/search?q=hysys&type=repositories) | 150 repos | weak — .hsc binary, needs HYSYS license |
| [Modelica Standard Library](https://github.com/modelica/ModelicaStandardLibrary) + [LBNL Buildings](https://github.com/lbl-srg/modelica-buildings) | thousands of schematics | connect() parsing; adjacent domain — pretraining only |

### Tier C — Convertible via digitization (images/PDFs)

| Source | Size | Notes |
|---|---|---|
| [PID_dataset (Gupta, ASU)](https://zenodo.org/records/8028570) | 6.7 GB real industry/scraped P&IDs | CC-BY-4.0; needs digitizer |
| [Digitize-PID](https://huggingface.co/datasets/hamzas/digitize-pid-yolo) (TCS) | 500 synthetic P&IDs | HF mirror has symbol bboxes only; original Google Drive has lines |
| [DOE/NETL FEED studies (OSTI)](https://www.osti.gov/biblio/1897679) | dozens of reports w/ P&ID appendices (exemplar: 44.6 MB MTR CO2-capture FEED) | US-gov open; digitize drawings |
| [NETL Bituminous Baseline](https://www.netl.doe.gov/sites/default/files/netl-file/BitBase_FinRep_Rev2.pdf) | BFDs + full stream tables | rare public case of topology **with parameters** |
| [IEAGHG reports](https://publications.ieaghg.org/) | 100+ reports w/ PFDs + heat/mass balances | copyright IEAGHG; open download |
| [NREL/PNNL design reports](https://docs.nrel.gov/docs/fy19osti/71949.pdf) | dozens; PFDs + stream tables | US-gov open |
| [Azure P&ID digitization pipeline](https://github.com/Azure-Samples/digitization-of-piping-and-instrument-diagrams) | code only (symbol/line/arrow/graph) | MIT; includes flow-direction detection |

Digitization accuracy bound (PID2Graph paper): ~75% edge mAP / ~84% node mAP. Every Tier C sample needs rule-based repair + human spot-check.

### Tier D — Convertible via LLM extraction (text)

| Source | Size | Notes |
|---|---|---|
| [USPTO Patent Grant XML bulk](https://data.uspto.gov/bulkdata/datasets/PTGRXML) | all US grants 1976→, weekly XML | filter CPC C/B01; US-gov open |
| [Google Patents on BigQuery](https://github.com/google/patents-public-data) | millions of patents | data live; examples repo archived 2026-04 |
| [BIGPATENT](https://arxiv.org/pdf/1906.03741) | 1.3M patents, CPC-tagged | HF-hosted |
| [HUPD](https://huggingface.co/datasets/HUPD/hupd) | 4.5M applications, 125 GB | **CC-BY-NC-SA** — excludes commercial use |
| [Text-to-Flowsheet code (ETH)](https://zenodo.org/records/19910216) | pipeline code + 101-flowsheet expert benchmark ([paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d6dd00060f)) | CC-BY-4.0; text→graph→Aspen; the extraction engine for this tier |
| [Northwestern processdesign wiki](https://processdesign.mccormick.northwestern.edu/index.php/Main_Page) + [Penn CBE senior design reports](https://repository.upenn.edu/cbe_sdr/70/) | ~15–20 + plausibly hundreds of capstone reports | full designs w/ flowsheets + economics; per-item licenses |

### Tier E — Synthetic generators with engineering validity

| Source | What it emits |
|---|---|
| SFILES2 generator + [augmentation](https://arxiv.org/pdf/2302.03379) | random-but-grammatical flowsheet topologies; branch-permutation multiplies any corpus (MIT) |
| [pyDEXPI generator](https://github.com/process-intelligence-research/pyDEXPI) | P&IDs composed from engineering module templates — validity by construction (AGPL) |
| [SynGameZero](https://github.com/grimmlab/SynGameZero) / [drl4procsyn](https://github.com/grimmlab/drl4procsyn) (TUM) | RL-synthesized flowsheets optimized for profit/separation feasibility — valid **and** purposeful topologies; narrow chemistry scope (MIT) |
| [Pgraph](https://github.com/tsyet12/Pgraph) / [P-Graph Studio](http://p-graph.org/downloads.html) | exhaustive enumeration of structurally feasible process networks (BSD-2) |
| [Rule-based P&ID autocorrection](https://arxiv.org/pdf/2502.18493) (33 rules) | the validity filter for everything above |

### Notable dead ends (verified absent)

- **Vogel 8,000-synthetic + 223-real autocompletion corpus** — papers real, data never released.
- **Graph-to-SFILES 100k dataset** (Zenodo 14840560) — record holds a 26 kB xlsx, no data.
- **Theisen 1,005 labeled real PFDs** — private (copyright).
- **Hirtreiter 312 real PFD/P&ID pairs** — explicitly withheld (copyright).
- **SynthPID (665 topology-preserving synthetic P&IDs)** — release repo 404s.
- **OPEN100** — HTML spec pages only; no downloadable P&IDs despite press claims.
- **Ullmann's / Kirk-Othmer / Meyers** — content-rich, paywalled, no TDM rights.
- Standards bodies (CFIHOS, ISO 15926/POSC Caesar, AutomationML, Proteus XSD) — schemas and dictionaries, zero flowsheet instances. Useful as grammar, not data.

---

## 3. Build plan

**Canonical representation:** SFILES 2.0 string ⇄ NetworkX directed graph (SFILES2 toolkit, MIT). Node = unit operation with class; edge = directed stream; control loops as SFILES 2.0 control notation. Attach DEXPI-derived attributes where available. Canonical SFILES string doubles as the dedup key.

**Phase 1 — Simulator harvest (weeks 1–4).** Highest real-data yield per engineering hour.
- DWSIM parser (.dwxmz = zipped open XML): FOSSEE 275 + mirror 194 + built-ins 186 → ~450–500 unique after dedup.
- Aspen .bkp parser (ASCII block/stream sections): GitHub 313 + NREL + PSEcommunity → ~250 unique.
- Pyomo Arc extraction: IDAES + WaterTAP + BioSTEAM → ~80–100.
- COCO via COM automation → ~90.
- DEXPI XML import via pyDEXPI → ~83 cases.
- PID2Graph real subsets → 72.
- **Yield: ~1,000–1,200 real semantic flowsheets.** Modelica connect() parsing adds thousands of adjacent-domain schematics for pretraining only.

**Phase 2 — Existing corpora + synthetic scale-up (weeks 3–6).**
- Ingest Hirtreiter Zenodo corpus (10k–100k PFD→P&ID pairs) directly.
- Run SFILES2 and pyDEXPI generators for 50k–100k grammatical topologies; run RL generators for purposeful ones.
- Branch-permutation augmentation ×5–10 on all real data.

**Phase 3 — Extraction mining (weeks 4–12, parallel).**
- Text: ETH Text-to-Flowsheet pipeline over CPC-filtered USPTO XML, DOE/NREL/IEAGHG reports, capstane reports. Target 5k–50k extracted flowsheets.
- Images: digitizer (Relationformer-style, trained on PID2Graph + PID_dataset) over FEED-study and report drawings. Expect ~75% edge accuracy pre-repair.

**Phase 4 — Validity layer (continuous).** Non-negotiable given Phase 3 noise.
- 33-rule P&ID autocorrection + SFILES grammar checks.
- Simulator round-trip: auto-instantiate sampled graphs in DWSIM; converged mass balance = strong validity signal.
- Canonical-SFILES dedup across all tiers; provenance + license tag on every sample.

**Phase 5 — Training + evaluation.**
- Pretrain on synthetic + Modelica; mid-train on extracted (Tier C/D); fine-tune on the ~1k real set.
- Benchmarks: held-out real flowsheets (perplexity, validity rate, graph edit distance); control-structure top-k vs published 74.8%/89.2% (Hirtreiter) and 73.2% (Graph-to-SFILES); rule-violation rate; DWSIM convergence rate of generated designs.

**Side quests with outsized payoff:** email TU Delft PIR group for the withheld 312/1,005/100k corpora; email IIT Bombay for SynthPID; confirm FOSSEE license.

**License handling:** train-set is license-mixed. Keep pyDEXPI (AGPL) as an isolated generation service. Exclude HUPD (NC) from any commercial track. PID2Graph is CC-BY-SA — assume share-alike on derived datasets. Per-sample provenance tags make later license filtering cheap.

---

## 4. Feasibility recommendation

**Proceed, with scoped ambition.**

| Capability | Outcome quality | Confidence |
|---|---|---|
| Syntactically valid flowsheet generation | High — grammar + validity layer make this near-guaranteed | High |
| Flowsheet autocompletion / topology suggestion | Good — matches published results at this data scale | High |
| PFD→P&ID control-structure generation | Good — 73–89% top-5 is the demonstrated band | High |
| Flowsheet error detection/correction | Good — rules + learned model | Medium |
| Novel, globally coherent process synthesis | Weak — model will interpolate seen topologies | Medium |
| Parameterized design (conditions, sizing, economics) | Not achievable from public data — parameters essentially absent | High |

**Why the ceiling:** generative quality tracks corpus size and the public real-topology corpus tops out near 10³ (vs 10⁸ for molecules). Synthetic data raises validity, not design insight; extracted data raises coverage, not precision. The withheld academic corpora and any industrial partner data are the only levers that move the ceiling.

**Recommendation:** build the copilot (autocomplete + control-structure + correction) on Phases 1–2 first — it is achievable and immediately useful, and the corpus infrastructure it forces (canonical representation, validity layer, provenance) is exactly what a bigger design model needs if better data ever lands. Treat Phase 3 extraction as an experiment, gated on validity-layer pass rates, not a dependency.
