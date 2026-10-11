# Structure in the model or structure after it: a literature review for Task B

Date: 2026-10-11 · Scope: BLUEPRINT §WS5.3 v2 and milestone PoC-1 · Author: project

## 0. Summary

The question is whether a flowsheet-to-P&ID model should predict engineering
objects (control loops over a known process graph) or predict characters and
have a verifier reject what it gets wrong. The literature answers it in three
parts.

First, the domain literature has already run half of the experiment. Schulze
Balhorn, Degens, and Schweidtmann (2025) replaced the sequence encoder of the
original PFD→P&ID transformer with a graph encoder and measured the effect at
three training-set sizes. With one thousand training flowsheets the graph
model reached 28.4% top-5 against 0.9% for the sequence model. With one
hundred thousand it lost, 79.0% against 89.2%, and the authors traced the loss
to the graph model's inability to reproduce the input topology in its output:
80.5% of its outputs reconstructed the PFD correctly, against 99.2% for the
sequence model [2]. A design that renders the output from the input graph
rather than regenerating it keeps the first result and removes the cause of
the second. Nobody in the domain has built one. The same authors list
constrained decoding and hybrid systems with expert rules as future work and
have shipped the rule set [7] but not the model.

Second, the general literature on structured generation says where structure
should enter. Grammar masks at decoding time guarantee syntax but distort the
model's distribution [19] and do not reliably improve semantic accuracy [20].
Constraints in the output space itself, as in set prediction [25, 26],
select-then-attach graph decoders [28], and entities-then-constraints CAD
models [30], remove whole classes of error by construction. Verifier feedback
used for post-training rather than for filtering produces the largest reported
gains: Autodesk's CAD constraint generator went from 34% fully constrained
sketches after supervised fine-tuning to 93% after alignment against the
constraint solver [31].

Third, the engineering theory needed to build the verifier and the masks is
old and explicit: Luyben's nine-step procedure, Skogestad's top-down procedure
with its degrees-of-freedom analysis [40], and the thirty-three rules the
Delft group encoded on DEXPI graphs [7].

The direction this review recommends combines them. Encode the PFD as a graph.
Predict the control layer as a set of loops, each anchored to a PFD node with a
pointer to the actuated stream, under hard masks from degrees-of-freedom
analysis. Render the SFILES string deterministically from graph plus loops.
Then align the loop model against the rule verifier and, where available, a
simulator, by policy optimization on verifiable rewards. Section 5 specifies
it; section 6 ties it to PoC-1 with corrected baselines.

One correction surfaced on the way. The figures 74.8% and 89.2% that this
repository had been treating as top-1 and top-5 are both top-5, at 10,000 and
100,000 training pairs respectively; the corresponding top-1 figures are 37.7%
and 48.6% [1, 2]. Our 100k model therefore sits 6.6 points under the published
top-1 and 8.1 under the published top-5, not 33 and 8. Every table in the
repository is corrected alongside this review.

## 1. The question, stated precisely

A P&ID is a PFD plus a control layer. The current model (`gpd/models/seq2seq.py`)
reads the PFD as a token sequence and writes the P&ID as a token sequence. Three
properties of that design are at issue.

1. *Copying.* Most of the output is the input. The decoder must reproduce the
   PFD exactly before it can be scored on the control layer.
2. *Error type.* A wrong token produces an unparseable or wrong-topology string.
   Engineering mistakes and syntax mistakes are the same event.
3. *Verification.* The rules that say which outputs are acceptable are applied
   after generation. They can reject; they cannot inform.

The alternative puts structure into the model: the output space is the set of
engineering decisions, the invariants are masks on that space, and the verifier
contributes to training rather than to filtering. The review asks what evidence
exists for each of these moves, in this domain and in neighbouring ones.

## 2. What the domain literature has established

### 2.1 Sequence-to-sequence control-structure generation

Hirtreiter, Schulze Balhorn, and Schweidtmann [1] cast control-structure
generation as translation between SFILES 2.0 strings. A transformer is
pre-trained on synthetic PFD→P&ID pairs produced by a rule-based generator
that combines process patterns from the literature, then fine-tuned on real
pairs. The headline figures, on synthetic test data, are 74.8% top-5 with
10,000 training pairs and 89.2% with 100,000. Fine-tuning on 312 real PFDs
produced results the authors describe only as indicating the need for a
larger dataset and for hybrid approaches. The real pairs were not released.

Vogel, Schulze Balhorn, and Schweidtmann [3] trained a decoder-only
transformer on SFILES strings to autocomplete flowsheets, the same group
having defined SFILES 2.0 [4]. Schulze Balhorn, Caballero, and Schweidtmann
[5] used the same string-to-string framing for autocorrection, reaching 80%
top-1 and 84% top-5 on synthetic test data.

These three papers fix the framing that this repository inherited: strings
in, strings out, synthetic data, exact match.

### 2.2 The graph-encoder experiment

Graph-to-SFILES [2] keeps the sequence decoder and replaces the encoder with a
graph neural network over the PFD topology. The authors compare GATv2,
GraphConv, and a Graph Transformer with Laplacian positional encodings, and a
combined model that performs best. The decoder is unchanged. No copy or pointer
mechanism is used. No constraints are applied at decoding. Predicted strings
are canonicalized before scoring because the model does not always emit
canonical form.

The results at three training-set sizes, synthetic test split of 1,000:

| Training pairs | Model | top-1 | top-5 |
|---|---|---|---|
| 1,000 | sequence encoder [1] | 0.3% | 0.9% |
| 1,000 | graph encoder [2] | 15.2% | 28.4% |
| 10,000 | sequence encoder | 37.7% | 74.8% |
| 10,000 | graph encoder | 41.9% | 73.2% |
| 100,000 | sequence encoder | 48.6% | 89.2% |
| 100,000 | graph encoder | 46.1% | 79.0% |

Four findings matter for us.

- At 1,000 pairs the graph encoder is the difference between a model that
  works and one that does not. This is the regime real data will occupy.
- At 100,000 pairs the sequence model wins, and the authors locate the cause:
  PFD reconstruction accuracy of 80.5% for the graph model versus 99.2% for the
  sequence model. The graph model is worse at copying its input, and copying is
  most of the task. A model that does not have to copy would not pay this.
- Top-1 is capped by the one-to-many mapping: the generator admits several
  valid control structures for some patterns, and the test set contains one.
  The gap between top-1 and top-5 is therefore partly irreducible for any
  model scored by exact match, and the right primary metric is at the loop
  level.
- Fine-tuning on 250 real industrial and academic P&IDs gave very low scores.
  Training on 100 synthetic flowsheets gave under 1%. Neither architecture
  transfers from a few hundred examples.

The authors' stated future work includes constrained beam search restricted
to units present in the input, fusion of sequence and graph encoders, process
attributes beyond topology, and hybrid systems with expert rules for safety.
They explicitly keep the sequence output "as an inductive bias for
efficiency" and do not propose predicting loops or graphs directly. The
dataset is oil-and-gas oriented, about 52 nodes and 61 edges per flowsheet,
with a vocabulary of 113 tokens.

### 2.3 Rules, graphs, and the DEXPI line

The same group's rule-based autocorrection [7] encodes 33 engineering rules as
graph rewrites on DEXPI P&IDs loaded through pyDEXPI, demonstrates five on one
diagram, and argues for rules over learned models on the grounds of data
scarcity, explainability, and hallucination. The follow-up graph-based
generative methodology for utility-system P&IDs [6] trains on synthetic
erroneous-versus-corrected pairs and reports a gap when tested on five
industrial diagrams. Oeing et al. [9] applied node classification on DEXPI
graphs to flag deviations from common design. Alimin et al. [8] build a
knowledge graph from pyDEXPI output and attach an LLM to it by retrieval.

Taken together: the rule set exists, the graph schema exists, and the group's
own conclusion from the data is that learned models need help from both.

### 2.4 Process synthesis by reinforcement learning

Göttl, Grimm, and Burger [10, 11] framed flowsheet synthesis as a game played
against a thermodynamic simulator, with an agent combining a policy network
and tree search; the hierarchical version structures the action space into
levels. Stops et al. [12] built an actor-critic agent on a graph
representation with a GNN, placing units as discrete actions and setting
design variables as continuous ones, rewarded by a flowsheet simulation.
These establish that a simulator can serve as the reward for a structured
generator in this domain, that the action space should be the engineering
decision (place a unit, connect a stream), and that the approach is confined
to small chemistries by simulation cost.

### 2.5 Language-model agents

Gowaikar et al. [13] produce P&IDs from natural-language descriptions with a
multi-step agent and evaluate soundness and completeness. The ETH
text-to-flowsheet pipeline [14] digitizes process descriptions into
simulatable Aspen flowsheets and fills missing parameters by black-box
optimization. Later work drives simulators through tool-calling agents
[15]. These systems address extraction and parameterization; none learns
control-structure placement from data, and none reports exact-match figures
comparable to the above. They are complementary: a simulator wrapped as a tool
is also a verifier.

## 3. What the structured-generation literature has established

The useful way to read this literature is by where the structure enters.

### 3.1 At decoding time: grammar constraints

PICARD [16], Synchromesh [17], and the grammar-constrained decoding of Geng
et al. [18] mask tokens that cannot extend a valid prefix, with Outlines and
XGrammar as the production implementations. They guarantee syntactic validity
with no retraining. Two results temper them. Park et al. [19] prove that
masking changes the distribution the model samples from: the output is valid
but no longer the model's most likely valid output, and their grammar-aligned
decoding corrects this at the price of slow convergence. StructureBench [20],
evaluating eleven small models under three constrained-decoding frameworks,
finds that constraints reliably deliver syntactic validity but do not
reliably improve semantic correctness and can hurt small models on complex
grammars. Ctrl-G [21] offers a tractable alternative for constraints
expressible as automata, by conditioning a distilled HMM rather than masking
the LLM directly.

Reading for Task B: constrained decoding closes the 1.8% of our outputs that
do not parse and nothing else. It is a floor, not a method.

### 3.2 At training time: constraints in the loss or the output layer

The semantic loss of Xu et al. [22] penalizes probability mass on invalid
outputs, computed exactly through circuit compilation of the constraint.
Semantic probabilistic layers [23] go further and make the output layer itself
a distribution that assigns zero mass outside the constraint while remaining
trainable by maximum likelihood; the 2024 follow-up extends the loss to
structured prediction [24]. These are exact for constraints that compile to
tractable circuits, which bounds their reach: degree constraints and
one-actuator-per-loop compile; global degrees-of-freedom balance over an
arbitrary graph does not, in general.

### 3.3 In the output space

*Set prediction.* DETR [25] predicts a fixed-size set with a bipartite
matching loss, removing the need to order outputs. SPN4RE [26] applies the
same construction to relation triples in text. A control layer is a set of
loops, and the Delft authors' own observation that several valid structures
exist per PFD is an ordering ambiguity that a set loss dissolves.

*Pointers.* Pointer networks [27] let a decoder select an element of the input
rather than a vocabulary item. The actuated valve of a loop is an element of
the PFD.

*Select then attach.* Jin, Barzilay, and Jaakkola [28] decode molecular graphs
by alternating two decisions, which motif to add and where to attach it, and
translate graphs to graphs with that decoder. A loop is a motif; its
attachment is the measured node and the actuated stream.

*Edit rather than rewrite.* The Levenshtein Transformer [29] generates by
insertion and deletion on an existing sequence. A P&ID is an edit of a PFD; a
model that emits edits cannot corrupt what it does not touch.

*Entities then constraints.* Vitruvion [30] generates CAD sketches as a
sequence of primitives followed by a sequence of constraints that reference
them by index, in the exact format a CAD solver consumes. This is the closest
published analogue to "PFD then loops that reference PFD nodes."

*Graphs with hard structural constraints.* GCPN [32] masks valence-violating
bonds during molecular graph construction. DiGress [33] generates graphs by
discrete diffusion; ConStruct [34] adds a projector that keeps every sample
inside a constraint set for edge-deletion-invariant properties such as
planarity, acyclicity, and degree bounds, raising validity by up to 71 points.
The analog-circuit line (LaMAGIC [36], AnalogGenie and AnalogGenie-Lite [35])
reports that sequence encodings of graphs scale better than adjacency
prediction and that a graph-aware sequence design improves validity by 23.5%
to 33.6% relative to prior methods. For flowsheet autocompletion, where the output is new
topology, these are the relevant designs; for Task B, where the topology is
given, they are not needed.

### 3.4 Verifiers: for selection, and for training

LEVER [37] trains a verifier on execution results to re-rank sampled programs
and gains 4.6 to 10.9 points over the base model. It is the cheap use of a
verifier: no change to the generator.

The expensive use is to train against it. GRPO [38] and its descendants
optimize a policy on verifiable rewards; in code generation the reward is
compiler or test feedback, with dense variants [41] addressing sparsity, and
the surveys [42] catalogue the shaping tricks. The decisive analogue for this
project is Autodesk's constraint generator [31]: starting from Vitruvion
pre-trained on SketchGraphs, supervised fine-tuning on solver-verified
sketches reached 34% fully constrained outputs; alignment against the solver
(RLOO, DPO, expert iteration) reached 93%; the base model managed 8.9%. The
verifier was a constraint solver that reports under-, fully-, and
over-constrained status, solvability, and geometry shift. A degrees-of-freedom
analysis reports the same three states for a control structure.

### 3.5 Engineering theory available as structure

Luyben, Tyreus, and Luyben's nine-step procedure and Skogestad's top-down
procedure [40] both begin by counting degrees of freedom: the number of
manipulated variables fixes the number of variables that can be controlled,
inventories (levels, pressures) must be controlled, one flow in every recycle
loop must be fixed, and the remaining degrees of freedom are assigned to
economic objectives. These are computable from topology alone. The 33 rules of
[7] are a second, finer-grained source, already expressed as graph patterns.
Together they supply three things a structured model needs: masks on what
may be predicted, a verifier that scores what was predicted, and, through the
synthetic generator of [1], a label source whose rules are the ones being
learned.

## 4. Synthesis

The evidence sorts by where structure enters and what it buys.

| Where structure enters | What it buys | What it costs | Evidence | Tried in this domain |
|---|---|---|---|---|
| Decoding masks | syntax validity | distribution distortion; no semantic gain | [16–20] | listed as future work [2] |
| Graph encoder | 30× accuracy at 1k pairs | loses at 100k through copy errors | [2] | yes |
| Output space = loops over PFD nodes | no copying; errors are engineering errors; set loss removes ordering; masks are exact | must write `render`/`parse` faithfully; new metric | [25–30] | no |
| Hard masks from DOF / degree rules | zero invalid outputs by construction | masks must be correct | [32, 34, 40] | no |
| Verifier re-ranking | 5–10 points, no retraining | needs a verifier | [37] | no |
| Verifier-aligned post-training | 34% → 93% in the closest analogue | RL plumbing; reward design | [31, 38, 41] | no |
| Simulator reward | physical validity | slow, sparse, narrow chemistry | [10–12] | for synthesis, not control |

Three gaps are visible. No one has moved the output space to loops. No one has
used the rule set or a DOF analysis as masks. No one has post-trained a
flowsheet model against a verifier. Each has a published analogue in a
neighbouring field with a large effect size, and each is cheap relative to
the data collection it would replace.

## 5. Recommended direction

**Name.** Graph-anchored loop prediction with verifier alignment.

**Claim.** A model that predicts control loops as a set over the PFD graph,
under degrees-of-freedom masks, rendered to SFILES deterministically and
post-trained against the rule verifier, will match or exceed the published
sequence model at 10,000 pairs and exceed the published graph model at 1,000
pairs, with every output parseable and rule-consistent.

**Why this and not the alternatives.**
- Over scaling the sequence model: scaling buys top-5 but not real-data
  transfer; the Delft group's fine-tuning on 250 real diagrams failed for both
  architectures [2].
- Over constrained decoding alone: it fixes 1.8% of outputs and distorts the
  rest [19, 20].
- Over graph-native generation (diffusion, autoregressive graph decoders): the
  topology is an input for Task B, not an output; generating it is the error
  source [2]. These designs are right for autocompletion (Task A), later.
- Over LLM agents: they do not learn placement conventions from data and
  report no comparable accuracy [13–15]; a simulator-as-tool is a reward, not
  a generator.

**Specification.**

*Encoder.* Graph transformer over PFD nodes with edge-aware attention bias
and structural features (class, in/out degree, depth from feed, recycle
membership, column tag); Graphormer-style [39] or the Delft combined GNN [2].
The sequence tokenizer already in the repository supplies node classes.

*Output.* A set of loops. Each loop: measured node (pointer over PFD nodes),
variable ∈ {F, T, P, L}, kind ∈ {controller, indicator, recorder, motor},
actuated target (pointer over actuator-capable PFD nodes ∪ "insert valve on
incident stream *s*"). Decoded either non-autoregressively with a bipartite
matching loss [25, 26] or autoregressively with select-then-attach [28]; the
first is simpler and matches the set semantics, the second gives a natural
top-k. Start with per-node multi-label heads plus a pointer head, which is the
degenerate set decoder with one query per PFD node; move to learned queries if
multi-loop nodes are mis-handled.

*Masks.* One controller per actuated stream; indicators carry no actuator;
level loops only on inventory-holding classes; pressure loops only where a
vapor-side manipulated stream exists; total actuated streams ≤ manipulable
streams per connected component (the DOF count). These are exact and cheap;
they are the subset of [40] and [7] that depends on topology alone.

*Rendering.* `render(pfd_graph, loops) → SFILES 2.0`, inverse `parse_loops`.
Contract test on the entire Hirtreiter corpus: `render(parse_loops(pid)) ==
canonical(pid)` at ≥ 99%. This is the component with execution risk, because
it must reproduce the generator's string conventions; it is also the
component that makes every downstream number comparable with the literature.

*Training.* Stage 1, supervised: per-loop losses from `parse_loops` labels on
the 1k and 10k Hirtreiter splits, DEXPI real diagrams, pyDEXPI synthetic
diagrams. Stage 2, alignment: GRPO or RLOO [38, 31] with reward = rule pass
rate [7] + DOF balance + loop-level F1 against the label, on-policy samples
from the stage-1 model. Stage 3, if the simulator lane lands: sparse
convergence reward from DWSIM, last, because it is slow and noisy.

*Metrics.* Loop-level precision, recall, F1 (primary); rendered exact match
top-1 and top-5 on the Delft test split (comparability); parse rate; rule
violation rate; and the 1k-pair result, which is the number that predicts
real-data behaviour.

**What is new relative to Graph-to-SFILES [2].** The output space (loops, not
strings), the masks (DOF and rule constraints inside the decoder), and the
training signal (verifier alignment). The encoder is theirs or close to it;
reusing it isolates the contribution.

**Risks.** `render` fidelity, mitigated by the contract test. Masks encoding
the wrong convention, mitigated by measuring mask violations against the
ground-truth loops before training. Reward hacking under alignment, mitigated
by holding loop-F1 in the reward and by the Autodesk recipe of SFT on
verifier-passing data before RL [31]. The ceiling imposed by topology-only
data, which no architecture lifts; the WS1 harvest is the answer to that, and
the IR already carries per-node attributes for when it arrives.

## 6. Consequences for PoC-1

The bake-off in BLUEPRINT §8 is the right experiment; its bars were set
against mis-read baselines and need restating.

- The published sequence model reaches 37.7% / 74.8% (top-1 / top-5) at 10k
  and 48.6% / 89.2% at 100k. Our 100k run reaches 42.0% / 81.1%. Our 10k run
  reaches 16.5% / 35.8%, far under the published 10k figure, which means our
  seq2seq baseline is under-trained at 10k and must be brought to convergence
  before it serves as a comparator.
- The published graph model reaches 15.2% / 28.4% at 1k and 41.9% / 73.2% at
  10k.
- The decisive regime is 1,000 pairs. The bar for the loop model there is the
  published graph-encoder result, 28.4% top-5; a structured output that cannot
  beat an unstructured graph encoder at 1k has no claim to data efficiency.
- At 10k the bar is the better of the two published models: 41.9% top-1 and
  74.8% top-5, rendered strings, same test split.
- Loop-level F1 is reported alongside, and parse rate is expected to be 100%
  by construction.

Revised rule: switch if the loop model clears both bars (1k: top-5 ≥ 28.4%;
10k: top-5 ≥ 74.8% and top-1 ≥ 41.9%); keep both and run the curve if it
clears one; one week of diagnosis then set aside if it clears neither.
Verifier alignment (stage 2) is run only on whichever model survives, and its
gain is reported separately so that the architecture result and the training
result are not confounded.

## 7. Corrections made to the repository with this review

- `gpd/eval/baselines.json`: published metrics restated as top-1 / top-5 at
  10k and 100k; Graph-to-SFILES figures added.
- `docs/poc0.md`, `README.md`, `BLUEPRINT.md`: the "33 points under on top-1"
  statement replaced; the top-1 gap is 6.6 points at 100k. Decision text left
  intact: both pass bars are still met and Track B is still not triggered.
- `BLUEPRINT.md` §8: PoC-1 decision rule restated with the 1k bar.

## References

1. Hirtreiter, E., Schulze Balhorn, L., Schweidtmann, A. M. *Toward automatic generation of control structures for process flow diagrams with large language models.* AIChE Journal 70(1), e18259 (2024). [arXiv:2211.05583](https://arxiv.org/abs/2211.05583). Data: [Zenodo 7658798](https://zenodo.org/records/7658798).
2. Schulze Balhorn, L., Degens, K., Schweidtmann, A. M. *Graph-to-SFILES: Control structure prediction from process topologies using generative artificial intelligence.* Computers & Chemical Engineering (2025). [arXiv:2412.00508](https://arxiv.org/abs/2412.00508). Supplement: [Zenodo 14840560](https://zenodo.org/records/14840560).
3. Vogel, G., Schulze Balhorn, L., Schweidtmann, A. M. *Learning from flowsheets: A generative transformer model for autocompletion of flowsheets.* Computers & Chemical Engineering (2023). [arXiv:2208.00859](https://arxiv.org/abs/2208.00859).
4. Vogel, G., Schulze Balhorn, L., Hirtreiter, E., Schweidtmann, A. M. *SFILES 2.0: an extended text-based flowsheet representation.* Code: [process-intelligence-research/SFILES2](https://github.com/process-intelligence-research/SFILES2).
5. Schulze Balhorn, L., Caballero, M., Schweidtmann, A. M. *Toward autocorrection of chemical process flowsheets using large language models.* Computer Aided Chemical Engineering 53, 3109–3114 (2024). [arXiv:2312.02873](https://arxiv.org/abs/2312.02873).
6. Schulze Balhorn, L., Goldstein, D. P., et al. *A graph-based generative artificial intelligence methodology for autocorrection of utility-system P&IDs.* Chemie Ingenieur Technik (2026). DOI 10.1002/cite.70135.
7. *Rule-based autocorrection of Piping and Instrumentation Diagrams (P&IDs) on graphs.* Systems and Control Transactions 4, 1656–1661 (2025). [arXiv:2502.18493](https://arxiv.org/abs/2502.18493).
8. Alimin, A. A., Goldstein, D. P., Schulze Balhorn, L., Schweidtmann, A. M. *Talking like Piping and Instrumentation Diagrams (P&IDs).* Systems and Control Transactions (2025). [arXiv:2502.18928](https://arxiv.org/abs/2502.18928).
9. Oeing, J., Brandt, K., Wiedau, M., Tolksdorf, G., Welscher, W., Kockmann, N. *Graph learning in machine-readable plant topology data.* Chemie Ingenieur Technik 95(7) (2023). [TU Dortmund repository](https://eldorado.tu-dortmund.de/items/5d6d2f9d-f967-4bde-bec3-a53963c30369/full).
10. Göttl, Q., Grimm, D. G., Burger, J. *Automated synthesis of steady-state continuous processes using reinforcement learning.* Frontiers of Chemical Science and Engineering 16, 288–302 (2022). [arXiv:2101.04422](https://arxiv.org/abs/2101.04422).
11. Göttl, Q., Tönges, Y., Grimm, D. G., Burger, J. *Automated flowsheet synthesis using hierarchical reinforcement learning: proof of concept.* Chemie Ingenieur Technik (2021).
12. Stops, L., Leenhouts, R., Gao, Q., Schweidtmann, A. M. *Flowsheet generation through hierarchical reinforcement learning and graph neural networks.* AIChE Journal 69(1), e17938 (2023). [arXiv:2207.12051](https://arxiv.org/abs/2207.12051).
13. Gowaikar, S., Iyengar, S., Segal, S., Kalyanaraman, S. *An agentic approach to automatic creation of P&ID diagrams from natural language descriptions.* AAAI 2025 Workshop AI2ASE. [arXiv:2412.12898](https://arxiv.org/abs/2412.12898).
14. *Text-to-flowsheet: an LLM-assisted pipeline for expert-level digitization and automated simulation of chemical processes.* (2026). [PMC13251679](https://pmc.ncbi.nlm.nih.gov/articles/PMC13251679/). Code: [Zenodo 19910216](https://zenodo.org/records/19910216).
15. *Context is all you need: Towards autonomous model-based process design using agentic AI in flowsheet simulations.* (2026). [arXiv:2603.12813](https://arxiv.org/abs/2603.12813).
16. Scholak, T., Schucher, N., Bahdanau, D. *PICARD: Parsing incrementally for constrained auto-regressive decoding from language models.* EMNLP 2021. [arXiv:2109.05093](https://arxiv.org/abs/2109.05093).
17. Poesia, G., et al. *Synchromesh: Reliable code generation from pre-trained language models.* ICLR 2022. [arXiv:2201.11227](https://arxiv.org/abs/2201.11227).
18. Geng, S., et al. *Grammar-constrained decoding for structured NLP tasks without finetuning.* EMNLP 2023. [arXiv:2305.13971](https://arxiv.org/abs/2305.13971).
19. Park, K., Wang, J., Berg-Kirkpatrick, T., Polikarpova, N., D'Antoni, L. *Grammar-aligned decoding.* NeurIPS 2024. [arXiv:2405.21047](https://arxiv.org/abs/2405.21047).
20. Xiong, X., et al. *StructureBench: A unified benchmark suite for multi-scenario structured generation tasks with on-device models.* IJCAI 2026. [proceedings](https://www.ijcai.org/proceedings/2026/267).
21. Zhang, H., Kung, P.-N., Yoshida, M., Van den Broeck, G., Peng, N. *Adaptable logical control for large language models (Ctrl-G).* NeurIPS 2024. [arXiv:2406.13892](https://arxiv.org/abs/2406.13892). Predecessor: Zhang et al., *Tractable control for autoregressive language generation (GeLaTo)*, ICML 2023.
22. Xu, J., Zhang, Z., Friedman, T., Liang, Y., Van den Broeck, G. *A semantic loss function for deep learning with symbolic knowledge.* ICML 2018. [arXiv:1711.11157](https://arxiv.org/abs/1711.11157).
23. Ahmed, K., Teso, S., Chang, K.-W., Van den Broeck, G., Vergari, A. *Semantic probabilistic layers for neuro-symbolic learning.* NeurIPS 2022. [arXiv:2206.00426](https://arxiv.org/abs/2206.00426).
24. *Semantic loss functions for neuro-symbolic structured prediction.* (2024). [arXiv:2405.07387](https://arxiv.org/abs/2405.07387).
25. Carion, N., et al. *End-to-end object detection with transformers (DETR).* ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872).
26. Sui, D., et al. *Joint entity and relation extraction with set prediction networks.* (2020). [arXiv:2011.01675](https://arxiv.org/abs/2011.01675).
27. Vinyals, O., Fortunato, M., Jaitly, N. *Pointer networks.* NeurIPS 2015. [arXiv:1506.03134](https://arxiv.org/abs/1506.03134).
28. Jin, W., Barzilay, R., Jaakkola, T. *Hierarchical graph-to-graph translation for molecules.* ICLR 2020. [arXiv:1907.11223](https://arxiv.org/abs/1907.11223). And *Hierarchical generation of molecular graphs using structural motifs.* ICML 2020. [arXiv:2002.03230](https://arxiv.org/abs/2002.03230).
29. Gu, J., Wang, C., Zhao, J. *Levenshtein transformer.* NeurIPS 2019. [arXiv:1905.11006](https://arxiv.org/abs/1905.11006).
30. Seff, A., Zhou, W., Richardson, N., Adams, R. P. *Vitruvion: A generative model of parametric CAD sketches.* ICLR 2022. [arXiv:2109.14124](https://arxiv.org/abs/2109.14124).
31. Casey, E., et al. *Aligning constraint generation with design intent in parametric CAD.* ICCV 2025. [arXiv:2504.13178](https://arxiv.org/abs/2504.13178).
32. You, J., Liu, B., Ying, R., Pande, V., Leskovec, J. *Graph convolutional policy network for goal-directed molecular graph generation.* NeurIPS 2018. [arXiv:1806.02473](https://arxiv.org/abs/1806.02473).
33. Vignac, C., et al. *DiGress: Discrete denoising diffusion for graph generation.* ICLR 2023. [arXiv:2209.14734](https://arxiv.org/abs/2209.14734).
34. Madeira, M., Vignac, C., Thanou, D., Frossard, P. *Generative modelling of structurally constrained graphs (ConStruct).* NeurIPS 2024. [arXiv:2406.17341](https://arxiv.org/abs/2406.17341).
35. Gao, J., et al. *AnalogGenie: A generative engine for automatic discovery of analog circuit topologies.* ICLR 2025. [arXiv:2503.00205](https://arxiv.org/abs/2503.00205). And *AnalogGenie-Lite*, ICML 2025.
36. *LaMAGIC: Language-model-based topology generation for analog integrated circuits.* (2024). [Duke CEI](https://cei.pratt.duke.edu/research/electronic-design-automation/lamagic-language-model-based-topology-generation-analog).
37. Ni, A., et al. *LEVER: Learning to verify language-to-code generation with execution.* ICML 2023. [arXiv:2302.08468](https://arxiv.org/abs/2302.08468).
38. Shao, Z., et al. *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models* (GRPO). (2024). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). DeepSeek-AI, *DeepSeek-R1* (2025). [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
39. Ying, C., et al. *Do transformers really perform badly for graph representation? (Graphormer).* NeurIPS 2021. [arXiv:2106.05234](https://arxiv.org/abs/2106.05234).
40. Skogestad, S. *Control structure design for complete chemical plants.* Computers & Chemical Engineering 28, 219–234 (2004). [NTNU PDF](https://skoge.folk.ntnu.no/publications/2004/control_structure_design_cce/old-submitted/control_structure.pdf). Larsson, T., Skogestad, S. *Plantwide control: A review and a new design procedure.* Modeling, Identification and Control 21 (2000). Luyben, W. L., Tyreus, B. D., Luyben, M. L. *Plantwide Process Control.* McGraw-Hill (1998).
41. *VeRPO: Verifiable dense reward policy optimization for code generation.* (2026). [arXiv:2601.03525](https://arxiv.org/abs/2601.03525).
42. *Reward engineering for software tasks: A survey of reinforcement learning approaches.* (2026). [arXiv:2601.19100](https://arxiv.org/abs/2601.19100).
