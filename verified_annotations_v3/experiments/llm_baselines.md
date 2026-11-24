# Baselines with vision-language models

The question these baselines answer: how far does a general vision-language model (VLM) get when it is asked to do a
component's job directly, from the drawing, without training on the 40 training pages? Each baseline gets the same
inputs as the component it is compared with and is scored by the same code ([`eval/`](eval/)) on the 25 test pages.

## Common setup

- **Models.** The strongest model of three providers at the time of the run (Anthropic, OpenAI, Google) and one
  open-weight VLM run locally (for example Qwen3-VL), so that the open baseline can be reproduced. Record the exact model
  identifier and the date of every run.
- **Decoding.** Temperature 0 where the API allows it; otherwise the provider's default, and say so. One run per setting;
  repeat one setting three times on 5 test pages to measure the spread.
- **Output.** Every prompt asks for JSON in the schema given below. A response that does not parse is retried twice and
  then counted as an empty answer.
- **Resolution.** The pages are 3,262 to 5,100 pixels on the long side, and APIs shrink large images, which blurs small
  symbols and text. Give crops, or tiles of at most 1,536 pixels with 15% overlap, and map every coordinate back to page
  pixels.
- **Rules in the prompt.** Give the labeling rules of this dataset that matter for the task (from
  [`../README.md`](../README.md)): one box per symbol, a valve with its actuator and its tag circle is one box, text and
  dashed frames are not symbols, crossing pipes are connected only where a filled dot is drawn, the class list
  ([`../symbol_classes/classes.csv`](../symbol_classes/classes.csv): name and one-line definition).
- **Logging.** Keep every request and raw response (`results/llm/<baseline>/<model>/<page>/...json`), the tokens and
  the cost, so that the paper can report cost per page.
- **No test labels in the prompt**, and no examples from the test pages.

## B1. Symbol classification (compared with the symbol classifier)

- **Input**, for every annotated box of the test pages (about 1,770): the crop of the box with a margin of 8% of its
  longer side, the box drawn in red; a second, wider crop (three box sizes around it) so that the tag and letters such as
  M, LO or FC are visible; the 63 classes with their definitions and the actuator values.
- **Two settings:** (a) zero-shot; (b) with a legend sheet: one reference image per class
  ([`../symbol_classes/class_reference/`](../symbol_classes/class_reference/)), the counterpart of the classifier trained
  with legend symbols.
- **Output:** `{"boxes": [{"id": <box id>, "class": "<class>", "actuator": "<actuator or ->", "normal_position": "open|closed|n/a"}]}`;
  up to 10 boxes per request.
- **Scores:** class accuracy, class macro-F1, actuator accuracy, class and actuator both right, fault-tree type accuracy,
  open/closed accuracy: the metrics of `symbol_classifier/` on the same boxes.

## B2. Drawing to graph (compared with the whole pipeline)

- **Input:** the page shrunk to the model's largest accepted size, plus tiles of the page at full resolution (2 x 2 or
  3 x 2, 15% overlap), in one request, so that the model can follow a pipe from tile to tile.
- **Output:** `{"symbols": [{"id", "class", "tag", "bbox": [x0,y0,x1,y1]}], "connections": [[id1, id2], ...]}` in page
  pixels; a connection joins two symbols that a pipe connects without passing another symbol (through tees and dotted
  crossings).
- **Scores:** symbols with `eval/score.py symbols` (IoU >= 0.5) and with the lenient matching of `eval/score.py pairs
  --match center` (the predicted box center lies inside the reference box), which separates finding a symbol from
  boxing it; connections with the connected-pair F1 of `eval/score.py pairs`. The pipeline's end-to-end output is scored
  with the same pair F1 for the comparison.

## B3. Graph assembly only (compared with the topology assembler)

- **Input:** the page (whole and in tiles) with every annotated symbol box drawn and labeled with a large number
  (set-of-mark prompting).
- **Output:** `{"connections": [[n1, n2], ...]}`.
- **Scores:** connected-pair F1 (`eval/score.py pairs`), against the assembler on annotated symbols with annotated lines
  and with the pipeline's lines (`eval/score.py graph`). This measures pipe tracing and the crossing convention without
  detection errors.

## B4. Text recognition (compared with PP-OCRv5)

- **Input:** the crop of every annotated text line of the test pages (about 2,600), not turned upright, up to 20 per
  request.
- **Output:** `{"lines": [{"id", "text", "rot": "0|cw|ccw|180"}]}`.
- **Scores:** `eval/score.py text` with the annotated boxes as predicted boxes (detection is then perfect, so exact match,
  character error rate and reading direction measure recognition), against PP-OCRv5 recognition on the same crops.

## B5. Candidate verification (compared with the relational re-scorer)

- **Input:** the candidate pool of task 1 (`results/symbol_detection/candidates/`), each candidate as a crop with its box
  drawn and a wider crop around it; the question is whether the box is a correct symbol box under the labeling rules
  (one symbol, tight around its ink). Up to 20 candidates per request.
- **Output:** `{"candidates": [{"id", "symbol": true|false, "confidence": 0..1}]}`; then the selection of the re-scorer
  (score >= 0.15, overlap and containment suppression) on the confidence.
- **Scores:** `eval/score.py symbols`. This is the most expensive baseline (several thousand candidates per model).

## B6. Symbol boxes from the model (compared with the base detector)

Models that output boxes (Gemini, Qwen-VL) run on the tiles and return every symbol box; scored with
`eval/score.py symbols`. It can be folded into the symbol part of B2.

## Optional: questions on the graph

Questions generated from the reference graph of the test pages (which components a given pump is directly connected
to; which valves lie between two tagged components; which valves isolate a component), answered by a language model
from (i) the page image, (ii) the pipeline's LLM JSON, (iii) the reference LLM JSON; scored by the F1 of the answer
set. This measures what the graph representation gives a language model.

## Order

B1, B2 and B3 first: they compare the three main parts of the paper (classifier, whole pipeline, assembler). Then B4, B5
and B6.
