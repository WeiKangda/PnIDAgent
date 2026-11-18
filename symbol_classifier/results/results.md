# Symbol classifier: held-out results

Every number is on boxes the model did not see in training: 3,483 boxes on 65 pages, each predicted once.

- new family: each of the 7 document families is held out in turn (the model has never seen that drawing style).
- new page: 5 folds of whole pages; the other pages of the same family are in training. GE-Hitachi ABWR and NuScale have one page each, so they are a new family in both protocols.

## Overall

| model | test | input | class | class macro-F1 | actuator | class + actuator | fault-tree type | open/closed |
|---|---|---|---:|---:|---:|---:|---:|---:|
| most frequent class | new family | - | 37.3 | 1.2 |  |  |  |  |
| CLIP ViT-B/32 features + logistic regression | new family | image | 57.4 | 16.0 |  |  |  |  |
| EfficientNetV2-S features + logistic regression | new family | image | 57.6 | 14.5 |  |  |  |  |
| EfficientNetV2-S features + logistic regression | new family | image + text | 57.7 | 14.3 | 77.8 | 49.7 | 59.5 | 88.7 |
| ResNet-18, boxes only | new family | image only | 61.6 | 15.2 | 79.7 | 54.9 | 67.9 | 95.4 |
| ResNet-18, boxes only | new family | image + text | 61.7 | 15.2 | 79.8 | 55.0 | 67.9 | 95.0 |
| ResNet-18 + legend symbols | new family | image only | 71.7 | 33.2 | 79.8 | 64.8 | 74.6 | 94.6 |
| ResNet-18 + legend symbols | new family | image + text | 71.8 | 33.1 | 80.2 | 64.9 | 74.9 | 94.5 |
| ResNet-18 + legend symbols | new page | image only | 84.3 | 50.2 | 93.5 | 82.3 | 87.8 | 95.3 |
| ResNet-18 + legend symbols | new page | image + text | 84.4 | 50.1 | 93.7 | 82.5 | 88.2 | 95.3 |

Accuracy in %. Macro-F1 is over the 45 classes with at least 5 boxes. Actuator accuracy is on the 2,347 boxes that have an actuator (valves, dampers, pumps). The fault-tree type (MOV, AOV, XVM, CKV, CRB, ...) follows from class + actuator (`ft_component_types.csv`); its accuracy is on the 2,963 boxes that have one. Open/closed accuracy is on the 1,626 boxes with a normal position.

## Boxes not on the review sheet

The 479 boxes on review.xlsx have a class or position that is still to be confirmed.

| model | test | input | boxes | class | fault-tree type | open/closed |
|---|---|---|---:|---:|---:|---:|
| ResNet-18, boxes only | new family | image only | 3004 | 61.1 | 67.7 | 99.4 |
| ResNet-18, boxes only | new family | image + text | 3004 | 61.2 | 67.7 | 99.4 |
| ResNet-18 + legend symbols | new family | image only | 3004 | 70.7 | 73.4 | 98.9 |
| ResNet-18 + legend symbols | new family | image + text | 3004 | 70.7 | 73.6 | 99.0 |
| ResNet-18 + legend symbols | new page | image only | 3004 | 82.9 | 86.5 | 97.3 |
| ResNet-18 + legend symbols | new page | image + text | 3004 | 83.1 | 87.0 | 97.3 |

## Class accuracy per family

| model | test | input | AFW, other plants | APR1400 DCD | Davis-Besse reliability report | GE-Hitachi ABWR | NUREG/CR-4550 (Surry) | NuScale | Surry FSAR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| most frequent class | new family | - | 54.2 | 14.8 | 55.2 | 47.1 | 45.5 | 23.8 | 43.0 |
| CLIP ViT-B/32 features + logistic regression | new family | image | 78.6 | 26.9 | 84.7 | 62.9 | 72.8 | 57.1 | 64.1 |
| EfficientNetV2-S features + logistic regression | new family | image | 79.1 | 30.2 | 85.0 | 59.3 | 68.4 | 57.1 | 58.6 |
| EfficientNetV2-S features + logistic regression | new family | image + text | 79.2 | 30.6 | 85.0 | 58.6 | 67.5 | 54.8 | 59.2 |
| ResNet-18, boxes only | new family | image only | 82.2 | 35.4 | 83.8 | 63.6 | 70.9 | 54.8 | 67.3 |
| ResNet-18, boxes only | new family | image + text | 82.1 | 35.6 | 83.8 | 63.6 | 71.2 | 59.5 | 67.3 |
| ResNet-18 + legend symbols | new family | image only | 82.8 | 60.9 | 83.8 | 65.0 | 71.8 | 61.9 | 70.8 |
| ResNet-18 + legend symbols | new family | image + text | 82.7 | 61.1 | 83.8 | 64.3 | 71.8 | 64.3 | 71.6 |
| ResNet-18 + legend symbols | new page | image only | 88.2 | 85.4 | 99.4 | 62.9 | 71.2 | 61.9 | 77.8 |
| ResNet-18 + legend symbols | new page | image + text | 88.2 | 85.7 | 99.4 | 63.6 | 71.2 | 61.9 | 77.8 |

## Confidence

Share of boxes whose most likely class has at least this probability, and the class accuracy on them. `predict.py` marks the boxes below 0.7 with `needs_review`.

| model | test | input | p >= 0.5 | p >= 0.7 | p >= 0.9 |
|---|---|---|---|---|---|
| ResNet-18, boxes only | new family | image only | 64.1% of boxes, 82.1% right | 46.2% of boxes, 91.2% right | 1.4% of boxes, 60.0% right |
| ResNet-18, boxes only | new family | image + text | 64.2% of boxes, 82.1% right | 46.5% of boxes, 91.0% right | 1.4% of boxes, 66.7% right |
| ResNet-18 + legend symbols | new family | image only | 63.6% of boxes, 89.6% right | 42.5% of boxes, 94.6% right | 0.8% of boxes, 86.2% right |
| ResNet-18 + legend symbols | new family | image + text | 64.2% of boxes, 89.2% right | 43.0% of boxes, 94.5% right | 1.0% of boxes, 82.9% right |
| ResNet-18 + legend symbols | new page | image only | 81.3% of boxes, 95.2% right | 68.9% of boxes, 97.9% right | 5.9% of boxes, 97.1% right |
| ResNet-18 + legend symbols | new page | image + text | 81.7% of boxes, 95.3% right | 69.6% of boxes, 97.9% right | 7.6% of boxes, 97.4% right |

## Per class (ResNet-18 + legend symbols, new family, image + text)

| class | boxes | precision | recall | F1 |
|---|---:|---:|---:|---:|
| VLV_GATE | 1298 | 87.7 | 99.5 | 93.3 |
| VLV_CHECK | 494 | 85.3 | 95.1 | 90.0 |
| BREAKER | 181 | 91.7 | 67.4 | 77.7 |
| VLV_GLOBE | 132 | 85.0 | 25.8 | 39.5 |
| HX | 124 | 50.4 | 47.6 | 49.0 |
| PUMP_CENTRIFUGAL | 113 | 65.7 | 81.4 | 72.7 |
| ORIFICE | 103 | 32.1 | 33.0 | 32.5 |
| VLV_BUTTERFLY | 92 | 70.0 | 7.6 | 13.7 |
| INSTR | 78 | 39.1 | 34.6 | 36.7 |
| TANK | 76 | 50.4 | 76.3 | 60.7 |
| STEAM_GEN | 73 | 81.6 | 54.8 | 65.6 |
| FILTER | 68 | 98.0 | 70.6 | 82.1 |
| OFFPAGE | 52 | 65.2 | 28.8 | 40.0 |
| FAN | 48 | 84.6 | 68.8 | 75.9 |
| COIL | 40 | 100.0 | 100.0 | 100.0 |
| VLV_SAFETY_RELIEF | 39 | 30.0 | 30.8 | 30.4 |
| VLV_STOP_CHECK | 37 | 0.0 | 0.0 | 0.0 |
| TRANSFORMER | 37 | 80.0 | 54.1 | 64.5 |
| DAMPER_EH | 30 | 87.5 | 93.3 | 90.3 |
| DAMPER_PNEUMATIC | 27 | 85.7 | 22.2 | 35.3 |
| TURBINE | 22 | 7.1 | 4.5 | 5.6 |
| PUMP_TURBINE_DRIVEN | 22 | 0.0 | 0.0 | 0.0 |
| SPRAY | 21 | 50.0 | 38.1 | 43.2 |
| OTHER | 20 | 0.0 | 0.0 | 0.0 |
| PENETRATION | 18 | 0.0 | 0.0 | 0.0 |
| BLOCK | 18 | 12.8 | 27.8 | 17.5 |
| MOTOR | 17 | 36.4 | 70.6 | 48.0 |
| STRAINER | 16 | 0.0 | 0.0 | 0.0 |
| DAMPER_NO_OPERATOR | 15 | 0.0 | 0.0 | 0.0 |
| PUMP_VERTICAL | 13 | 33.3 | 30.8 | 32.0 |
| COMPRESSOR | 12 | 0.0 | 0.0 | 0.0 |
| MOISTURE_SEPARATOR | 12 | 75.0 | 100.0 | 85.7 |
| CARBON_ADSORBER | 12 | 92.3 | 100.0 | 96.0 |
| VLV_PILOT_RELIEF | 12 | 0.0 | 0.0 | 0.0 |
| CONDENSER | 11 | 10.0 | 9.1 | 9.5 |
| CHILLER | 11 | 0.0 | 0.0 | 0.0 |
| GENERATOR | 10 | 0.0 | 0.0 | 0.0 |
| DEMIN | 10 | 0.0 | 0.0 | 0.0 |
| VLV_3WAY | 9 | 0.0 | 0.0 | 0.0 |
| AHU | 9 | 0.0 | 0.0 | 0.0 |
| PANEL | 7 | 0.0 | 0.0 | 0.0 |
| INVERTER | 6 | 0.0 | 0.0 | 0.0 |
| DRYER | 5 | 0.0 | 0.0 | 0.0 |
| VLV_ANGLE | 5 | 0.0 | 0.0 | 0.0 |
| SWITCH | 5 | 0.0 | 0.0 | 0.0 |

## Most frequent confusions (same model)

Examples of each: `confusions.png`.

| true class | predicted | boxes |
|---|---|---:|
| VLV_GLOBE | VLV_GATE | 94 |
| VLV_BUTTERFLY | SWITCH | 42 |
| INSTR | GENERATOR | 37 |
| VLV_STOP_CHECK | VLV_CHECK | 28 |
| STEAM_GEN | TANK | 27 |
| BREAKER | VLV_CHECK | 26 |
| BREAKER | SWITCH | 22 |
| ORIFICE | HX | 22 |
| VLV_BUTTERFLY | DAMPER_NO_OPERATOR | 22 |
| VLV_SAFETY_RELIEF | VLV_CHECK | 22 |
| DAMPER_PNEUMATIC | ORIFICE | 21 |
| HX | BLOCK | 18 |
| ORIFICE | VLV_GATE | 17 |
| VLV_CHECK | CHARGER | 15 |
| FAN | PUMP_CENTRIFUGAL | 15 |
| OFFPAGE | INSTR | 14 |
| ORIFICE | OTHER | 14 |
| PENETRATION | ORIFICE | 13 |
| VLV_PILOT_RELIEF | VLV_GATE | 12 |
| SPRAY | VLV_SAFETY_RELIEF | 12 |
