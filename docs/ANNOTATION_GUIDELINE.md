# CROPSTATE Annotation Guideline

## 1. Task definition

The primary vision task is six-class **image classification**. Use object detection only as an optional auxiliary task for panicle or plant-part localization.

## 2. Stage taxonomy

| ID | Label | BBCH range | Minimum visible evidence |
|---:|---|---|---|
| 0 | Establishment | 00-19 | young plants, few leaves/tillers, sparse canopy |
| 1 | Tillering | 20-29 | multiple tillers emerging from plant base, no visible panicle |
| 2 | Stem/Booting | 30-49 | elongated stems or swollen boot, no fully emerged panicle |
| 3 | Reproductive | 50-69 | panicle emergence, heading, flowering or anthesis evidence |
| 4 | Grain Filling | 70-79 | green panicles with developing grains |
| 5 | Ripening | 80-89 | yellowing panicles/grains, senescence or mature appearance |

## 3. Reject or mark uncertain

Reject an image from supervised ground truth when:

- morphology needed for the label is outside the crop;
- the image is too blurred, overexposed, or heavily occluded;
- multiple fields or plants at different stages dominate one image;
- only background color is available as evidence;
- the label is inferred only from filename or approximate date;
- a patch removes panicles or stems required to distinguish later stages.

Use `uncertain` during annotation and adjudicate before training.

## 4. Patch and overlap rules

Each patch must record `parent_image_id`. All patches from one parent image, capture burst, or overlapping tile group must remain in the same split. When exact overlap data are available, also store crop coordinates.

## 5. Ground-truth evidence priority

1. Expert morphological assessment and BBCH description.
2. Field log with date, variety, days after sowing/transplanting, and observed structures.
3. Agreement of two trained annotators.
4. Caption or filename only as supporting evidence, never as the sole label.

## 6. Double annotation

Independently annotate at least 20-30% of the held-out dataset. Report Cohen's kappa for categorical labels or Krippendorff's alpha when missing/uncertain annotations occur. Preserve disagreements and record adjudication.

## 7. Knowledge-chunk annotation

For each agricultural chunk, record:

- topic;
- direct applicable stages;
- adjacent or multi-stage applicability;
- explicit contraindicated stages;
- source organization and publication version;
- authority score and review status.

Compatibility vector convention:

- `1.0`: directly applicable;
- `0.6`: adjacent or weakly applicable;
- `0.5`: general evidence;
- `0.25`: unknown applicability;
- `0.0`: incompatible or contraindicated.

Values are hyperparameters and must be included in sensitivity analysis.
