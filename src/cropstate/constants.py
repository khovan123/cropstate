STAGE_NAMES = [
    "establishment",
    "tillering",
    "stem_booting",
    "reproductive",
    "grain_filling",
    "ripening",
]
STAGE_TO_ID = {name: i for i, name in enumerate(STAGE_NAMES)}
ID_TO_STAGE = {i: name for name, i in STAGE_TO_ID.items()}

STAGE_DISPLAY_NAMES = {
    "establishment": "Establishment",
    "tillering": "Tillering",
    "stem_booting": "Stem/booting",
    "reproductive": "Reproductive",
    "grain_filling": "Grain filling",
    "ripening": "Ripening",
}

STAGE_BBCH_RANGES = {
    "establishment": "00-19",
    "tillering": "20-29",
    "stem_booting": "30-49",
    "reproductive": "50-69",
    "grain_filling": "70-79",
    "ripening": "80-89",
}

STAGE_ALIASES = {
    "s01": "establishment",
    "s01_establishment": "establishment",
    "establishment": "establishment",
    "germination_leaf_development": "establishment",
    "s02": "tillering",
    "s02_tillering": "tillering",
    "tillering": "tillering",
    "s03": "stem_booting",
    "s03_stem_booting": "stem_booting",
    "stem_booting": "stem_booting",
    "stem_boot": "stem_booting",
    "stem_elongation_booting": "stem_booting",
    "stem_elongation_and_booting": "stem_booting",
    "s04": "reproductive",
    "s04_reproductive": "reproductive",
    "reproductive": "reproductive",
    "heading_flowering": "reproductive",
    "heading_and_flowering": "reproductive",
    "s05": "grain_filling",
    "s05_grain_filling": "grain_filling",
    "grain_filling": "grain_filling",
    "grain_development": "grain_filling",
    "s06": "ripening",
    "s06_ripening": "ripening",
    "ripening": "ripening",
}

NON_TRAINING_STAGE_ALIASES = {
    "s07": "uncertain",
    "s07_uncertain": "uncertain",
    "uncertain": "uncertain",
    "s08": "unusable",
    "s08_unusable": "unusable",
    "unusable": "unusable",
}
