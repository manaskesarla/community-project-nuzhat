"""Map free-text SpectrumIdentity labels to the project classes.

Classes: PET, PE, PP, PS, PVC, PA, OTHER_PLASTIC, NON_PLASTIC.  Anything that cannot be
resolved (Unknown / Umweltprobe / empty) returns None and is dropped.
"""
import re

import pandas as pd

TARGET = ["PET", "PE", "PP", "PS", "PVC", "PA"]
OTHER, NON = "OTHER_PLASTIC", "NON_PLASTIC"

# order matters: first match wins. Copolymers/blends are checked first so they are NOT
# swallowed by a base-polymer rule (e.g. SAN, ABS, EVA, PBT, polyesterurethane).
_EXCLUDE_TARGET = re.compile(
    r"urethane|butylene|buthylene|butyr|polybutyratadipat|\bpbt\b|\bpbs\b|"
    r"styren?e_|styrol-|styrol_|styrolbutadien|acrylnitril|acrylonitrile|\bsan\b|\babs\b|\bsbc\b|\bseb\b|"
    r"ethylene_(vinyl|methacrylic|propylene|acrylic|ethyl)|ethylen-|ethylene/vinyl|vinylidene|\be/p\b|\beva\b|"
    r"verbund|oxidized.*wax|wax|polyimide|polyetherether|\bpan\b|aramid|vinyl_alcohol|vinyl alcohol|\bpva\b|"
    r"vinyl_acetate|ethylene_oxide|\bpei\b|sulfone|sulphone|polyacetal|\bpom\b",
    re.I)
_RULES = [
    ("PVC", re.compile(r"^pvc$|vinyl ?chlorid|polyvinylchlorid", re.I)),
    ("PET", re.compile(r"^pet( |-)?g?$|^pest$|\bpet\b|terephthal|tehylene_terephthal|^polyester$|fibre_polyester|umwelt/pest|^pes$", re.I)),
    ("PS", re.compile(r"^e?ps$|polystyr|(?<!co)polystyren", re.I)),
    ("PP", re.compile(r"^pp$|polypropyl|poly_?propyl", re.I)),
    ("PA", re.compile(r"^pa(6|66|12|11)?$|polyamid|polymaide|nylon|fibre_polyamide", re.I)),
    ("PE", re.compile(r"^pe$|^[hlm]?l?d?pe$|^u?ldpe$|^tcl?dpe$|^tpcldpe$|^lldpe$|^pe-oxidized$|"
                      r"polyethylene(?!_terephth)|polytehylene(?!_terephth)|polyethylen|density polyethylene|"
                      r"polyethylene_(high|low)", re.I)),
]
_NATURAL = re.compile(
    r"cotton|linen|wool|fur_|hair|silk|flax|hemp|jute|kapok|cocoanut|honeycomb|broodcomb|feather|coal|amber|"
    r"tomato|cellulose|cigarette|rosin|paraffin|grass|turf|urtica|poplar|cashmere|mohair|merino|natur|"
    r"viscose|latex|camel|quartz|biotite|feldspar|muscovite|silica|chitin|sand|fibre_acetate|fibre_indian", re.I)
_UNKNOWN = re.compile(r"^(unknown|umweltprobe|na|nan|)$", re.I)


def map_label(raw) -> str | None:
    s = "" if pd.isna(raw) else str(raw).strip()
    if _UNKNOWN.match(s):
        return None
    excluded = bool(_EXCLUDE_TARGET.search(s))
    if not excluded:
        for cls, rx in _RULES:
            if rx.search(s):
                return cls
    if _NATURAL.search(s):
        return NON
    return OTHER
