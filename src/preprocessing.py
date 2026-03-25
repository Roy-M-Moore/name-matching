import pandas as pd
import numpy as np
import regex as re
from unidecode import unidecode

LEGAL_SUFFIXES = {
    "bv", "nv", "sa", "ag", "vof", "cv",
    "stichting", "maatschap",
    "vzw", "sl", "lp", "llp", "ohg", "ev",
    "sarl", "sca", "scs", "gmbh", "llc",
    "aps", "as", "ivs", "ks", "fmba",
    "ltd", "limited", "plc", "inc", "corp", 
    "kg", "kgaa", "oy",
    "srl", "spa", "pte", "ab", "as", "eg", "sro"
}

def basic_normalize(name: str) -> str:
    
    if not isinstance(name, str):
        return ""
    
    # lowercase
    name = name.lower()
    
    # unicode normalization (müller -> muller)
    name = unidecode(name)
    
    # handle initials with dots: A.B.C -> ABC, A.B.C. -> ABC
    name = re.sub(
        r'\b(?:[a-z][./])+[a-z][./]?',
        lambda m: re.sub(r'[./&]', '', m.group(0)),
        name
    )
    
    # replace punctuation with space (keep word chars, spaces)
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def remove_legal_suffixes(name: str) -> str:
    
    tokens = name.split()
    
    tokens = [
        t for t in tokens
        if t not in LEGAL_SUFFIXES
    ]
    
    return " ".join(tokens)

def remove_single_chars(name: str) -> str:

    tokens = name.split()

    long_tokens = [t for t in tokens if len(t) > 1]

    # drop only if signal remains
    if long_tokens:
        return " ".join(long_tokens)
    
    return " ".join(tokens)

def preprocess_company_name(name: str) -> str:
    
    name = basic_normalize(name)
    name = remove_legal_suffixes(name)
    # name = remove_single_chars(name)
    
    return name
