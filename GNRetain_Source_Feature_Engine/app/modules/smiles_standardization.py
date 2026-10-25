"""Local RDKit-only SMILES handling for GNRetain runtime inference."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SmilesValidation:
    is_valid: bool
    canonical_smiles: str | None
    message: str
    is_mixture_or_salt: bool


def parse_smiles(smiles: str):
    """Parse a SMILES string with RDKit and return a molecule or None."""
    if smiles is None:
        return None
    value = str(smiles).strip()
    if not value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def detect_salt_or_mixture(smiles: str) -> bool:
    """Return True when the input contains multiple disconnected fragments."""
    mol = parse_smiles(smiles)
    if mol is None:
        return False
    return len(Chem.GetMolFrags(mol)) > 1


def largest_organic_fragment(smiles: str) -> str | None:
    """Return the largest organic fragment as canonical SMILES, if available."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    try:
        frag = chooser.choose(mol)
    except Exception:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if frag is None or frag.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical isomeric SMILES without changing fragment composition."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def standardize_smiles(smiles: str) -> str:
    """Normalize, choose the largest organic fragment, and canonicalize locally."""
    if smiles is None:
        raise ValueError("SMILES is required.")
    value = str(smiles).strip()
    if not value:
        raise ValueError("SMILES is empty.")
    mol = parse_smiles(value)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {value}")
    try:
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)
    except Exception:
        pass
    canonical = largest_organic_fragment(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    if canonical is None:
        raise ValueError(f"Could not standardize SMILES: {value}")
    return canonical


def validate_smiles(smiles: str) -> SmilesValidation:
    """Validate a SMILES string without raising for routine user-input errors."""
    if smiles is None or not str(smiles).strip():
        return SmilesValidation(False, None, "SMILES is empty.", False)
    mol = parse_smiles(smiles)
    if mol is None:
        return SmilesValidation(False, None, "SMILES could not be parsed by RDKit.", False)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    is_mix = len(Chem.GetMolFrags(mol)) > 1
    return SmilesValidation(True, canonical, "SMILES parsed successfully.", is_mix)
