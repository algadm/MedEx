from typing import Dict, List, Optional
import re
from datetime import datetime

class TMJValidator:
    def __init__(self):
        self.medication_database = set([
            "cetirizine", "melatonin", "ibuprofen", "acetaminophen", "klonopin", "amitriptyline",
            "ativan", "doxycycline", "advil", "amitriptyline", "cyclosporin", "adderall",
            "sertraline", "synthroid", "vicodin", "naproxen", "celebrex", "relafen", "flexeril",
            "parafon forte", "soma", "robaxin", "topamax", "replax", "oracea", "lortab", "tylenol",
        ])
        
        self.wilkes_stages = set(["I", "II", "III", "IV", "V"])
        
    def validate_medications(self, medications: List[Dict]) -> List[str]:
        """Validate medication entries against known database."""
        errors = []
        for med in medications:
            name = med.get('name', '').lower()
            if name == 'sig':
                errors.append("'Sig' incorrectly identified as medication")
            if name not in self.medication_database:
                errors.append(f"Unknown medication: {name}")
            if not med.get('dosage'):
                errors.append(f"Missing dosage for {name}")
        return errors

    def validate_wilkes_classification(self, diagnosis: str) -> bool:
        """Validate Wilkes classification in diagnosis."""
        match = re.search(r'Wilkes\s+([IV]+)', diagnosis)
        if not match:
            return False
        stage = match.group(1)
        return stage in self.wilkes_stages

    def validate_measurements(self, measurements: Dict) -> List[str]:
        """Validate clinical measurements."""
        errors = []
        
        # Validate Maximum Vertical Opening (MVO)
        mvo = measurements.get('maximum_vertical_opening')
        if mvo:
            if not (35 <= float(mvo) <= 50):
                errors.append(f"MVO {mvo}mm outside normal range (35-50mm)")
                
        # Validate pain scores
        for side in ['left', 'right']:
            score = measurements.get(f'{side}_pain_score')
            if score and not (0 <= int(score) <= 10):
                errors.append(f"Invalid pain score for {side}: {score}")
                
        return errors

    def validate_dates(self, dates: Dict[str, str]) -> List[str]:
        """Validate date formatting and logical sequence."""
        errors = []
        try:
            for date_type, date_str in dates.items():
                datetime.strptime(date_str, '%m/%d/%y')
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
        return errors

# Example usage
validator = TMJValidator()

# Validate a clinical note extraction
extraction = {
    "medications": [
        {"name": "cetirizine", "dosage": "10mg", "frequency": "daily"},
        {"name": "sig", "dosage": "unknown"}  # This should trigger an error
    ],
    "diagnosis": "Wilkes IV bilateral",
    "measurements": {
        "maximum_vertical_opening": "45",
        "right_pain_score": "10",
        "left_pain_score": "10"
    }
}

medication_errors = validator.validate_medications(extraction["medications"])
wilkes_valid = validator.validate_wilkes_classification(extraction["diagnosis"])
measurement_errors = validator.validate_measurements(extraction["measurements"])

print("Validation Results:")
print(f"Medication Errors: {medication_errors}")
print(f"Wilkes Classification Valid: {wilkes_valid}")
print(f"Measurement Errors: {measurement_errors}")