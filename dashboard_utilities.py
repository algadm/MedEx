import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def set_age_data(df):
    # Age groups calculation
    age_bins = [0, 19, 40, 100]
    age_labels = ['12-19', '20-40', '40+']
    age_groups = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels)
    age_dist = age_groups.value_counts(normalize=True).sort_index() * 100
    
    return age_dist

def set_sleep_data(df):
    # Sleep disorder data (including chronic fatigue)
    has_sleep_disorder = df['sleep_disorder_type'].notna() & (df['sleep_disorder_type'] != "")
    has_chronic_fatigue = df['chronic_fatigue_present'] == "True"
    sleep_disorder_percentage = (has_sleep_disorder | has_chronic_fatigue).mean() * 100
    
    return sleep_disorder_percentage

def set_tenderness_data(df):
    # Tenderness/Stiffness/Soreness data
    tenderness_metrics = ["muscle_tenderness_present", "muscle_stiffness_present", "muscle_soreness_present"]
    tenderness_percentage = (df[tenderness_metrics] == "True").any(axis=1).mean() * 100
    
    return tenderness_percentage

def set_migraine_data(df):
    no_migraine_headache = 0
    headache_only = 0
    migraine_only = 0
    migraine_and_headache = 0

    # Iterate through the dataframe to categorize patients
    for idx, row in df.iterrows():
        migraine_history = row['migraine_history']
        headache_intensity = row['headache_intensity']
        
        if pd.isna(migraine_history) or migraine_history == "":
            if pd.isna(headache_intensity) or headache_intensity == 0:
                no_migraine_headache += 1
            else:
                headache_only += 1
        else:
            if pd.isna(headache_intensity) or headache_intensity == 0:
                migraine_only += 1
            else:
                migraine_and_headache += 1

    # Calculate percentages
    total_patients = len(df)
    no_migraine_headache_pct = (no_migraine_headache / total_patients) * 100
    headache_only_pct = (headache_only / total_patients) * 100
    migraine_only_pct = (migraine_only / total_patients) * 100
    migraine_and_headache_pct = (migraine_and_headache / total_patients) * 100
    
    return no_migraine_headache_pct, headache_only_pct, migraine_only_pct, migraine_and_headache_pct

def set_left_stick_data(df):
    metrics = ["headache_intensity", "average_daily_pain_intensity", 
              "diet_score", "tmj_pain_rating", "disability_rating"]
    
    metrics_titles = ["Headache\nIntensity", "Daily Pain\nIntensity", "Diet\nScore", "TMJ pain\nRating", "Disability\nRating"]
    
    means = df[metrics].mean()
    std_devs = df[metrics].std()
    
    return metrics_titles, means, std_devs

def set_middle_stick_data(df):
    def extract_mm_value(value):
        """
        Extract numerical value from a string like '33mm'.
        Returns NaN if no match is found.
        """
        match = re.search(r'\d+', str(value))
        return float(match.group(0)) if match else np.nan

    # Apply the function to extract numerical values
    max_opening = df['maximum_opening'].apply(extract_mm_value) if 'maximum_opening' in df else pd.Series([np.nan] * len(df))
    max_opening_no_pain = df['maximum_opening_without_pain'].apply(extract_mm_value) if 'maximum_opening_without_pain' in df else pd.Series([np.nan] * len(df))
    
    return max_opening, max_opening_no_pain

def set_right_stick_data(df):
    # Define the list of possible headache locations
    possible_locations = ["frontal", "temporal", "posterior", "top of the head", "temple"]

    # Initialize a dictionary to count occurrences
    location_counts = {location: 0 for location in possible_locations}

    # Iterate over each patient's headache_location
    for locations in df['headache_location']:
        if pd.notna(locations) and locations != "":
            # Check for each possible location in the whole string
            for location in possible_locations:
                if location in locations.lower():
                    location_counts[location] += 1

    # Calculate percentages
    total_patients = len(df)
    location_percentages = {location: (count / total_patients) * 100 
                            for location, count in location_counts.items()}
    
    return location_percentages

def set_joint_pain_data(df):
    # Define the list of possible joint pain areas
    possible_areas = ["TMJ", "Neck", "Shoulder", "Back"]

    # Initialize a dictionary to count occurrences
    joint_pain_counts = {area: 0 for area in possible_areas}

    # Iterate over each patient's joint_pain_areas
    for areas in df['joint_pain_areas']:
        if pd.notna(areas) and areas != "":
            # Combine left and right TMJ into a single "TMJ" category
            if "left TMJ" in areas or "right TMJ" in areas:
                joint_pain_counts["TMJ"] += 1
            # Check for other joint pain areas
            for area in possible_areas:
                if area != "TMJ" and area.lower() in areas.lower():
                    joint_pain_counts[area] += 1

    # Calculate percentages
    total_patients = len(df)
    joint_pain_percentages = {area: (count / total_patients) * 100 
                              for area, count in joint_pain_counts.items()}
    
    return joint_pain_percentages

def set_lower_stick_data(df):
    # Define the list of possible severities
    possible_severities = ["low", "mild", "moderate", "severe"]

    # Initialize a dictionary to count occurrences
    severity_counts = {severity: 0 for severity in possible_severities}

    # Iterate over each patient's muscle_pain_score
    for score in df['muscle_pain_score']:
        if pd.notna(score) and score != "":
            # Convert to lowercase for case-insensitive matching
            score_lower = score.lower()
            # Check for each possible severity in the score
            for severity in possible_severities:
                if severity in score_lower:
                    severity_counts[severity] += 1
                    break  # Stop checking once a match is found

    # Calculate percentages
    total_patients = len(df)
    severity_percentages = {severity: (count / total_patients) * 100 
                            for severity, count in severity_counts.items()}
    
    return severity_percentages

def set_upper_donuts_data(df):
    bool_metrics = ["earache_present", "tinnitus_present", "vertigo_present", 
                "hearing_loss_present", "jaw_crepitus", "jaw_clicking"]

    # Calculate percentages for boolean metrics
    bool_percentages = {metric: (df[metric] == "True").mean() * 100 
                        for metric in bool_metrics}

    # Calculate percentage for jaw issues (crepitus or clicking)
    jaw_issues = (df['jaw_crepitus'] != "") | (df['jaw_clicking'] != "")
    jaw_issues_percentage = jaw_issues.mean() * 100

    # Calculate percentage for mental health issues (anxiety, depression, or stress)
    mental_health_issues = (df['anxiety_present'] == "True") | \
                            (df['depression_present'] == "True") | \
                            (df['stress_present'] == "True")
    mental_health_percentage = mental_health_issues.mean() * 100
    
    return bool_percentages, jaw_issues_percentage, mental_health_percentage

def extract_months_from_pain_onset(pain_onset_str):
    """
    Convert pain_onset_date strings (e.g., "3 years ago", "1 year and 9 months ago") into total months.
    """
    if pd.isna(pain_onset_str) or pain_onset_str == "":
        return 0
    
    # Extract years and months using regex
    years_match = re.search(r'(\d+)\s*year', pain_onset_str)
    months_match = re.search(r'(\d+)\s*month', pain_onset_str)
    
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    
    return years * 12 + months

def set_pain_onset_data(df):
    # Define age bins (10-year intervals)
    age_bins = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]
    age_labels = [f'{start}-{start+9}' for start in age_bins[:-1]]  # ['10-19', '20-29', ..., '90-99']

    # Convert pain_onset_date to months
    df['pain_onset_months'] = df['pain_onset_date'].apply(extract_months_from_pain_onset)

    # Add a column for age bins
    df['age_bin'] = pd.cut(df['patient_age'], bins=age_bins, labels=age_labels, right=False)

    # Group by age bin and calculate mean and SD of patient ages
    age_distribution_stats = df.groupby('age_bin', observed=False)['patient_age'].agg(['mean', 'std']).reset_index()

    # Group by age bin and calculate mean and SD of pain onset duration (in years)
    pain_onset_stats = df.groupby('age_bin', observed=False)['pain_onset_months'].agg(['mean', 'std']).reset_index()
    pain_onset_stats = pain_onset_stats.dropna(subset=['mean', 'std'])
    pain_onset_stats['mean'] /= 12  # Convert mean to years
    pain_onset_stats['std'] /= 12  # Convert SD to years

    # Merge the two datasets
    merged_stats = pd.merge(age_distribution_stats, pain_onset_stats, on='age_bin', suffixes=('_age', '_pain'))

    # Filter out age bins with no data
    merged_stats = merged_stats.dropna(subset=['mean_age', 'mean_pain'])

    # Data for plotting
    age_bins_plot = merged_stats['age_bin']
    mean_ages = merged_stats['mean_age']
    std_ages = merged_stats['std_age']
    mean_pain = merged_stats['mean_pain']
    std_pain = merged_stats['std_pain']
    
    return age_bins_plot, mean_ages, std_ages, mean_pain, std_pain

def set_disc_displacement_data(df):
    """
    Calculate disc displacement percentages for left and right TMJ.
    
    Args:
        df (pd.DataFrame): Processed dataframe from process_summaries()
    
    Returns:
        tuple: (left_tmj_data, right_tmj_data) where each is a dictionary of percentages
    """
    # Initialize counts
    left_counts = {
        "w/o reduction": 0,
        "w/ reduction": 0,
        "reduction not specified": 0,
        "no displacement": 0
    }
    
    right_counts = {
        "w/o reduction": 0,
        "w/ reduction": 0,
        "reduction not specified": 0,
        "no displacement": 0
    }
    
    # Iterate through the dataframe
    for _, row in df.iterrows():
        disc_displacement = str(row['disc_displacement'])
        
        # Left TMJ
        if "left TMJ without reduction" in disc_displacement:
            left_counts["w/o reduction"] += 1
        elif "left TMJ with reduction" in disc_displacement:
            left_counts["w/ reduction"] += 1
        elif "left TMJ" in disc_displacement:
            left_counts["reduction not specified"] += 1
        else:
            left_counts["no displacement"] += 1
        
        # Right TMJ
        if "right TMJ without reduction" in disc_displacement:
            right_counts["w/o reduction"] += 1
        elif "right TMJ with reduction" in disc_displacement:
            right_counts["w/ reduction"] += 1
        elif "right TMJ" in disc_displacement:
            right_counts["reduction not specified"] += 1
        else:
            right_counts["no displacement"] += 1
    
    # Convert counts to percentages
    total_patients = len(df)
    left_tmj_data = {k: (v / total_patients) * 100 for k, v in left_counts.items()}
    right_tmj_data = {k: (v / total_patients) * 100 for k, v in right_counts.items()}
    
    return left_tmj_data, right_tmj_data


def set_muscle_pain_data(df):
    # Define pain level categories and their order
    pain_levels = ["mild", "mild to moderate", "moderate", "moderate to severe", "severe"]

    # Initialize a dictionary to count occurrences of each pain level
    pain_counts = {level: 0 for level in pain_levels}

    # Parse muscle_pain_score to count pain levels
    for score in df['muscle_pain_score']:
        if pd.isna(score) or score == "":
            continue
        
        # Normalize the text to lowercase
        score = score.lower()
        
        # Check for each pain level
        if "mild" in score or "low" in score:
            pain_counts["mild"] += 1
        if "mild to moderate" in score:
            pain_counts["mild to moderate"] += 1
        if "moderate" in score and "moderate to severe" not in score and "high moderate" not in score:
            pain_counts["moderate"] += 1
        if "moderate to severe" in score or "high moderate to low severe" in score or "high moderate" in score:
            pain_counts["moderate to severe"] += 1
        if "severe" in score and "moderate to severe" not in score:
            pain_counts["severe"] += 1

    # Convert counts to percentages
    total_patients = len(df)
    pain_percentages = {level: (count / total_patients) * 100 for level, count in pain_counts.items()}
    
    clean_pain_levels = [
        level.replace("mild to moderate", "mild\nto\nmoderate").replace("moderate to severe", "moderate\nto\nsevere")
        for level in pain_levels
    ]

    return clean_pain_levels, pain_percentages