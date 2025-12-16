# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:43:14 2025

@author: mep24db
"""

from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    gpu_ids=[0])
tracker.start()

# --- Your code here ---
sum([i**2 for i in range(10_000_000)])
# -----------------------

emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO₂eq")

