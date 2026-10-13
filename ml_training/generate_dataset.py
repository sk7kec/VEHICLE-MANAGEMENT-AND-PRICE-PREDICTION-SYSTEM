"""
Synthetic Dataset Generator for Indian Used Vehicle Market

Generates a realistic synthetic dataset approximating patterns in the Indian
used-vehicle resale market. This data is for development/testing only and
does NOT represent actual market transactions.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import random
import csv
import math

SEED = 42
NUM_RECORDS = 3000

# Vehicle specifications: (make, model, fuel_types, transmission_types, engine_cc_range, base_price_range_lakh)
VEHICLE_SPECS = [
    # Budget segment
    ("Maruti Suzuki", "Alto", ["Petrol", "CNG"], ["Manual"], (796, 1000), (1.5, 3.5)),
    ("Maruti Suzuki", "Swift", ["Petrol"], ["Manual", "Automatic"], (1197, 1197), (3.5, 7.0)),
    ("Maruti Suzuki", "Baleno", ["Petrol"], ["Manual", "Automatic"], (1197, 1197), (4.5, 8.0)),
    ("Maruti Suzuki", "Dzire", ["Petrol", "CNG"], ["Manual", "Automatic"], (1197, 1197), (4.0, 7.5)),
    ("Maruti Suzuki", "WagonR", ["Petrol", "CNG"], ["Manual", "Automatic"], (998, 1197), (2.5, 5.5)),
    ("Maruti Suzuki", "Vitara Brezza", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1462, 1498), (5.0, 10.0)),
    ("Hyundai", "i20", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1197, 1497), (4.0, 9.0)),
    ("Hyundai", "Grand i10", ["Petrol", "Diesel", "CNG"], ["Manual"], (1086, 1197), (2.5, 5.5)),
    ("Hyundai", "Creta", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1497, 1582), (7.0, 16.0)),
    ("Hyundai", "Venue", ["Petrol", "Diesel"], ["Manual", "Automatic"], (998, 1497), (5.5, 11.0)),
    ("Hyundai", "Verna", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1497, 1582), (6.0, 13.0)),
    ("Tata", "Nexon", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1199, 1497), (5.0, 12.0)),
    ("Tata", "Punch", ["Petrol"], ["Manual", "Automatic"], (1199, 1199), (4.0, 8.0)),
    ("Tata", "Altroz", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1199, 1497), (4.5, 8.5)),
    ("Tata", "Harrier", ["Diesel"], ["Manual", "Automatic"], (1956, 1956), (10.0, 20.0)),
    ("Honda", "City", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1497, 1498), (6.0, 14.0)),
    ("Honda", "Amaze", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1199, 1498), (4.0, 9.0)),
    ("Toyota", "Innova Crysta", ["Petrol", "Diesel"], ["Manual", "Automatic"], (2393, 2694), (12.0, 25.0)),
    ("Toyota", "Fortuner", ["Petrol", "Diesel"], ["Manual", "Automatic"], (2694, 2755), (18.0, 40.0)),
    ("Toyota", "Glanza", ["Petrol"], ["Manual", "Automatic"], (1197, 1197), (4.0, 7.5)),
    ("Mahindra", "XUV700", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1997, 2184), (10.0, 22.0)),
    ("Mahindra", "Scorpio", ["Diesel"], ["Manual", "Automatic"], (2179, 2184), (8.0, 18.0)),
    ("Mahindra", "Thar", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1497, 2184), (8.0, 16.0)),
    ("Kia", "Seltos", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1353, 1497), (7.0, 16.0)),
    ("Kia", "Sonet", ["Petrol", "Diesel"], ["Manual", "Automatic"], (998, 1493), (5.5, 12.0)),
    ("Ford", "EcoSport", ["Petrol", "Diesel"], ["Manual", "Automatic"], (1497, 1498), (5.0, 10.0)),
    ("Volkswagen", "Polo", ["Petrol"], ["Manual", "Automatic"], (999, 1197), (4.0, 8.0)),
    ("Volkswagen", "Taigun", ["Petrol"], ["Manual", "Automatic"], (999, 1498), (7.0, 15.0)),
    ("Skoda", "Kushaq", ["Petrol"], ["Manual", "Automatic"], (999, 1498), (7.0, 15.0)),
    ("Skoda", "Slavia", ["Petrol"], ["Manual", "Automatic"], (999, 1498), (7.5, 16.0)),
    # Premium segment
    ("BMW", "3 Series", ["Petrol", "Diesel"], ["Automatic"], (1998, 2998), (25.0, 50.0)),
    ("BMW", "X1", ["Petrol", "Diesel"], ["Automatic"], (1499, 1995), (20.0, 40.0)),
    ("Mercedes-Benz", "C-Class", ["Petrol", "Diesel"], ["Automatic"], (1496, 2996), (28.0, 55.0)),
    ("Mercedes-Benz", "GLA", ["Petrol", "Diesel"], ["Automatic"], (1332, 2143), (22.0, 42.0)),
    ("Audi", "A4", ["Petrol", "Diesel"], ["Automatic"], (1395, 1984), (24.0, 45.0)),
    ("Audi", "Q3", ["Petrol", "Diesel"], ["Automatic"], (1395, 1984), (22.0, 38.0)),
]


def generate_dataset(output_path, num_records=NUM_RECORDS, seed=SEED):
    """Generate the synthetic vehicle dataset."""
    random.seed(seed)
    records = []

    for _ in range(num_records):
        spec = random.choice(VEHICLE_SPECS)
        make, model, fuel_types, trans_types, engine_range, price_range_lakh = spec

        fuel_type = random.choice(fuel_types)
        transmission = random.choice(trans_types)
        engine_cc = random.randint(engine_range[0], engine_range[1])

        year = random.randint(2005, 2024)
        age = 2024 - year

        # KM driven: correlated with age
        avg_km_per_year = random.randint(8000, 18000)
        km_driven = int(age * avg_km_per_year * random.uniform(0.6, 1.4))
        km_driven = max(500, km_driven)

        previous_owners = min(4, max(0, int(random.gauss(age / 5, 0.8))))

        # Calculate price based on base range with realistic depreciation
        base_price_lakh = random.uniform(price_range_lakh[0], price_range_lakh[1])

        # Depreciation: ~15-20% first year, ~10-12% subsequent years
        depreciation = 1.0
        for y in range(age):
            if y == 0:
                depreciation *= random.uniform(0.78, 0.87)
            elif y < 3:
                depreciation *= random.uniform(0.85, 0.92)
            elif y < 7:
                depreciation *= random.uniform(0.88, 0.95)
            else:
                depreciation *= random.uniform(0.92, 0.97)

        # High mileage penalty
        if km_driven > 100000:
            depreciation *= random.uniform(0.85, 0.95)
        if km_driven > 150000:
            depreciation *= random.uniform(0.88, 0.95)

        # Ownership history impact
        depreciation *= (1 - previous_owners * random.uniform(0.02, 0.05))

        # Transmission premium
        if transmission == "Automatic":
            depreciation *= random.uniform(1.02, 1.08)

        # Diesel premium for larger vehicles
        if fuel_type == "Diesel" and engine_cc > 1400:
            depreciation *= random.uniform(1.01, 1.05)

        # CNG discount
        if fuel_type == "CNG":
            depreciation *= random.uniform(0.92, 0.98)

        selling_price = base_price_lakh * depreciation * 100000  # Convert to INR
        selling_price = max(30000, round(selling_price, -2))  # Round to nearest 100

        # Add some noise
        noise = random.uniform(0.92, 1.08)
        selling_price = round(selling_price * noise, -2)
        selling_price = max(50000, selling_price)

        records.append({
            "make": make,
            "model": model,
            "year": year,
            "km_driven": km_driven,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "engine_cc": engine_cc,
            "previous_owners": previous_owners,
            "selling_price": selling_price,
        })

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "make", "model", "year", "km_driven", "fuel_type",
            "transmission", "engine_cc", "previous_owners", "selling_price"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} records -> {output_path}")
    return records


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output = os.path.join(project_root, "dataset", "vehicles.csv")
    generate_dataset(output)
