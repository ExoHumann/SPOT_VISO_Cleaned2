#!/usr/bin/env python3
"""
Test script for SpotLoader: load raw data, group by class, and print all objects.
Also demonstrates creating SpotJsonObject instances and applying filters.
"""

from spot_loader import SpotLoader
from SPOT_Filters import SpotJsonObject

class SpotFilter:
    """
    A class to filter SpotJsonObject instances based on various criteria.
    """
    def __init__(self, objects: list[SpotJsonObject]):
        self.objects = objects

    def filter_by_class(self, class_name: str) -> list[SpotJsonObject]:
        """Filter objects by Class."""
        return [obj for obj in self.objects if obj["Class"] == class_name]

    def filter_by_axis(self, axis_name: str) -> list[SpotJsonObject]:
        """Filter objects by Axis@Name."""
        return [obj for obj in self.objects if obj["Axis@Name"] == axis_name]

    def filter_by_station_gt(self, value: float) -> list[SpotJsonObject]:
        """Filter objects where StationValue > value."""
        def get_station_val(obj):
            val = obj["StationValue"]
            if isinstance(val, list):
                val = val[0] if val else 0
            try:
                return float(val) if val else 0
            except (ValueError, TypeError):
                return 0
        return [obj for obj in self.objects if get_station_val(obj) > value]

    def filter_by_name_contains(self, substring: str) -> list[SpotJsonObject]:
        """Filter objects where Name contains substring."""
        return [obj for obj in self.objects if obj["Name"] and substring.lower() in (obj["Name"] or "").lower()]

def main():
    # Initialize SpotLoader with GIT folder and MAIN branch
    loader = SpotLoader(master_folder="GIT", branch="MAIN", verbose=True)

    # Load raw data and group by class
    loader.load_raw().group_by_class()

    # Create SpotJsonObject instances from raw rows
    spot_objects = [SpotJsonObject(row) for row in loader._raw_rows]

    # Create SpotFilter instance
    filter_obj = SpotFilter(spot_objects)

    # Print all objects grouped by class
    print("\n" + "="*80)
    print("GROUPED OBJECTS BY CLASS")
    print("="*80)

    for class_name, rows in sorted(loader._by_class.items()):
        print(f"\nClass: {class_name} ({len(rows)} rows)")
        print("-" * (len(class_name) + 10))

        for i, row in enumerate(rows, 1):
            print(f"  Row {i}: {row}")

    print(f"\nTotal classes: {len(loader._by_class)}")
    total_rows = sum(len(rows) for rows in loader._by_class.values())
    print(f"Total rows: {total_rows}")

    # Demonstrate filtering with SpotFilter
    print("\n" + "="*80)
    print("FILTERED OBJECTS USING SpotFilter")
    print("="*80)

    # Example filter: PierObject instances
    pier_objects = filter_obj.filter_by_class("PierObject")
    print(f"\nFiltered PierObject: {len(pier_objects)} objects")
    for i, obj in enumerate(pier_objects[:5], 1):  # Show first 5
        print(f"  Pier {i}: Name={obj.get('Name', 'N/A')}, Axis={obj.get('Axis@Name', 'N/A')}")

    # Example filter: Objects with specific axis
    axis_ra_objects = filter_obj.filter_by_axis("RA")
    print(f"\nFiltered objects with Axis@Name='RA': {len(axis_ra_objects)} objects")
    for i, obj in enumerate(axis_ra_objects[:5], 1):  # Show first 5
        print(f"  RA {i}: Class={obj.get('Class', 'N/A')}, Name={obj.get('Name', 'N/A')}")

    # Example filter: Objects with StationValue > 100
    high_station_objects = filter_obj.filter_by_station_gt(100)
    print(f"\nFiltered objects with StationValue > 100: {len(high_station_objects)} objects")
    for i, obj in enumerate(high_station_objects[:5], 1):  # Show first 5
        print(f"  High Station {i}: Class={obj.get('Class', 'N/A')}, Name={obj.get('Name', 'N/A')}, Station={obj.get('StationValue', 'N/A')}")

    # Example filter: Objects with Name containing 'Pier'
    pier_name_objects = filter_obj.filter_by_name_contains("Pier")
    print(f"\nFiltered objects with Name containing 'Pier': {len(pier_name_objects)} objects")
    for i, obj in enumerate(pier_name_objects[:5], 1):  # Show first 5
        print(f"  Pier Name {i}: Class={obj.get('Class', 'N/A')}, Name={obj.get('Name', 'N/A')}")

if __name__ == "__main__":
    main()
