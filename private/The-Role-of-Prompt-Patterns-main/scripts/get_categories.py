import json
import os

# Path to the vulnerabilities.json file
file_path = os.path.expanduser("~/Desktop/AblationStudy/smartbugs-curated/vulnerabilities.json")

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract unique category names
categories = set()
for item in data:
     for vuln in item["vulnerabilities"]:
        if vuln['category']:
            categories.add(vuln['category'])

# Print the unique categories
print("Unique Categories:")
for category in sorted(categories):
    print(category)