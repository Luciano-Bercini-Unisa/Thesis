# scorri tutti i file .sol in un dataset e per ogni file conta il numero di # commenti e il numero di linee di codice
import os
import json
import re

def count_comments_and_loc(code):

    comment_count = 0
    loc_count = 0
    in_block_comment = False

    for line in code:
        stripped = line.strip()

        # Salta le linee vuote
        if not stripped:
            continue

        # Gestione commenti multi-linea
        if in_block_comment:
            comment_count += 1
            if "*/" in stripped:
                in_block_comment = False
            continue

        # Inizio di un commento multi-linea
        if stripped.startswith("/*"):
            comment_count += 1
            if "*/" not in stripped:  # blocco non terminato nella stessa linea
                in_block_comment = True
            continue

        # Commenti singola linea
        if stripped.startswith("//"):
            comment_count += 1
            continue

        # Linee con codice e commento inline
        if "//" in stripped:
            comment_count += 1
            loc_count += 1
            continue

        # Linea di codice pura
        loc_count += 1

    return comment_count, loc_count

comment_count = tot_comment_count = 0
loc_count = tot_loc_count = 0
for category in [
        "access_control", "arithmetic", "bad_randomness", "denial_of_service",
        "front_running", "reentrancy", "short_addresses", "time_manipulation",
        "unchecked_low_level_calls"
    ]:
    print(f"Processing category: {category}")
    category_path = os.path.join("/Users/gerardo/Desktop/AblationStudy/smartbugs-curated/dataset/", category)

    for file_name in os.listdir(category_path):
        if file_name.endswith(".sol"):
            file_path = os.path.join(category_path, file_name)
            with open(file_path, "r") as file:
                code = file.readlines()
            comment_count, loc_count = count_comments_and_loc(code)   
            tot_comment_count += comment_count
            tot_loc_count += loc_count  

print(f"Comments: {tot_comment_count}, LOC: {tot_loc_count}")