import json, os

# Lista ordinata delle vulnerabilità da usare come intestazioni (righe e colonne)
VULNERABILITIES = [
    "access_control",
    "arithmetic",
    "bad_randomness",
    "denial_of_service",
    "front_running",
    "reentrancy",
    "short_addresses",
    "time_manipulation",
    "unchecked_low_level_calls"
]

# Mappiamo tutto in lowercase con underscore per coerenza
def normalize(vuln):
    return vuln.strip().lower().replace(" ", "_")

def parse_findings(findings_raw):
    findings = {}
    for line in findings_raw.strip().split("\n"):
        if ": " in line:
            key, val = line.split(": ", 1)
            try:
                findings[normalize(key.strip())] = int(val.strip())
            except ValueError:
                print(f"Warning: could not parse count for line: {line}")
    return findings

def votes(number_of_votes, prompt):
    lower_case = prompt.lower()
    # Carica tutte le 5 esecuzione di original, 1_ORIGINAL, 2_ORIGINAL, 3_ORIGINAL, 4_ORIGINAL, 5_ORIGINAL
    with open(f"./results/{lower_case}/1_{prompt}_output.json", "r") as f:
        data1 = json.load(f)
    with open(f"./results/{lower_case}/2_{prompt}_output.json", "r") as f:
        data2 = json.load(f)
    with open(f"./results/{lower_case}/3_{prompt}_output.json", "r") as f:
        data3 = json.load(f)
    with open(f"./results/{lower_case}/4_{prompt}_output.json", "r") as f:
        data4 = json.load(f)
    with open(f"./results/{lower_case}/5_{prompt}_output.json", "r") as f:
        data5 = json.load(f)


    # crea un file se non esiste
    if not os.path.exists(f"./results/majority_voting/{lower_case}_{number_of_votes}_votes_output.json"):
        with open(f"./results/majority_voting/{lower_case}_{number_of_votes}_votes_output.json", "w") as f:
            json.dump([], f, indent=4)

        # apri il file json presente nella cartella results/majority_voting di nome majority_voting_output.json in modalità modifica
    with open(f"./results/majority_voting/{lower_case}_{number_of_votes}_votes_output.json", "r") as f:
        votes = json.load(f)

    # Scorri gli item dei dati in contemporaneo
    for item1, item2, item3, item4, item5 in zip(data1, data2, data3, data4, data5):
        results = {}
        findinds1 = parse_findings(item1["findings"])
        findings2 = parse_findings(item2["findings"])
        findings3 = parse_findings(item3["findings"])
        findings4 = parse_findings(item4["findings"])
        findings5 = parse_findings(item5["findings"])
        # scorri le findings e fai il voto
        combined_findings = {}
        for key in set(findinds1.keys()):
            combined_findings[key] = 1 if sum([
                findinds1.get(key, 0),
                findings2.get(key, 0),
                findings3.get(key, 0),
                findings4.get(key, 0),
                findings5.get(key, 0)
            ]) >= number_of_votes else 0  # Maggiore o uguale a 3 voti positivi
        results["number_of_votes"] = number_of_votes
        results["prompt"] = item1["prompt"]  # Assumiamo che il prompt sia lo stesso per tutti gli item
        results["file_name"] = item1["file_name"]  # Assumiamo che il file_name sia lo stesso per tutti gli item
        results["category"] = item1["category"]  # Assumiamo che la categoria sia la stessa per tutti gli item
        results["findings"] = "\n".join([f"{k}: {v}" for k, v in combined_findings.items()])
        votes.append(results)
    # Salva il file json con i risultati
    with open(f"./results/majority_voting/{lower_case}_{number_of_votes}_votes_output.json", "w") as f:
        json.dump(votes, f, indent=4)
    print("Majority voting completed and results saved to majority_voting_output.json")


votes(number_of_votes=5, prompt="ORIGINAL")  #  
votes(number_of_votes=4, prompt="ORIGINAL")  #  
votes(number_of_votes=3, prompt="ORIGINAL")  #  

votes(number_of_votes=5, prompt="VARIANT_1")  #  
votes(number_of_votes=4, prompt="VARIANT_1")  #  
votes(number_of_votes=3, prompt="VARIANT_1")  #  

votes(number_of_votes=5, prompt="VARIANT_2")  #  
votes(number_of_votes=4, prompt="VARIANT_2")  #  
votes(number_of_votes=3, prompt="VARIANT_2")  #  

votes(number_of_votes=5, prompt="VARIANT_3")  #  
votes(number_of_votes=4, prompt="VARIANT_3")  #  
votes(number_of_votes=3, prompt="VARIANT_3")  #  