#"Access Control: 1  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0"

labels_contract = {
    "access_control": set(),
    "arithmetic": set(),
    "bad_randomness": set(),
    "denial_of_service": set(),     
    "front_running": set(),
    "reentrancy": set(),
    "short_addresses": set(),
    "time_manipulation": set(),
    "unchecked_low_level_calls": set()
}

labels_findings = {
    "access_control": "Access Control: 1  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "arithmetic": "Access Control: 0  \nArithmetic: 1  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "bad_randomness": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 1  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "denial_of_service": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 1  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "front_running":"Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 1  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "reentrancy": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 1  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "short_addresses": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 1  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 0",
    "time_manipulation": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 1  \nUnchecked Low Level Calls: 0",
    "unchecked_low_level_calls": "Access Control: 0  \nArithmetic: 0  \nBad Randomness: 0  \nDenial Of Service: 0  \nFront Running: 0  \nReentrancy: 0  \nShort Addresses: 0  \nTime Manipulation: 0  \nUnchecked Low Level Calls: 1"
}

# scorri tutti gli item del file json 
def analyze_perfect_detection(data):
    for item in data:
        # per ogni key nel dizionario labels_contract
        for key in labels_contract.keys():
            # se la chiave è presente nell'item, aggiungi il valore alla lista corrispondente
            if key == item["category"] and item["findings"] == labels_findings[key]:
                labels_contract[key].add(item['file_name'])
    
    return labels_contract

#apri il file json
def load_data(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

#cicla da 1 a 5
for i in range(1, 6):
    # carica il file json
    json = load_data(f"./results/original/uncommented_gpt-4.1/{i}_ORIGINAL_output.json")
    # esegui l'analisi
    result = analyze_perfect_detection(json)
    

# stampa il risultato
for category, contracts in result.items():
    print(f"{category}: {len(contracts)} contracts")
    for contract in contracts:
        print(f"  - {contract}")

total_lines = 0
total_contracts = 0
# scorri il dizionario labels_contract e conta il numero di righe di ogni contratto
# cicla sulla chiave del dizionario labels_contract
for key, contracts in labels_contract.items():
    # prendi la lista dei contratti
    contracts_list = list(contracts)
    # incrementa il numero totale di contratti
    total_contracts += len(contracts_list)
    # per ogni contratto nella lista dei contratti
    for contract in contracts_list:
        # apri il file del contratto
        with open(f"./smartbugs-curated/dataset_cleaned/{key}/{contract}", 'r') as file:
            # leggi il file
            lines = file.readlines()
            # somma il numero di righe del contratto al totale
            total_lines += len(lines)
            # stampa il numero di righe del contratto
            print(f"{contract} has {len(lines)} lines")

# stampa il numero totale di righe
print(f"Total lines in all contracts: {total_lines}")   
# stampa la media delle righe per contratto
print(f"Average lines per contract: {total_lines / total_contracts if total_contracts > 0 else 0:.2f}")