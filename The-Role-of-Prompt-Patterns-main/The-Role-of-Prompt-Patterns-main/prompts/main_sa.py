from openai import OpenAI
from keys import OPEN_AI_KEY
from prompts import ORIGINAL_PROMPT_SA, ORIGINAL_PROMPT_SA_RP
import os, argparse, json
# Imposta la tua chiave API di OpenAI

def esegui_prompt(model, formatted_prompt):
    try:
        
        client = OpenAI(
            api_key=OPEN_AI_KEY,
            organization="org-o7zOuoixFz94QuEm3bqKytHM",
            project="proj_CHdbaL1Dgy1GGODBmtcyXKRr")
        # setup utilizzato da me
        #msg = [{"role": "user", "content": formatted_prompt}]

        # setup utilizzato nel RP
        msg = [{"role": "system", "content": "You are a semantic analyzer of text."}]
        msg.append({"role": "user", "content": formatted_prompt})

        print("\nEseguendo il prompt con il modello:", msg)
        response = client.chat.completions.create(
            model = model,  # Specifica il modello che vuoi usare
            messages = msg,
            temperature = 0.7  # Controlla la creatività della risposta
        )

        print("\nRisultato del prompt:")
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Errore durante l'esecuzione del prompt: {e}")
        return f"Errore durante l'esecuzione del prompt: {e}"

def main(filename, model):

    output_file = f"../results/{filename}.json"

    # Carica i risultati esistenti
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        for result in results:
            if result["findings"] != []:
                print(f"File {result['file_name']} already processed. Skipping.")
                continue
            formatted_prompt = ORIGINAL_PROMPT_SA_RP.format(input=result["result"])
            #formatted_prompt = ORIGINAL_PROMPT_SA.format(input=result["result"])
            print(formatted_prompt)
            finding = esegui_prompt(model, formatted_prompt)
            result["findings"] = finding
            # Scrive su file dopo ogni iterazione
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    except json.JSONDecodeError:
        print(f"Warning: Failed to decode existing JSON file. Starting fresh.")


                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threats in the dataset.")
    parser.add_argument('--filename', type=str, help='Filename to process', required=True)
    parser.add_argument('--model', type=str, help='Model to use for the classification.', default='gpt-4.1')

    args = parser.parse_args()
    main(args.filename, args.model)


# python3 main_sa.py --filename 1_VARIANT_1_output