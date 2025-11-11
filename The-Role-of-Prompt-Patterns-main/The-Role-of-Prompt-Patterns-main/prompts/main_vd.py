from openai import OpenAI
from keys import OPEN_AI_KEY
from prompts import PROMPT_VD_VARIANT_1, PROMPT_VD_VARIANT_2, PROMPT_VD_VARIANT_3, ORIGINAL_PROMPT_VD, PROMPT_VD_FEW_SHOTS, PROMPT_VD_FEW_SHOTS_1, PROMPT_VD_FEW_SHOTS_2, PROMPT_VD_FEW_SHOTS_3, ORIGINAL_PROMPT_VD_RP
import os, argparse, json
# Imposta la tua chiave API di OpenAI

def esegui_prompt(model, formatted_prompt, temperature):
    try:
        
        client = OpenAI(
            api_key=OPEN_AI_KEY,
            organization="org-o7zOuoixFz94QuEm3bqKytHM",
            project="proj_CHdbaL1Dgy1GGODBmtcyXKRr")
        # setup utilizzato da me
        #msg = [{"role": "user", "content": formatted_prompt}]

        # setup utilizzato nel RP
        msg = [{"role": "system", "content": "You are a vulnerability detector for a smart contract."}]
        msg.append({"role": "user", "content": formatted_prompt})

        print("\nEseguendo il prompt con il modello:", msg)
        response = client.chat.completions.create(
            model = model,  # Specifica il modello che vuoi usare
            messages = msg,
            temperature = temperature  # Controlla la creatività della risposta
        )

        print("\nRisultato del prompt:")
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Errore durante l'esecuzione del prompt: {e}")
        return f"Errore durante l'esecuzione del prompt: {e}"

def main(iteration, prompt, dataset, output, model, temperature):
    PROMPTS = {
        "RP": ORIGINAL_PROMPT_VD_RP,
        "ORIGINAL": ORIGINAL_PROMPT_VD,
        "VARIANT_1": PROMPT_VD_VARIANT_1,
        "VARIANT_2": PROMPT_VD_VARIANT_2,
        "VARIANT_3": PROMPT_VD_VARIANT_3,
        "FEWSHOTS_1": PROMPT_VD_FEW_SHOTS_1,
        "FEWSHOTS_2": PROMPT_VD_FEW_SHOTS_2,
        "FEWSHOTS_3": PROMPT_VD_FEW_SHOTS_3,
        "FEWSHOTS": PROMPT_VD_FEW_SHOTS
    }
    output_file = f"../results/{iteration}_{prompt}_{output}.json"

    # Crea il file JSON se non esiste
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump([], f)
    else:
        print(f"Output file {output_file} already exists. Appending results.")

    # Carica i risultati esistenti
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
            processed_files = {item["file_name"] for item in results}
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode existing JSON file. Starting fresh.")
        results = []

    for category in [
        "access_control", "arithmetic", "bad_randomness", "denial_of_service",
        "front_running", "reentrancy", "short_addresses", "time_manipulation",
        "unchecked_low_level_calls"
    ]:
        print(f"Processing category: {category}")
        category_path = os.path.join(dataset, category)

        for file_name in os.listdir(category_path):
            if file_name.endswith(".sol"):
                file_path = os.path.join(category_path, file_name)
                with open(file_path, "r") as file:
                    code = file.read()
                tmp_prompt = PROMPTS[prompt]
                formatted_prompt = f"{tmp_prompt}".replace("{input}", code)
                #formatted_prompt = tmp_prompt.format(input=code)
                print(formatted_prompt)

                try:
                    if file_name in processed_files:
                        print(f"Skipping already processed file: {file_name}")
                        continue
                    else:
                        result = esegui_prompt(model=model, formatted_prompt=formatted_prompt, temperature=temperature)
                except Exception as e:
                    print(f"Error while processing {file_name}: {e}")
                    continue

                item = {
                    "iteration": iteration,
                    "model": model,
                    "dataset": dataset,
                    "prompt": prompt,
                    "file_name": file_name,
                    "category": category,
                    "result": result,
                    "findings": []
                }

                results.append(item)

                # Scrive su file dopo ogni iterazione
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threats in the dataset.")
    parser.add_argument('--dataset', type=str, help='Input dataset', default="/Users/gerardo/Desktop/AblationStudy/smartbugs-curated/dataset_cleaned/")
    parser.add_argument('--output', type=str, help='Output file name for the experiment', default="output")
    parser.add_argument('--model', type=str, help='Model to use for the classification.', default='gpt-4.1')
    parser.add_argument('--iteration', type=int, help='Number of the iteration of the experiment, e.g., 1, 2, ..., 5')
    parser.add_argument('--prompt', type=str, help='Prompt to use for the experiment, e.g., VARIANT_1, VARIANT_2, VARIANT_3')
    parser.add_argument('--temperature', type=str, help='temperature', default="0.7")

    
    args = parser.parse_args()
    main(args.iteration, args.prompt, args.dataset, args.output, args.model, args.temperature)

# python3 main_vd.py --iteration 1 --prompt VARIANT_1 --model gpt-4.1