import requests

# ==== CONFIGURATION ====
API_KEY = "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"  # Replace with your actual key
MODEL_NAME = "gpt-4o-mini"  # Example OpenRouter GPT-4 model

# ==== INPUT DATA (Replace with your Step 5 results) ====
usable_area = 45.0            # in square meters (m²)
solar_capacity = 6.9          # in kW
installation_cost = 483000    # in ₹
annual_savings = 66240        # in ₹
payback_period = 7.3          # in years

# ==== PROMPT TEMPLATE ====
prompt = f"""
Summarize the solar feasibility for a rooftop with:
- Usable area: {usable_area:.1f} m²
- Estimated solar capacity: {solar_capacity:.2f} kW
- Installation cost: ₹{installation_cost:,}
- Annual savings: ₹{annual_savings:,}
- Payback period: {payback_period:.1f} years

Provide a concise report highlighting the potential benefits and any concerns for solar panel installation.
"""

def generate_summary_openrouter(api_key, model, prompt_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
        }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant that summarizes solar feasibility reports."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

def main():
    summary = generate_summary_openrouter(API_KEY, MODEL_NAME, prompt)
    print("\n===== Solar Feasibility Summary =====\n")
    print(summary)

if __name__ == "__main__":
    main()
