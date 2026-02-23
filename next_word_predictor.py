# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "transformers",
#     "torch",
#     "plotly",
# ]
# ///
# __generated_with = "0.19.11"

import marimo

__generated_with = "0.19.11"
app = marimo.App(app_title="Monkey_LLM")

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # LLMs: Advanced Autocomplete ..|

    Okay, I admit the title is a bit clickbait-y. Many argue that Large Language Models (LLMs) can reason, exhibit thoughtfulness, and perform complex tasks far beyond simple autocomplete. But when you strip down all the engineering aroudn it, **LLMs are just an autocomplete tool**.

    My motive here is to understand LLMs underlying mechanics helps us grasp its true potential and limitations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## "Monkey See, Monkey Do" 🐒
    ---
    I like to think of LLMs as incredibly advanced versions of the predictive text on your phone. Despite appearances, an LLM doesn't "know" facts or "understand" feelings like a human. Instead, it's a **massive pattern recognition engine**.

    Imagine an LLM as a **highly intelligent, very observant monkey**.
    """)
    return


@app.cell
def _():
    mo.image(src="static/img1.png")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### How the Monkey Learns
    ---
    The journey from a chaotic monkey to a genius, using banana rewards, starts here with :
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "**1. Untrained AI**": mo.md(
                """
                You start with a highly intelligent monkey with a perfect, photographic memory, but zero knowledge of the world. At this stage, if you ask it to type, it will only produce random gibberish.
                """
            ),
            "**2. Training**": mo.md(
                """
                You give the monkey millions of books, articles, and websites. You play a game: show it a sentence with the last word missing, and it has to guess that word. If it predicts correctly, it gets a banana. After billions of repetitions, the monkey memorizes the statistical patterns of human language. It doesn't know what a sky is, or what blue looks like, but it knows with 99% certainty that "blue" follows "The sky is..."
                """
            ),
            "**3. Fine-Tuning**": mo.md(
                """
                Now the monkey knows how to speak perfectly, but it just babbles randomly based on internet patterns. You need to teach it to be a helpful assistant. You show it specific examples of questions and answers to teach it the specific pattern of a Q&A conversation.
                """
            ),
            "**4. RLHF (Reinforcement Learning from Human Feedback)**": mo.md(
                """
                To make the monkey safe and helpful, you bring in human graders. You ask: "How do I bake a cake?" and the monkey types three responses. You give it bananas for a clear, step-by-step recipe and no bananas for a rude or dangerous response.
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Illusion of Genius ✨
    ---

    When you type in a complex question, the monkey looks at its massive memory, remembers what types of answers earned the most bananas, and types out a brilliant response. It seems like a genius, but it's really just playing the ultimate pattern-matching game.

    Ultimately, LLMs predict the most likely next word based on their training. They match patterns and imitate human language without having a deep understanding of the concepts themselves.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Try it yourself!
    ---
    Experiment with different models and see the word predictions for yourself.
    """)
    return


@app.cell
def _():
    model_options = ["gpt2", "gpt2-xl", "meta-llama/Llama-3-8B", "Qwen/Qwen2-7B"]
    model_selector = mo.ui.dropdown(
        options=model_options, value="gpt2", label="Select Monkey: "
    )

    text_input = mo.ui.text(
        placeholder="Type a sentence fragment...",
        label="Input Text: ",
        full_width=True,
    )

    n_words = mo.ui.slider(start=1, stop=20, label="Top K Predictions", value=5)

    n_gen_words = mo.ui.slider(start=1, stop=5, label="Words to Generate", value=1)

    predict_button = mo.ui.run_button(label="Predict Word", kind="success")

    controls = mo.vstack(
        [
            mo.md("### Interactive Controls"),
            mo.hstack([model_selector, n_words, n_gen_words], justify="start"),
            text_input,
            predict_button,
        ]
    ).callout()

    controls
    return model_selector, n_gen_words, n_words, predict_button, text_input


@app.function
@mo.cache
def load_llm_model(
    model_name: str, cache_dir: str = ".model_cache/", verbose: bool = False
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=str(cache_path)
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model


@app.function
def get_model_info(model, tokenizer, model_name: str):
    params = model.num_parameters()
    vocab_size = len(tokenizer)
    return {
        "Model Name": model_name,
        "Parameters": f"{params:,}",
        "Vocabulary Size": f"{vocab_size:,}",
        "Device": str(model.device),
    }


@app.cell
def _predict_next_word(n_gen_words, n_words):
    def predict_next_word(
        model,
        tokenizer,
        text_input_value: str,
    ) -> list[dict]:
        import torch

        input_text = text_input_value.strip()
        if not input_text:
            return []

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get Top K candidates for the FIRST next word
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k=n_words.value)

        results = []
        for i in range(n_words.value):
            prob = top_probs[i].item()
            token_id = top_indices[i].unsqueeze(0)

            # Start the sequence with this candidate
            current_ids = torch.cat([inputs.input_ids[0], token_id], dim=0)

            # Generate subsequent words greedily
            if n_gen_words.value > 1:
                # Use model.generate for simplicity for the remaining tokens
                generated = model.generate(
                    current_ids.unsqueeze(0),
                    max_new_tokens=n_gen_words.value - 1,
                    do_sample=False, # Greedy
                    pad_token_id=tokenizer.eos_token_id
                )
                full_seq_ids = generated[0]
            else:
                full_seq_ids = current_ids

            # Decode only the generated part
            # Skip the input_text part
            generated_tokens = full_seq_ids[len(inputs.input_ids[0]):]
            generated_text = tokenizer.decode(generated_tokens).strip()
            # Convert to list of words
            word_list = generated_text.split()

            results.append({
                "Word": word_list,
                "Probability": round(prob, 6),
                "Probability (%)": round(prob * 100, 4),
            })

        return results

    return (predict_next_word,)


@app.cell
def _(model_selector, predict_button, predict_next_word, text_input):
    prediction_results = []
    model_info = None
    if predict_button.value and text_input.value.strip():
        with mo.status.spinner(title="Loading model...") as _spinner:
            # Load model and tokenizer
            tokenizer, model = load_llm_model(model_selector.value)
            model_info = get_model_info(model, tokenizer, model_selector.value)
            _spinner.update("Predicting next word...")

            # Do the prediction
            prediction_results = predict_next_word(
                model, tokenizer, text_input.value
            )
    return model_info, prediction_results


@app.cell
def _(model_info, prediction_results, text_input):
    import plotly.graph_objects as go

    if prediction_results:
        # Create Plotly Chart
        # Join words back into strings for the chart's labels
        words = [" ".join(r["Word"]) for r in prediction_results][::-1]
        probs = [r["Probability (%)"] for r in prediction_results][::-1]

        fig = go.Figure(
            go.Bar(
                x=probs,
                y=words,
                orientation="h",
                marker_color="Red",
                text=[f"{p:.2f}%" for p in probs],
                textposition="auto",
                hovertemplate="Word: %{y}<br>Probability: %{x}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(text="Top Predictions", font=dict(size=18)),
            xaxis_title="Probability (%)",
            yaxis_title="Next Word(s)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,
            xaxis=dict(range=[0, 100]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Use the highlight function for each word in the prediction
        most_likely_words = prediction_results[0]["Word"]
        highlighted_words = " ".join([highlight(w) for w in most_likely_words])

        full_sentence = f"{text_input.value.strip()} {highlighted_words}"

        if model_info:
            info_card = mo.vstack(
                [
                    mo.md("### Model Details"),
                    mo.ui.table(
                        [
                            {"Detail": k, "Value": v}
                            for k, v in model_info.items()
                        ],
                        label="Model and Tokenizer Information",
                        selection=None,
                    ),
                ]
            ).callout(kind="info")


        chart_card = mo.vstack(
            [
                mo.md(f"### Prediction: {full_sentence}"),
                info_card,
                mo.ui.plotly(fig),
            ]
        ).callout()
    else:
        chart_card = mo.md("")

    chart_card
    return


@app.function(hide_code=True)
def highlight(text):
    import random 
    r = lambda: random.randint(0,255)
    color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
    return f"<span style='background-color: {color}; padding: 0 4px; border-radius: 4px;'>{text}</span>"


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
