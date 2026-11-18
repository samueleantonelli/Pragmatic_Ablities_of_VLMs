import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def extract_features(df, images_dir, model, processor, target_layers, text_only=False):
    # Extract last-token hidden states across target layers
    layer_embeddings = {l: [] for l in target_layers}

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Extracting (text_only={text_only})",
    ):
        description = row["description"]

        if text_only:
            messages = [{"role": "user", "content": [{"type": "text", "text": description}]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(text=[text_input], return_tensors="pt").to("cuda")
        else:
            image_path = f"{images_dir}/{row['image_file']}"
            raw_image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": description},
                    ],
                }
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[text_input],
                images=[raw_image],
                return_tensors="pt",
            ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Hidden states: tuple of shape (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states

            for l in target_layers:
                # Last token bottleneck representation
                vec = hidden_states[l][0, -1, :].cpu().to(torch.float32).numpy()
                layer_embeddings[l].append(vec)

    for l in target_layers:
        layer_embeddings[l] = np.array(layer_embeddings[l])

    return layer_embeddings


def fit_probe(X_train, y_train):
    # Linear probe with class balancing (default regularisation)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf


def evaluate_probe(clf, X, y):
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    return {
        "acc": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "auroc": roc_auc_score(y, probs),
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--preds_out", type=str, default="test_probe_predictions.csv")
    parser.add_argument("--layer_selection_out", type=str, default="layer_selection_val.csv")
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Fix the layer instead of selecting it on the validation split",
    )
    args = parser.parse_args()

    # Load splits (must be partitioned strictly by image_id to prevent leakage)
    df = pd.read_csv(args.data_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    y_train = train_df["discriminative_label"].values
    y_val = val_df["discriminative_label"].values
    y_test = test_df["discriminative_label"].values

    # Model layers: even layers up to 26, plus final layer 27
    target_layers = list(range(0, 28, 2)) + [27]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 1. Multimodal embeddings for all layers, on train / validation / test
    feats_train_mm = extract_features(
        train_df, args.images_dir, model, processor, target_layers, text_only=False
    )
    feats_val_mm = extract_features(
        val_df, args.images_dir, model, processor, target_layers, text_only=False
    )
    feats_test_mm = extract_features(
        test_df, args.images_dir, model, processor, target_layers, text_only=False
    )

    # 2. Layer selection on the VALIDATION split only: highest validation AUROC.
    #    The test split is not used for this choice.
    val_scores = []
    for l in target_layers:
        clf = fit_probe(feats_train_mm[l], y_train)
        val_scores.append(
            {"layer": l, "val_auroc": evaluate_probe(clf, feats_val_mm[l], y_val)["auroc"]}
        )
    val_scores_df = pd.DataFrame(val_scores)
    val_scores_df.to_csv(args.layer_selection_out, index=False)

    if args.layer is None:
        best_layer = int(val_scores_df.loc[val_scores_df["val_auroc"].idxmax(), "layer"])
    else:
        best_layer = args.layer
    print(f"\nSelected layer (highest validation AUROC): {best_layer}")
    print(f"Validation AUROC per layer saved to {args.layer_selection_out}")

    # 3. Test-set results at the selected layer
    print(f"\n--- Layer {best_layer} Probing Results (test split) ---")
    clf_mm = fit_probe(feats_train_mm[best_layer], y_train)
    res_mm = evaluate_probe(clf_mm, feats_test_mm[best_layer], y_test)
    preds_mm = res_mm["preds"]
    print(f"Multimodal (Img+Text)  -> Acc: {res_mm['acc']:.2f}, F1: {res_mm['f1']:.2f}, AUROC: {res_mm['auroc']:.2f}")

    # Random labels control: same layer, shuffled training labels
    rng = np.random.default_rng(0)
    y_train_rand = rng.permutation(y_train)
    clf_rnd = fit_probe(feats_train_mm[best_layer], y_train_rand)
    res_rnd = evaluate_probe(clf_rnd, feats_test_mm[best_layer], y_test)
    print(f"Random Labels Control  -> Acc: {res_rnd['acc']:.2f}, F1: {res_rnd['f1']:.2f}, AUROC: {res_rnd['auroc']:.2f}")

    # Text-only ablation: same layer, no image
    feats_train_to = extract_features(
        train_df, args.images_dir, model, processor, [best_layer], text_only=True
    )
    feats_test_to = extract_features(
        test_df, args.images_dir, model, processor, [best_layer], text_only=True
    )
    clf_to = fit_probe(feats_train_to[best_layer], y_train)
    res_to = evaluate_probe(clf_to, feats_test_to[best_layer], y_test)
    preds_to = res_to["preds"]
    print(f"Text-Only Ablation     -> Acc: {res_to['acc']:.2f}, F1: {res_to['f1']:.2f}, AUROC: {res_to['auroc']:.2f}")

    # Per-trial test predictions, input to analysis/error_and_linguistic.py
    pred_df = pd.DataFrame({
        "image_file": test_df["image_file"].values,
        "description": test_df["description"].values,
        "discriminative_label": y_test,
        "mm_pred": preds_mm,
        "to_pred": preds_to,
    })
    pred_df["mm_correct"] = (pred_df["mm_pred"] == pred_df["discriminative_label"]).astype(int)
    pred_df["to_correct"] = (pred_df["to_pred"] == pred_df["discriminative_label"]).astype(int)
    pred_df.to_csv(args.preds_out, index=False)
    print(f"Per-trial predictions saved to {args.preds_out}")


if __name__ == "__main__":
    main()
