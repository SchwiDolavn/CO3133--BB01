import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients, visualization
import os

def explain_prediction_lime(text, model, tokenizer, label_encoder, scenario_name="model", device="cpu"):
    model.to(device)
    model.eval()

    class_names = list(label_encoder.classes_)
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()} 
    
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        return probs.cpu().numpy()
    
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=500)
    
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME - Case: {scenario_name}")
    plt.tight_layout()
    file_name = f'../result/lime_explanation_{scenario_name}.png'
    plt.savefig(file_name)
    plt.close()

def explain_prediction_captum(text, model, tokenizer, label_encoder, scenario_name, model_type, device="cpu"):
    model.to(device)

    if 'Bi-LSTM' in model_type:
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
    else:
        model.eval()
    
    for param in model.parameters():
        param.requires_grad = True
        
    def custom_forward(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class_idx = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0, pred_class_idx].item()

    def custom_forward(inputs_ids, mask):
        out = model(input_ids=inputs_ids, attention_mask=mask)
        return out.logits if hasattr(out, 'logits') else out[0]

    if 'DistilBERT' in model_type:
        embed_layer = model.distilbert.embeddings
    elif 'BERT' in model_type:
        embed_layer = model.bert.embeddings
    elif 'Bi-LSTM' in model_type:
        embed_layer = model.embedding
    else:
        raise ValueError("Không tìm thấy lớp Embedding cho mô hình này.")

    lig = LayerIntegratedGradients(custom_forward, embed_layer)

    ref_token_id = tokenizer.pad_token_id
    baselines = input_ids * 0 + ref_token_id

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        additional_forward_args=(attention_mask,),
        target=pred_class_idx,
        return_convergence_delta=True
    )

    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    attributions_sum = attributions_sum / torch.norm(attributions_sum)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    vis_record = visualization.VisualizationDataRecord(
        word_attributions=attributions_sum.cpu().detach().numpy(),
        pred_prob=pred_prob,
        pred_class=label_encoder.classes_[pred_class_idx],
        true_class="N/A", # Trong thực tế lúc test ta không cần show nhãn thật ở đây
        attr_class=label_encoder.classes_[pred_class_idx],
        attr_score=attributions_sum.sum().item(),
        raw_input_ids=tokens,
        convergence_score=delta.item()
    )

    html_content = visualization.visualize_text([vis_record])
    html_str = html_content.data 
    
    file_path = f"../result/captum_explanation_{scenario_name}.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_str)