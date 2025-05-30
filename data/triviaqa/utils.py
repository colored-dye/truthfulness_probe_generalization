def get_few_shot_prompt(few_shot_dataset, template, rng, n_shots):
    few_shot_indices = rng.choice(len(few_shot_dataset), n_shots, replace=False).tolist()
    few_shot_prompt = ""
    for i, ind in enumerate(few_shot_indices):
        rec = few_shot_dataset[ind]
        text = template.format(question=rec['question'], answer=rec['answer']['normalized_value'])
        few_shot_prompt += text + "\n\n"
    return few_shot_prompt
