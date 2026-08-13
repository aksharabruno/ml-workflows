from dependency import *  # noqa: F401,F403


def model_generation_5(model_name, num_labels):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels=num_labels,
    )

    model.to("cuda")


    # Setup LoRA (PEFT) config
    lora_r = 8
    lora_alpha = 32
    target_modules = ["query", "value"]  # common for transformer attention layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    print("Applying LoRA / PEFT...")
    model = get_peft_model(model, lora_config)

    # Show parameters counts
    total_params, trainable_params = compute_params(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.4f} % )")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=args.report_to,
    )

    # Opt: use bitsandbytes 8-bit optimizer if available (lower CPU memory)
    optimizer = None
    if BNB_AVAILABLE:
        try:
            optimizer = AdamW8bit(model.parameters(), lr=args.learning_rate)
            print("Using bitsandbytes AdamW8bit optimizer")
        except Exception as e:
            print("Could not create AdamW8bit optimizer, falling back. Error:", e)
            optimizer = None

    return model, optimizer, training_args
