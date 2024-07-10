
DATASET           = 'Kaludi/Customer-Support-Responses'
MODEL_NAME        = 'google-t5/t5-base'
# pretext           = "Generate meaningful customer support response: "
pretext           = "Assure the customer and provide specific help: "
max_source_length = 512
max_target_length = 128
BATCH_SIZE        = 4
learning_rate     = 1e-4 # According to the T5 docs, 1e-4 and 3e-4 work well for most problems
EPOCHS            = 10
SAVE_DIR          = './models'
SAVE_AS           = 't5-customer-support'
TRAINING_ARGS     = {
    'output_dir'                  : SAVE_DIR,
    'evaluation_strategy'         : 'epoch',
    'save_strategy'               : "epoch",
    'learning_rate'               : learning_rate,
    'per_device_train_batch_size' : BATCH_SIZE,
    'per_device_eval_batch_size'  : BATCH_SIZE,
    'num_train_epochs'            : EPOCHS,
    'weight_decay'                : 0.01,
    'save_total_limit'            : 2,
    'fp16'                        : True,
    'load_best_model_at_end'      : True,
    'report_to'                   : "tensorboard",
    'predict_with_generate'       : True,
    'generation_max_length'       : 50,
    # 'metric_for_best_model'       : "bleu",
    # 'greater_is_better'           : True,
}