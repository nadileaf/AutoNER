import os
from invoke import run

ROOT = os.path.dirname(__file__)
GREEN = "`tput setaf 2`"
COLOR_RESET = "`tput sgr0`"


# ==============================  Required  =============================
# PROJECT_NAME = 'BC5CDR'
PROJECT_NAME = 'PEOPLE'

DATA_ROOT = os.path.join(ROOT, 'data', PROJECT_NAME)
DATA_RAW_ROOT = os.path.join(ROOT, 'data', PROJECT_NAME, 'required')
DATA_GENERATE_ROOT = os.path.join(ROOT, 'data', PROJECT_NAME, 'generate')
EMBEDDING_ROOT = os.path.join(ROOT, 'embedding')
MODEL_ROOT = os.path.join(ROOT, 'models', PROJECT_NAME)

TRAINING_TEXT = os.path.join(DATA_RAW_ROOT, 'training.txt')
DEV_TEXT = os.path.join(DATA_RAW_ROOT, 'dev.txt')
TEST_TEXT = os.path.join(DATA_RAW_ROOT, 'test.txt')
DICT_CORE = os.path.join(DATA_RAW_ROOT, 'dict_core.txt')
DICT_FULL = os.path.join(DATA_RAW_ROOT, 'dict_full.txt')

EMBEDDING_TXT_FILE = os.path.join(EMBEDDING_ROOT, 'embedding_zh_300.txt')
# ========================================================================


# ==============================  Generated DataSet  =============================
EMBEDDING_PKL_FILE = os.path.join(EMBEDDING_ROOT, 'embedding_zh_300.pk')

TRAINING_SET = os.path.join(DATA_GENERATE_ROOT, 'truth_train.ck')
DEV_SET = os.path.join(DATA_GENERATE_ROOT, 'truth_dev.ck')
TEST_SET = os.path.join(DATA_GENERATE_ROOT, 'truth_test.ck')

TRAINING_PKL_FILE = os.path.join(DATA_GENERATE_ROOT, 'train_0.pk')
DEV_PKL_FILE = os.path.join(DATA_GENERATE_ROOT, 'dev.pk')
TEST_PKL_FILE = os.path.join(DATA_GENERATE_ROOT, 'test.pk')

PRED_TEST_TXT_FILE = os.path.join(DATA_ROOT, 'generate/pred_test_text.txt')

CHECKPOINT_DIR = os.path.join(MODEL_ROOT, 'checkpoint')
CHECKPOINT_NAME = 'autoner'
# ========================================================================

if not os.path.isfile(EMBEDDING_PKL_FILE):
    run(f"echo {GREEN}=== Pickle Embedding Vectors ==={COLOR_RESET}")
    run(f"python preprocess_partial_ner/save_emb.py --input_embedding {EMBEDDING_TXT_FILE} --output_embedding {EMBEDDING_PKL_FILE}")

run(f"mkdir -p {MODEL_ROOT}", hide=True, warn=True)
run("make", hide=True, warn=True)

if not os.path.isfile(TRAINING_SET):
    run(f"echo {GREEN}=== Generate Training Dataset from Dictionaries ==={COLOR_RESET}")
    run(f"bin/generate {TRAINING_TEXT} {DICT_CORE} {DICT_FULL} {TRAINING_SET}")
    run(f"echo {GREEN}=== Generate DEV Dataset from Dictionaries ==={COLOR_RESET}")
    run(f"bin/generate {DEV_TEXT} {DICT_CORE} {DICT_FULL} {DEV_SET}")
    run(f"echo {GREEN}=== Generate TEST Dataset from Dictionaries ==={COLOR_RESET}")
    run(f"bin/generate {TEST_TEXT} {DICT_CORE} {DICT_FULL} {TEST_SET}")

if not os.path.isfile(TRAINING_PKL_FILE):
    run(f"echo {GREEN}=== Encoding Dataset ==={COLOR_RESET}")
    run(f"""
        python preprocess_partial_ner/encode_folder.py \
         --input_train {TRAINING_SET} \
         --input_testa {DEV_SET} \
         --input_testb {TEST_SET} \
         --pre_word_emb {EMBEDDING_PKL_FILE} \
         --output_folder {DATA_GENERATE_ROOT}/
         """)


run(f"echo {GREEN}=== Training AutoNER Model ==={COLOR_RESET}")
run(f"""
python train_partial_ner.py \
    --cp_root {CHECKPOINT_DIR} \
    --checkpoint_name {CHECKPOINT_NAME} \
    --eval_dataset {TEST_PKL_FILE} \
    --train_dataset {TRAINING_PKL_FILE} \
    --update SGD \
    --lr 0.05 \
    --hid_dim 300 \
    --droprate 0.5 \
    --sample_ratio 1.0 \
    --word_dim 300 \
    --epoch 50
""")


run(f"echo {GREEN}=== Encode Training Dataset ==={COLOR_RESET}")
run(f"""
python preprocess_partial_ner/encode_test.py \
    --input_data {TRAINING_TEXT} \
    --checkpoint_folder {CHECKPOINT_DIR}/{CHECKPOINT_NAME} \
    --output_file {TEST_PKL_FILE}
""")


run(f"echo {GREEN}=== Evaluate AutoNER Model ==={COLOR_RESET}")
run(f"""
python test_partial_ner.py \
    --input_corpus {TEST_PKL_FILE} \
    --checkpoint_folder {CHECKPOINT_DIR}/{CHECKPOINT_NAME} \
    --output_text {PRED_TEST_TXT_FILE} \
    --hid_dim 300 \
    --droprate 0.5 \
    --word_dim 200
""")
