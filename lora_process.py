import subprocess
import re
from datetime import datetime, timedelta
import time
from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import json
def nc_imdb_eval(text):

    def extract_numbers(label, index):
        numbers = re.findall(r'\d+', label)
        if len(numbers) == 0:
            print(index, label)
            return [0]
        return [int(number) for number in numbers]

    labels = []
    preds = []
    for i in range(len(text)):
        preds.append(extract_numbers(text[i]['predict'], i))
        labels.append(extract_numbers(text[i]['label'], i))


    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    preds = mlb.transform(preds)

    ma = f1_score(labels, preds, average='macro')
    mi = f1_score(labels, preds, average='micro')
    print(f"ma: {ma * 100}%")
    print(f"mi: {mi * 100}%")


def nc_arxiv_eval(text):
    cs_list = [
        'cs.NA', 'cs.MM', 'cs.LO', 'cs.CY', 'cs.CR', 'cs.DC', 'cs.HC', 'cs.CE', 'cs.NI', 'cs.CC',
        'cs.AI', 'cs.MA', 'cs.GL', 'cs.NE', 'cs.SC', 'cs.AR', 'cs.CV', 'cs.GR', 'cs.ET', 'cs.SY',
        'cs.CG', 'cs.OH', 'cs.PL', 'cs.SE', 'cs.LG', 'cs.SD', 'cs.SI', 'cs.RO', 'cs.IT', 'cs.PF',
        'cs.CL', 'cs.IR', 'cs.MS', 'cs.FL', 'cs.DS', 'cs.OS', 'cs.GT', 'cs.DB', 'cs.DL', 'cs.DM'
    ]

    def extract_numbers(label, index):
        numbers = re.findall(r'\d+', label)
        if len(numbers) == 0:
            print(index, label)
            return [0]
        return [int(number) for number in numbers]

    def find_indices(input_string, index):
        indices = []
        for item in cs_list:
            if item in input_string:
                indices.append(cs_list.index(item) + 1)
        if len(indices) == 0:
            return extract_numbers(input_string, index)
            # return [0]
        return indices

    labels = []
    preds = []
    for i in range(len(text)):
        x = find_indices(text[i]['predict'], i)[0]
        y = extract_numbers(text[i]['label'], i)[0]
        if x > 40:
            x = 0
        preds.append(x)
        labels.append(y)


    ma = f1_score(labels, preds, average='macro')
    mi = f1_score(labels, preds, average='micro')

    print(f"ma: {ma * 100}%")
    print(f"mi: {mi * 100}%")


def eval_result(name, text):
    if "nc_arxiv" in name:
        nc_arxiv_eval(text)
    if "nc_imdb" in name:
        nc_imdb_eval(text)


path="/xxx/xxx"


CUDA_VISIBLE_DEVICES = 0
llama_model = "Meta-Llama-3-8B"
for data_item in [["train_mix_dataset",2000,2000,1000]]:
    # ---------------------------Train---------------------------
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
    adapter_name = ""
    dataset = data_item[0]
    num = data_item[1]

    epochs = 2
    output_dir = f"saves/{llama_model}/lora/lora_{formatted_time}-checkpoint-{round(num * 0.99) * epochs}"

    save_steps = data_item[2]
    eval_steps = data_item[3]
    max_samples = 120000
    cutoff_len = 1200
    train_cmd = (
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u {path}/LLaMA-Factory/src/train_bash.py --stage sft  --do_train  --model_name_or_path "
        f"{path}/LM/{llama_model} "
        f"{adapter_name}  --dataset "
        f"{dataset}  --dataset_dir data  --template alpaca  "
        "--finetuning_type lora  --lora_target q_proj,v_proj  --output_dir "
        f"{output_dir}  --overwrite_cache  --overwrite_output_dir  --cutoff_len "
        f"{cutoff_len}  --preprocessing_num_workers 16  --per_device_train_batch_size 1  --per_device_eval_batch_size 1  "
        "--gradient_accumulation_steps 1  --lr_scheduler_type cosine  --logging_steps 10  --warmup_steps 20  "
        f"--save_steps {save_steps}  --eval_steps {eval_steps}  --evaluation_strategy steps  --load_best_model_at_end  "
        f"--learning_rate 5e-5 --lora_rank 32  --lora_alpha 64  --lora_dropout 0.05 --num_train_epochs {epochs}  "
        f"--max_samples {max_samples}  --val_size 0.01 --plot_loss")

    train_process = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = train_process.communicate()

    output = output.decode("utf-8")
    error = error.decode("utf-8")
    with open("out/log-train-model.txt", 'w') as file:
        file.write(output + '/n' + error)

    train_result = output.split("***** train metrics *****")[1].split("Figure saved at")[0]

    # ---------------------------Test---------------------------
    start_time = time.time()

    result = []
    test_datasets = ["test_nc_imdb","test_nc_arxiv"]
    for index,test_dataset in enumerate(test_datasets):
        mix_test_output_path = f"{path}/LLaMA-Factory/saves/{llama_model}/base-predict/lora_{formatted_time}-checkpoint-{round(num * 0.99) * epochs}/{test_dataset}"
        mix_test_cmd = (
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u {path}/LLaMA-Factory/src/train_bash.py --stage sft --do_predict --model_name_or_path "
            f"{path}/LM/{llama_model} "
            f"--adapter_name_or_path {output_dir}"
            f" --dataset {test_dataset} "
            " --dataset_dir data --template alpaca --output_dir "
            f"{mix_test_output_path} --do_sample true "
            "--temperature 0.2 --overwrite_cache --overwrite_output_dir --cutoff_len 1200 "
            "--per_device_eval_batch_size 1 --max_new_tokens 800 --predict_with_generate")


        test_mix_process = subprocess.Popen(mix_test_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = test_mix_process.communicate()
        output = output.decode("utf-8")
        with open("out/log-train-test_mix_process.txt", 'w') as file:
            file.write(output)

        mix_dataset_text = []
        result.append(mix_dataset_text)
        text=mix_dataset_text
        name=test_datasets[index]
        eval_result(name,text)

    # ---------------------------Evaluate---------------------------
    for i in range(len(result)):
        text=result[i]
        name=test_datasets[i]
        eval_result(name,text)

    print("*"*200)


end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time = timedelta(seconds=execution_time_seconds)
formatted_time = str(execution_time)



print("test time：", formatted_time)
